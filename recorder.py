import vmas
import cbor2
from tqdm import tqdm
import torch
import numpy as np
import os
import glob
import re
from tensordict import TensorDict

# BenchMARL / TorchRL imports
try:
    from benchmarl.algorithms import MappoConfig, MasacConfig, QmixConfig, IppoConfig
    from benchmarl.models.mlp import MlpConfig
    from torchrl.envs.libs.vmas import VmasEnv
except ImportError:
    print("Warning: BenchMARL/TorchRL not installed. BenchMARL models will not work.")

class Env:
    def __init__(self, env_kwargs):
        self.env = vmas.make_env(**env_kwargs)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        actions = list(actions)
        return self.env.step(actions)
    
    def agents(self):
        return self.env.agents
    
    def landmarks(self):
        return self.env.world.landmarks

class Model:
    def __init__(self, name, env_kwargs, env):
        self.device = env_kwargs.get('device', 'cpu')
        scenario_name = env_kwargs.get('scenario')
        self.num_agents = env_kwargs.get('n_agents', 1)
        self.num_envs = env_kwargs.get('num_envs', 1)
        self.name = name
        
        # Legacy key mapping for your custom models
        if scenario_name == "simple_spread":
            key = "spread"
        elif scenario_name == "food_collection":
            key = "collection"
        else:
            key = scenario_name

        # ====================================================================
        # BENCHMARL MODELS (MAPPO, QMIX, IPPO, MASAC)
        # ====================================================================
        if name in ["mappo", "ippo", "qmix", "masac"]:
            print(f"Loading BenchMARL model: {name} for {scenario_name}...")
            
            # 1. Find the latest checkpoint
            # Assuming structure: baselines/results/{task_name}/{algo}_{task}_mlp_{seed}/checkpoints/
            # If running from 'baselines/' dir, search 'results/'
            search_path = os.path.join("results", scenario_name)
            
            if not os.path.exists(search_path):
                # Fallback: maybe running from root?
                search_path = os.path.join("baselines", "results", scenario_name)
            
            # Find experiment folder (e.g. mappo_simple_spread_mlp_...)
            # We look for folders starting with the algo name
            exp_folders = glob.glob(os.path.join(search_path, f"{name}*"))
            
            if not exp_folders:
                raise FileNotFoundError(f"No experiment folder found for '{name}' in {search_path}")
            
            # Pick the most recent experiment folder
            latest_exp_folder = max(exp_folders, key=os.path.getmtime)
            print(f"  Experiment dir: {latest_exp_folder}")
            
            # Find checkpoints inside
            ckpt_pattern = os.path.join(latest_exp_folder, "checkpoints", "*.pt")
            checkpoints = glob.glob(ckpt_pattern)
            
            if not checkpoints:
                raise FileNotFoundError(f"No .pt checkpoints found in {latest_exp_folder}")
            
            # Sort by checkpoint step number (checkpoint_12345.pt)
            def extract_step(path):
                match = re.search(r"checkpoint_(\d+).pt", path)
                return int(match.group(1)) if match else -1
            
            latest_ckpt = max(checkpoints, key=extract_step)
            print(f"  Checkpoint: {latest_ckpt}")

            # 2. Reconstruct Policy Architecture
            if name == "mappo":
                algo_config = MappoConfig.get_from_yaml()
            elif name == "ippo":
                algo_config = IppoConfig.get_from_yaml()
            elif name == "qmix":
                algo_config = QmixConfig.get_from_yaml()
            elif name == "masac":
                algo_config = MasacConfig.get_from_yaml()
            
            # Must match training config [256, 256]
            model_config = MlpConfig.get_from_yaml()
            model_config.num_cells = [256, 256]

            # Create dummy env to get the input/output specs
            dummy_env = VmasEnv(
                scenario=scenario_name,
                num_envs=self.num_envs,
                n_agents=self.num_agents,
                continuous_actions=True, 
                device=self.device,
                **{k:v for k,v in env_kwargs.items() if k not in ['scenario', 'num_envs', 'n_agents', 'device', 'continuous_actions']}
            )

            # Build the algorithm to get the policy module
            algorithm = algo_config.get_algorithm(
                experiment=None, 
                env_spec=dummy_env.full_spec, 
                model_config=model_config
            )
            
            # Get the policy (Actor)
            self.actor = algorithm.get_policy_for_collection()

            # 3. Load Weights
            loaded_dict = torch.load(latest_ckpt, map_location=self.device)
            
            # BenchMARL saves the full state; we need to extract the model weights
            if 'model_state_dict' in loaded_dict:
                try:
                    algorithm.load_state_dict(loaded_dict['model_state_dict'])
                except RuntimeError:
                    # Robust loading: filter for matching keys
                    print("  Warning: Strict load failed, attempting partial load...")
                    curr_state = algorithm.state_dict()
                    filtered_dict = {k: v for k, v in loaded_dict['model_state_dict'].items() 
                                     if k in curr_state and v.shape == curr_state[k].shape}
                    algorithm.load_state_dict(filtered_dict, strict=False)
            else:
                 self.actor.load_state_dict(loaded_dict)

            self.actor.eval()
            self.actor.to(self.device)

        # ====================================================================
        # EXISTING MODELS (GSA, pH-MARL, InfoMARL)
        # ====================================================================
        elif name == "GSA":
            # Assuming local files
            self.actor = torch.load('./models/GSA_' + key + '.pth', map_location=self.device)
            self.actor.na = self.num_agents
            self.actor.eval()
        elif name == "pH-MARL":
            self.actor = torch.load('./models/pH-MARL_' + key + '.pth', map_location=self.device)
            self.actor.na = self.num_agents
            self.actor.eval()
        elif "infomarl" in name:
            from stable_baselines3 import PPO
            from wrapper import VMASVecEnv
            
            # Handle naming variants (infomarl, infomarl_nomask, etc.)
            variant = name.replace("infomarl", "")
            if variant.startswith("_"): variant = variant[1:]
            
            # Construct filename
            fname_key = f"{variant}_{key}" if variant else key
            filename = f'./models/infomarl_{fname_key}.zip'.replace("__", "_")
            
            print(f"Loading InfoMarl model from: {filename}")
            
            # Wrap env for SB3 compatibility
            vec_env = VMASVecEnv(env.env, rnd_nums=True) 
            model = PPO.load(filename, map_location=self.device)
            model.policy.to(self.device)
            
            # Patch policy specs
            model.policy.observation_space = vec_env.observation_space
            model.policy.action_space = vec_env.action_space
            
            if hasattr(model.policy, "pi_features_extractor"):
                model.policy.pi_features_extractor.actor.number_agents = self.num_agents
                model.policy.pi_features_extractor.actor.number_food = env_kwargs.get('n_food', 0)
            
            model.policy.eval()
            self.actor = model

            # Fix for log_std if missing
            with torch.no_grad():
                if not hasattr(model.policy, "log_std"):
                     model.policy.log_std = torch.nn.Parameter(
                        torch.zeros(np.prod(vec_env.action_space.shape), dtype=torch.float32)
                    )
        else:
            raise ValueError(f"Unknown model name: {name}")
        
    
    def __call__(self, obs):
        with torch.no_grad():
            # ----------------------------------------------------------------
            # BENCHMARL INFERENCE
            # ----------------------------------------------------------------
            if self.name in ["mappo", "ippo", "qmix", "masac"]:
                # Stack obs list [(n_envs, dim), ...] -> (n_envs, n_agents, dim)
                stacked_obs = torch.stack(obs, dim=1).to(self.device)
                
                # Wrap in TensorDict
                input_td = TensorDict({
                    "agents": TensorDict({
                        "observation": stacked_obs
                    }, batch_size=self.num_envs, device=self.device)
                }, batch_size=self.num_envs, device=self.device)
                
                # Forward pass
                output_td = self.actor(input_td)
                
                # Extract actions
                if "action" in output_td["agents"].keys():
                    actions = output_td["agents", "action"]
                elif "loc" in output_td["agents"].keys():
                    actions = output_td["agents", "loc"]
                else:
                     raise KeyError("Could not find action/loc in model output")

                # Transpose back to VMAS format: (n_agents, n_envs, dim)
                return actions.transpose(0, 1)

            # ----------------------------------------------------------------
            # INFOMARL INFERENCE
            # ----------------------------------------------------------------
            elif "infomarl" in self.name:
                rnd_nums = torch.rand(self.num_envs, self.num_agents, device=self.device).unsqueeze(-1)
                obs = torch.stack(obs, dim=0).transpose(1, 0)
                obs = torch.cat([obs, rnd_nums], dim=-1)
                actions, _ = self.actor.predict(obs, deterministic=True)
                return actions.transpose(1, 0, 2)
            
            # ----------------------------------------------------------------
            # GSA / pH-MARL INFERENCE
            # ----------------------------------------------------------------
            else:
                obs = torch.stack(obs).transpose(1, 0).to(self.device)
                actions = self.actor(obs).mean
                return actions.transpose(1, 0)

def record_episodes(fp, env_kwargs, model_name, max_steps):
    cbor2.dump(env_kwargs, fp)
    cbor2.dump({"model": model_name, "max_steps": max_steps}, fp)

    env = Env(env_kwargs)
    model = Model(model_name, env_kwargs, env)
    obs = env.reset()
    
    for step in tqdm(range(max_steps), desc=f"Recording {model_name}"):
        actions = model(obs)
        
        # Extract visual data
        agent_data = {e.name: e.state.pos.cpu().numpy().tolist() for e in env.agents()}
        landmarks = {l.name: l.state.pos.cpu().numpy().tolist() for l in env.landmarks()}
        
        # Safe action serialization
        if isinstance(actions, torch.Tensor):
            action_data = actions.cpu().numpy().tolist()
        elif isinstance(actions, TensorDict):
            action_data = actions.cpu().numpy().tolist()
        else:
            action_data = np.array(actions).tolist()
            
        cbor2.dump(
            {
                'step': step,
                'agent_data': agent_data,
                'landmarks': landmarks,
                'actions': action_data,
            },
            fp,
        )
        obs, rewards, dones, infos = env.step(actions)
        
    # Final frame
    agent_data = {e.name: e.state.pos.cpu().numpy().tolist() for e in env.agents()}
    landmarks = {l.name: l.state.pos.cpu().numpy().tolist() for l in env.landmarks()}
    cbor2.dump(
        {'step': max_steps, 'agent_data': agent_data, 'landmarks': landmarks},
        fp,
    )
    
    fp.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record episodes.")
    parser.add_argument('--output', type=str, required=True, help='Output file path (.cbor)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenario_name', type=str, required=True, help='Scenario: simple_spread or food_collection')
    parser.add_argument('--n_agents', type=int, default=4)
    parser.add_argument('--n_food', type=int, default=6)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--model_name', type=str, required=True, help='Model: mappo, qmix, ippo, infomarl, etc.')
    
    args = parser.parse_args()
    
    env_kwargs = {
        'scenario': args.scenario_name,
        'n_agents': args.n_agents,
        'n_food': args.n_food,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': args.seed,
        'continuous_actions': True,
        'num_envs': args.num_envs,
        'terminated_truncated': False,
        'respawn_food': True,
    }
    
    print(f"Recording {args.model_name} on {args.scenario_name}...")
    with open(args.output, 'wb') as fp:
        record_episodes(fp, env_kwargs, args.model_name, args.max_steps)