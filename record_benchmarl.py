import argparse
import cbor2
import re
import os
import vmas
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# BenchMARL / TorchRL imports
try:
    from benchmarl.algorithms import MappoConfig, MasacConfig, QmixConfig, IppoConfig
    from benchmarl.models.mlp import MlpConfig
    from benchmarl.environments import VmasTask
    from benchmarl.experiment import Experiment, ExperimentConfig
    from tensordict import TensorDict
    from torchrl.envs.utils import ExplorationType, set_exploration_type
except ImportError:
    print("Warning: BenchMARL/TorchRL not installed.")

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

class BenchMARLModel:
    def __init__(self, checkpoint_path, env_kwargs, algorithm="mappo"):
        self.device = env_kwargs.get('device', 'cpu')
        self.num_agents = env_kwargs.get('n_agents', 1)
        self.num_envs = env_kwargs.get('num_envs', 1)
        self.algorithm = algorithm
        
        scenario_name = env_kwargs.get('scenario')
        
        # 1. Map Scenario
        if scenario_name == "simple_spread":
            task = VmasTask.SIMPLE_SPREAD.get_from_yaml()
        elif scenario_name == "food_collection":
            task = VmasTask.FOOD_COLLECTION.get_from_yaml()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        if "n_agents" in task.config:
            task.config["n_agents"] = self.num_agents
        
        # 2. Map Algorithm
        if algorithm == "mappo":
            algo_config = MappoConfig.get_from_yaml()
        elif algorithm == "ippo":
            algo_config = IppoConfig.get_from_yaml()
        elif algorithm == "qmix":
            algo_config = QmixConfig.get_from_yaml()
        elif algorithm == "masac":
            algo_config = MasacConfig.get_from_yaml()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # 3. Model Config
        model_config = MlpConfig.get_from_yaml()
        model_config.num_cells = [256, 256]
        critic_model_config = MlpConfig.get_from_yaml()
        critic_model_config.num_cells = [256, 256]
        
        # 4. Experiment Config
        temp_dir = Path("/tmp/benchmarl_restore_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        exp_config = ExperimentConfig.get_from_yaml()
        exp_config.restore_file = str(checkpoint_path)
        exp_config.train_device = self.device
        exp_config.sampling_device = self.device
        exp_config.loggers = []
        exp_config.save_folder = str(temp_dir)
        
        # Important: Tell BenchMARL to use discrete head for QMIX
        if algorithm == "qmix":
            exp_config.prefer_continuous_actions = False
        
        self.experiment = Experiment(
            task=task,
            algorithm_config=algo_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=0,
            config=exp_config,
        )
        
        self.policy = self.experiment.policy
        self.policy.eval()
    
    def __call__(self, obs):
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            # VMAS -> TensorDict
            obs_tensor = torch.stack(obs, dim=1).to(self.device)
            
            td = TensorDict({
                "agents": TensorDict({
                    "observation": obs_tensor,
                }, batch_size=[self.num_envs], device=self.device),
            }, batch_size=[self.num_envs], device=self.device)
            
            # Forward Pass
            td = self.policy(td)
            
            # Extract Action
            if "action" in td["agents"].keys():
                actions = td["agents", "action"]
            else:
                actions = td["agents", "loc"]

            # QMIX OUTPUT:
            # actions is [n_envs, n_agents] (Indices)
            # OR [n_envs, n_agents, 1] (Indices unsqueezed)
            # We just need to ensure it's clean for VMAS Discrete mode.
            if self.algorithm == "qmix":
                if actions.ndim == 3 and actions.shape[-1] == 1:
                    actions = actions.squeeze(-1)
                
                # Check if it output one-hot logits instead of indices
                if actions.shape[-1] == 5 or actions.shape[-1] == 9: 
                     actions = torch.argmax(actions, dim=-1)

            return actions.transpose(0, 1)

# ============================================================================
# UTILS
# ============================================================================

def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    # Match both 'checkpoint_123.pt' and 'ckpt_f123_r-45.pt'
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    
    def get_frame_count(p):
        # This regex looks for 'checkpoint_' OR 'ckpt_f' followed by digits
        match = re.search(r'(?:checkpoint_|ckpt_f)(\d+)', p.name)
        return int(match.group(1)) if match else 0
    
    # Filter out files that didn't match our pattern if necessary
    valid_checkpoints = [c for c in checkpoints if re.search(r'(?:checkpoint_|ckpt_f)(\d+)', c.name)]
    if not valid_checkpoints:
        return None

    return max(valid_checkpoints, key=get_frame_count)

def find_all_experiments(results_dir: Path):
    experiments = {}
    
    for scenario in ["simple_spread", "food_collection"]:
        scenario_dir = results_dir / scenario
        if not scenario_dir.exists():
            continue

        for algo in ["mappo", "masac", "qmix", "ippo"]:
            exp_folders = list(scenario_dir.glob(f"{algo}*"))
            
            for exp_dir in exp_folders:
                checkpoint_dir = exp_dir / "checkpoints"
                if not checkpoint_dir.exists():
                    continue
                
                best_ckpt = find_best_checkpoint(checkpoint_dir)
                if not best_ckpt:
                    continue
                
                # Use the same logic to extract the frame count for the experiment dict
                match = re.search(r'(?:checkpoint_|ckpt_f)(\d+)', best_ckpt.name) 
                frame_count = int(match.group(1)) if match else 0
                
                key = (algo, scenario)
                if key not in experiments or frame_count > experiments[key]['frame_count']:
                    experiments[key] = {
                        "algorithm": algo,
                        "scenario": scenario,
                        "checkpoint": best_ckpt,
                        "exp_name": exp_dir.name,
                        "frame_count": frame_count,
                    }
    
    return list(experiments.values())

def record_episodes(output_path, env_kwargs, checkpoint_path, algorithm, max_steps):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    env_kwargs_copy = env_kwargs.copy()
    
    if algorithm == "qmix":
        env_kwargs_copy['continuous_actions'] = False
    
    env = Env(env_kwargs_copy)
    model = BenchMARLModel(checkpoint_path, env_kwargs_copy, algorithm)
    
    print(f"  Recording to: {output_path}")
    
    with open(output_path, 'wb') as fp:
        cbor2.dump(env_kwargs_copy, fp)
        cbor2.dump({"model": f"benchmarl_{algorithm}", "max_steps": max_steps}, fp)

        obs = env.reset()
        for step in tqdm(range(max_steps), desc="  Steps", leave=False):
            actions = model(obs)
            
            agent_data = {e.name: e.state.pos.cpu().numpy().tolist() for e in env.agents()}
            landmark_data = {l.name: l.state.pos.cpu().numpy().tolist() for l in env.landmarks()}
            
            if isinstance(actions, torch.Tensor):
                action_data = actions.cpu().numpy().tolist()
            else:
                action_data = np.array(actions).tolist()
            
            cbor2.dump({
                'step': step,
                'agent_data': agent_data,
                'landmarks': landmark_data,
                'actions': action_data,
            }, fp)
            
            obs, rewards, dones, infos = env.step(actions)
        
        cbor2.dump({
            'step': max_steps,
            'agent_data': {e.name: e.state.pos.cpu().numpy().tolist() for e in env.agents()},
            'landmarks': {l.name: l.state.pos.cpu().numpy().tolist() for l in env.landmarks()},
        }, fp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='baselines/results_old')
    # Default output dir matches your existing ones
    parser.add_argument('--output_dir', type=str, default='eval_data1') 
    parser.add_argument('--algorithms', nargs='+', default=['mappo', 'masac', 'qmix', 'ippo'])
    parser.add_argument('--scenarios', nargs='+', default=['simple_spread', 'food_collection'])
    parser.add_argument('--n_agents', type=int, default=4)
    parser.add_argument('--n_food', type=int, default=6)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Scanning {results_dir}...")
    experiments = find_all_experiments(results_dir)
    
    experiments = [e for e in experiments 
                   if e['algorithm'] in args.algorithms and e['scenario'] in args.scenarios]
    
    if not experiments:
        print(f"No matching experiments found in {results_dir}")
        return
    
    print(f"Found {len(experiments)} valid checkpoints.")
    
    for exp in experiments:
        algo = exp['algorithm']
        scenario = exp['scenario']
        checkpoint = exp['checkpoint']
        
        scenario_short = scenario.replace("simple_spread", "spread").replace("food_collection", "collection")
        
        # CHANGED: Now saves as .dat
        output_file = output_dir / f"{algo}_{scenario_short}_{args.n_agents}.dat"
        
        print(f"\nProcessing {algo} on {scenario}...")
        
        env_kwargs = {
            'scenario': scenario,
            'n_agents': args.n_agents,
            'n_food': args.n_food,
            'device': args.device,
            'seed': args.seed,
            'continuous_actions': True,
            'num_envs': args.num_envs,
            'terminated_truncated': False,
            'respawn_food': True,
        }
        
        try:
            record_episodes(output_file, env_kwargs, checkpoint, algo, args.max_steps)
            print("  Success!")
        except Exception as e:
            print(f"  Failed: {e}")
            # import traceback
            # traceback.print_exc()

    print("\nAll tasks finished.")

if __name__ == "__main__":
    main()