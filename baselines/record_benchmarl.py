
import argparse
import pickle
from pathlib import Path
from typing import Optional
import re

import vmas
from tqdm import tqdm
import torch


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
        from benchmarl.algorithms import MappoConfig, QmixConfig, MasacConfig
        from benchmarl.environments import VmasTask
        from benchmarl.models.mlp import MlpConfig
        from benchmarl.experiment import Experiment, ExperimentConfig
        
        self.device = env_kwargs.get('device', 'cpu')
        self.num_agents = env_kwargs.get('n_agents', 1)
        self.num_envs = env_kwargs.get('num_envs', 1)
        self.algorithm = algorithm
        
        scenario_name = env_kwargs.get('scenario')
        if scenario_name == "simple_spread":
            task = VmasTask.SIMPLE_SPREAD.get_from_yaml()
        elif scenario_name == "food_collection":
            task = VmasTask.FOOD_COLLECTION.get_from_yaml()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        if "n_agents" in task.config:
            task.config["n_agents"] = self.num_agents
        
        if algorithm == "mappo":
            algo_config = MappoConfig.get_from_yaml()
        elif algorithm == "qmix":
            algo_config = QmixConfig.get_from_yaml()
        elif algorithm == "masac":
            algo_config = MasacConfig.get_from_yaml()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        model_config = MlpConfig.get_from_yaml()
        critic_model_config = MlpConfig.get_from_yaml()
        
        exp_config = ExperimentConfig.get_from_yaml()
        exp_config.restore_file = str(checkpoint_path)
        exp_config.train_device = self.device
        exp_config.sampling_device = self.device
        exp_config.loggers = []
        exp_config.save_folder = "/tmp/benchmarl_temp"
        Path("/tmp/benchmarl_temp").mkdir(parents=True, exist_ok=True)
        
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
        from tensordict import TensorDict
        from torchrl.envs.utils import ExplorationType, set_exploration_type
        
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            obs_tensor = torch.stack(obs, dim=1).to(self.device)
            
            td = TensorDict({
                "agents": TensorDict({
                    "observation": obs_tensor,
                }, batch_size=[self.num_envs]),
            }, batch_size=[self.num_envs])
            
            td = self.policy(td)
            actions = td["agents", "action"]
            return actions.transpose(0, 1)


def find_best_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None
    
    def get_frame_count(p):
        match = re.search(r'checkpoint_(\d+)\.pt', p.name)
        return int(match.group(1)) if match else 0
    
    return max(checkpoints, key=get_frame_count)


def find_all_experiments(results_dir: Path):
    experiments = {}
    
    for algo in ["mappo", "masac", "qmix"]:
        algo_dir = results_dir / algo
        if not algo_dir.exists():
            continue
        
        for scenario in ["simple_spread", "food_collection"]:
            for exp_dir in algo_dir.glob(f"{scenario}*"):
                checkpoint_dir = exp_dir / "checkpoints"
                if not checkpoint_dir.exists():
                    continue
                
                best_ckpt = find_best_checkpoint(checkpoint_dir)
                if not best_ckpt:
                    continue
                
                match = re.search(r'checkpoint_(\d+)\.pt', best_ckpt.name)
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
    """Record episodes and save to pickle."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    env_kwargs_copy = env_kwargs.copy()
    if algorithm == "qmix":
        env_kwargs_copy['continuous_actions'] = False
    
    env = Env(env_kwargs_copy)
    model = BenchMARLModel(checkpoint_path, env_kwargs_copy, algorithm)
    
    all_frames = []
    obs = env.reset()
    
    for step in tqdm(range(max_steps), desc="Recording", leave=False):
        actions = model(obs)
        
        agent_data = {e.name: e.state.pos.cpu().numpy().tolist() for e in env.agents()}
        landmark_data = {l.name: l.state.pos.cpu().numpy().tolist() for l in env.landmarks()}
        action_data = actions.cpu().numpy().tolist() if isinstance(actions, torch.Tensor) else actions.tolist()
        
        all_frames.append({
            'step': step,
            'agent_data': agent_data,
            'landmarks': landmark_data,
            'actions': action_data,
        })
        
        obs, rewards, dones, infos = env.step(actions)
    
    # Final state
    all_frames.append({
        'step': max_steps,
        'agent_data': {e.name: e.state.pos.cpu().numpy().tolist() for e in env.agents()},
        'landmarks': {l.name: l.state.pos.cpu().numpy().tolist() for l in env.landmarks()},
    })
    
    # Save as pickle
    data = {
        'env_kwargs': env_kwargs_copy,
        'meta': {"model": f"benchmarl_{algorithm}", "max_steps": max_steps},
        'frames': all_frames,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    # Verify
    with open(output_path, 'rb') as f:
        loaded = pickle.load(f)
        print(f"  Saved & verified: {loaded['meta']['model']}, {len(loaded['frames'])} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='baselines/results')
    parser.add_argument('--output_dir', type=str, default='recordings')
    parser.add_argument('--algorithms', nargs='+', default=['mappo', 'masac', 'qmix'])
    parser.add_argument('--scenarios', nargs='+', default=['simple_spread', 'food_collection'])
    parser.add_argument('--n_agents', type=int, default=4)
    parser.add_argument('--n_food', type=int, default=6)
    parser.add_argument('--max_steps', type=int, default=400)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dry_run', action='store_true')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    experiments = find_all_experiments(results_dir)
    experiments = [
        e for e in experiments 
        if e['algorithm'] in args.algorithms and e['scenario'] in args.scenarios
    ]
    
    if not experiments:
        print(f"No experiments found in {results_dir}")
        return
    
    print(f"Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  {exp['algorithm']}/{exp['scenario']}: {exp['checkpoint'].name}")
    
    if args.dry_run:
        return
    
    print("\n" + "=" * 60)
    
    for exp in experiments:
        algo = exp['algorithm']
        scenario = exp['scenario']
        checkpoint = exp['checkpoint']
        
        scenario_short = scenario.replace("simple_spread", "spread").replace("food_collection", "collection")
        output_file = output_dir / f"benchmarl_{algo}_{scenario_short}_{args.n_agents}.pkl"
        
        print(f"\nRecording {algo} on {scenario}")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Output: {output_file}")
        
        env_kwargs = {
            'scenario': scenario,
            'n_agents': args.n_agents,
            'device': args.device,
            'seed': args.seed,
            'continuous_actions': True,
            'num_envs': args.num_envs,
            'terminated_truncated': False,
        }
        
        if scenario == "food_collection":
            env_kwargs['n_food'] = args.n_food
            env_kwargs['respawn_food'] = True
        
        try:
            record_episodes(output_file, env_kwargs, checkpoint, algo, args.max_steps)
            print(f"  Done")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All recordings complete!")


if __name__ == "__main__":
    main()