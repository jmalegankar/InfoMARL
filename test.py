from vmas import make_env
import cbor2
from tqdm import tqdm
import torch

class Env:
    def __init__(self, env_kwargs):
        self.env = make_env(**env_kwargs)
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        actions = list(actions.transpose(0, 1))
        return self.env.step(actions)
    
    def agents(self):
        return self.env.agents
    
    def landmarks(self):
        return self.env.world.landmarks

class Model:
    def __init__(self, name, env_kwargs):
        self.device = env_kwargs.get('device', 'cpu')
        scenario_name = env_kwargs.get('scenario_name')
        num_agents = env_kwargs.get('n_agents', 1)
        if scenario_name == "simple_spread":
            key = "spread"
        elif scenario_name == "food_collection":
            key = "collection"
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        if name == "GSA":
            from functions import GSA_actor
            self.actor = torch.load('./models/GSA_' + key + '.pth', map_location=self.device)
            self.actor.na = num_agents
        elif name == "pH-MARL":
            from functions import LEMURS_actor
            self.actor = torch.load('./models/pH-MARL_' + key + '.pth', map_location=self.device)
            self.actor.na = num_agents
        else:
            raise ValueError(f"Unknown model name: {name}")
        
        self.actor.eval()
    
    def __call__(self, obs):
        with torch.no_grad():
            obs = torch.stack(obs).transpose(1,0).to(self.device)
            actions = self.actor(obs).mean
        return actions

def record_episodes(fp, env_kwargs, model_name, max_steps):
    cbor2.dump(
        env_kwargs,
        fp,
    )
    cbor2.dump(
        {
            "model": model_name,
            "max_steps": max_steps,
        },
        fp,
    )

    env = Env(env_kwargs)
    model = Model(model_name, env_kwargs)
    obs = env.reset()
    for step in tqdm(range(max_steps)):
        actions = model(obs)
        agent_data = {e.name:e.state.pos.cpu().numpy().tolist() for e in env.agents()}
        landmarks = {l.name:l.state.pos.cpu().numpy().tolist() for l in env.landmarks()}
        action_data = actions.cpu().numpy().tolist()
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
    agent_data = {e.name:e.state.pos.cpu().numpy().tolist() for e in env.agents()}
    landmarks = {l.name:l.state.pos.cpu().numpy().tolist() for l in env.landmarks()}
    cbor2.dump(
        {
            'step': max_steps,
            'agent_data': agent_data,
            'landmarks': landmarks,
        },
        fp,
    )
    
    fp.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record episodes for a given environment and model.")
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scenario_name', type=str, required=True, help='Scenario name')
    parser.add_argument('--n_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--n_food', type=int, default=6, help='Number of food items')
    parser.add_argument('--max_steps', type=int, default=400, help='Maximum steps per episode')
    parser.add_argument('--num_envs', type=int, default=64, help='Number of environments to run')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (GSA or pH-MARL)')
    
    args = parser.parse_args()
    
    env_kwargs = {
        'scenario_name': args.scenario_name,
        'n_agents': args.n_agents,
        'n_food': args.n_food,
        'device': 'cuda',
        'seed': args.seed,
        'continuous_actions': True,
        'num_envs': args.num_envs,
        'terminated_truncated': False,
        'respawn_food': True,
    }
    
    with open(args.output, 'wb') as fp:
        record_episodes(fp, env_kwargs, args.model_name, args.max_steps)