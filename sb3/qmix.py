import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class QMixReplayBuffer:
    def __init__(self, capacity: int, n_envs: int, n_agents: int, obs_shape: tuple, device: str = "cpu"):
        self.capacity = capacity
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.obs_shape = obs_shape
        self.device = device
        
        self.buffer_size = capacity
        
        # Pre-allocate tensors
        self.observations = torch.zeros(self.buffer_size, n_agents, *obs_shape, device=device)
        self.actions = torch.zeros(self.buffer_size, n_agents, dtype=torch.long, device=device)
        self.rewards = torch.zeros(self.buffer_size, n_agents, device=device)
        self.next_observations = torch.zeros(self.buffer_size, n_agents, *obs_shape, device=device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.bool, device=device)
        
        global_state_size = n_agents * np.prod(obs_shape)
        self.global_states = torch.zeros(self.buffer_size, global_state_size, device=device)
        self.next_global_states = torch.zeros(self.buffer_size, global_state_size, device=device)
        
        self.position = 0
        self.size = 0
        
    def push(self, obs, actions, rewards, next_obs, dones):
        batch_size = obs.shape[0]
        
        for i in range(batch_size):
            if self.position >= self.capacity:
                self.position = 0
                
            self.observations[self.position] = obs[i]
            self.actions[self.position] = actions[i] 
            self.rewards[self.position] = rewards[i]
            self.next_observations[self.position] = next_obs[i]
            self.dones[self.position] = dones[i]
            
            self.global_states[self.position] = obs[i].flatten()
            self.next_global_states[self.position] = next_obs[i].flatten()
            
            self.position += 1
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            self.global_states[indices],
            self.next_global_states[indices]
        )
    
    def __len__(self):
        return self.size

class QMixAgentNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim - 1  # Remove random number dimension
        
        self.network = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs):
        obs_processed = obs[..., :-1]  # Remove random number
        return self.network(obs_processed)

class QMixMixingNetwork(nn.Module):
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_q_vals, global_state):
        batch_size = agent_q_vals.shape[0]
        
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)
        
        w2 = torch.abs(self.hyper_w2(global_state))
        
        b1 = self.hyper_b1(global_state)
        b2 = self.hyper_b2(global_state)
        
        agent_q_vals = agent_q_vals.unsqueeze(-1)
        hidden = F.elu(torch.sum(w1 * agent_q_vals, dim=1) + b1)
        
        global_q_val = torch.sum(w2 * hidden, dim=1, keepdim=True) + b2
        
        return global_q_val

class QMixVMAS:
    def __init__(
        self,
        env,
        lr: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        target_update_freq: int = 500,
        buffer_size: int = 100000,
        batch_size: int = 256,
        train_freq: int = 4,
        action_grid_size: int = 7,
        device: str = "cpu"
    ):
        self.env = env
        self.n_envs = env.num_envs
        self.n_agents = env.num_agents
        self.obs_dim = env.observation_space.shape[-1]
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.device = device
        
        self.action_grid_size = action_grid_size
        self.action_dim = action_grid_size ** 2
        
        self.action_map = self._create_action_mapping()
        
        # Create networks
        self.agent_networks = nn.ModuleList([
            QMixAgentNetwork(self.obs_dim, self.action_dim, hidden_dim=64)
            for _ in range(self.n_agents)
        ]).to(device)
        
        self.target_agent_networks = nn.ModuleList([
            QMixAgentNetwork(self.obs_dim, self.action_dim, hidden_dim=64)
            for _ in range(self.n_agents)
        ]).to(device)
        
        global_state_dim = self.n_agents * (self.obs_dim - 1)
        
        self.mixing_network = QMixMixingNetwork(
            self.n_agents, global_state_dim, hidden_dim=32
        ).to(device)
        
        self.target_mixing_network = QMixMixingNetwork(
            self.n_agents, global_state_dim, hidden_dim=32
        ).to(device)
        
        self.update_target_networks()
        
        all_params = list(self.agent_networks.parameters()) + list(self.mixing_network.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        
        self.replay_buffer = QMixReplayBuffer(
            buffer_size, self.n_envs, self.n_agents, 
            (self.obs_dim,), device
        )
        
        self.training_step = 0
        
    def _create_action_mapping(self):
        actions = []
        for x in np.linspace(-1, 1, self.action_grid_size):
            for y in np.linspace(-1, 1, self.action_grid_size):
                actions.append([x, y])
        return torch.tensor(actions, dtype=torch.float32, device=self.device)
    
    def discrete_to_continuous(self, discrete_actions):
        return self.action_map[discrete_actions]
    
    def get_epsilon(self, step: int):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               max(0, 1 - step / self.epsilon_decay_steps)
    
    def select_actions(self, observations, step: int, explore: bool = True):
        with torch.no_grad():
            batch_size = observations.shape[0]
            epsilon = self.get_epsilon(step) if explore else 0.0
            
            discrete_actions = torch.zeros(batch_size, self.n_agents, dtype=torch.long, device=self.device)
            
            for i, agent_net in enumerate(self.agent_networks):
                if explore and random.random() < epsilon:
                    discrete_actions[:, i] = torch.randint(
                        0, self.action_dim, (batch_size,), device=self.device
                    )
                else:
                    q_vals = agent_net(observations[:, i])
                    discrete_actions[:, i] = q_vals.argmax(dim=-1)
            
            continuous_actions = self.discrete_to_continuous(discrete_actions)
            
            return discrete_actions, continuous_actions
    
    def update_target_networks(self):
        for target, main in zip(self.target_agent_networks, self.agent_networks):
            target.load_state_dict(main.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        obs, actions, rewards, next_obs, dones, global_states, next_global_states = \
            self.replay_buffer.sample(self.batch_size)
        
        # Current Q-values
        current_q_vals = []
        for i, agent_net in enumerate(self.agent_networks):
            agent_q_vals = agent_net(obs[:, i])
            chosen_q_vals = agent_q_vals.gather(1, actions[:, i:i+1])
            current_q_vals.append(chosen_q_vals.squeeze(1))
        
        current_q_vals = torch.stack(current_q_vals, dim=1)
        
        # Remove random numbers from global states
        global_states_clean = self._remove_random_numbers_from_global_state(global_states)
        next_global_states_clean = self._remove_random_numbers_from_global_state(next_global_states)
        
        current_global_q = self.mixing_network(current_q_vals, global_states_clean)
        
        # Target Q-values
        with torch.no_grad():
            next_q_vals = []
            for i, target_agent_net in enumerate(self.target_agent_networks):
                next_agent_q_vals = target_agent_net(next_obs[:, i])
                max_next_q_vals = next_agent_q_vals.max(1)[0]
                next_q_vals.append(max_next_q_vals)
            
            next_q_vals = torch.stack(next_q_vals, dim=1)
            target_global_q = self.target_mixing_network(next_q_vals, next_global_states_clean)
            
            team_rewards = rewards.sum(dim=1, keepdim=True)
            targets = team_rewards + self.gamma * target_global_q * (~dones).unsqueeze(1).float()
        
        # Train
        loss = F.mse_loss(current_global_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_networks.parameters()) + list(self.mixing_network.parameters()),
            max_norm=10.0
        )
        self.optimizer.step()
        
        if self.training_step % self.target_update_freq == 0:
            self.update_target_networks()
        
        self.training_step += 1
        
        return {
            'loss': loss.item(),
            'epsilon': self.get_epsilon(self.training_step)
        }
    
    def _remove_random_numbers_from_global_state(self, global_states):
        reshaped = global_states.view(-1, self.n_agents, self.obs_dim)
        cleaned = reshaped[..., :-1]
        return cleaned.reshape(global_states.shape[0], -1)
    
    def learn(self, total_timesteps: int, log_interval: int = 1000, progress_bar: bool = True):
        print(f"Training QMix for {total_timesteps} timesteps...")
        
        obs = self.env.reset()
        obs = torch.from_numpy(obs).to(self.device)
        step = 0
        episode_rewards = torch.zeros(self.n_envs, device=self.device)
        episode_count = 0
        
        while step < total_timesteps:
            discrete_actions, continuous_actions = self.select_actions(obs, step, explore=True)
            
            next_obs, rewards, dones, infos = self.env.step(continuous_actions.cpu().numpy())
            
            # Convert numpy arrays to tensors for replay buffer
            next_obs_tensor = torch.from_numpy(next_obs).to(self.device)
            rewards_tensor = torch.from_numpy(rewards).to(self.device)
            dones_tensor = torch.from_numpy(dones).to(self.device)
            
            self.replay_buffer.push(obs, discrete_actions, rewards_tensor, next_obs_tensor, dones_tensor)
            
            if step > self.batch_size and step % self.train_freq == 0:
                self.train_step()
            
            if rewards_tensor.dim() == 1:
                episode_rewards += rewards_tensor  # Already summed
            else:
                episode_rewards += rewards_tensor.sum(dim=1)
            
            if dones_tensor.any():
                finished_episodes = dones_tensor.nonzero().squeeze(-1)
                for env_idx in finished_episodes:
                    episode_count += 1
                    if progress_bar and episode_count % log_interval == 0:
                        print(f"Step {step}, Episodes: {episode_count}, "
                              f"Reward: {episode_rewards[env_idx].item():.2f}, "
                              f"Epsilon: {self.get_epsilon(step):.3f}")
                    episode_rewards[env_idx] = 0
            
            obs = next_obs_tensor
            step += 1
    
    def save(self, path: str):
        torch.save({
            'agent_networks': [net.state_dict() for net in self.agent_networks],
            'mixing_network': self.mixing_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'action_map': self.action_map
        }, f"{path}_qmix.zip")
        print(f"QMix model saved to {path}_qmix.zip")
    
    def load(self, path: str):
        checkpoint = torch.load(f"{path}_qmix.zip", map_location=self.device)
        
        for i, state_dict in enumerate(checkpoint['agent_networks']):
            self.agent_networks[i].load_state_dict(state_dict)
        
        self.mixing_network.load_state_dict(checkpoint['mixing_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        self.action_map = checkpoint['action_map']
        
        self.update_target_networks()
        print(f"QMix model loaded from {path}_qmix.zip")