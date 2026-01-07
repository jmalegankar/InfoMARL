from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

import torch
import torch.nn as nn
import warnings

from bridge_attn import DiamondAttention

def get_permuted_env_random_numbers(env_random_numbers, number_agents, num_envs, device):
    """
    Permute the random numbers for each agent in the environment.
    """
    permutation_indices = torch.zeros(number_agents, number_agents, dtype=torch.long, device=device)
    for i in range(number_agents):
        other_agents = sorted([j for j in range(number_agents) if j != i])
        permutation_indices[i] = torch.tensor([i] + other_agents)
    expanded_rand = env_random_numbers.unsqueeze(1).expand(-1, number_agents, -1)
    permuted_rand = torch.gather(expanded_rand, dim=2, index=permutation_indices.unsqueeze(0).expand(num_envs, -1, -1))
    return permuted_rand


def env_parser_multi_give_way(obs: torch.Tensor, number_agents: int):
    """
    Parse observations from multi_give_way environment.
    
    Observation structure per agent (from source code):
    - Agent position (2)
    - Agent velocity (2)
    - Relative position to goal (2)
    - Distance to goal (1)
    - Random number (1) if using rnd_nums wrapper
    
    Total: 8 dimensions per agent
    """
    random_numbers = obs[..., -1]
    obs = obs.view(-1, obs.shape[-1])
    
    cur_pos = obs[:, 0:2]
    cur_vel = obs[:, 2:4]
    goal_rel_pos = obs[:, 4:6]
    goal_distance = obs[:, 6:7]
    
    return cur_pos, cur_vel, goal_rel_pos, goal_distance, random_numbers


class RandomAgentPolicy(nn.Module):
    def __init__(self, number_agents, agent_dim, goal_dim, hidden_dim):
        super().__init__()
        self.number_agents = number_agents
        self.hidden_dim = hidden_dim
        self.training = True
        
        # Network for processing current agent (pos + vel)
        self.cur_agent_embedding = nn.Sequential(
            nn.Linear(agent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Network for processing goal information (rel_pos + distance)
        self.goal_embedding = nn.Sequential(
            nn.Linear(goal_dim + 1, hidden_dim),  # +1 for distance
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Network to create "virtual" self-representations at different hierarchy levels
        # This allows the agent to reason about "cautious me" vs "aggressive me"
        self.self_hierarchy_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Diamond attention for self-modulation based on hierarchy
        self.self_attention = DiamondAttention(
            hidden_dim=hidden_dim,
            num_heads=1,
            dropout=0.0,
        )
        
        # Goal attention
        self.goal_attention = DiamondAttention(
            hidden_dim=hidden_dim,
            num_heads=1,
            dropout=0.0,
        )
    
    def forward(self, obs):
        """Extract features from observations using the attention mechanism"""
        num_envs = obs.shape[0]
        
        cur_pos, cur_vel, goal_rel_pos, goal_distance, random_numbers = \
            env_parser_multi_give_way(obs, self.number_agents)
        
        # Get permuted random numbers for hierarchy
        random_numbers_permuted = get_permuted_env_random_numbers(
            random_numbers, self.number_agents, num_envs, cur_pos.device
        ).view(-1, self.number_agents)
        
        # Prepare current agent features
        cur_agent = torch.cat((cur_pos, cur_vel), dim=-1)
        
        # Prepare goal features  
        goal_features = torch.cat((goal_rel_pos, goal_distance), dim=-1)
        
        # Encode current agent and goal
        cur_agent_embeddings = self.cur_agent_embedding(cur_agent)  # (batch*agents, hidden)
        goal_embeddings = self.goal_embedding(goal_features)  # (batch*agents, hidden)
        
        # Create "virtual" representations of self at different hierarchy levels
        # based on hierarchy
        self_hierarchy_reps = self.self_hierarchy_projection(cur_agent_embeddings)
        
        # Reshape for attention (batch*agents, n_agents, hidden)
        # Each agent gets n_agents copies of its own representation
        self_hierarchy_reps = self_hierarchy_reps.unsqueeze(1).expand(-1, self.number_agents, -1)
        
        # Create hierarchy mask: agents mask representations with higher random numbers
        hierarchy_mask = ~(random_numbers_permuted >= random_numbers_permuted[:, 0].view(-1, 1))
        
        # Self-attention with hierarchy mask
        # This modulates the agent's confidence based on its hierarchy position
        cur_agent_for_attn = cur_agent_embeddings.unsqueeze(1)  # (batch*agents, 1, hidden)
        _, agent_modulated, self_weights = self.self_attention(
            cur_agent_for_attn,
            self_hierarchy_reps,
            cur_agent_for_attn,
            self_hierarchy_reps,
            key_mask=hierarchy_mask,
        )
        
        if not self.training:
            self.self_attention_weights = self_weights
        else:
            del self_weights
        
        # Goal attention
        goal_for_attn = goal_embeddings.unsqueeze(1)  # (batch*agents, 1, hidden)
        _, goal_refined, goal_weights = self.goal_attention(
            agent_modulated,
            goal_for_attn,
            agent_modulated,
            goal_for_attn,
        )
        
        if not self.training:
            self.goal_attention_weights = goal_weights
        else:
            del goal_weights
        
        # Concatenate features for final processing
        latent = torch.cat((agent_modulated.squeeze(1), goal_refined.squeeze(1)), dim=-1)
        
        return latent.view(num_envs, self.number_agents, self.hidden_dim * 2)


class CriticPolicy(nn.Module):
    def __init__(self, obs_size, hidden_dim):
        super().__init__()
        self.obs_size = obs_size
        self.layer = nn.Sequential(
            nn.Linear(obs_size, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.SiLU(),
        )
    
    def forward(self, obs):
        obs = obs[..., :-1]
        obs = obs.reshape(-1, self.obs_size)
        obs = self.layer(obs)
        return obs


class InfoMARLExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for InfoMARL on multi_give_way scenario.

    """
    
    def __init__(
        self,
        observation_space,
        agent_dim=2,
        goal_dim=2,
        hidden_dim=64,
        critic=False,
    ):
        super(InfoMARLExtractor, self).__init__(observation_space, hidden_dim*2)
        self.n_agents = observation_space.shape[0]
        
        if critic:
            self.critic = CriticPolicy(
                obs_size=self.n_agents * (observation_space.shape[-1] - 1),
                hidden_dim=hidden_dim,
            )
        else:
            self.actor = RandomAgentPolicy(
                number_agents=self.n_agents,
                agent_dim=agent_dim,
                goal_dim=goal_dim,
                hidden_dim=hidden_dim,
            )
    
    def forward(self, observations):
        """
        Forward pass through the feature extractor.
        """
        if hasattr(self, 'critic'):
            return self.critic(observations)
        else:
            return self.actor(observations)


from stable_baselines3.common.distributions import make_proba_distribution


class InfoMARLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule=3e-4,
        net_arch=None,
        activation_fn=nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class=InfoMARLExtractor,
        features_extractor_kwargs=None,
        normalize_images: bool = True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            warnings.warn(
                "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                "you should now pass directly a dictionary and not a list "
                "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
            )
            net_arch = net_arch[0]
        
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        
        # Feature extractors not shared
        self.share_features_extractor = False
        self.features_extractor_kwargs.update({"critic": False})
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.pi_features_extractor = self.features_extractor
        self.features_extractor_kwargs.update({"critic": True})
        self.vf_features_extractor = self.make_features_extractor()
        
        self.log_std_init = log_std_init
        dist_kwargs = None
        
        assert not (squash_output and not use_sde), "squash_output=True only available with use_sde=True"
        
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }
        else:
            dist_kwargs = {}
        
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs
        
        self.action_dist = make_proba_distribution(
            action_space, self.log_std_init, self.dist_kwargs
        )
        
        self._build(lr_schedule)
    
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[-1]),
            nn.Flatten(),
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )