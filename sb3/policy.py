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

def env_parser(obs: torch.Tensor, number_agents: int, number_food: int, agent_dim: int, landmark_dim: int):
    """
    Parse observations from food collection environment.
    
    Observation structure per agent:
    - Agent move possibitlity (4)
    - Enemy/Food data (number_food * landmark_dim)
    - Other agents data ((number_agents - 1) * agent_dim)
    - Agent heatlh , sheild, unit type (3)
    - Random number (1) if using rnd_nums wrapper
    """
    random_numbers = obs[..., -1]
    obs = obs[..., :-1]
    cur_agent_vec = torch.cat((
        obs[..., :4],
        obs[..., -(agent_dim -4):],
    ), dim=-1).view(-1, agent_dim)
    landmarks_start = 4
    landmarks_end = landmarks_start + number_food * landmark_dim
    landmarks = obs[..., landmarks_start:landmarks_end].view(-1, number_agents, number_food, landmark_dim)
    other_agents_start = landmarks_end
    other_agents_end = other_agents_start + (number_agents - 1) * agent_dim
    other_agents = obs[..., other_agents_start:other_agents_end].view(-1, number_agents - 1, agent_dim)

    return cur_agent_vec, landmarks, other_agents, random_numbers


class RandomAgentPolicy(nn.Module):
    def __init__(self, number_agents, number_food, agent_dim, landmark_dim, hidden_dim):
        super().__init__()
        self.number_agents = number_agents
        self.agent_dim = agent_dim
        self.landmark_dim = landmark_dim
        self.number_food = number_food
        self.hidden_dim = hidden_dim
        self.training = True  # Add this to control behavior

        # Network architecture for processing agent observations
        self.cur_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Network architecture for processing landmark positions
        self.landmark_embedding = nn.Sequential(
            nn.Linear(self.landmark_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
       
        # Network for processing all agents' positions
        self.all_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Diamond attention mechanism for cross attention
        self.agent_landmark = DiamondAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        self.cur_agent = DiamondAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        self.agent_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 6),
        )

        self.cur_landmark = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, obs):
        """Extract features from observations using the attention mechanism"""
        cur_agent, landmarks, other_agents, random_numbers = env_parser(obs, self.number_agents, self.number_food, self.agent_dim, self.landmark_dim)

        random_numbers = get_permuted_env_random_numbers(
            random_numbers, self.number_agents, obs.shape[0], cur_agent.device
        ).view(-1, self.number_agents)

        all_agents_list = torch.cat((cur_agent.unsqueeze(1), other_agents), dim=1)
        # Encode current agent
        cur_agent_embeddings = self.cur_agent_embedding(cur_agent)
        
        # Encode landmarks
        landmark_embeddings = self.landmark_embedding(
            landmarks.reshape(-1, self.landmark_dim)
        ).view(-1, self.number_food, self.hidden_dim)
        
        # Encode all agents
        all_agents_embeddings = self.all_agent_embedding(
            all_agents_list.view(-1, self.agent_dim)
        ).view(-1, self.number_agents, self.hidden_dim)
        # Create attention mask for cross attention
        agents_mask = ~(random_numbers >= random_numbers[:, 0].view(-1, 1))

        agent_emb, landmark_emb, cross_weights = self.agent_landmark(
            landmark_embeddings,
            all_agents_embeddings,
            landmark_embeddings,
            all_agents_embeddings,
            key_mask=agents_mask,
        )
        
        if not self.training:
            self.cross_attention_weights = cross_weights
        else:
            del cross_weights

        landmark_emb = self.cur_landmark(landmark_emb).view(-1, self.number_food) # (batch_size * num_agents, number_food)

        _, cur_agent_embeddings, _ = self.cur_agent(
            cur_agent_embeddings.unsqueeze(1),
            agent_emb,
            cur_agent_embeddings.unsqueeze(1),
            agent_emb,
            key_mask=agents_mask,
        )

        cur_agent_embeddings = self.agent_processor(cur_agent_embeddings.squeeze(1)) # (batch_size * num_agents, 6)

        # Concatenate features for final processing
        latent = torch.cat((cur_agent_embeddings, landmark_emb), dim=-1)

        return latent.view(-1, self.number_agents, 6 + self.number_food)


class CriticPolicy(nn.Module):
    def __init__(self, obs_size, hidden_dim, out_size):
        super().__init__()
        self.obs_size = obs_size
        self.layer = nn.Sequential(
            nn.Linear(obs_size, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_size),
            nn.SiLU(),
        )
    
    def forward(self, obs):
        obs = obs[..., :-1]
        obs = obs.reshape(-1, self.obs_size)
        obs = self.layer(obs)
        return obs

class InfoMARLExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for InfoMARL.

    Observation space is expected to be (n_agents, n_features+1).
    The last entry is the random number of the agent.
    """

    def __init__(
        self,
        observation_space,
        agent_dim=8,
        landmark_dim=8,
        hidden_dim=64,
        critic=False,
    ):
        self.n_agents = observation_space.shape[0]
        self.n_food = (observation_space.shape[1] - self.n_agents * agent_dim - 1) // landmark_dim
        super(InfoMARLExtractor, self).__init__(observation_space, 6 + self.n_food)
        if critic:
            self.critic = CriticPolicy(
                obs_size=self.n_agents*(observation_space.shape[-1]-1),
                hidden_dim=hidden_dim,
                out_size=6 + self.n_food,
            )
        else:
            self.actor = RandomAgentPolicy(
                number_agents=self.n_agents,
                number_food=self.n_food,
                agent_dim=agent_dim,
                landmark_dim=landmark_dim,
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


from stable_baselines3.common.distributions import (
    make_proba_distribution,
)

class InfoMARLActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule = 3e-4,
        net_arch = None,
        activation_fn = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class = InfoMARLExtractor,
        features_extractor_kwargs = None,
        normalize_images: bool = True,
        optimizer_class = torch.optim.Adam,
        optimizer_kwargs = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
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
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = dict(pi=[], vf=[])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # Feature extractor is never shared between pi and vf
        self.share_features_extractor = False
        self.features_extractor_kwargs.update({"critic": False})
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        self.pi_features_extractor = self.features_extractor
        self.features_extractor_kwargs.update({"critic": True})
        self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (squash_output and not use_sde), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
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

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space, self.log_std_init, self.dist_kwargs
        )

        self._build(lr_schedule)
    
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Sequential(
            nn.Flatten(),
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]