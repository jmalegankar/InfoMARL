from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

import torch
import torch.nn as nn
import warnings

from bridge_attn import BridgeAttention
from utils import env_parser, get_permuted_env_random_numbers

class RandomAgentPolicy(nn.Module):
    def __init__(self, number_agents, agent_dim, landmark_dim, hidden_dim):
        super().__init__()
        self.number_agents = number_agents
        self.agent_dim = agent_dim
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim
        self.training = True  # Add this to control behavior

        # Network architecture for processing agent observations
        self.cur_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Network architecture for processing landmark positions
        self.landmark_embedding = nn.Sequential(
            nn.Linear(landmark_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
       
        # Network for processing all agents' positions
        self.all_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Bridge attention mechanism for cross attention
        self.agent_landmark = BridgeAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        self.cur_agent = BridgeAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        self.cur_landmark = BridgeAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

    def forward(self, obs):
        """Extract features from observations using the attention mechanism"""
        cur_pos, cur_vel, landmarks, other_agents, random_numbers = env_parser(obs, self.number_agents)
        random_numbers = get_permuted_env_random_numbers(
            random_numbers, self.number_agents, obs.shape[0], cur_pos.device
        ).view(-1, self.number_agents)
        cur_agent = torch.cat((cur_pos, cur_vel), dim=-1)
        all_agents_list = torch.cat((cur_pos.unsqueeze(1), other_agents), dim=1)

        # Encode current agent
        cur_agent_embeddings = self.cur_agent_embedding(cur_agent)
        
        # Encode landmarks
        landmark_embeddings = self.landmark_embedding(
            landmarks.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        # Encode all agents
        all_agents_embeddings = self.all_agent_embedding(
            all_agents_list.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        # Create attention mask for cross attention
        agents_mask = ~(random_numbers >= random_numbers[:, 0].view(-1, 1))

        landmark_emb, agent_emb, cross_weights = self.agent_landmark(
            all_agents_embeddings,
            landmark_embeddings,
            agents_mask,
        )
        
        if not self.training:
            self.cross_attention_weights = cross_weights

        landmark_emb, _, landmark_weights = self.cur_landmark(
            landmark_embeddings,
            cur_agent_embeddings.unsqueeze(1),
        )

        if not self.training:
            self.landmark_attention_weights = landmark_weights

        cur_agent_embeddings, _, _ = self.cur_agent(
            agent_emb,
            cur_agent_embeddings.unsqueeze(1),
            agents_mask,
        )

        # Concatenate features for final processing
        latent = torch.cat((cur_agent_embeddings.squeeze(1), landmark_emb.squeeze(1)), dim=-1)

        return latent.view(-1, self.number_agents, self.hidden_dim*2)


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
    Custom feature extractor for InfoMARL.

    Observation space is expected to be (n_agents, n_features+1).
    The last entry is the random number of the agent.
    """

    def __init__(
        self,
        observation_space,
        agent_dim=2,
        landmark_dim=2,
        hidden_dim=64,
        critic=False,
    ):
        super(InfoMARLExtractor, self).__init__(observation_space, hidden_dim*2)
        self.n_agents = observation_space.shape[0]
        if critic:
            self.critic = CriticPolicy(
                obs_size=self.n_agents*(observation_space.shape[-1]-1),
                hidden_dim=hidden_dim,
            )
        else:
            self.actor = RandomAgentPolicy(
                number_agents=self.n_agents,
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
            net_arch = dict(pi=[64, 64], vf=[64, 64])

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
            nn.Linear(self.mlp_extractor.latent_dim_pi, self.action_space.shape[-1]),
            nn.Flatten(),
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]