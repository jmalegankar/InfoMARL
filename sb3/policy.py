from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

import torch
import torch.nn as nn
import warnings

from bridge_attn import DiamondAttention

import random

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

def env_parser(obs:torch.Tensor, num_good:int, num_adversaries:int):
    random_numbers = obs[..., -1]
    adv_rnd = random_numbers[..., :num_adversaries]
    good_rnd = random_numbers[..., num_adversaries:num_adversaries+num_good]
    cur_pos = obs[: ,:, 0:2].view(-1, num_adversaries+num_good, 2)
    cur_vel = obs[: ,:, 2:4].view(-1, num_adversaries+num_good, 2)
    lmsize = 4+num_good*2
    landmarks = obs[:, :, 4:lmsize].reshape(-1, num_good+num_adversaries, num_good, 2)
    oth_adv = obs[:, :num_adversaries, lmsize:lmsize+2*num_adversaries-2].reshape(-1, num_adversaries-1, 2)
    goods = obs[:, :num_adversaries, lmsize+2*num_adversaries-2:-1].reshape(-1, num_good, 2)
    advs = obs[:, num_adversaries:, lmsize:lmsize+2*num_adversaries].reshape(-1, num_adversaries, 2)
    oth_good = obs[:, num_adversaries:, lmsize+2*num_adversaries:-1].reshape(-1, num_good-1, 2)
    return cur_pos, cur_vel, landmarks, oth_adv, goods, advs, oth_good, adv_rnd, good_rnd

class RandomAgentPolicy(nn.Module):
    def __init__(self, num_good, num_adversaries, agent_dim, landmark_dim, hidden_dim):
        super().__init__()
        self.num_good = num_good
        self.num_adversaries = num_adversaries
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

        self.oth_adv_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.good_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.adv_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.oth_good_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Attention mechanism for data mixing
        self.adv_good_lmk = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )
        self.adv_good_lmk_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.good_adv = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

        self.good_adv_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
       
        # Diamond attention mechanism for cross attention
        self.good_landmark = DiamondAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        self.good_adversary = DiamondAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        self.adversary_landmark = DiamondAttention(
            hidden_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
        )

        # Attention mechanism for current projection
        self.adv_good_curr = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

        self.good_adv_curr_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.adv_curr = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

        self.adv_curr_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.good_lmk = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

        self.good_lmk_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.good_curr = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=1,
            dropout=0.0,
            batch_first=True,
        )

        self.good_curr_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.adv_latent_processing = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

        self.good_latent_processing = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )

    def forward(self, obs):
        """Extract features from observations using the attention mechanism"""
        cur_pos, cur_vel, landmarks, oth_adv, goods, advs, oth_good, adv_rnd, good_rnd = env_parser(
            obs, self.num_good, self.num_adversaries
        )

        adv_rnd = get_permuted_env_random_numbers(
            adv_rnd, self.num_adversaries, obs.shape[0], obs.device
        ).view(-1, self.num_adversaries)

        good_rnd = get_permuted_env_random_numbers(
            good_rnd, self.num_good, obs.shape[0], obs.device
        ).view(-1, self.num_good)

        cur_agent = torch.cat((cur_pos, cur_vel), dim=-1)
        # Encode current agent
        cur_agent_embeddings = self.cur_agent_embedding(cur_agent)

        cur_adv_embeddings = cur_agent_embeddings[:, :self.num_adversaries, :].reshape(-1, self.hidden_dim)
        cur_good_embeddings = cur_agent_embeddings[:, self.num_adversaries:, :].reshape(-1, self.hidden_dim)
        
        # Encode landmarks
        landmark_embeddings = self.landmark_embedding(landmarks)

        landmark_embeddings_adv = landmark_embeddings[:, :self.num_adversaries, :, :].reshape(-1, self.num_good, self.hidden_dim)
        landmark_embeddings_good = landmark_embeddings[:, self.num_adversaries:, :, :].reshape(-1, self.num_good, self.hidden_dim)

        # Encode other agents
        oth_adv = torch.cat((cur_pos[:, :self.num_adversaries, :].reshape(-1, 1, 2), oth_adv), dim=-2)
        oth_adv_embeddings = self.oth_adv_embedding(oth_adv)

        # Encode goods
        goods_embeddings = self.good_embedding(goods)

        # Encode adversaries
        advs_embeddings = self.adv_embedding(advs)

        # Encode other goods
        oth_good = torch.cat((cur_pos[:, self.num_adversaries:, :].reshape(-1, 1, 2), oth_good), dim=-2)
        oth_good_embeddings = self.oth_good_embedding(oth_good)

        # Create attention mask for cross attention
        good_mask = ~(good_rnd >= good_rnd[:, 0].view(-1, 1))
        adv_mask = ~(adv_rnd >= adv_rnd[:, 0].view(-1, 1))

        goods_lmk, _ = self.adv_good_lmk(goods_embeddings, landmark_embeddings_adv, landmark_embeddings_adv)
        goods_lmk = self.adv_good_lmk_mixer(
            torch.cat((goods_lmk, goods_embeddings), dim=-1)
        )

        adv_good_emb, adv_emb, adv_good_weights = self.good_adversary(
            goods_lmk,
            oth_adv_embeddings,
            goods_lmk,
            oth_adv_embeddings,
            key_mask=adv_mask,
        )

        good_adv, _ = self.good_adv(
            oth_good_embeddings,
            advs_embeddings,
            advs_embeddings,
        )
        good_adv = self.good_adv_mixer(
            torch.cat((good_adv, oth_good_embeddings), dim=-1)
        )

        landmark_emb, good_emb, good_lmk_weights = self.good_landmark(
            landmark_embeddings_good,
            good_adv,
            landmark_embeddings_good,
            good_adv,
            key_mask=good_mask,
        )

        if not self.training:
            self.cross_attention_weights = (adv_good_weights, good_lmk_weights)

        
        adv_good_curr, _ = self.adv_good_curr(
            cur_adv_embeddings.unsqueeze(1),
            adv_good_emb,
            adv_good_emb,
            key_padding_mask=adv_mask,
            need_weights=False,
        )

        adv_good_curr = self.good_adv_curr_mixer(
            torch.cat((adv_good_curr.squeeze(1), cur_adv_embeddings), dim=-1
        ))

        adv_curr, adv_weights = self.adv_curr(
            cur_adv_embeddings.unsqueeze(1),
            adv_emb,
            adv_emb,
            need_weights=True,
        )

        adv_curr = self.adv_curr_mixer(
            torch.cat((adv_curr.squeeze(1), cur_adv_embeddings), dim=-1)
        )

        good_lmk, good_lmk_weights = self.good_lmk(
            cur_good_embeddings.unsqueeze(1),
            landmark_emb,
            landmark_emb,
            need_weights=True,
        )

        good_lmk = self.good_lmk_mixer(
            torch.cat((good_lmk.squeeze(1), cur_good_embeddings), dim=-1)
        )

        good_curr, _ = self.good_curr(
            cur_good_embeddings.unsqueeze(1),
            good_emb,
            good_emb,
            key_padding_mask=good_mask,
            need_weights=False,
        )

        good_curr = self.good_curr_mixer(
            torch.cat((good_curr.squeeze(1), cur_good_embeddings), dim=-1)
        )

        if not self.training:
            self.landmark_attention_weights = (adv_weights, good_lmk_weights)

        latent_adv = torch.cat((adv_curr, adv_good_curr), dim=-1)
        latent_good = torch.cat((good_curr, good_lmk), dim=-1)

        latent_adv = self.adv_latent_processing(latent_adv)
        latent_good = self.good_latent_processing(latent_good)

        latent = torch.cat((latent_adv, latent_good), dim=-2)

        return latent.view(-1, self.num_adversaries+self.num_good, self.hidden_dim)


class CriticPolicy(nn.Module):
    def __init__(self, obs_size, hidden_dim):
        super().__init__()
        self.obs_size = obs_size
        self.layer = nn.Sequential(
            nn.Linear(obs_size, hidden_dim*4),
            nn.SiLU(),
            nn.Linear(hidden_dim*4, hidden_dim*4),
            nn.SiLU(),
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
        )
    
    def forward(self, obs):
        obs = obs[..., :-1]
        obs = obs.reshape(-1, self.obs_size)
        obs = self.layer(obs)
        return obs

class ActionNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_adversaries, num_good):
        super(ActionNet, self).__init__()
        self.num_adversaries = num_adversaries
        self.num_good = num_good
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.good_fc = nn.Linear(input_dim, output_dim)
        self.adversary_fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the action network.
        """
        good_actions = self.good_fc(x[:, :self.num_good, :])
        adv_actions = self.adversary_fc(x[:, self.num_good:, :])
        actions = torch.cat((adv_actions, good_actions), dim=1).view(-1, self.output_dim* (self.num_adversaries + self.num_good))
        return actions

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
        num_good=1,
        num_adversaries=1,
        critic=False,
    ):
        super(InfoMARLExtractor, self).__init__(observation_space, hidden_dim)
        self.num_good = num_good
        self.num_adversaries = num_adversaries
        self.n_agents = num_good + num_adversaries
        if critic:
            self.critic = CriticPolicy(
                obs_size=self.n_agents*(observation_space.shape[-1]-1),
                hidden_dim=hidden_dim,
            )
        else:
            self.actor = RandomAgentPolicy(
                num_good=num_good,
                num_adversaries=num_adversaries,
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

        # Update log_prob method of the distribution to flip the sign for adversaries
        num_adversaries = self.features_extractor_kwargs.get('num_adversaries', 0)
        self.counter = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.counter.requires_grad = False
        class SignFlipFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor, counter):
                ctx.save_for_backward(counter.clone().detach())
                return input_tensor

            @staticmethod
            def backward(ctx, grad_output):
                counter = ctx.saved_tensors[0]
                grad_output = grad_output.clone()
                if num_adversaries > 0:
                    # Flip the sign for adversaries
                    grad_output[:, :num_adversaries*2] *= -1
                    # grad_output[:, num_adversaries*2:] *= counter/150
                    # grad_output[:, :num_adversaries*2] *= (300 - counter) / 150
                return grad_output, None

        def new_log_prob(actions):
            self.counter += 1
            self.counter %= 300
            log_prob = self.action_dist.distribution.log_prob(actions)
            log_prob = SignFlipFunction.apply(log_prob, self.counter)
            return log_prob.sum(dim=-1)
        
        self.action_dist.log_prob = new_log_prob


        self._build(lr_schedule)
    
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = ActionNet(
            input_dim=self.mlp_extractor.latent_dim_pi,
            output_dim=self.action_space.shape[-1],
            num_adversaries=self.features_extractor_kwargs.get('num_adversaries', 0),
            num_good=self.features_extractor_kwargs.get('num_good', 0),
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]