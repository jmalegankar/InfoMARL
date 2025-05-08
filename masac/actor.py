
import torch
import torch.nn as nn
from torch.distributions import Normal


def env_parser(obs:torch.Tensor, number_agents:int=3):
    #cur agents pos
    cur_pos = obs[: ,0:2]
    #print("cur_pos", cur_pos, cur_pos.shape)
    #cur agents vel
    cur_vel = obs[: ,2:4]
    #print("cur_vel", cur_vel, cur_vel.shape)
    #landmarks pos 
    landmarks = obs[:, 4:4 + 2 * number_agents]
    #print("landmarks", landmarks, landmarks.shape)
    #other agents pos
    other_agents = obs[:, 4 + 2 * number_agents:]
    #print("other_agents", other_agents, other_agents.shape)
    if number_agents == 1:
        landmarks = landmarks.unsqueeze(-2)
        other_agents = other_agents.unsqueeze(-2)
        num_envs = cur_pos.shape[0]
        return cur_pos.view(num_envs, 2), cur_vel.view(num_envs, 2), landmarks.contiguous(), other_agents.contiguous().view(num_envs, 0, 2)
    else:
        return cur_pos, cur_vel, landmarks.contiguous().reshape(-1, number_agents, 2), other_agents.contiguous().reshape(-1, (number_agents - 1), 2)

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
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Network architecture for processing landmark positions
        self.landmark_embedding = nn.Sequential(
            nn.Linear(landmark_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.landmark_value = nn.Sequential(
            nn.Linear(landmark_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        
        # Network for processing all agents' positions
        self.all_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        
        # Attention mechanisms
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1, batch_first=True)
        self.processor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.landmark_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1, batch_first=True)
        
        # Policy network (actor)
        self.mean_processor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2)  # Output mean for 2D actions
        )
        
        self.log_std_processor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2)  # Output log_std for 2D actions
        )
        
        # Value network (critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1)  # Output state value
        )

        # Initialize weights appropriately
        self.log_std_processor[-1].bias.data.fill_(-0.5)  # Start with moderate exploration
        self.mean_processor[-1].bias.data.fill_(0.0)      # Initialize to zero mean output
    
    def get_features(self, obs, random_numbers):
        """Extract features from observations using the attention mechanism"""
        cur_pos, cur_vel, landmarks, other_agents = env_parser(obs, self.number_agents)
        cur_agent = torch.cat((cur_pos, cur_vel), dim=-1)
        all_agents_list = torch.cat((cur_pos.unsqueeze(1), other_agents), dim=1)

        # Encode current agent
        cur_agent_embeddings = self.cur_agent_embedding(cur_agent)
        
        # Encode landmarks
        landmark_embeddings = self.landmark_embedding(
            landmarks.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        landmark_value = self.landmark_value(
            landmarks.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        # Encode all agents
        all_agents_embeddings = self.all_agent_embedding(
            all_agents_list.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        # Create attention mask for cross attention
        agents_mask = ~(random_numbers >= random_numbers[:, 0].view(-1, 1))
        
        # Cross attention between landmarks and all agents
        attention_output, _ = self.cross_attention(
            query=landmark_embeddings,
            key=all_agents_embeddings,
            value=all_agents_embeddings,
            attn_mask=agents_mask.unsqueeze(-2).repeat(1, self.number_agents, 1),
            need_weights=False
        )

        attention_output = self.processor(attention_output)

        # Attention between current agent and processed landmarks
        attention_output = self.landmark_attention(
            cur_agent_embeddings.unsqueeze(dim=-2), 
            attention_output, 
            landmark_value, 
            need_weights=False
        )[0].squeeze(dim=-2)

        # Concatenate features for final processing
        latent = torch.cat((cur_agent_embeddings, attention_output), dim=-1)
        
        return latent

    def get_policy_dist(self, latent):
        """Get policy distribution parameters from latent features"""
        mean = self.mean_processor(latent)
        
        # Get log_std with constraints for numerical stability
        log_std = self.log_std_processor(latent)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent extreme values
        
        return mean, log_std
    
    def get_value(self, latent):
        """Get state value from latent features"""
        return self.value_head(latent)
    
    def evaluate_actions(self, obs, actions, random_numbers):
        """Evaluate actions for PPO update"""
        latent = self.get_features(obs, random_numbers)
        
        # Get policy distribution
        mean, log_std = self.get_policy_dist(latent)
        std = log_std.exp()
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # For actions already in [-1, 1] from tanh squashing
        # We need to apply inverse tanh to get the original sampled actions 
        actions_clipped = torch.clamp(actions, -0.999, 0.999)  # Avoid numerical issues
        unsquashed_actions = 0.5 * torch.log((1 + actions_clipped) / (1 - actions_clipped))
        
        # Get log probs from the normal distribution
        log_prob_normal = dist.log_prob(unsquashed_actions)
        
        # Correction for tanh squashing (log det of Jacobian)
        log_prob_correction = torch.log(1 - actions.pow(2) + 1e-6)
        
        # Total log prob
        action_log_probs = (log_prob_normal - log_prob_correction).sum(-1)
        
        # Get entropy (approximation for squashed distribution)
        entropy = dist.entropy().sum(-1)
        
        # Get state values
        values = self.get_value(latent).squeeze(-1)
        
        return action_log_probs, values, entropy
    
    def forward(self, obs, random_numbers):
        """Forward pass for action selection and evaluation with debugging"""
        latent = self.get_features(obs, random_numbers)
        
        # Get policy distribution parameters
        mean, log_std = self.get_policy_dist(latent)
        std = log_std.exp()
        
        # Debug distribution parameters
        if torch.rand(1).item() < 0.01:  # Only print for 1% of forward passes to avoid spam
            print(f"\n--- DEBUG: Policy Distribution ---")
            print(f"Mean - Mean: {mean.mean().item():.6f}, Min: {mean.min().item():.6f}, Max: {mean.max().item():.6f}")
            print(f"Std - Mean: {std.mean().item():.6f}, Min: {std.min().item():.6f}, Max: {std.max().item():.6f}")
        
        # Create normal distribution
        dist = Normal(mean, std)
        
        # Sample actions or use mean based on training mode
        if self.training:
            unsquashed_actions = dist.sample()
            sampling_mode = "sampling (training)"
        else:
            unsquashed_actions = mean  # Use deterministic actions during evaluation
            sampling_mode = "using mean (evaluation)"
        
        # Apply tanh squashing for bounded actions [-1, 1]
        actions = torch.tanh(unsquashed_actions)
        
        # Calculate log probabilities with correction for squashing
        log_prob_normal = dist.log_prob(unsquashed_actions)
        log_prob_correction = torch.log(1 - actions.pow(2) + 1e-6)
        log_probs = (log_prob_normal - log_prob_correction).sum(-1)
        
        # Get state values
        values = self.get_value(latent).squeeze(-1)
        
        # Debug occasionally
        if torch.rand(1).item() < 0.01:  # Only print for 1% of forward passes
            print(f"Action selection mode: {sampling_mode}")
            print(f"Unsquashed actions - Mean: {unsquashed_actions.mean().item():.6f}, "
                f"Min: {unsquashed_actions.min().item():.6f}, Max: {unsquashed_actions.max().item():.6f}")
            print(f"Actions (after tanh) - Mean: {actions.mean().item():.6f}, "
                f"Min: {actions.min().item():.6f}, Max: {actions.max().item():.6f}")
            print(f"Log probs - Mean: {log_probs.mean().item():.6f}, Min: {log_probs.min().item():.6f}, "
                f"Max: {log_probs.max().item():.6f}")
            print(f"Values - Mean: {values.mean().item():.6f}, Min: {values.min().item():.6f}, "
                f"Max: {values.max().item():.6f}")
            print("--- End of policy debug ---\n")
    
        return actions, log_probs, values
        
    def train(self, mode=True):
        """Override train method to set training flag"""
        super().train(mode)
        self.training = mode
        return self
        
    def eval(self):
        """Override eval method to set training flag to False"""
        super().eval()
        self.training = False
        return self