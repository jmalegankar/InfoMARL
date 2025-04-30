import torch
import torch.nn as nn

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

        self.cur_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        self.landmark_embedding = nn.Sequential(
            nn.Linear(landmark_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.landmark_value = nn.Sequential(
            nn.Linear(landmark_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        self.all_agent_embedding = nn.Sequential(
            nn.Linear(self.agent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1, batch_first=True)
        self.landmark_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=1, batch_first=True)
        self.mean_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )
        self.std_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )
    
    def get_mean_std(self, obs, random_numbers):
        cur_pos, cur_vel, landmarks, other_agents = env_parser(obs, self.number_agents)
        batch_size = cur_pos.shape[0]
        cur_agent = torch.cat((cur_pos, cur_vel), dim=-1)
        all_agents_list = torch.cat((cur_pos.unsqueeze(1), other_agents), dim=1)

 
        cur_agent_embeddings = self.cur_agent_embedding(cur_agent)
        landmark_embeddings = self.landmark_embedding(
            landmarks.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        landmark_value = self.landmark_value(
            landmarks.view(-1, 2)
        ).view(-1, self.number_agents, self.hidden_dim)

        all_agents_embeddings = self.all_agent_embedding(
            all_agents_list.view(-1, 2)  
        ).view(-1, self.number_agents, self.hidden_dim)

        agents_mask = ~(random_numbers >= random_numbers[:, 0].view(-1,1))
        attention_output, _ = self.cross_attention(
            query=landmark_embeddings,
            key=all_agents_embeddings,
            value=all_agents_embeddings,
            attn_mask = agents_mask.unsqueeze(-2).repeat(1, self.number_agents, 1),
            need_weights=False
        )

        attention_output = self.landmark_attention(attention_output, landmark_embeddings, landmark_value, need_weights=False)[0]

        # Take landmark value with maximum cosine similarity for cur_agent_embeddings and attention_output
        attention_output = attention_output / torch.norm(attention_output, dim=-1, keepdim=True)
        cur_agent_embeddings = cur_agent_embeddings / torch.norm(cur_agent_embeddings, dim=-1, keepdim=True)
        cosine_similarity = torch.bmm(attention_output, cur_agent_embeddings.unsqueeze(-1)).squeeze(-1)
        max_indices = torch.argmax(cosine_similarity, dim=-1)
        latent = attention_output[torch.arange(batch_size, device=max_indices.device), max_indices, :]

        mean = self.mean_processor(latent)
        log_std = self.std_processor(latent)
        return mean, log_std

    def forward(self, obs, random_numbers):
        mean, log_std = self.get_mean_std(obs, random_numbers)

        log_std = torch.clamp(log_std, min=-5, max=1)
        log_std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, log_std)

        x_t = normal.rsample()
        
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t) - torch.log((1 - action.pow(2)) + 1e-8)
        
        return action, log_prob.sum(dim=-1).clamp(max=0)