import torch
from torch import nn


class DiamondAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        if self.head_dim * num_heads != hidden_dim:
            raise ValueError(f"hidden_dim must be divisible by num_heads (got {hidden_dim} and {num_heads})")
        
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.lvalue_projection = nn.Linear(hidden_dim, hidden_dim)

        self.left_residual = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.right_residual = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, lvalue: torch.Tensor, rvalue: torch.Tensor, attn_mask=None, key_mask=None):
        """
        Forward pass for the diamond attention mechanism.
        
        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            rvalue (torch.Tensor): The right value tensor.
            lvalue (torch.Tensor): The left value tensor.
            attn_mask (torch.Tensor, optional): An optional attention mask.
            key_mask (torch.Tensor, optional): An optional key mask.

        Returns:
            tuple: A tuple containing the left attention output, right attention output, and attention weights.
        """
        attn_right, weights = self.attn(query, key, rvalue, attn_mask=attn_mask, need_weights=True, key_padding_mask=key_mask)
        # Apply the query projection
        attn_left = torch.bmm(weights.transpose(2, 1), self.lvalue_projection(lvalue))
        attn_left = torch.cat((attn_left, rvalue), dim=-1)
        attn_left = self.left_residual(attn_left)
        attn_right = torch.cat((attn_right, lvalue), dim=-1)
        attn_right = self.right_residual(attn_right)

        return attn_left, attn_right, weights

class BridgeAttention(nn.Module):
    def __init__(self, hidden_dim:int, num_heads:int, dropout:float=0.0):
        super(BridgeAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.task_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.task_residue = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.agent_residue = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, agent_emb, task_emb, key_mask=None):
        """
        agent_emb: (batch_size, num_agents, hidden_dim)
        task_emb: (batch_size, num_tasks, hidden_dim)
        """
        batch_size, num_agents, _ = agent_emb.shape
        _, num_tasks, _ = task_emb.shape

        weighted_agent_emb, weights = self.attn(
            task_emb,
            agent_emb,
            agent_emb,
            key_padding_mask=key_mask,
            need_weights=True,
        ) # ( batch_size, num_tasks, hidden_dim ), ( batch_size, num_tasks, num_agents )

        weighted_task_emb = torch.bmm(weights.transpose(2, 1), self.task_processor(task_emb)) # ( batch_size, num_agents, hidden_dim )

        task_residue = self.task_residue(torch.cat((weighted_agent_emb, task_emb), dim=-1)) # ( batch_size, num_tasks, hidden_dim )
        agent_residue = self.agent_residue(torch.cat((weighted_task_emb, agent_emb), dim=-1)) # ( batch_size, num_agents, hidden_dim )

        return task_residue, agent_residue, weights # ( batch_size, num_tasks, hidden_dim ), ( batch_size, num_agents, hidden_dim ), ( batch_size, num_tasks, num_agents )