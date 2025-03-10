import torch as th
import numpy as np
import os
from typing import Dict, Any, Optional

def update_target_networks(target_net, source_net, tau):
    """
    Soft update of target network parameters
    
    Args:
        target_net: Target network to update
        source_net: Source network to copy from
        tau: Interpolation parameter (0 = no update, 1 = complete replacement)
    """
    with th.no_grad():
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def save_model(state_dict: Dict[str, Any], path: str) -> None:
    """
    Save model state dictionary to disk
    
    Args:
        state_dict: State dictionary containing model parameters
        path: Path to save model to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    th.save(state_dict, path)

def load_model(path: str, device: Optional[th.device] = None) -> Dict[str, Any]:
    """
    Load model state dictionary from disk
    
    Args:
        path: Path to load model from
        device: Device to place tensors on
        
    Returns:
        state_dict: State dictionary containing model parameters
    """
    state_dict = th.load(path, map_location=device)
    return state_dict

def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation
    
    Args:
        rewards: Rewards [T, B, N]
        values: Value estimates [T, B, N]
        next_values: Next value estimates [T, B, N]
        dones: Done flags [T, B]
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: Advantage estimates
        returns: Return estimates
    """
    T, B, N = rewards.shape
    advantages = th.zeros((T, B, N), device=rewards.device)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t].unsqueeze(-1)
            next_value = next_values[t]
        else:
            next_non_terminal = 1.0 - dones[t].unsqueeze(-1)
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae
    
    returns = advantages + values
    return advantages, returns

def linear_annealing(start_value, end_value, current_step, total_steps):
    """
    Linear annealing schedule
    
    Args:
        start_value: Initial value
        end_value: Final value
        current_step: Current step
        total_steps: Total number of steps
        
    Returns:
        value: Annealed value
    """
    fraction = min(current_step / total_steps, 1.0)
    return start_value + fraction * (end_value - start_value)