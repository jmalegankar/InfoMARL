"""
Observation parser utility for the randomized attention policy
"""

import torch

def parse_observation(obs, n_agents, n_landmarks):
    """
    Parse observation for randomized attention policy
    
    For VMAS simple_spread environment:
    - Extract agent state (position, velocity)
    - Extract landmark positions
    - Format them properly for the attention mechanism
    
    Args:
        obs: Raw observation tensor
        n_agents: Number of agents in the environment
        n_landmarks: Number of landmarks in the environment
        
    Returns:
        Dictionary containing parsed observation components
    """
    # Parse based on observation structure from VMAS simple_spread
    # Assuming obs contains: [position, velocity, relative positions to landmarks]
    
    agent_pos = obs[:2]  # x, y position
    agent_vel = obs[2:4]  # x, y velocity
    
    # Extract landmark relative positions
    landmarks = []
    for i in range(n_landmarks):
        # In simple_spread, landmark relative positions start after agent state
        # Each landmark has 2 values (x, y relative position)
        start_idx = 4 + i * 2
        landmark_rel_pos = obs[start_idx:start_idx+2]
        landmarks.append(landmark_rel_pos)
    
    landmark_positions = torch.stack(landmarks) if landmarks else torch.zeros((0, 2))
    
    return {
        "position": agent_pos,
        "velocity": agent_vel,
        "landmarks": landmark_positions
    }