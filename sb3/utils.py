import torch

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

def env_parser(obs:torch.Tensor, number_agents:int):
    random_numbers = obs[..., -1]
    obs = obs.view(-1, obs.shape[-1])
    cur_pos = obs[: ,0:2]
    cur_vel = obs[: ,2:4]
    landmarks = obs[:, 4:4 + 2 * number_agents].contiguous().reshape(-1, number_agents, 2) + cur_pos.unsqueeze(1)
    other_agents = obs[:, 4 + 2 * number_agents:-1].contiguous().reshape(-1, (number_agents - 1), 2) + cur_pos.unsqueeze(1)
    return cur_pos, cur_vel, landmarks, other_agents, random_numbers