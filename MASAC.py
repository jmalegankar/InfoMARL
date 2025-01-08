import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import vmas
from vmas.simulator.utils import save_video
from collections import deque
import matplotlib.pyplot as plt
import os

###################################
# 0) Utility Functions
###################################
GLOBAL_SEED = 10

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    

def parse_observation(obs: torch.Tensor, n_agents: int, n_landmarks: int):
    # 2 => velocity, 2 => position, 2*n_landmarks => landmarks, leftover => other agents
    velocity = obs[0:2]
    position = obs[2:4]
    landmark_chunk = obs[4 : 4 + 2*n_landmarks]
    landmark_positions = landmark_chunk.view(n_landmarks, 2)
    leftover_start = 4 + 2*n_landmarks
    leftover = obs[leftover_start:]
    n_other = n_agents - 1
    if leftover.size(0) != 2 * n_other:
        raise ValueError(
            f"parse_observation mismatch: leftover={leftover.size(0)}, expected={2*n_other}\nobs={obs}"
        )
    other_agents = leftover.view(n_other, 2)
    return {
        "velocity": velocity,
        "position": position,
        "landmarks": landmark_positions,
        "other_agents": other_agents
    }

###################################
# 1) Environment Wrapper
###################################
class RandomAgentCountEnv:
    def __init__(
        self,
        scenario_name="simple_spread",
        agent_count_dict={1: 0.4, 3: 0.2, 5: 0.2, 7: 0.2},
        device="cpu",
        continuous_actions=True,
        max_steps=None,
        seed=None
    ):
        self.scenario_name = scenario_name
        self.agent_count_dict = agent_count_dict
        self.device = device
        self.continuous_actions = continuous_actions
        self.max_steps = max_steps
        self.seed = seed
        self.agent_counts = list(agent_count_dict.keys())
        self.agent_probs = list(agent_count_dict.values())
        self.agent_rng = np.random.default_rng(seed)
        self.num_landmarks = None
        self.current_num_agents = None
        self.env = None
        self.render_mode = False

        self.reset()

    def sample_agent_count(self):
        return self.agent_rng.choice(self.agent_counts, p=self.agent_probs)
    
    def make_env(self, n_agents):
        return vmas.make_env(
            scenario=self.scenario_name,
            num_envs=1,
            n_agents=n_agents,
            continuous_actions=self.continuous_actions,
            device=self.device,
            seed=self.seed,
            max_steps=self.max_steps,
        )
    
    def reset(self):
        self.current_num_agents = int(self.sample_agent_count())
        self.num_landmarks = self.current_num_agents
        self.env = self.make_env(self.current_num_agents)
        obs = self.env.reset()  # list of length n_agents
        return obs

    def enable_render(self):
        self.render_mode = True
        self.env.render()
    
    def disable_render(self):
        self.render_mode = False
    
    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        n_agents = self.num_agents
        for r in rewards:
            r.div_(n_agents)
        
        if self.render_mode:
            self.env.render()

        return obs, rewards, dones, infos
    
    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
    
    @property
    def num_agents(self):
        return self.current_num_agents


###################################
# 2) Robust Replay Buffer
###################################
class ReplayBuffer:
    """
    Stores transitions with variable agent counts in a dictionary.
    """
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def add(self, n_agents, n_landmarks, obs_list, actions, rewards, next_obs_list, dones, random_numbers):
        transition = {
            "n_agents": n_agents,
            "n_landmarks": n_landmarks,
            "obs_list": obs_list,
            "actions": actions,          # shape [n_agents, action_dim]
            "rewards": rewards,         # list of length n_agents
            "next_obs_list": next_obs_list,
            "dones": dones,              # list of length n_agents
            "random_numbers": random_numbers       # list of length n_agents
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        indices = torch.randint(0, len(self.buffer), (batch_size,), device='cpu').tolist()
        batch = [self.buffer[idx] for idx in indices]
        return batch

    def __len__(self):
        return len(self.buffer)
    

###################################
# 3) Critic
###################################
class CustomQFuncCritic(nn.Module):
    def __init__(self, agent_dim, action_dim, landmark_dim, hidden_dim):
        super().__init__()
        self.agent_dim = agent_dim
        self.action_dim = action_dim
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim

        self.agent_fc = nn.Linear(agent_dim + action_dim, hidden_dim)
        self.landmark_fc = nn.Linear(landmark_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):

        agent_inputs = torch.cat([state['agent_states'], action], dim=-1)
        agent_features = self.agent_fc(agent_inputs)  # => [n_agents, hidden_dim]

        landmark_features = self.landmark_fc(state['landmark_positions'])  # => [n_landmarks, hidden_dim]

        all_features = torch.cat([agent_features, landmark_features], dim=0).unsqueeze(0)
        attn_output, _ = self.attention(all_features, all_features, all_features)
        aggregated = attn_output.mean(dim=1)  # => shape [1, hidden_dim]
        q_value = self.output_layer(aggregated)  # => shape [1,1]
        return q_value.squeeze()


########################################################
# 3) ACTOR: TANH-SQUASHED GAUSSIAN
########################################################
class RandomizedAttentionSACPolicy(nn.Module):

    def __init__(self, agent_dim, landmark_dim, hidden_dim, action_dim, device="cpu"):
        super().__init__()
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(12345)

        self.agent_dim = agent_dim
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Initial attention
        self.init_embed = nn.Linear(agent_dim + landmark_dim, hidden_dim)
        self.init_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Consensus attention
        self.consensus_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Output => (mean, log_std)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.logstd_layer = nn.Linear(hidden_dim, action_dim)

    ################################
    # Tanh-Normal sampling
    ################################
    def tanh_normal_sample(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)

        # diagonal Gaussian log-prob
        var = std.pow(2)
        log_prob_gauss = -0.5 * (((pre_tanh - mean)**2) / (var + 1e-8) + 2*log_std + np.log(2*np.pi))
        log_prob_gauss = log_prob_gauss.sum(dim=-1, keepdim=True)  # => shape [1,1]

        # Tanh correction
        # - sum( log( d/dx tanh(x) ) ) = sum( log( 1 - tanh^2(x) ) ), but we do the stable version:
        # formula: 2 * ( log(2) - x - softplus(-2x) )
        correction = 2.0 * (np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        correction = correction.sum(dim=-1, keepdim=True)  # => [1,1]

        log_prob = log_prob_gauss - correction
        return action, log_prob

    ################################
    # initial_state_estimation
    ################################
    def initial_state_estimation(self, agent_state, landmark_positions):

        n_landmarks = landmark_positions.size(1)
        agent_state_exp = agent_state.unsqueeze(1).repeat(1, n_landmarks, 1)
        tokens = torch.cat([agent_state_exp, landmark_positions], dim=-1)  # => [1,n_landmarks, agent_dim+landmark_dim]

        tokens = tokens.squeeze(0)  # => [n_landmarks, agent_dim+landmark_dim]
        embed = self.init_embed(tokens).unsqueeze(0)  # => [1, n_landmarks, hidden_dim]
        att_out, _ = self.init_attention(embed, embed, embed, need_weights=False)
        temp_state = att_out  # => [1, n_landmarks, hidden_dim]
        message = att_out.mean(dim=1, keepdim=True)  # => [1,1,hidden_dim]
        return temp_state, message

    ################################
    # consensus_process
    ################################
    def create_mask(self, rnd, rnd_list):
        rnd_tensor = rnd_list.clone().detach().to(dtype=torch.float32)  # Convert list to tensor
        mask = (rnd_tensor.unsqueeze(0) > rnd).logical_or(rnd_tensor.unsqueeze(1) > rnd)  # Compare as per M_ij definition
        return mask.float()
    
    def consensus_process(self, temp_state, messages, random_numbers, agent_random_number):

        Q = temp_state  # => [1,n_agents,hidden_dim]
        msg_batch = messages.transpose(1,0)      # => [1,n_agents,hidden_dim]

        # Mask (True => ignore)
        # mask = (random_numbers > agent_random_number)
        mask = self.create_mask(agent_random_number, random_numbers)
        att_out, _ = self.consensus_attention(Q, msg_batch, msg_batch, attn_mask=mask, need_weights=False)
        att_out = att_out.mean(dim=1)
        mean = self.mean_layer(att_out)   # => [1, action_dim]
        log_std = self.logstd_layer(att_out)  # => [1, action_dim]
        return mean, log_std

    ################################
    # sample_actions_and_logp/forward
    ################################
    def sample_actions_and_logp(self, obs_list, n_agents, n_landmarks, random_numbers=None):
        """
        returns:
          actions => [n_agents, action_dim] in [-1,1]
          log_prob => scalar = sum of agent log-probs
        """
        temp_states = []
        msg_list = []

        # 1) init state for each agent
        for i in range(n_agents):
            obs = obs_list[i]
            parsed = parse_observation(obs, n_agents, n_landmarks)
            agent_s = torch.cat([parsed["velocity"], parsed["position"]], dim=0).unsqueeze(0)
            landmark_p = parsed["landmarks"].unsqueeze(0)
            temp_s, msg = self.initial_state_estimation(agent_s, landmark_p)
            temp_states.append(temp_s)
            msg_list.append(msg.squeeze(0))  # => [hidden_dim]

        messages_tensor = torch.stack(msg_list, dim=0)  # => [n_agents, hidden_dim]
        if random_numbers is not None:
            random_tensor = random_numbers
        else:
            random_tensor = torch.rand(n_agents, dtype=torch.float32, device=self.device, generator=self.rng)  # => [n_agents]

        # 2) consensus => produce actions
        all_actions = []
        all_logps = []
        for i in range(n_agents):
            mean_i, log_std_i = self.consensus_process(
                temp_state=temp_states[i],
                messages=messages_tensor,
                random_numbers=random_tensor,
                agent_random_number=random_tensor[i]
            )
            action_i, logp_i = self.tanh_normal_sample(mean_i, log_std_i)
            # action_i => [1, action_dim], logp_i => [1,1]
            all_actions.append(action_i.squeeze(0))   # => [action_dim]
            all_logps.append(logp_i.squeeze(0))       # => scalar

        actions_tensor = torch.stack(all_actions, dim=0)   # => [n_agents, action_dim]
        log_prob_joint = torch.stack(all_logps).sum()      # => scalar
        return actions_tensor, log_prob_joint, random_tensor


###################################
# 4) Train MASAC Agent
###################################
def train_masac_like(
    policy, policy_target,
    critic1, critic1_target,
    critic2, critic2_target,
    policy_optim, critic1_optim, critic2_optim,
    alpha,  # can be float or nn.Parameter
    alpha_optim,  # optional
    target_entropy,  # e.g. - (n_agents * action_dim)
    replay_batch,
    gamma, tau,
    device="cpu",
):
    q_losses = []
    policy_losses = []
    alpha_losses = []

    for transition in replay_batch:
        n_agents = transition["n_agents"]
        n_landmarks = transition["n_landmarks"]
        obs_list = transition["obs_list"]
        actions = transition["actions"].to(device)  # => [n_agents, action_dim]
        rewards = transition["rewards"]
        next_obs_list = transition["next_obs_list"]
        dones = transition["dones"]
        random_numbers = transition["random_numbers"]

        rew_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        done_t = dones.clone().detach().to(dtype=torch.float32, device=device)

        # Build dict for critics
        agent_states = []
        for i in range(n_agents):
            obs_t = obs_list[i].to(device)
            parsed = parse_observation(obs_t, n_agents, n_landmarks)
            agent_states.append(torch.cat([parsed["velocity"], parsed["position"]], dim=0))
        agent_states_tensor = torch.stack(agent_states, dim=0)

        parsed_0 = parse_observation(obs_list[0].to(device), n_agents, n_landmarks)
        landmark_positions = parsed_0["landmarks"].to(device)

        state_dict = {
            "agent_states": agent_states_tensor,
            "landmark_positions": landmark_positions,
        }

        # Q1(s,a), Q2(s,a)
        q1_val = critic1(state_dict, actions)
        q2_val = critic2(state_dict, actions)

        # Next state
        agent_states_next = []
        for i in range(n_agents):
            obs_nt = next_obs_list[i].to(device)
            parsed_n = parse_observation(obs_nt, n_agents, n_landmarks)
            agent_states_next.append(torch.cat([parsed_n["velocity"], parsed_n["position"]], dim=0))
        agent_states_next_t = torch.stack(agent_states_next, dim=0)
        parsed_0_next = parse_observation(next_obs_list[0].to(device), n_agents, n_landmarks)
        landmark_next = parsed_0_next["landmarks"].to(device)
        next_state_dict = {
            "agent_states": agent_states_next_t,
            "landmark_positions": landmark_next
        }

        with torch.no_grad():
            next_actions, next_logp, _ = policy_target.sample_actions_and_logp(
                next_obs_list, n_agents, n_landmarks, random_numbers=random_numbers
            )
            q1_next = critic1_target(next_state_dict, next_actions)
            q2_next = critic2_target(next_state_dict, next_actions)
            q_next_min = torch.min(q1_next, q2_next)

            sum_reward = rew_t.sum()  # or rew_t.mean()
            done_any = done_t.max()
            backup = sum_reward + gamma*(1.0 - done_any)*(q_next_min - alpha*next_logp)

        # Critic losses
        critic1_loss = F.mse_loss(q1_val, backup.detach())
        critic2_loss = F.mse_loss(q2_val, backup.detach())
        critic_loss = critic1_loss + critic2_loss

        critic1_optim.zero_grad()
        critic2_optim.zero_grad()
        critic_loss.backward()
        critic1_optim.step()
        critic2_optim.step()

        # Policy update
        curr_actions, logp_curr, _ = policy.sample_actions_and_logp(obs_list, n_agents, n_landmarks)
        q1_pi = critic1(state_dict, curr_actions)
        q2_pi = critic2(state_dict, curr_actions)
        q_pi_min = torch.min(q1_pi, q2_pi)
        policy_loss = alpha*logp_curr - q_pi_min

        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        # alpha update (if learnable)
        alpha_loss_val = 0.0
        if alpha_optim is not None and isinstance(alpha, nn.Parameter):
            # alpha_loss = - alpha * (logp + target_entropy)
            alpha_loss_val = -(alpha*(logp_curr + target_entropy).detach())
            alpha_optim.zero_grad()
            alpha_loss_val.backward()
            alpha_optim.step()
            alpha_loss_val = alpha_loss_val.item()

        q_losses.append(critic_loss.item())
        policy_losses.append(policy_loss.item())
        alpha_losses.append(alpha_loss_val)

        # Soft update critics
        with torch.no_grad():
            for tp, sp in zip(critic1_target.parameters(), critic1.parameters()):
                tp.data.copy_(tau*sp.data + (1 - tau)*tp.data)
            for tp, sp in zip(critic2_target.parameters(), critic2.parameters()):
                tp.data.copy_(tau*sp.data + (1 - tau)*tp.data)

    return (
        float(np.mean(q_losses)),
        float(np.mean(policy_losses)),
        float(np.mean(alpha_losses)),
    )

def train_masac(
    env_wrapper,
    policy, policy_target,
    critic1, critic1_target,
    critic2, critic2_target,
    buffer,
    agent_dim=4,
    landmark_dim=2,
    action_dim=2,
    n_episodes=1000,
    max_steps=25,
    batch_size=32,
    gamma=0.99,
    lr_actor=1e-3,
    lr_critic=1e-3,
    tau=0.005,
    alpha=0.2,            # can be float or nn.Parameter
    learnable_alpha=False,
    target_entropy=-4.0,  # e.g. - (n_agents*action_dim)
    lr_alpha=1e-3,
    device="cpu",
    save_interval=100,
    save_dir="checkpoints",
    video_dir="videos",
    eval_interval=100,
    num_eval_episodes=5,
    save_video_every=10,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    policy_optim = optim.Adam(policy.parameters(), lr=lr_actor)
    critic1_optim = optim.Adam(critic1.parameters(), lr=lr_critic)
    critic2_optim = optim.Adam(critic2.parameters(), lr=lr_critic)

    alpha_optim = None
    if learnable_alpha:
        alpha = nn.Parameter(torch.tensor(alpha, device=device, requires_grad=True))
        alpha_optim = optim.Adam([alpha], lr=lr_alpha)

    os.makedirs("save_dir", exist_ok=True)

    #visualzation
    rewards_history = []
    moving_avg_rewards = []
    moving_avg_window = 100
    rewards_q = deque(maxlen=moving_avg_window)

    for ep in range(n_episodes):
        frames = []

        obs_list = env_wrapper.reset()
        n_agents = env_wrapper.num_agents
        obs_list = [o.float().to(device).squeeze() for o in obs_list]

        ep_reward = 0.0
        for step in range(max_steps):
            with torch.no_grad():
                actions_t, _, random_t = policy.sample_actions_and_logp(obs_list, n_agents, env_wrapper.num_landmarks)
            # convert each agent's action => [1, action_dim]
            actions_for_env = []
            for i in range(n_agents):
                actions_for_env.append(actions_t[i].unsqueeze(0))

            next_obs_list, rewards, dones, _ = env_wrapper.step(actions_for_env)
            next_obs_list = [o.float().to(device).squeeze() for o in next_obs_list]
            ep_reward += sum(rewards)

            try:
                frame = env_wrapper.env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                print(f"Failed to render frame at episode {ep+1}, step {step+1}: {e}")

            buffer.add(
                n_agents=n_agents,
                n_landmarks=env_wrapper.num_landmarks,
                obs_list=obs_list,
                actions=actions_t.detach(),
                rewards=rewards,
                next_obs_list=next_obs_list,
                dones=dones,
                random_numbers=random_t
            )

            obs_list = next_obs_list

            if any(dones):
                break

            # train step
            if len(buffer) >= batch_size:
                replay_batch = buffer.sample(batch_size)
                q_loss, p_loss, alpha_loss = train_masac_like(
                    policy, policy_target,
                    critic1, critic1_target,
                    critic2, critic2_target,
                    policy_optim, critic1_optim, critic2_optim,
                    alpha, alpha_optim,
                    target_entropy,
                    replay_batch,
                    gamma, tau,
                    device=device,
                )
        
        rewards_history.append(ep_reward)
        rewards_q.append(ep_reward)
        moving_avg = np.mean(rewards_q)
        moving_avg_rewards.append(moving_avg)



        if ep % save_interval == 0 or ep == n_episodes:
            checkpoint = {
                'episode': ep,
                'policy_state_dict': policy.state_dict(),
                'policy_target_state_dict': policy_target.state_dict(),
                'critic1_state_dict': critic1.state_dict(),
                'critic1_target_state_dict': critic1_target.state_dict(),
                'critic2_state_dict': critic2.state_dict(),
                'critic2_target_state_dict': critic2_target.state_dict(),
                'policy_optimizer_state_dict': policy_optim.state_dict(),
                'critic1_optimizer_state_dict': critic1_optim.state_dict(),
                'critic2_optimizer_state_dict': critic2_optim.state_dict(),
            }
            if learnable_alpha:
                checkpoint['alpha'] = alpha.detach().cpu().numpy()
                checkpoint['alpha_optimizer_state_dict'] = alpha_optim.state_dict()
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_ep{ep}.pth"))
            print(f"Saved checkpoint at episode {ep}")

        if (ep + 1) % save_video_every == 0 or (ep + 1) == n_episodes:
            video_filename = os.path.join(save_dir, "videos", f"episode_{ep+1}")
            
            # Ensure the "videos" directory exists
            os.makedirs(os.path.dirname(video_filename), exist_ok=True)
            
            # Save the video using save_video
            if len(frames) > 0:
                save_video(video_filename, frames, fps=30)
                print(f"Saved episode {ep+1} video to {video_filename}")
            else:
                print(f"No frames captured for episode {ep+1}. Video not saved.")


        if ep % 10 == 0 or ep == 1:
            print(f"[Episode {ep+1}/{n_episodes}] Reward={ep_reward} AgentCount={n_agents} MAvg={moving_avg}")



    # After training, plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, label='Episode Reward')
    plt.plot(moving_avg_rewards, label=f'Moving Avg Reward ({moving_avg_window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_rewards.png"))
    plt.show()

    # Save the final models
    final_checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'policy_target_state_dict': policy_target.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic1_target_state_dict': critic1_target.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
        'critic2_target_state_dict': critic2_target.state_dict(),
    }
    if learnable_alpha:
        final_checkpoint['alpha'] = alpha.detach().cpu().numpy()
        final_checkpoint['alpha_optimizer_state_dict'] = alpha_optim.state_dict()

    torch.save(final_checkpoint, os.path.join(save_dir, "final_model.pth"))
    print("Final model saved.")

    print("Multi-Agent SAC training (with tanh bounding) complete!")


########################################################
# 6) EXAMPLE USAGE
########################################################
if __name__ == "__main__":
    set_global_seed(GLOBAL_SEED)

    device = "cpu"

    env_wrapper = RandomAgentCountEnv(
        scenario_name="simple_spread",
        agent_count_dict={1: 0.2, 3: 0.4, 5: 0.4},
        device=device,
        continuous_actions=True,
        max_steps=None,
        seed=42
    )

    # We'll guess agent_dim=4, landmark_dim=2, action_dim=2 from your scenario
    agent_dim = 4
    landmark_dim = 2
    action_dim = 2
    hidden_dim = 32

    # Critic networks
    critic1 = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic2 = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic1_target = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic2_target = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # Stochastic policy with tanh bounding
    policy = RandomizedAttentionSACPolicy(
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)

    policy_target = RandomizedAttentionSACPolicy(
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)
    policy_target.load_state_dict(policy.state_dict())

    buffer = ReplayBuffer(capacity=50000)

    train_masac(
        env_wrapper,
        policy, policy_target,
        critic1, critic1_target,
        critic2, critic2_target,
        buffer,
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        action_dim=action_dim,
        n_episodes=100,
        max_steps=50,
        batch_size=128,
        gamma=0.99,
        lr_actor=3e-4,
        lr_critic=3e-4,
        tau=0.0001,
        alpha=0.2,
        learnable_alpha=False,
        target_entropy=-4.0,
        lr_alpha=1e-3,
        device=device
    )
