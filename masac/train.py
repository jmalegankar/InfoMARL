import torch as th
import torch.nn.functional as F
from masac.critic import CustomQFuncCritic
from masac.buffer import ReplayBuffer, StateBuffer
from masac.actor import RandomizedAttentionPolicy

import imageio
import os
from typing import List, Tuple, Dict

TensorFuture = th.jit.Future[th.Tensor]


@th.jit.interface
class PolicyImpl:
    def sample_actions_and_logp(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        pass

@th.jit.interface
class CriticImpl:
    def forward(self, obs: Dict[str, th.Tensor], action: th.Tensor) -> th.Tensor:
        pass

@th.jit.script
def _future_critic_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    state_buffer: StateBuffer,
    gamma: float,
) -> th.Tensor:
    with th.no_grad():
        next_actions, _ = policy.sample_actions_and_logp(state_buffer.next_obs)
        next_q1 = critic1.forward(state_buffer.next_obs, next_actions)
        next_q2 = critic2.forward(state_buffer.next_obs, next_actions)
        next_q = th.minimum(next_q1, next_q2) # => [n_agents]
        target_q = state_buffer.reward + (~state_buffer.done) * gamma * next_q
    q1 = critic1.forward(state_buffer.obs, state_buffer.action).view(-1)
    q2 = critic2.forward(state_buffer.obs, state_buffer.action).view(-1)
    critic1_loss = F.mse_loss(q1, target_q, reduction="none").view(-1)
    critic2_loss = F.mse_loss(q2, target_q, reduction="none").view(-1)
    return critic1_loss + critic2_loss


@th.jit.script
def _future_policy_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    state_buffer: StateBuffer,
    alpha: float,
) -> th.Tensor:
    actions, logp = policy.sample_actions_and_logp(state_buffer.obs)
    q1 = critic1.forward(state_buffer.obs, actions)
    q2 = critic2.forward(state_buffer.obs, actions)
    q = th.minimum(q1, q2)
    policy_loss = alpha * logp - q
    return policy_loss.view(-1)


@th.jit.script
def critic_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    replay_batch: List[StateBuffer],
    gamma: float,
) -> th.Tensor:
    futures = th.jit.annotate(List[TensorFuture], [])
    for state_buffer in replay_batch:
        futures.append(th.jit.fork(_future_critic_loss, policy, critic1, critic2, state_buffer, gamma))
    
    critic_loss = th.mean(
        th.cat([th.jit.wait(future) for future in futures], dim=0)
    )

    return critic_loss


@th.jit.script
def policy_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    replay_batch: List[StateBuffer],
    alpha: float,
) -> th.Tensor:
    futures = th.jit.annotate(List[TensorFuture], [])
    for state_buffer in replay_batch:
        futures.append(th.jit.fork(_future_policy_loss, policy, critic1, critic2, state_buffer, alpha))
    
    policy_loss = th.mean(
        th.cat([th.jit.wait(future) for future in futures], dim=0)
    )

    return policy_loss

def masac_train(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    replay_batch: List[StateBuffer],
    alpha: float,
    gamma: float,
    policy_optim: th.optim.Optimizer,
    critic1_optim: th.optim.Optimizer,
    critic2_optim: th.optim.Optimizer,
) -> None:
    critic1_optim.zero_grad()
    critic2_optim.zero_grad()

    critic_loss_val = critic_loss(policy, critic1, critic2, replay_batch, gamma)

    critic_loss_val.backward()
    critic1_optim.step()
    critic2_optim.step()

    policy_optim.zero_grad()

    policy_loss_val = policy_loss(policy, critic1, critic2, replay_batch, alpha)

    policy_loss_val.backward()
    policy_optim.step()


def soft_update(target, source, tau):
    with th.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data = tau * source_param.data + (1 - tau) * target_param.data

def train(
    env_wrapper,
    policy,
    critic1, critic1_target,
    critic2, critic2_target,
    buffer,
    n_episodes=1000,
    batch_size=32,
    gamma=0.99,
    lr_actor=1e-3,
    lr_critic=1e-3,
    tau=0.005,
    alpha=0.2,
    train_interval=100,
    save_interval=100,
    save_dir="checkpoints",
    video_dir="videos",
    eval_interval=100,
    num_eval_episodes=10,
    device="cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    policy_optim = th.optim.Adam(policy.parameters(), lr=lr_actor)
    critic1_optim = th.optim.Adam(critic1.parameters(), lr=lr_critic)
    critic2_optim = th.optim.Adam(critic2.parameters(), lr=lr_critic)
    
    os.makedirs("save_dir", exist_ok=True)

    num_timesteps = 0

    for episode in range(n_episodes):
        policy.eval()
        critic1.eval()
        critic2.eval()
        critic1_target.eval()
        critic2_target.eval()

        obs = env_wrapper.reset()
        num_agents = env_wrapper.num_agents
        done = False

        while not done:
            with th.no_grad():
                actions, _ = policy.sample_actions_and_logp(obs)
                next_obs, rewards, done, _ = env_wrapper.step(actions.unsqueeze(1))
                buffer.add(obs, actions, rewards, next_obs, done)
                obs = next_obs

            num_timesteps += num_agents

            if buffer.size >= batch_size and num_timesteps % train_interval == 0:
                replay_batch = buffer.sample(batch_size, device=device)
                policy.train()
                critic1.train()
                critic2.train()
                masac_train(
                    policy,
                    critic1,
                    critic2,
                    replay_batch,
                    alpha,
                    gamma,
                    policy_optim,
                    critic1_optim,
                    critic2_optim,
                )
                policy.eval()
                critic1.eval()
                critic2.eval()

                soft_update(critic1_target, critic1, tau)
                soft_update(critic2_target, critic2, tau)

        if episode % save_interval == 0:
            th.save(
                {
                    "policy": policy.state_dict(),
                    "critic1": critic1.state_dict(),
                    "critic2": critic2.state_dict(),
                    "critic1_target": critic1_target.state_dict(),
                    "critic2_target": critic2_target.state_dict(),
                    "policy_optim": policy_optim.state_dict(),
                    "critic1_optim": critic1_optim.state_dict(),
                    "critic2_optim": critic2_optim.state_dict(),
                },
                os.path.join(save_dir, f"checkpoint_{episode}.pt"),
            )

        if episode % eval_interval == 0:
            with th.no_grad():
                episode_reward = 0
                for _ in range(num_eval_episodes):
                    frames = []
                    obs = env_wrapper.reset()
                    frames.append(env_wrapper.env.render(mode="rgb_array"))
                    done = False
                    while not done:
                        actions, _ = policy.sample_actions_and_logp(obs)
                        obs, rewards, done, _ = env_wrapper.step(actions.unsqueeze(1))
                        episode_reward += rewards.mean().item()
                        frames.append(env_wrapper.env.render(mode="rgb_array"))
                
                imageio.mimsave(os.path.join(video_dir, f"episode_{episode}.gif"), frames, fps=10)


            print(f"Episode {episode} | Reward: {episode_reward} | Steps: {num_timesteps}")


if __name__ == "__main__":
    from masac.simple_spread import Scenario
    from masac.env import RandomAgentCountEnv
    from masac.actor import RandomizedAttentionPolicy
    from masac.critic import CustomQFuncCritic
    from masac.buffer import ReplayBuffer

    device = "cpu"

    env = RandomAgentCountEnv(
        scenario_name=Scenario(),
        agent_count_dict={1: 1.0, 3: 0.0, 5: 0.0},
        seed=42,
        device="cpu",
        max_steps=100,
    )

    agent_dim = 2
    landmark_dim = 2
    action_dim = 2
    hidden_dim = 64

    # Critic networks
    critic1 = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic2 = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic1_target = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic2_target = CustomQFuncCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # Stochastic policy with tanh bounding
    policy = RandomizedAttentionPolicy(
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)

    policy = th.jit.script(policy)
    critic1 = th.jit.script(critic1)
    critic2 = th.jit.script(critic2)

    buffer = th.jit.script(ReplayBuffer(50000, device))

    train(
        env_wrapper=env,
        policy=policy,
        critic1=critic1,
        critic1_target=critic1_target,
        critic2=critic2,
        critic2_target=critic2_target,
        buffer=buffer,
        n_episodes=1000,
        batch_size=256,
        gamma=0.95,
        lr_actor=3e-4,
        lr_critic=1e-3,
        tau=0.001,
        alpha=0.01,
        train_interval=64,
        save_interval=100,
        save_dir="checkpoints",
        video_dir="videos",
        eval_interval=10,
        num_eval_episodes=10,
        device=device,
    )