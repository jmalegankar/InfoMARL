import torch as th
import torch.nn.functional as F
from masac.critic import CustomQFuncCritic
from masac.buffer import ReplayBuffer, StateBuffer
from masac.actor import RandomizedAttentionPolicy

from typing import List

TensorFuture = th.jit.Future[th.Tensor]

@th.jit.script
def _future_critic_loss(
    policy: RandomizedAttentionPolicy,
    critic1: CustomQFuncCritic,
    critic2: CustomQFuncCritic,
    state_buffer: StateBuffer,
    gamma: float,
) -> th.Tensor:
    with th.no_grad():
        next_actions, _ = policy.sample_actions_and_logp(state_buffer.next_obs)
        next_q1 = critic1(state_buffer.next_obs, next_actions)
        next_q2 = critic2(state_buffer.next_obs, next_actions)
        next_q = th.minimum(next_q1, next_q2) # => [n_agents]
        target_q = state_buffer.reward + (1 - state_buffer.done) * gamma * next_q
    q1 = critic1(state_buffer.obs, state_buffer.action)
    q2 = critic2(state_buffer.obs, state_buffer.action)
    critic1_loss = F.mse_loss(q1, target_q)
    critic2_loss = F.mse_loss(q2, target_q)
    return critic1_loss + critic2_loss


@th.jit.script
def _future_policy_loss(
    policy: RandomizedAttentionPolicy,
    critic1: CustomQFuncCritic,
    critic2: CustomQFuncCritic,
    state_buffer: StateBuffer,
    alpha: float,
) -> th.Tensor:
    actions, logp = policy.sample_actions_and_logp(state_buffer.obs)
    q1 = critic1(state_buffer.obs, actions)
    q2 = critic2(state_buffer.obs, actions)
    q = th.minimum(q1, q2)
    policy_loss = alpha * logp - q
    return policy_loss

@th.jit.script
def masac_train(
    policy: RandomizedAttentionPolicy,
    policy_target: RandomizedAttentionPolicy,
    critic1: CustomQFuncCritic,
    critic2: CustomQFuncCritic,
    critic1_target: CustomQFuncCritic,
    critic2_target: CustomQFuncCritic,
    replay_batch: List[StateBuffer],
    alpha: float,
    gamma: float,
    tau: float,
    policy_optim: th.optim.Optimizer,
    critic1_optim: th.optim.Optimizer,
    critic2_optim: th.optim.Optimizer,
):
    critic1_optim.zero_grad()
    critic2_optim.zero_grad()

    futures = th.jit.annotate(List[TensorFuture], [])
    for state_buffer in replay_batch:
        futures.append(th.jit.fork(_future_critic_loss, policy, critic1, critic2, state_buffer, gamma))
    
    critic_loss = th.mean(
        th.cat([th.jit.wait(future) for future in futures], dim=0)
    )

    critic_loss.backward()
    critic1_optim.step()
    critic2_optim.step()

    policy_optim.zero_grad()
    futures = th.jit.annotate(List[TensorFuture], [])
    for state_buffer in replay_batch:
        futures.append(th.jit.fork(_future_policy_loss, policy, critic1, critic2, state_buffer, alpha))

    policy_loss = th.mean(
        th.cat([th.jit.wait(future) for future in futures], dim=0)
    )

    policy_loss.backward()
    policy_optim.step()

    with th.no_grad():
        for param, target_param in zip(policy.parameters(), policy_target.parameters()):
            target_param.data = tau * param.data + (1 - tau) * target_param.data
        for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
            target_param.data = tau * param.data + (1 - tau) * target_param.data
        for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
            target_param.data = tau * param.data + (1 - tau) * target_param.data


def train(
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
    pass


if __name__ == "__main__":
    from masac.simple_spread import Scenario
    from masac.env import RandomAgentCountEnv
    from masac.actor import RandomizedAttentionPolicy
    from masac.critic import CustomQFuncCritic
    from masac.buffer import ReplayBuffer

    device = "cpu"

    env = RandomAgentCountEnv(
        scenario_name=Scenario(),
        agent_count_dict={1: 0.2, 3: 0.4, 5: 0.4},
        seed=42,
        device="cpu",
    )

    agent_dim = 2
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
    policy = RandomizedAttentionPolicy(
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)

    policy_target = RandomizedAttentionPolicy(
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    ).to(device)
    policy_target.load_state_dict(policy.state_dict())

    policy = th.jit.script(policy)
    policy_target = th.jit.script(policy_target)
    critic1 = th.jit.script(critic1)
    critic2 = th.jit.script(critic2)
    critic1_target = th.jit.script(critic1_target)
    critic2_target = th.jit.script(critic2_target)

    buffer = th.jit.script(ReplayBuffer(50000, device))

    train(
        env_wrapper=env,
        policy=policy,
        policy_target=policy_target,
        critic1=critic1,
        critic1_target=critic1_target,
        critic2=critic2,
        critic2_target=critic2_target,
        buffer=buffer,
        agent_dim=agent_dim,
        landmark_dim=landmark_dim,
        action_dim=action_dim,
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
    )