# train_mlp_single.py
import torch as th
import torch.nn.functional as F
from typing import List, Tuple, Dict
import os
import imageio

from masac.buffer import ReplayBuffer, StateBuffer
from masac.env import RandomAgentCountEnv
from masac.simple_spread import Scenario  # or scenario_name="simple_spread"

# Import the new MLP actor/critic
from actor_mlp import MLPActor
from critic_mlp import MLPQCritic

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
    alpha: float,
) -> th.Tensor:
    """
    For each transition, compute the soft Q backup target:
        target = r + gamma * (min(Q1,Q2)(s', a') - alpha*logp(a'))
    Then compute MSE for Q1,Q2 to that target.
    """
    with th.no_grad():
        next_actions, logp_next = policy.sample_actions_and_logp(state_buffer.next_obs)
        q1_next = critic1.forward(state_buffer.next_obs, next_actions)
        q2_next = critic2.forward(state_buffer.next_obs, next_actions)
        q_next = th.minimum(q1_next, q2_next)
        target_q = state_buffer.reward + (~state_buffer.done) * gamma * (q_next - alpha * logp_next.squeeze(-1))

    q1 = critic1.forward(state_buffer.obs, state_buffer.action)
    q2 = critic2.forward(state_buffer.obs, state_buffer.action)
    critic1_loss = F.mse_loss(q1, target_q, reduction="none")
    critic2_loss = F.mse_loss(q2, target_q, reduction="none")

    return critic1_loss + critic2_loss

@th.jit.script
def _future_policy_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    state_buffer: StateBuffer,
    alpha: float,
) -> th.Tensor:
    """
    For the policy update, we do:
        L_pi = E[alpha * logp(a) - min(Q1,Q2)(s,a)]
    """
    actions, logp = policy.sample_actions_and_logp(state_buffer.obs)
    q1 = critic1.forward(state_buffer.obs, actions)
    q2 = critic2.forward(state_buffer.obs, actions)
    q = th.minimum(q1, q2)
    # shape matching
    policy_loss = alpha * logp.squeeze(-1) - q
    return policy_loss.view(-1)

@th.jit.script
def critic_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    replay_batch: List[StateBuffer],
    gamma: float,
    alpha: float,
) -> th.Tensor:
    """
    Sums up the critic loss over the batch (forking each transition in parallel for speed).
    """
    futures = th.jit.annotate(List[TensorFuture], [])
    for state_buffer in replay_batch:
        futures.append(th.jit.fork(_future_critic_loss, policy, critic1, critic2, state_buffer, gamma, alpha))

    combined = th.cat([th.jit.wait(fut) for fut in futures], dim=0)
    return th.mean(combined)

@th.jit.script
def policy_loss(
    policy: PolicyImpl,
    critic1: CriticImpl,
    critic2: CriticImpl,
    replay_batch: List[StateBuffer],
    alpha: float,
) -> th.Tensor:
    """
    Sums up the policy loss over the batch.
    """
    futures = th.jit.annotate(List[TensorFuture], [])
    for state_buffer in replay_batch:
        futures.append(th.jit.fork(_future_policy_loss, policy, critic1, critic2, state_buffer, alpha))

    combined = th.cat([th.jit.wait(fut) for fut in futures], dim=0)
    return th.mean(combined)

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
    """
    Single training iteration: update critics, then update policy.
    """
    critic1_optim.zero_grad()
    critic2_optim.zero_grad()
    c_loss = critic_loss(policy, critic1, critic2, replay_batch, gamma, alpha)
    c_loss.backward()
    critic1_optim.step()
    critic2_optim.step()

    policy_optim.zero_grad()
    p_loss = policy_loss(policy, critic1, critic2, replay_batch, alpha)
    p_loss.backward()
    policy_optim.step()

def soft_update(target, source, tau):
    with th.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data = tau * s_param.data + (1 - tau) * t_param.data

def train(
    env_wrapper,
    policy,
    critic1, critic1_target,
    critic2, critic2_target,
    buffer,
    n_episodes=500,
    batch_size=64,
    gamma=0.99,
    lr_actor=3e-4,
    lr_critic=3e-4,
    tau=0.005,
    alpha=0.2,
    train_interval=100,
    save_interval=100,
    save_dir="checkpoints",
    video_dir="videos",
    eval_interval=100,
    num_eval_episodes=5,
    device="cpu",
):
    """
    Main training loop using the same style as your original train.py.
    """
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

            # Training step
            if buffer.size >= batch_size and (num_timesteps % train_interval == 0):
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
                # Soft update
                soft_update(critic1_target, critic1, tau)
                soft_update(critic2_target, critic2, tau)
                policy.eval()
                critic1.eval()
                critic2.eval()

        # Save checkpoints
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

        # Evaluation
        if episode % eval_interval == 0:
            total_reward = 0.0
            with th.no_grad():
                for _ in range(num_eval_episodes):
                    frames = []
                    obs = env_wrapper.reset()
                    frames.append(env_wrapper.env.render(mode="rgb_array"))
                    done = False
                    ep_reward = 0.0
                    while not done:
                        actions, _ = policy.sample_actions_and_logp(obs)
                        next_obs, rewards, done, _ = env_wrapper.step(actions.unsqueeze(1))
                        ep_reward += rewards.mean().item()
                        obs = next_obs
                        frames.append(env_wrapper.env.render(mode="rgb_array"))

                    total_reward += ep_reward

                avg_reward = total_reward / num_eval_episodes
                imageio.mimsave(os.path.join(video_dir, f"episode_{episode}.gif"), frames, fps=10)
            print(f"Episode {episode} | Avg. Reward: {avg_reward} | Steps: {num_timesteps}")

    print("Training complete.")

if __name__ == "__main__":
    # Example: single-agent random env config => only 1 agent

    device = "cpu"
    
    env = RandomAgentCountEnv(
        scenario_name=Scenario(),
        agent_count_dict={1: 1.0},
        seed=42,
        device="cpu",
        max_steps=100,
    )

    # Build MLP actor and critics
    agent_dim = 2
    landmark_dim = 2
    action_dim = 2
    hidden_dim = 64

    from actor_mlp import MLPActor
    from critic_mlp import MLPQCritic
    from masac.buffer import ReplayBuffer

    policy = MLPActor(agent_dim, landmark_dim, hidden_dim, action_dim).to("cpu")
    critic1 = MLPQCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to("cpu")
    critic2 = MLPQCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to("cpu")
    critic1_target = MLPQCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to("cpu")
    critic2_target = MLPQCritic(agent_dim, action_dim, landmark_dim, hidden_dim).to("cpu")
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # Wrap in torchscript if desired
    policy = th.jit.script(policy)
    critic1 = th.jit.script(critic1)
    critic2 = th.jit.script(critic2)

    # Replay buffer
    buffer = th.jit.script(ReplayBuffer(100000, device))

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
        lr_actor=1e-2,
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
