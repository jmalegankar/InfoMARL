import torch
import numpy as np
from train import env, model
from matplotlib.animation import PillowWriter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_attention(attentions, output_path='attention_heatmap.gif', fps=5):
    num_frames = len(attentions)
    
    fig, ax = plt.subplots()
    heatmap = ax.imshow(attentions[0], vmin=0, vmax=1, cmap='viridis')
    cbar = fig.colorbar(heatmap)
    ax.set_xlabel('Target agent')
    ax.set_ylabel('Source agent')
    
    def update(frame):
        heatmap.set_data(attentions[frame])
        ax.set_title(f'Frame {frame + 1}/{num_frames}')
        return [heatmap]
    
    anim = FuncAnimation(
        fig, update,
        frames=num_frames,
        blit=True,
        interval=1000 / fps
    )
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved animation to {output_path}")

device = model.device

# 1) reset and warm‐up
obs = env.reset()
attn_buffer = []

done = False
while not done:
    actions, _states = model.predict(obs, deterministic=True)
    
    
    obs_tensor = torch.as_tensor(obs, device=device).float()

    # run through policy and stash the weights
    _ = model.policy.actor(obs_tensor)
    
    #shape = (num_envs * num_agents, num_heads, q_len, k_len)
    w = model.policy.actor.last_attn_weights.cpu().numpy()
    
    # reshape/aggregate:
    # for simplicity take env 0 only, and drop the head dimension
    num_envs = env.num_envs
    na = model.policy.actor.number_agents
    w = w.reshape(num_envs, na, -1, -1)[0, :, 0, :]  # → shape (num_agents, num_agents)
    # .mean(axis=0) for all envs
    
    attn_buffer.append(w)
    
    obs, rewards, dones, infos = env.step(actions)
    done = dones.any()

# 7) now render to GIF
animate_attention(attn_buffer, output_path='agent_attention.gif', fps=5)
print("Done! See agent_attention.gif")
