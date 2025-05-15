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

obs = env.reset()
attn_buffer = []
done = False

while not done:
    actions, _ = model.predict(obs, deterministic=True)

    obs_tensor = torch.as_tensor(obs, device=device).float()

   
    with torch.no_grad():
        _ = model.policy.features_extractor(obs_tensor)


    w = model.policy.features_extractor.actor.last_landmark_weights
    w = w.cpu().numpy() # → (batch*agents, 1, n_agents)

    # reshape into (n_envs, n_agents, 1, n_agents)
    n_envs = getattr(env, "num_envs", None) or obs.shape[0]
    n_agents = model.policy.features_extractor.n_agents
    w = w.reshape(n_envs, n_agents, 1, n_agents)

    # drop the “1” dimension (our single‐query token) → (n_envs, n_agents, n_agents)
    w = w[:, :, 0, :]

    # pick env‐0’s matrix
    mat = w[0]

    # normalize to [0,1] (optional but keeps colors consistent)
    mat = mat / mat.max()

    attn_buffer.append(mat)

    obs, _, dones, _ = env.step(actions)
    done = dones.any()

# 11) once collected, render out the GIF
animate_attention(attn_buffer, output_path='agent_attention.gif', fps=5)
print("✅ Saved agent_attention.gif")
