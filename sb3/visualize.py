import vmas
import wrapper
import policy
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from matplotlib.animation import FuncAnimation
import os
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.animation import FFMpegWriter



class AttentionAnimator:
    def __init__(self):
        self.env = None
        self.model = None
        self.cross_attention_weights = []
        self.landmark_weights = None
        self.env_frames = []
        self.scenario = None
        self.max_steps = None
        self.n_agents = None
        self.env_idx = 0
    
    def create_env(self, sim, env_idx, **kwargs):
        if sim == "vmas":
            env = vmas.make_env(
                scenario=kwargs["scenario"],
                n_agents=kwargs["n_agents"],
                num_envs=kwargs["num_envs"],
                continuous_actions=kwargs["continuous_actions"],
                max_steps=kwargs["max_steps"],
                seed=kwargs["seed"],
                device=kwargs["device"],
                terminated_truncated=kwargs["terminated_truncated"],
            )
            env = wrapper.VMASVecEnv(env, rnd_nums=True)
            self.env = env
            self.env_idx = env_idx
            self.scenario = kwargs["scenario"]
            self.max_steps = kwargs["max_steps"]
            self.n_agents = kwargs["n_agents"]
            self.num_envs = kwargs["num_envs"]
            self.landmark_weights = []
            
            
    
    def attach_and_load_model(self, model_name, path, **kwargs):
        if model_name == "ppo":
            self.model = PPO(
                policy=kwargs["policy"],
                env=self.env,
                device=kwargs["device"],
                verbose=kwargs["verbose"],
                batch_size=kwargs["batch_size"],
                n_epochs=kwargs["n_epochs"],
                max_grad_norm=kwargs["max_grad_norm"],
                gamma=kwargs["gamma"],
                n_steps=kwargs["n_steps"],
            )
        self.model.load(path)


    def collect_data(self):
        obs = self.env.reset()
        for step in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            actor = self.model.policy.features_extractor.actor
            cross_attention_weights = actor.cross_attention_weights
            cross_attention_weights = cross_attention_weights.view(self.num_envs, self.n_agents, self.n_agents, self.n_agents)
            cross_attention_weights = cross_attention_weights[self.env_idx].cpu().numpy()

            frame = self.env.render(
                mode="rgb_array",
                agent_index_focus=None
            )

            self.env_frames.append(frame)
            self.cross_attention_weights.append(cross_attention_weights)

            if self.scenario == "simple_spread":
                #do same for landmark weights(landmarks are same as number of agents)
                landmark_weights = actor.landmark_attention_weights
                landmark_weights = landmark_weights.view(self.num_envs, self.n_agents, 1, self.n_agents)
                landmark_weights = landmark_weights[self.env_idx].cpu().numpy()
                self.landmark_weights.append(landmark_weights)
            
            # take a step in the environment
            obs, rewards, dones, infos = self.env.step(action)
            
    
    def create_mp4(self, path):
        # Create figure once
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(f"{self.scenario} Scenario")

        
        writer = FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)

        
        with writer.saving(fig, path, 300): 
            for frame_idx, frame in enumerate(self.env_frames):
                ax.imshow(frame)

                
                ax.text(
                    0.5, 1.02, f"Frame: {frame_idx + 1}", ha='center', va='bottom', 
                    transform=ax.transAxes, fontsize=14, color='white', fontweight='bold'
                )

                writer.grab_frame()

        plt.close(fig)
             
        
            

if __name__ == "__main__":
    # Example usage
    
    animator = AttentionAnimator()
    animator.create_env(
        sim="vmas",
        env_idx=0,
        scenario="simple_spread",
        n_agents=4,
        num_envs=40,
        continuous_actions=True,
        max_steps=100,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        terminated_truncated=False,
    )
    animator.attach_and_load_model(
        model_name="ppo",
        path="/Users/jmalegaonkar/Desktop/InfoMARL-1/sb3/ppo_infomarl.zip",
        policy=policy.InfoMARLActorCriticPolicy,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
        batch_size=400,
        n_epochs=10,
        max_grad_norm=10,
        gamma=0.99,
        n_steps=100
    )
    animator.collect_data()
    animator.create_mp4("attention_animation.mp4")
