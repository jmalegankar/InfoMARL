import numpy as np
import torch
from vmas import render_interactively

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.num_good = kwargs.get("n_agents_good", 5)
        self.num_adversaries = kwargs.get("n_agents_adversaries", 3)
        self.num_agents = self.num_adversaries + self.num_good
        self.obs_agents = kwargs.get("obs_agents", True)
        self.ratio = kwargs.get("ratio", 3)  # ratio = 3, 4, 5
        self.device = device
        self.collection_radius = kwargs.get("collection_radius", 0.05)

        self.num_landmarks = self.num_good

        world = World(batch_dim=batch_dim, device=device)

        # Add agents
        for i in range(self.num_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=False,
                shape=Sphere(radius=(0.075 if i < self.num_adversaries else 0.05)),
                color=(Color.RED if i < self.num_adversaries else Color.BLUE),
                adversary=(True if i < self.num_adversaries else False),
                max_speed=(2.0 if i < self.num_adversaries else 3.0),
                movable=True,
                u_range=4.0
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(self.num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                movable=False,
                shape=Sphere(radius=0.03),
                color=Color.GREEN,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                self.ratio
                * torch.rand(
                    (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32
                )
                - 0.5 * self.ratio,
                batch_index=env_index,
            )
            agent.color = Color.RED if agent.adversary else Color.BLUE
            agent.mass = 1.0
            agent.set_vel(
                torch.zeros(
                    (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                self.ratio
                * torch.rand(
                    (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p), device=self.world.device, dtype=torch.float32
                )
                - 0.5 * self.ratio,
                batch_index=env_index,
            )

    def reward(self, agent: Agent):
        # Agents are rewarded based on minimum agent distance to each landmark
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            for a in self.world.agents:
                if a.adversary:
                    continue
                for i, food in enumerate(self.world.landmarks):
                    dist_to_food = torch.linalg.vector_norm(
                        a.state.pos - food.state.pos, dim=1
                    )

                    closest = torch.min(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    a.state.pos - food.state.pos, dim=1
                                )
                                for a in self.world.agents
                            ],
                            dim=-1,
                        ),
                        dim=-1,
                    )[0]
                    self.rew -= closest

                    # Check which environments have collected this food
                    newly_collected = (dist_to_food < self.collection_radius)
                    
                    if newly_collected.any():                        
                        # Give reward for collection
                        self.rew += newly_collected.float() * 20.0
                        
                        # Handle collected food
                        # if self.respawn_food:
                        # Respawn food at new random location
                        new_pos = self.ratio * torch.rand(
                            self.world.batch_dim,
                            self.world.dim_p,
                            device=self.world.device,
                            dtype=torch.float32,
                        ) - 0.5 * self.ratio
                        
                        # Only update position in environments where food was collected
                        current_pos = food.state.pos.clone()
                        current_pos[newly_collected] = new_pos[newly_collected]
                        food.set_pos(current_pos, batch_index=None)
                            
            for a in self.world.agents:
                if not a.adversary:
                    continue
                for i, food in enumerate(self.world.agents):
                    if food.adversary:
                        continue
                    dist_to_food = torch.linalg.vector_norm(
                        a.state.pos - food.state.pos, dim=1
                    )
                    closest = torch.min(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(
                                    a.state.pos - food.state.pos, dim=1
                                )
                                for a in self.world.agents
                            ],
                            dim=-1,
                        ),
                        dim=-1,
                    )[0]
                    self.rew += closest

                    # Check which environments have collected this food
                    newly_collected = (dist_to_food < self.collection_radius)
                    
                    if newly_collected.any():                        
                        # Give reward for collection
                        self.rew -= newly_collected.float() * 50.0
                        
                        # Handle collected food
                        new_pos = self.ratio * torch.rand(
                            self.world.batch_dim,
                            self.world.dim_p,
                            device=self.world.device,
                            dtype=torch.float32,
                        ) - 0.5 * self.ratio
                        
                        # Only update position in environments where food was collected
                        current_pos = food.state.pos.clone()
                        current_pos[newly_collected] = new_pos[newly_collected]
                        food.set_pos(current_pos, batch_index=None)            

        return self.rew
                

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos,
            agent.state.vel,
        ]

        for food in self.world.landmarks:
            observations.append(
                food.state.pos
            )
        
        for other_agent in self.world.agents:
            if other_agent == agent:
                continue
            observations.append(other_agent.state.pos)
        
        return torch.cat(observations, dim=-1)


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)