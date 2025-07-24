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
        self.ratio = kwargs.get("ratio", 2)  # ratio = 2, 3, 4, 5
        self.device = device

        self.num_landmarks = self.num_good

        world = World(batch_dim=batch_dim, device=device)

        # Add agents
        for i in range(self.num_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
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

    # return all agents that are not adversaries
    def good_agents(self):
        return [agent for agent in self.world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self):
        return [agent for agent in self.world.agents if agent.adversary]
    
    def agent_reward(self, agent: Agent):
        pos_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        closest = torch.min(
            torch.stack(
                [
                    torch.linalg.vector_norm(
                        landmark.state.pos - agent.state.pos, dim=1
                    )
                    for landmark in self.world.landmarks
                ],
                dim=1,
            ),
            dim=-1,
        )[0]
        pos_rew -= closest

        for landmark in self.world.landmarks:
            found = self.world.is_overlapping(landmark, agent)
            pos_rew[found] += 20
            while torch.where(found)[0].shape[0] != 0:
                landmark.set_pos(
                    self.ratio
                    * torch.rand(
                        self.world.dim_p, device=self.world.device, dtype=torch.float32
                    )
                    - 0.5 * self.ratio,
                    batch_index=torch.where(found)[0][0],
                )
                found = self.world.is_overlapping(landmark, agent)
        
        if agent.collide:
            for a in self.world.agents:
                if a != agent and a.adversary:
                    killed = self.world.is_overlapping(a, agent)
                    pos_rew[killed] -= 5
                    while torch.where(killed)[0].shape[0] != 0:
                        a.set_pos(
                            self.ratio
                            * torch.rand(
                                self.world.dim_p, device=self.world.device, dtype=torch.float32
                            )
                            - 0.5 * self.ratio,
                            batch_index=torch.where(killed)[0][0],
                        )
                        killed = self.world.is_overlapping(a, agent)
        return pos_rew

    def adversary_reward(self, agent: Agent):
        adv_rew = torch.zeros(
            self.world.batch_dim, device=self.world.device, dtype=torch.float32
        )

        closest = torch.min(
            torch.stack(
                [
                    torch.linalg.vector_norm(
                        agent.state.pos - evader.state.pos, dim=1
                    )
                    for evader in self.good_agents()
                ],
                dim=1,
            ),
            dim=-1,
        )[0]
        adv_rew -= closest

        if agent.collide:
            for a in self.world.agents:
                if a != agent and not a.adversary:
                    killed = self.world.is_overlapping(a, agent)
                    adv_rew[killed] += 15

        return adv_rew

    def reward(self, agent: Agent):
        if agent.adversary:
            return -self.adversary_reward(agent)
        else:
            return self.agent_reward(agent)                

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