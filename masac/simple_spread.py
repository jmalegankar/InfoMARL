#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_agents = kwargs.pop("n_agents", 3)
        obs_agents = kwargs.pop("obs_agents", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.obs_agents = obs_agents

        world = World(batch_dim=batch_dim, device=device)
        # set any world properties first
        num_landmarks = num_agents
        # Add agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=0.15),
                color=Color.BLUE,
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}",
                collide=False,
                color=Color.BLACK,
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

    # def reward(self, agent: Agent):
    #     is_first = agent == self.world.agents[0]
    #     if is_first:
    #         # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    #         self.rew = torch.zeros(
    #             self.world.batch_dim,
    #             device=self.world.device,
    #             dtype=torch.float32,
    #         )
    #         for single_agent in self.world.agents:
    #             for landmark in self.world.landmarks:
    #                 closest = torch.min(
    #                     torch.stack(
    #                         [
    #                             torch.linalg.vector_norm(
    #                                 a.state.pos - landmark.state.pos, dim=1
    #                             )
    #                             for a in self.world.agents
    #                         ],
    #                         dim=-1,
    #                     ),
    #                     dim=-1,
    #                 )[0]
    #                 self.rew -= closest

    #             if single_agent.collide:
    #                 for a in self.world.agents:
    #                     if a != single_agent:
    #                         self.rew[self.world.is_overlapping(a, single_agent)] -= 1

    #     return self.rew

    def reward(self, agent: Agent):
        # Agent reward is based on the distance to the closest landmark
        rew = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
        )
        # Negative distance to closest landmark
        for landmark in self.world.landmarks:
            rew -= torch.min(
                torch.linalg.vector_norm(
                    agent.state.pos - landmark.state.pos, dim=-1
                ),
                dim=-1,
            )[0]
        
        # Penalize collisions
        if agent.collide:
            for a in self.world.agents:
                if a != agent:
                    rew[self.world.is_overlapping(a, agent)] -= 1
        
        return rew


    def observation(self, agent: Agent):
        # get positions of all landmarks in this agent's reference frame
        obs = []
        for landmark in self.world.landmarks:  # world.entities:
            obs.append(torch.concat((landmark.state.pos - agent.state.pos, agent.state.vel), dim=-1))
        # distance to all other agents
        pos = []
        agent_idx = 0
        for idx, other in enumerate(self.world.agents):
            if other == agent:
                agent_idx = idx
            pos.append(other.state.pos - agent.state.pos)
        return {
            "obs": torch.concat(obs, dim=0),
            "agent_states": torch.concat(pos, dim=0),
            "idx": torch.tensor(agent_idx, device=self.world.device, dtype=torch.long),
        }


if __name__ == "__main__":
    from vmas import make_env
    env = make_env(
        scenario=Scenario(),
        num_envs=1,
        n_agents=3,
        continuous_actions=True,
        device="cpu",
        seed=42,
        max_steps=100,
    )
    print(env.reset())