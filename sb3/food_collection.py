import torch
from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        num_agents = kwargs.pop("n_agents", 3)
        num_food = kwargs.pop("n_food", 5)
        obs_agents = kwargs.pop("obs_agents", True)
        collection_radius = kwargs.pop("collection_radius", 0.05)
        respawn_food = kwargs.pop("respawn_food", True)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.obs_agents = obs_agents
        self.collection_radius = collection_radius
        self.respawn_food = respawn_food
        self.num_food = num_food

        world = World(batch_dim=batch_dim, device=device)
        
        # Create agents
        for i in range(num_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                shape=Sphere(radius=0.075),
                color=Color.BLUE,
                max_speed=None,  # No speed limit
            )
            world.add_agent(agent)
        
        # Create food items (landmarks)
        for i in range(num_food):
            food = Landmark(
                name=f"food_{i}",
                collide=False,
                shape=Sphere(radius=0.05),
                color=Color.GREEN,
                movable=False,
            )
            world.add_landmark(food)

        # Initialize tracking variables
        self.food_collected = torch.zeros(
            (batch_dim, num_food), device=device, dtype=torch.bool
        )
        self.total_food_collected = torch.zeros(
            batch_dim, device=device, dtype=torch.long
        )
        self.food_positions = torch.zeros(
            (batch_dim, num_food, world.dim_p), device=device, dtype=torch.float32
        )
        
        # Track if game is done (all food collected)
        self.game_done = torch.zeros(batch_dim, device=device, dtype=torch.bool)

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
            
            # Reset velocity
            agent.set_vel(
                torch.zeros(
                    (
                        (1, self.world.dim_p)
                        if env_index is not None
                        else (self.world.batch_dim, self.world.dim_p)
                    ),
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        # Reset food items
        for i, food in enumerate(self.world.landmarks):
            new_pos = torch.zeros(
                (
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p)
                ),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(-1.0, 1.0)
            
            food.set_pos(new_pos, batch_index=env_index)
            
            # Update tracking
            if env_index is not None:
                self.food_positions[env_index, i] = new_pos
                self.food_collected[env_index, i] = False
                self.total_food_collected[env_index] = 0
                self.game_done[env_index] = False
            else:
                self.food_positions[:, i] = new_pos
                self.food_collected[:, i] = False
                self.total_food_collected[:] = 0
                self.game_done[:] = False

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        
        if is_first:
            # Initialize shared reward
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            
            # Check food collection for all agents
            for a in self.world.agents:
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
                        # Mark food as collected in those environments
                        self.food_collected[:, i] |= newly_collected
                        self.total_food_collected += newly_collected.long()
                        
                        # Give reward for collection
                        self.rew += newly_collected.float() * 20.0
                        
                        # Handle collected food
                        # if self.respawn_food:
                        # Respawn food at new random location
                        new_pos = torch.zeros(
                            (self.world.batch_dim, self.world.dim_p),
                            device=self.world.device,
                            dtype=torch.float32,
                        ).uniform_(-1.0, 1.0)
                        
                        # Only update position in environments where food was collected
                        current_pos = food.state.pos.clone()
                        current_pos[newly_collected] = new_pos[newly_collected]
                        food.set_pos(current_pos, batch_index=None)
                        
                        # Reset collection status for respawned food
                        self.food_collected[newly_collected, i] = False
            # Add collision penalties
            for a in self.world.agents:
                if a.collide:
                    for b in self.world.agents:
                        if a != b:
                            overlap = self.world.is_overlapping(a, b)
                            self.rew[overlap] -= 0.1
            
        
        return self.rew

    def observation(self, agent: Agent):
        """Return the observation for the agent."""
        observations = [
            agent.state.pos,  # Agent's position
            agent.state.vel,  # Agent's velocity
        ]
        
        # Add relative positions of all food items
        for i, food in enumerate(self.world.landmarks):
            # Create position tensor
            rel_pos = food.state.pos - agent.state.pos
            
            # Only include food that hasn't been collected
            # This maintains a fixed observation size but marks collected food
            # collected_mask = self.food_collected[:, i].unsqueeze(-1)
            # rel_pos = torch.where(
            #     collected_mask,
            #     torch.full_like(rel_pos, -999.0),  # Special marker for collected food
            #     rel_pos
            # )
            observations.append(rel_pos)
        
        # Add relative positions of other agents
        if self.obs_agents:
            for other in self.world.agents:
                if other != agent:
                    observations.append(other.state.pos - agent.state.pos)
        
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent) -> dict:
        return {
            "total_food_collected": self.total_food_collected,
            "food_collected_mask": self.food_collected,
            "game_done": self.game_done,
            "all_food_collected": self.food_collected.all(dim=1),
        }

    def done(self):
        if not self.respawn_food:
            return self.game_done
        else:
            # If respawning food, episode never ends based on food collection
            return torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.bool)


if __name__ == "__main__":
    render_interactively(
        __file__, 
        control_two_agents=True,
        n_agents=4,
        n_food=8,
        respawn_food=True,  # Set to False so game ends when all food collected
        collection_radius=0.15,  # Adjust this to control how close agents need to be
    )