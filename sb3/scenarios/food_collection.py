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
        collection_radius = kwargs.pop("collection_radius", 0.15)
        respawn_food = kwargs.pop("respawn_food", True)
        sparse_reward = kwargs.pop("sparse_reward", False)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.obs_agents = obs_agents
        self.collection_radius = collection_radius
        self.respawn_food = respawn_food
        self.sparse_reward = sparse_reward
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

        return world

    def reset_world_at(self, env_index: int = None):
        # Reset agents
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
            else:
                self.food_positions[:, i] = new_pos
                self.food_collected[:, i] = False
                self.total_food_collected[:] = 0

    def reward(self, agent: Agent):
        """Reward function for food collection with collision penalties."""
        is_first = agent == self.world.agents[0]
        
        if is_first:
            # Initialize shared reward
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )
            
            # Check food collection for all agents
            for i, food in enumerate(self.world.landmarks):
                if not self.food_collected[:, i].all():  # Only check if food not already collected
                    for a in self.world.agents:
                        dist_to_food = torch.linalg.vector_norm(
                            a.state.pos - food.state.pos, dim=1
                        )
                        
                        # Check which environments have collected this food
                        newly_collected = (dist_to_food < self.collection_radius) & ~self.food_collected[:, i]
                        
                        if newly_collected.any():
                            # Mark food as collected in those environments
                            self.food_collected[:, i] |= newly_collected
                            self.total_food_collected += newly_collected.long()
                            
                            # Give reward for collection
                            self.rew += newly_collected.float() * 1.0
                            
                            # Handle collected food
                            if self.respawn_food:
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
                            else:
                                # Move collected food far away
                                far_pos = torch.full(
                                    (self.world.batch_dim, self.world.dim_p),
                                    1000.0,
                                    device=self.world.device,
                                    dtype=torch.float32,
                                )
                                current_pos = food.state.pos.clone()
                                current_pos[newly_collected] = far_pos[newly_collected]
                                food.set_pos(current_pos, batch_index=None)
            
            # Add collision penalties
            for a in self.world.agents:
                if a.collide:
                    for b in self.world.agents:
                        if a != b:
                            overlap = self.world.is_overlapping(a, b)
                            self.rew[overlap] -= 0.1
            
            # If not using sparse rewards, add distance-based shaping
            if not self.sparse_reward:
                # Find closest uncollected food for each agent
                for a in self.world.agents:
                    min_dist = torch.full(
                        (self.world.batch_dim,), float('inf'), 
                        device=self.world.device
                    )
                    
                    for i, food in enumerate(self.world.landmarks):
                        # Only consider uncollected food
                        uncollected_mask = ~self.food_collected[:, i]
                        if uncollected_mask.any():
                            dist = torch.linalg.vector_norm(
                                a.state.pos - food.state.pos, dim=1
                            )
                            # Update minimum distance only for uncollected food
                            min_dist[uncollected_mask] = torch.min(
                                min_dist[uncollected_mask], 
                                dist[uncollected_mask]
                            )
                    
                    # Add small distance-based reward (negative of distance)
                    valid_dist = min_dist < float('inf')
                    self.rew[valid_dist] -= 0.01 * min_dist[valid_dist]
        
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
            
            # Mask out collected food by setting to zeros or large values
            collected_mask = self.food_collected[:, i].unsqueeze(-1)
            rel_pos = torch.where(
                collected_mask,
                torch.zeros_like(rel_pos),  # or torch.full_like(rel_pos, 100.0) for "far away"
                rel_pos
            )
            observations.append(rel_pos)
        
        # Add relative positions and velocities of other agents
        if self.obs_agents:
            for other in self.world.agents:
                if other != agent:
                    observations.append(other.state.pos - agent.state.pos)
        
        return torch.cat(observations, dim=-1)

    def info(self, agent: Agent) -> dict:
        """Return info dictionary with collection statistics."""
        return {
            "total_food_collected": self.total_food_collected,
            "food_collected_mask": self.food_collected,
        }


if __name__ == "__main__":
    render_interactively(
        __file__, 
        control_two_agents=True,
        n_agents=3,
        n_food=8,
        respawn_food=True,
        sparse_reward=False,
    )