import random
import numpy as np
from typing import List, Tuple
from .chromosome import FleetChromosome, DroneGene


class GeneticOperators:
    """
    Custom genetic operators tailored for spatial coordinate path planning,
    dynamic waypoint counts, and multi-segment operational profiles.
    """

    def __init__(
        self,
        grid_width: int = 50,
        grid_height: int = 50,
        min_waypoints: int = 4,
        max_waypoints: int = 12,
        mutation_rate: float = 0.08,
        crossover_rate: float = 0.85,
        tournament_size: int = 3,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.min_waypoints = min_waypoints
        self.max_waypoints = max_waypoints
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

    def select(self, population: List[FleetChromosome]) -> FleetChromosome:
        """Tournament selection operator."""
        aspirants = random.sample(population, self.tournament_size)
        return max(aspirants, key=lambda ind: ind.fitness)

    def crossover(
        self, parent1: FleetChromosome, parent2: FleetChromosome
    ) -> Tuple[FleetChromosome, FleetChromosome]:
        """
        Multi-segment dynamic crossover:
        - Route exchange between drones
        - Cut-and-splice trajectory crossover for variable length
        """
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        child1_drones = []
        child2_drones = []

        num_drones = min(len(parent1.drones), len(parent2.drones))

        for d_idx in range(num_drones):
            d1 = parent1.drones[d_idx]
            d2 = parent2.drones[d_idx]

            # Route level swap with 30% probability
            if random.random() < 0.3:
                child1_drones.append(d2.clone())
                child2_drones.append(d1.clone())
                continue

            # Cut-and-splice trajectory crossover
            w1, s1, p1 = d1.waypoints, d1.speeds, d1.sensor_powers
            w2, s2, p2 = d2.waypoints, d2.speeds, d2.sensor_powers

            cut1 = random.randint(1, max(1, len(w1) - 1))
            cut2 = random.randint(1, max(1, len(w2) - 1))

            new_w1 = w1[:cut1] + w2[cut2:]
            new_s1 = s1[:cut1] + s2[cut2:]
            new_p1 = p1[:cut1] + p2[cut2:]

            new_w2 = w2[:cut2] + w1[cut1:]
            new_s2 = s2[:cut2] + s1[cut1:]
            new_p2 = p2[:cut2] + p1[cut1:]

            # Enforce dynamic waypoint bounds [min_waypoints, max_waypoints]
            if len(new_w1) > self.max_waypoints:
                new_w1 = new_w1[:self.max_waypoints]
                new_s1 = new_s1[:self.max_waypoints]
                new_p1 = new_p1[:self.max_waypoints]
            elif len(new_w1) < self.min_waypoints:
                while len(new_w1) < self.min_waypoints:
                    new_w1.append((random.uniform(2, self.grid_width - 2), random.uniform(2, self.grid_height - 2)))
                    new_s1.append(random.choice(DroneGene.SPEED_LEVELS))
                    new_p1.append(random.choice(DroneGene.SENSOR_LEVELS))

            if len(new_w2) > self.max_waypoints:
                new_w2 = new_w2[:self.max_waypoints]
                new_s2 = new_s2[:self.max_waypoints]
                new_p2 = new_p2[:self.max_waypoints]
            elif len(new_w2) < self.min_waypoints:
                while len(new_w2) < self.min_waypoints:
                    new_w2.append((random.uniform(2, self.grid_width - 2), random.uniform(2, self.grid_height - 2)))
                    new_s2.append(random.choice(DroneGene.SPEED_LEVELS))
                    new_p2.append(random.choice(DroneGene.SENSOR_LEVELS))

            child1_drones.append(DroneGene(new_w1, new_s1, new_p1))
            child2_drones.append(DroneGene(new_w2, new_s2, new_p2))

        c1 = FleetChromosome(child1_drones)
        c2 = FleetChromosome(child2_drones)
        c1.clamp_all_bounds(self.grid_width, self.grid_height)
        c2.clamp_all_bounds(self.grid_width, self.grid_height)
        return c1, c2

    def mutate(self, individual: FleetChromosome) -> None:
        """
        Custom spatial mutation suite:
        - Gaussian Coordinate Jitter
        - Dynamic Waypoint Insertion & Deletion
        - Speed profile mutation
        - Sensor power mutation
        """
        for drone in individual.drones:
            # 1. Dynamic Insertion Mutation (explore detour or new area)
            if random.random() < (self.mutation_rate * 0.7) and len(drone.waypoints) < self.max_waypoints:
                idx = random.randint(0, len(drone.waypoints))
                if idx == 0 or idx == len(drone.waypoints):
                    new_pt = (random.uniform(2, self.grid_width - 2), random.uniform(2, self.grid_height - 2))
                else:
                    # Intermediate interpolation with noise
                    p_a = drone.waypoints[idx - 1]
                    p_b = drone.waypoints[idx]
                    mid_x = (p_a[0] + p_b[0]) / 2.0 + random.gauss(0, 3.0)
                    mid_y = (p_a[1] + p_b[1]) / 2.0 + random.gauss(0, 3.0)
                    new_pt = (mid_x, mid_y)

                drone.waypoints.insert(idx, new_pt)
                drone.speeds.insert(idx, random.choice(DroneGene.SPEED_LEVELS))
                drone.sensor_powers.insert(idx, random.choice(DroneGene.SENSOR_LEVELS))

            # 2. Dynamic Deletion Mutation (prune redundant waypoints)
            if random.random() < (self.mutation_rate * 0.7) and len(drone.waypoints) > self.min_waypoints:
                del_idx = random.randint(0, len(drone.waypoints) - 1)
                drone.waypoints.pop(del_idx)
                drone.speeds.pop(del_idx)
                drone.sensor_powers.pop(del_idx)

            # 3. Spatial Gaussian Jitter
            for i in range(len(drone.waypoints)):
                if random.random() < self.mutation_rate:
                    curr_x, curr_y = drone.waypoints[i]
                    jitter_x = curr_x + random.gauss(0, 2.5)
                    jitter_y = curr_y + random.gauss(0, 2.5)
                    drone.waypoints[i] = (jitter_x, jitter_y)

                # 4. Speed mutation
                if random.random() < self.mutation_rate:
                    drone.speeds[i] = random.choice(DroneGene.SPEED_LEVELS)

                # 5. Sensor power profile mutation
                if random.random() < self.mutation_rate:
                    drone.sensor_powers[i] = random.choice(DroneGene.SENSOR_LEVELS)

        individual.clamp_all_bounds(self.grid_width, self.grid_height)
