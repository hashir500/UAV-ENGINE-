import numpy as np
from typing import Dict, Any, Set, Tuple
from .environment import HazardGrid
from .chromosome import FleetChromosome, DroneGene


class MultiObjectiveFitnessEvaluator:
    """
    Evaluates a FleetChromosome across multiple competing objectives:
    1. Maximization: Disaster zone target coverage & detection rate
    2. Minimization: Fleet energy consumption
    3. Minimization: Total mission flight time (makespan)
    4. Minimization: Environmental hazard exposure (radiation/wind)
    5. Penalty Function: Hard constraint violations (NFZ intrusions & battery limits)
    """

    def __init__(
        self,
        grid: HazardGrid,
        w_coverage: float = 3.0,
        w_energy: float = 0.005,
        w_time: float = 0.5,
        w_hazard: float = 2.0,
        penalty_nfz: float = 80.0,
        penalty_battery: float = 0.02,
        battery_capacity: float = 18000.0,  # Joules per drone
    ):
        self.grid = grid
        self.w_coverage = w_coverage
        self.w_energy = w_energy
        self.w_time = w_time
        self.w_hazard = w_hazard
        self.penalty_nfz = penalty_nfz
        self.penalty_battery = penalty_battery
        self.battery_capacity = battery_capacity

        # Calculate total target points in grid for detection rate metric
        self.total_target_cells = int(np.sum(self.grid.targets > 0))
        self.total_target_value = float(np.sum(self.grid.targets))

    def _flight_power(self, speed: float) -> float:
        """Aerodynamic & propulsion power model (Watts). P = P_0 + c * v^2"""
        return 20.0 + 7.5 * (speed ** 2)

    def evaluate(self, chromosome: FleetChromosome) -> float:
        """
        Evaluates the chromosome, computes objective metrics, and assigns the fitness.
        """
        depot = self.grid.depot
        covered_cells: Set[Tuple[int, int]] = set()

        total_energy = 0.0
        max_flight_time = 0.0
        total_hazard_exposure = 0.0
        total_nfz_violations = 0
        total_battery_excess = 0.0

        for drone in chromosome.drones:
            drone_time = 0.0
            drone_energy = 0.0

            # Route: depot -> waypoint_1 -> ... -> waypoint_n -> depot
            route_points = [depot] + drone.waypoints + [depot]
            num_segments = len(route_points) - 1

            for seg_idx in range(num_segments):
                p_start = route_points[seg_idx]
                p_end = route_points[seg_idx + 1]

                # Leg distance
                dx = p_end[0] - p_start[0]
                dy = p_end[1] - p_start[1]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # Segment speed & sensor power
                # If segment is return to depot, use cruise speed and sensor OFF
                if seg_idx < len(drone.speeds):
                    speed = drone.speeds[seg_idx]
                    sensor_lvl = drone.sensor_powers[seg_idx]
                else:
                    speed = 2.0  # Cruise back to depot
                    sensor_lvl = 0

                leg_time = distance / max(0.5, speed)
                drone_time += leg_time

                # Energy = (P_flight + P_sensor) * time
                sensor_pwr = DroneGene.SENSOR_CONFIG[sensor_lvl]["power"]
                p_flight = self._flight_power(speed)
                leg_energy = (p_flight + sensor_pwr) * leg_time
                drone_energy += leg_energy

                # Rasterize path to check hazard exposure and NFZ collisions
                path_cells = self.grid.bresenham_line(p_start, p_end)
                for (cx, cy) in path_cells:
                    if 0 <= cx < self.grid.width and 0 <= cy < self.grid.height:
                        if self.grid.obstacles[cy, cx]:
                            total_nfz_violations += 1
                        total_hazard_exposure += self.grid.hazards[cy, cx]
                    else:
                        total_nfz_violations += 5  # Out of bounds penalty

                # Sensor coverage around destination waypoint
                sensor_radius = DroneGene.SENSOR_CONFIG[sensor_lvl]["radius"]
                if sensor_radius > 0:
                    scanned = self.grid.get_cells_in_radius(p_end, sensor_radius)
                    for sc in scanned:
                        covered_cells.add(sc)

            total_energy += drone_energy
            max_flight_time = max(max_flight_time, drone_time)

            if drone_energy > self.battery_capacity:
                total_battery_excess += (drone_energy - self.battery_capacity)

        # 1. Coverage Score & Target Detection
        coverage_score = 0.0
        detected_targets = 0
        for (cx, cy) in covered_cells:
            t_val = self.grid.targets[cy, cx]
            if t_val > 0:
                coverage_score += t_val
                detected_targets += 1

        detection_rate = (detected_targets / self.total_target_cells) if self.total_target_cells > 0 else 0.0

        # 2. Penalty Calculations
        penalty = (
            self.penalty_nfz * total_nfz_violations
            + self.penalty_battery * total_battery_excess
        )

        # 3. Composite Fitness Formula
        fitness = (
            self.w_coverage * coverage_score
            - self.w_energy * total_energy
            - self.w_time * max_flight_time
            - self.w_hazard * total_hazard_exposure
            - penalty
        )

        chromosome.fitness = fitness
        chromosome.objective_metrics = {
            "coverage_score": coverage_score,
            "detection_rate": detection_rate,
            "total_energy": total_energy,
            "makespan_time": max_flight_time,
            "hazard_exposure": total_hazard_exposure,
            "nfz_violations": total_nfz_violations,
            "battery_excess": total_battery_excess,
            "penalty": penalty,
            "covered_cells_count": len(covered_cells),
        }

        return fitness
