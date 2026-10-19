import copy
import random
from typing import List, Tuple, Dict, Any, Optional


class DroneGene:
    """
    Encodes the trajectory, speed schedule, and sensor profile for a single UAV.
    Supports dynamic waypoint lengths.
    """

    SPEED_LEVELS = [1.0, 2.0, 3.0]  # Slow (Economy), Medium (Cruise), Fast (Sprint)
    SENSOR_LEVELS = [0, 1, 2]       # 0: Off, 1: Standard (R=3.5), 2: High-Power (R=6.0)

    SENSOR_CONFIG = {
        0: {"radius": 0.0, "power": 0.0},
        1: {"radius": 3.5, "power": 25.0},
        2: {"radius": 6.0, "power": 60.0},
    }

    def __init__(
        self,
        waypoints: Optional[List[Tuple[float, float]]] = None,
        speeds: Optional[List[float]] = None,
        sensor_powers: Optional[List[int]] = None,
    ):
        self.waypoints: List[Tuple[float, float]] = waypoints if waypoints is not None else []
        self.speeds: List[float] = speeds if speeds is not None else []
        self.sensor_powers: List[int] = sensor_powers if sensor_powers is not None else []

    @classmethod
    def create_random(
        cls,
        grid_width: int = 50,
        grid_height: int = 50,
        min_waypoints: int = 4,
        max_waypoints: int = 10,
        depot: Tuple[int, int] = (2, 2)
    ) -> "DroneGene":
        """Instantiates a drone trajectory with a dynamically sampled number of waypoints."""
        num_pts = random.randint(min_waypoints, max_waypoints)
        waypoints = []
        for _ in range(num_pts):
            wx = random.uniform(2.0, grid_width - 3.0)
            wy = random.uniform(2.0, grid_height - 3.0)
            waypoints.append((wx, wy))

        speeds = [random.choice(cls.SPEED_LEVELS) for _ in range(num_pts)]
        sensor_powers = [random.choice(cls.SENSOR_LEVELS) for _ in range(num_pts)]

        return cls(waypoints, speeds, sensor_powers)

    def clone(self) -> "DroneGene":
        """Returns a deep copy of this drone's chromosome segment."""
        return DroneGene(
            waypoints=copy.deepcopy(self.waypoints),
            speeds=copy.deepcopy(self.speeds),
            sensor_powers=copy.deepcopy(self.sensor_powers),
        )

    def clamp_bounds(self, width: int = 50, height: int = 50) -> None:
        """Clamps all waypoints inside the grid boundaries."""
        clamped = []
        for (x, y) in self.waypoints:
            cx = max(0.0, min(float(width - 1), x))
            cy = max(0.0, min(float(height - 1), y))
            clamped.append((cx, cy))
        self.waypoints = clamped


class FleetChromosome:
    """
    Multi-segment dynamic chromosome representing the complete UAV fleet.
    Encodes multiple drones with variable waypoint counts and sensor/speed schedules.
    """

    def __init__(self, drones: Optional[List[DroneGene]] = None):
        self.drones: List[DroneGene] = drones if drones is not None else []
        self.fitness: float = -float("inf")
        self.objective_metrics: Dict[str, Any] = {}

    @classmethod
    def create_random(
        cls,
        num_drones: int = 3,
        grid_width: int = 50,
        grid_height: int = 50,
        min_waypoints: int = 4,
        max_waypoints: int = 10,
        depot: Tuple[int, int] = (2, 2)
    ) -> "FleetChromosome":
        """Initializes a random fleet chromosome."""
        drones = [
            DroneGene.create_random(grid_width, grid_height, min_waypoints, max_waypoints, depot)
            for _ in range(num_drones)
        ]
        return cls(drones)

    def clone(self) -> "FleetChromosome":
        """Creates a deep copy of the fleet chromosome."""
        clone_ind = FleetChromosome([d.clone() for d in self.drones])
        clone_ind.fitness = self.fitness
        clone_ind.objective_metrics = copy.deepcopy(self.objective_metrics)
        return clone_ind

    def clamp_all_bounds(self, width: int = 50, height: int = 50) -> None:
        for drone in self.drones:
            drone.clamp_bounds(width, height)
