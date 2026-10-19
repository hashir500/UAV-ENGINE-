import numpy as np
from typing import List, Tuple, Set


class HazardGrid:
    """
    Represents the 50x50 operational mission space for UAV surveillance
    and disaster management.
    
    Includes:
    - Base depot location (launch/recovery)
    - Disaster priority target zones (rewards to maximize)
    - Environmental hazard pockets (wind/radiation soft penalty)
    - No-Fly Zones / Obstacles (hard penalty)
    """

    def __init__(self, width: int = 50, height: int = 50, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        # Base depot (launch & return location)
        self.depot = (2, 2)

        # Layers
        self.obstacles = np.zeros((height, width), dtype=bool)   # True = Blocked / NFZ
        self.hazards = np.zeros((height, width), dtype=float)     # 0.0 to 1.0 intensity
        self.targets = np.zeros((height, width), dtype=float)     # 0.0 to 5.0 priority

        self._generate_environment()

    def _generate_environment(self) -> None:
        """Procedurally generate realistic disaster scenario features."""
        # 1. No-Fly Zones (NFZ) / Obstacles (e.g., collapsed buildings, radar domes)
        nfz_boxes = [
            (10, 15, 8, 5),   # x, y, w, h
            (25, 30, 6, 8),
            (35, 10, 7, 7),
            (18, 38, 10, 4),
            (38, 35, 5, 8),
        ]
        for (x, y, w, h) in nfz_boxes:
            x_end = min(self.width, x + w)
            y_end = min(self.height, y + h)
            self.obstacles[y:y_end, x:x_end] = True

        # Ensure depot is clear
        self.obstacles[0:5, 0:5] = False

        # 2. Hazard Pockets (Gaussian radiation / storm pockets)
        hazard_centers = [
            ((20, 20), 6.0, 0.9),   # (center_x, center_y), radius_sigma, max_intensity
            ((40, 22), 5.0, 0.8),
            ((12, 35), 4.5, 0.85),
            ((30, 42), 5.5, 0.75),
        ]
        y_indices, x_indices = np.indices((self.height, self.width))
        for (cx, cy), sigma, max_val in hazard_centers:
            dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
            gaussian = max_val * np.exp(-dist_sq / (2.0 * sigma ** 2))
            self.hazards = np.maximum(self.hazards, gaussian)

        # 3. High-Priority Disaster Target Clusters (Survivors, fire hotspots)
        target_clusters = [
            ((8, 25), 3, 5.0),    # critical survivors
            ((22, 12), 4, 4.0),   # high-value infrastructure
            ((32, 28), 5, 5.0),   # severe hazard inspection
            ((45, 42), 3, 3.5),   # secondary perimeter
            ((15, 45), 4, 4.5),   # rescue outpost
            ((42, 8), 3, 3.0),    # peripheral reconnaissance
        ]
        for (cx, cy), radius, priority in target_clusters:
            dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
            mask = dist <= radius
            # Apply priority where not an obstacle
            self.targets[mask & (~self.obstacles)] = np.maximum(
                self.targets[mask & (~self.obstacles)],
                priority * (1.0 - dist[mask & (~self.obstacles)] / (radius + 1.0))
            )

    @staticmethod
    def bresenham_line(p1: Tuple[int, int], p2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm to rasterize the straight flight path between two waypoints.
        Returns all visited integer grid coordinates.
        """
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        x2, y2 = int(round(p2[0])), int(round(p2[1]))

        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        curr_x, curr_y = x1, y1
        while True:
            points.append((curr_x, curr_y))
            if curr_x == x2 and curr_y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy
        return points

    def get_cells_in_radius(self, center: Tuple[float, float], radius: float) -> List[Tuple[int, int]]:
        """Returns valid integer grid coordinates within a circular sensor footprint."""
        cx, cy = center
        min_x = max(0, int(np.floor(cx - radius)))
        max_x = min(self.width - 1, int(np.ceil(cx + radius)))
        min_y = max(0, int(np.floor(cy - radius)))
        max_y = min(self.height - 1, int(np.ceil(cy + radius)))

        cells = []
        r_sq = radius ** 2
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r_sq:
                    cells.append((x, y))
        return cells
