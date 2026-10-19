import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import Dict, List, Optional
from .environment import HazardGrid
from .chromosome import FleetChromosome, DroneGene


class Visualizer:
    """
    Renders high-resolution graphical plots of UAV trajectories,
    sensor coverage heatmaps, fitness convergence curves, and evolution progression.
    """

    DRONE_COLORS = ["#00F5D4", "#FF007F", "#FEE440", "#7B2CBF", "#00BBF9"]

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_convergence(
        self,
        best_history: List[float],
        avg_history: List[float],
        filename: str = "convergence_curve.png",
    ) -> str:
        """Plots best vs average fitness across generations."""
        plt.figure(figsize=(10, 5), dpi=300)
        generations = range(1, len(best_history) + 1)

        plt.plot(generations, best_history, label="Best Fitness", color="#0077B6", linewidth=2.5)
        plt.plot(generations, avg_history, label="Population Average Fitness", color="#F77F00", linestyle="--", linewidth=1.8)
        plt.fill_between(generations, avg_history, best_history, color="#0077B6", alpha=0.1)

        plt.title("Genetic Algorithm Fitness Convergence (UAV Fleet Strategy Engine)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Generation", fontsize=11)
        plt.ylabel("Multi-Objective Fitness", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", loc="lower right")

        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_flight_paths(
        self,
        grid: HazardGrid,
        chromosome: FleetChromosome,
        title: str = "Evolved UAV Fleet Trajectories & Sensor Footprints",
        filename: str = "uav_flight_paths.png",
    ) -> str:
        """Plots the 2D disaster grid with obstacles, hazards, targets, and drone flight paths."""
        fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

        # 1. Base hazard field (soft background)
        hazard_cmap = plt.cm.YlOrRd
        hazard_display = np.ma.masked_where(grid.hazards < 0.05, grid.hazards)
        ax.imshow(
            hazard_display,
            cmap=hazard_cmap,
            origin="lower",
            alpha=0.45,
            extent=[0, grid.width, 0, grid.height],
        )

        # 2. No-Fly Zones / Obstacles (Hard constraints)
        y_obs, x_obs = np.where(grid.obstacles)
        for xo, yo in zip(x_obs, y_obs):
            ax.add_patch(
                Rectangle((xo, yo), 1, 1, facecolor="#2B2D42", edgecolor="#111", hatch="//", alpha=0.9)
            )

        # 3. High-Priority Target Clusters
        y_tgt, x_tgt = np.where(grid.targets > 0)
        tgt_vals = grid.targets[y_tgt, x_tgt]
        sc = ax.scatter(
            x_tgt + 0.5,
            y_tgt + 0.5,
            c=tgt_vals,
            cmap="viridis",
            s=tgt_vals * 22,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.85,
            zorder=3,
            label="Disaster Targets",
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Priority Value (1-5)", fontsize=10)

        # 4. Depot / Launch Site
        depot_x, depot_y = grid.depot
        ax.scatter(
            [depot_x],
            [depot_y],
            color="#2EC4B6",
            s=250,
            marker="s",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="Fleet Base / Depot",
        )

        # 5. UAV Trajectories & Sensor Circles
        depot = grid.depot
        for idx, drone in enumerate(chromosome.drones):
            color = self.DRONE_COLORS[idx % len(self.DRONE_COLORS)]
            route = [depot] + drone.waypoints + [depot]
            rx = [p[0] for p in route]
            ry = [p[1] for p in route]

            # Flight path line
            ax.plot(
                rx,
                ry,
                color=color,
                linewidth=2.2,
                alpha=0.9,
                zorder=4,
                label=f"Drone {idx + 1} ({len(drone.waypoints)} waypoints)",
            )

            # Waypoint markers and sensor footprints
            for w_idx, (wx, wy) in enumerate(drone.waypoints):
                ax.scatter(wx, wy, color=color, s=45, edgecolors="black", zorder=5)

                sensor_lvl = drone.sensor_powers[w_idx] if w_idx < len(drone.sensor_powers) else 0
                radius = DroneGene.SENSOR_CONFIG[sensor_lvl]["radius"]
                if radius > 0:
                    sensor_circle = Circle(
                        (wx, wy),
                        radius,
                        facecolor=color,
                        edgecolor=color,
                        linestyle="--",
                        alpha=0.15,
                        zorder=2,
                    )
                    ax.add_patch(sensor_circle)

        ax.set_xlim(0, grid.width)
        ax.set_ylim(0, grid.height)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("X Grid Coordinates", fontsize=10)
        ax.set_ylabel("Y Grid Coordinates", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_coverage_heatmap(
        self,
        grid: HazardGrid,
        chromosome: FleetChromosome,
        filename: str = "coverage_heatmap.png",
    ) -> str:
        """Renders the cumulative sensor scan density map across the grid."""
        coverage_map = np.zeros((grid.height, grid.width), dtype=float)

        for drone in chromosome.drones:
            for w_idx, (wx, wy) in enumerate(drone.waypoints):
                sensor_lvl = drone.sensor_powers[w_idx] if w_idx < len(drone.sensor_powers) else 0
                radius = DroneGene.SENSOR_CONFIG[sensor_lvl]["radius"]
                if radius > 0:
                    cells = grid.get_cells_in_radius((wx, wy), radius)
                    for cx, cy in cells:
                        coverage_map[cy, cx] += 1.0

        fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
        cmap = plt.cm.magma
        im = ax.imshow(coverage_map, cmap=cmap, origin="lower", extent=[0, grid.width, 0, grid.height])

        # Overlay targets
        y_tgt, x_tgt = np.where(grid.targets > 0)
        ax.scatter(x_tgt + 0.5, y_tgt + 0.5, color="cyan", s=10, alpha=0.6, label="Disaster Targets")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sensor Scan Overlap Count")
        ax.set_title("UAV Fleet Cumulative Sensor Coverage Heatmap", fontsize=12, fontweight="bold")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend(loc="upper right", fontsize=8)

        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_evolution_progression(
        self,
        grid: HazardGrid,
        snapshots: Dict[int, FleetChromosome],
        filename: str = "evolution_progression.png",
    ) -> str:
        """Plots side-by-side progression across generations (Gen 0 vs Mid vs Final)."""
        gen_keys = sorted(snapshots.keys())
        fig, axes = plt.subplots(1, len(gen_keys), figsize=(5.5 * len(gen_keys), 5), dpi=300)
        if len(gen_keys) == 1:
            axes = [axes]

        for ax, gen in zip(axes, gen_keys):
            ind = snapshots[gen]
            # Hazards
            hazard_display = np.ma.masked_where(grid.hazards < 0.05, grid.hazards)
            ax.imshow(hazard_display, cmap="YlOrRd", origin="lower", alpha=0.35, extent=[0, grid.width, 0, grid.height])

            # Obstacles
            y_obs, x_obs = np.where(grid.obstacles)
            for xo, yo in zip(x_obs, y_obs):
                ax.add_patch(Rectangle((xo, yo), 1, 1, facecolor="#2B2D42", hatch="//"))

            # Targets
            y_tgt, x_tgt = np.where(grid.targets > 0)
            ax.scatter(x_tgt + 0.5, y_tgt + 0.5, color="green", s=8, alpha=0.5)

            # Drones
            depot = grid.depot
            ax.scatter([depot[0]], [depot[1]], color="#2EC4B6", s=100, marker="s", edgecolors="black")

            for idx, drone in enumerate(ind.drones):
                color = self.DRONE_COLORS[idx % len(self.DRONE_COLORS)]
                route = [depot] + drone.waypoints + [depot]
                ax.plot([p[0] for p in route], [p[1] for p in route], color=color, linewidth=1.8, alpha=0.85)

            metrics = ind.objective_metrics
            det_rate = metrics.get("detection_rate", 0.0) * 100
            nfz = metrics.get("nfz_violations", 0)
            ax.set_title(f"Generation {gen}\nFit: {ind.fitness:.1f} | Det: {det_rate:.1f}% | NFZ: {nfz}", fontsize=10, fontweight="bold")
            ax.set_xlim(0, grid.width)
            ax.set_ylim(0, grid.height)
            ax.set_aspect("equal")

        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path

    def plot_baseline_comparison(
        self,
        ga_fitness_history: List[float],
        random_search_fitness_history: List[float],
        filename: str = "baseline_comparison.png",
    ) -> str:
        """Plots comparative convergence between GA and Random Search."""
        plt.figure(figsize=(10, 5), dpi=300)

        plt.plot(ga_fitness_history, label="Genetic Algorithm (Strategy Engine)", color="#2A9D8F", linewidth=2.5)
        plt.plot(random_search_fitness_history, label="Baseline Random Search", color="#E76F51", linestyle="--", linewidth=2.0)

        plt.title("Evolved Optimization vs. Baseline Random Search", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Generation / Iteration Window", fontsize=11)
        plt.ylabel("Multi-Objective Fitness", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", loc="lower right")

        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
