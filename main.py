import os
import sys
from src.environment import HazardGrid
from src.fitness import MultiObjectiveFitnessEvaluator
from src.operators import GeneticOperators
from src.ga_engine import GeneticAlgorithmEngine
from src.visualizer import Visualizer


def run_mission_simulation():
    print("=" * 70)
    print("   AUTONOMOUS UAV FLEET DISASTER MANAGEMENT STRATEGY ENGINE")
    print("   Genetic Algorithm Multi-Objective Optimization Simulation")
    print("=" * 70)

    # 1. Environment Setup (50x50 Hazard Grid)
    print("\n[1/4] Initializing 50x50 Hazard Grid...")
    grid = HazardGrid(width=50, height=50, seed=42)
    print(f"  -> Total Priority Target Cells: {int((grid.targets > 0).sum())}")
    print(f"  -> Total Target Value Points: {grid.targets.sum():.1f}")
    print(f"  -> No-Fly Zones / Obstacle Cells: {int(grid.obstacles.sum())}")
    print(f"  -> Base Depot: {grid.depot}")

    # 2. Strategy Engine Components
    print("\n[2/4] Configuring Strategy Engine & Fitness Evaluator...")
    evaluator = MultiObjectiveFitnessEvaluator(
        grid=grid,
        w_coverage=3.5,
        w_energy=0.004,
        w_time=0.4,
        w_hazard=2.2,
        penalty_nfz=90.0,
        penalty_battery=0.03,
        battery_capacity=18000.0,
    )

    operators = GeneticOperators(
        grid_width=50,
        grid_height=50,
        min_waypoints=4,
        max_waypoints=10,
        mutation_rate=0.08,
        crossover_rate=0.85,
        tournament_size=4,
    )

    # 3. Execution of Genetic Algorithm
    print("\n[3/4] Evolving Fleet Deployment Positions & Trajectories...")
    POPULATION_SIZE = 60
    GENERATIONS = 70
    NUM_DRONES = 3

    engine = GeneticAlgorithmEngine(
        grid=grid,
        evaluator=evaluator,
        operators=operators,
        pop_size=POPULATION_SIZE,
        generations=GENERATIONS,
        num_drones=NUM_DRONES,
        elitism_count=3,
        verbose=True,
    )

    results = engine.run()
    best_ind = results["best_individual"]
    metrics = best_ind.objective_metrics

    # 4. Result Metrics Summary
    print("\n" + "=" * 70)
    print("                  OPTIMAL EVOLVED MISSION PROFILE")
    print("=" * 70)
    print(f"  Final Multi-Objective Fitness  : {best_ind.fitness:.2f}")
    print(f"  Priority Disaster Coverage     : {metrics['coverage_score']:.1f} pts")
    print(f"  Target Detection Rate          : {metrics['detection_rate'] * 100:.2f}%")
    print(f"  Total Fleet Energy Consumed    : {metrics['total_energy']:.1f} Joules")
    print(f"  Mission Makespan (Flight Time) : {metrics['makespan_time']:.1f} seconds")
    print(f"  Hazard Exposure Accumulated    : {metrics['hazard_exposure']:.2f}")
    print(f"  No-Fly Zone (NFZ) Violations   : {metrics['nfz_violations']} cells")
    print(f"  Battery Capacity Exceeded      : {metrics['battery_excess']:.1f} J")
    print("-" * 70)

    for i, drone in enumerate(best_ind.drones):
        avg_spd = sum(drone.speeds) / len(drone.speeds)
        active_sensors = sum(1 for s in drone.sensor_powers if s > 0)
        print(
            f"  Drone {i + 1} | Waypoints: {len(drone.waypoints):2d} | "
            f"Avg Speed: {avg_spd:.1f} m/s | Active Sensor Waypoints: {active_sensors}/{len(drone.waypoints)}"
        )

    # 5. Visualizations
    print("\n[4/4] Generating Publication-Quality Plots...")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    vis = Visualizer(output_dir=results_dir)

    conv_path = vis.plot_convergence(results["best_fitness_history"], results["avg_fitness_history"])
    path_path = vis.plot_flight_paths(grid, best_ind)
    heat_path = vis.plot_coverage_heatmap(grid, best_ind)
    prog_path = vis.plot_evolution_progression(grid, results["snapshots"])

    print(f"  [OK] Convergence curve saved to      : {conv_path}")
    print(f"  [OK] UAV flight paths map saved to   : {path_path}")
    print(f"  [OK] Coverage heatmap saved to       : {heat_path}")
    print(f"  [OK] Evolution progression saved to  : {prog_path}")
    print("=" * 70)
    print("Simulation finished successfully!")


if __name__ == "__main__":
    run_mission_simulation()
