import os
import json
import matplotlib.pyplot as plt
from src.environment import HazardGrid
from src.fitness import MultiObjectiveFitnessEvaluator
from src.operators import GeneticOperators
from src.ga_engine import GeneticAlgorithmEngine, RandomSearchBaseline
from src.visualizer import Visualizer


def run_sensitivity_experiments():
    print("=" * 70)
    print("   RUNNING HYPERPARAMETER SENSITIVITY & BASELINE EXPERIMENTS")
    print("   Generating empirical data and plots for the Technical Report")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.path.join(results_dir)
    vis = Visualizer(output_dir=results_dir)

    grid = HazardGrid(width=50, height=50, seed=42)
    evaluator = MultiObjectiveFitnessEvaluator(grid=grid)

    # -------------------------------------------------------------
    # 1. Population Size Sensitivity Analysis
    # -------------------------------------------------------------
    print("\n[Experiment 1/4] Testing Population Sizes: [25, 60, 120]...")
    pop_sizes = [25, 60, 120]
    pop_results = {}

    plt.figure(figsize=(9, 5), dpi=300)
    for pop in pop_sizes:
        print(f"  -> Running Population Size = {pop}...")
        operators = GeneticOperators(grid_width=50, grid_height=50, mutation_rate=0.08, tournament_size=4)
        engine = GeneticAlgorithmEngine(
            grid=grid,
            evaluator=evaluator,
            operators=operators,
            pop_size=pop,
            generations=50,
            num_drones=3,
            verbose=False,
        )
        res = engine.run()
        pop_results[pop] = res["best_fitness_history"]
        plt.plot(res["best_fitness_history"], label=f"Population = {pop}", linewidth=2.0)

    plt.title("Impact of Population Size on GA Fitness Convergence", fontsize=12, fontweight="bold")
    plt.xlabel("Generation", fontsize=10)
    plt.ylabel("Best Fitness", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    pop_plot_path = os.path.join(results_dir, "sensitivity_population.png")
    plt.tight_layout()
    plt.savefig(pop_plot_path)
    plt.close()
    print(f"  [OK] Population plot saved to {pop_plot_path}")

    # -------------------------------------------------------------
    # 2. Mutation Rate Sensitivity Analysis
    # -------------------------------------------------------------
    print("\n[Experiment 2/4] Testing Mutation Rates: [0.02, 0.08, 0.22]...")
    mutation_rates = [0.02, 0.08, 0.22]
    mut_labels = ["Low (0.02)", "Moderate (0.08)", "High (0.22)"]

    plt.figure(figsize=(9, 5), dpi=300)
    for m_rate, m_lbl in zip(mutation_rates, mut_labels):
        print(f"  -> Running Mutation Rate = {m_rate} ({m_lbl})...")
        operators = GeneticOperators(grid_width=50, grid_height=50, mutation_rate=m_rate, tournament_size=4)
        engine = GeneticAlgorithmEngine(
            grid=grid,
            evaluator=evaluator,
            operators=operators,
            pop_size=50,
            generations=50,
            num_drones=3,
            verbose=False,
        )
        res = engine.run()
        plt.plot(res["best_fitness_history"], label=f"Mutation Rate = {m_lbl}", linewidth=2.0)

    plt.title("Impact of Mutation Rate on Exploration vs Exploitation", fontsize=12, fontweight="bold")
    plt.xlabel("Generation", fontsize=10)
    plt.ylabel("Best Fitness", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    mut_plot_path = os.path.join(results_dir, "sensitivity_mutation.png")
    plt.tight_layout()
    plt.savefig(mut_plot_path)
    plt.close()
    print(f"  [OK] Mutation plot saved to {mut_plot_path}")

    # -------------------------------------------------------------
    # 3. Selection Pressure Sensitivity Analysis (Tournament Size)
    # -------------------------------------------------------------
    print("\n[Experiment 3/4] Testing Tournament Sizes (Selection Pressure): [2, 4, 8]...")
    tourn_sizes = [2, 4, 8]
    tourn_labels = ["Weak Pressure (k=2)", "Balanced Pressure (k=4)", "Aggressive Pressure (k=8)"]

    plt.figure(figsize=(9, 5), dpi=300)
    for t_size, t_lbl in zip(tourn_sizes, tourn_labels):
        print(f"  -> Running Tournament Size = {t_size}...")
        operators = GeneticOperators(grid_width=50, grid_height=50, mutation_rate=0.08, tournament_size=t_size)
        engine = GeneticAlgorithmEngine(
            grid=grid,
            evaluator=evaluator,
            operators=operators,
            pop_size=50,
            generations=50,
            num_drones=3,
            verbose=False,
        )
        res = engine.run()
        plt.plot(res["best_fitness_history"], label=f"Selection: {t_lbl}", linewidth=2.0)

    plt.title("Impact of Selection Pressure on Convergence Rate", fontsize=12, fontweight="bold")
    plt.xlabel("Generation", fontsize=10)
    plt.ylabel("Best Fitness", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    tourn_plot_path = os.path.join(results_dir, "sensitivity_tournament.png")
    plt.tight_layout()
    plt.savefig(tourn_plot_path)
    plt.close()
    print(f"  [OK] Selection pressure plot saved to {tourn_plot_path}")

    # -------------------------------------------------------------
    # 4. Comparative Study: GA vs. Baseline Random Search
    # -------------------------------------------------------------
    print("\n[Experiment 4/4] Comparing GA with Baseline Random Search...")
    TOTAL_EVALUATIONS = 50 * 50  # 50 pop * 50 gens = 2500 evaluations

    # 4A. Run GA
    operators = GeneticOperators(grid_width=50, grid_height=50, mutation_rate=0.08, tournament_size=4)
    ga_engine = GeneticAlgorithmEngine(
        grid=grid,
        evaluator=evaluator,
        operators=operators,
        pop_size=50,
        generations=50,
        num_drones=3,
        verbose=False,
    )
    ga_res = ga_engine.run()

    # 4B. Run Random Search Baseline
    print(f"  -> Executing {TOTAL_EVALUATIONS} evaluations of Random Search...")
    rs_baseline = RandomSearchBaseline(
        grid=grid,
        evaluator=evaluator,
        total_evaluations=TOTAL_EVALUATIONS,
        num_drones=3,
    )
    rs_res = rs_baseline.run()

    # Downsample random search history to 50 generation checkpoints for direct comparison
    rs_history_gens = [
        rs_res["best_fitness_history"][min(len(rs_res["best_fitness_history"]) - 1, (i + 1) * 50 - 1)]
        for i in range(50)
    ]

    base_plot_path = vis.plot_baseline_comparison(
        ga_fitness_history=ga_res["best_fitness_history"],
        random_search_fitness_history=rs_history_gens,
        filename="baseline_comparison.png",
    )
    print(f"  [OK] Baseline comparison plot saved to {base_plot_path}")

    # Summary table output
    ga_best = ga_res["best_individual"]
    rs_best = rs_res["best_individual"]

    print("\n" + "=" * 70)
    print("            COMPARATIVE EVALUATION SUMMARY (GA vs RANDOM SEARCH)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'GA Solution':<18} | {'Random Search':<18}")
    print("-" * 70)
    print(f"{'Final Fitness':<30} | {ga_best.fitness:<18.2f} | {rs_best.fitness:<18.2f}")
    print(f"{'Disaster Target Coverage':<30} | {ga_best.objective_metrics['coverage_score']:<18.1f} | {rs_best.objective_metrics['coverage_score']:<18.1f}")
    print(f"{'Target Detection Rate':<30} | {ga_best.objective_metrics['detection_rate']*100:<17.1f}% | {rs_best.objective_metrics['detection_rate']*100:<17.1f}%")
    print(f"{'NFZ Obstacle Violations':<30} | {ga_best.objective_metrics['nfz_violations']:<18d} | {rs_best.objective_metrics['nfz_violations']:<18d}")
    print(f"{'Total Fleet Energy (J)':<30} | {ga_best.objective_metrics['total_energy']:<18.1f} | {rs_best.objective_metrics['total_energy']:<18.1f}")
    print(f"{'Hazard Exposure':<30} | {ga_best.objective_metrics['hazard_exposure']:<18.2f} | {rs_best.objective_metrics['hazard_exposure']:<18.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_sensitivity_experiments()
