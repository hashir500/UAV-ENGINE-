import time
from typing import List, Dict, Any, Tuple
from .environment import HazardGrid
from .chromosome import FleetChromosome
from .fitness import MultiObjectiveFitnessEvaluator
from .operators import GeneticOperators


class GeneticAlgorithmEngine:
    """
    Core Optimization Engine orchestrating population evolution,
    elitism, metrics logging, and convergence tracking.
    """

    def __init__(
        self,
        grid: HazardGrid,
        evaluator: MultiObjectiveFitnessEvaluator,
        operators: GeneticOperators,
        pop_size: int = 50,
        generations: int = 80,
        num_drones: int = 3,
        elitism_count: int = 2,
        verbose: bool = True,
    ):
        self.grid = grid
        self.evaluator = evaluator
        self.operators = operators
        self.pop_size = pop_size
        self.generations = generations
        self.num_drones = num_drones
        self.elitism_count = max(1, elitism_count)
        self.verbose = verbose

    def run(self) -> Dict[str, Any]:
        """Executes the Genetic Algorithm and returns convergence history & best solution."""
        start_time = time.time()

        # 1. Initialization
        population: List[FleetChromosome] = [
            FleetChromosome.create_random(
                num_drones=self.num_drones,
                grid_width=self.grid.width,
                grid_height=self.grid.height,
                min_waypoints=self.operators.min_waypoints,
                max_waypoints=self.operators.max_waypoints,
                depot=self.grid.depot,
            )
            for _ in range(self.pop_size)
        ]

        # Initial evaluation
        for ind in population:
            self.evaluator.evaluate(ind)

        best_individual = max(population, key=lambda ind: ind.fitness).clone()

        best_fitness_history = []
        avg_fitness_history = []
        metrics_history = []
        snapshots = {}

        # Snapshot Gen 0 (initial unevolved population)
        snapshots[0] = best_individual.clone()

        if self.verbose:
            print(f"--- Launching GA Optimization ({self.pop_size} pop, {self.generations} gens, {self.num_drones} drones) ---")
            print(f"Gen 0 | Best Fitness: {best_individual.fitness:.2f} | "
                  f"Coverage: {best_individual.objective_metrics['coverage_score']:.1f} | "
                  f"Violations: {best_individual.objective_metrics['nfz_violations']}")

        # 2. Generational Loop
        for gen in range(1, self.generations + 1):
            # Sort population descending by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Record stats
            gen_best_fitness = population[0].fitness
            gen_avg_fitness = sum(ind.fitness for ind in population) / self.pop_size

            best_fitness_history.append(gen_best_fitness)
            avg_fitness_history.append(gen_avg_fitness)
            metrics_history.append(population[0].objective_metrics)

            if gen_best_fitness > best_individual.fitness:
                best_individual = population[0].clone()

            # Snapshot intermediate states
            if gen == (self.generations // 2):
                snapshots[gen] = best_individual.clone()

            # Elitism: preserve top performers
            new_population: List[FleetChromosome] = [
                ind.clone() for ind in population[:self.elitism_count]
            ]

            # Breed remaining population
            while len(new_population) < self.pop_size:
                p1 = self.operators.select(population)
                p2 = self.operators.select(population)
                c1, c2 = self.operators.crossover(p1, p2)

                self.operators.mutate(c1)
                self.operators.mutate(c2)

                self.evaluator.evaluate(c1)
                self.evaluator.evaluate(c2)

                new_population.append(c1)
                if len(new_population) < self.pop_size:
                    new_population.append(c2)

            population = new_population

            if self.verbose and (gen % 10 == 0 or gen == self.generations):
                top_m = population[0].objective_metrics
                print(
                    f"Gen {gen:3d} | Best Fit: {population[0].fitness:7.2f} | Avg Fit: {gen_avg_fitness:7.2f} | "
                    f"Coverage: {top_m['coverage_score']:5.1f} ({top_m['detection_rate']*100:4.1f}%) | "
                    f"NFZ Viol: {top_m['nfz_violations']:2d} | Energy: {top_m['total_energy']:6.0f}J"
                )

        # Final snapshot
        snapshots[self.generations] = best_individual.clone()
        elapsed = time.time() - start_time

        if self.verbose:
            print(f"--- GA Completed in {elapsed:.2f}s | Optimal Fitness: {best_individual.fitness:.2f} ---")

        return {
            "best_individual": best_individual,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "metrics_history": metrics_history,
            "snapshots": snapshots,
            "elapsed_time": elapsed,
        }


class RandomSearchBaseline:
    """
    Baseline Random Search benchmark evaluating identical computational budget
    (pop_size * generations) for objective comparison against GA.
    """

    def __init__(
        self,
        grid: HazardGrid,
        evaluator: MultiObjectiveFitnessEvaluator,
        total_evaluations: int = 4000,
        num_drones: int = 3,
        min_waypoints: int = 4,
        max_waypoints: int = 12,
    ):
        self.grid = grid
        self.evaluator = evaluator
        self.total_evaluations = total_evaluations
        self.num_drones = num_drones
        self.min_waypoints = min_waypoints
        self.max_waypoints = max_waypoints

    def run(self) -> Dict[str, Any]:
        """Executes uniform random search."""
        best_individual: FleetChromosome = None
        best_fitness_history = []
        best_fitness = -float("inf")

        for i in range(self.total_evaluations):
            candidate = FleetChromosome.create_random(
                num_drones=self.num_drones,
                grid_width=self.grid.width,
                grid_height=self.grid.height,
                min_waypoints=self.min_waypoints,
                max_waypoints=self.max_waypoints,
                depot=self.grid.depot,
            )
            fit = self.evaluator.evaluate(candidate)
            if fit > best_fitness:
                best_fitness = fit
                best_individual = candidate.clone()
            best_fitness_history.append(best_fitness)

        return {
            "best_individual": best_individual,
            "best_fitness_history": best_fitness_history,
            "final_best_fitness": best_fitness,
        }
