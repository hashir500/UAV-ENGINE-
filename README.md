# Autonomous UAV Fleet Strategy Engine for Disaster Surveillance
**Multi-Objective Optimization via Custom Genetic Algorithm (GA)**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Overview
In modern disaster response and tactical surveillance, deploying a fleet of autonomous Unmanned Aerial Vehicles (UAVs) requires balancing competing, non-linear operational constraints:
* **Maximizing** high-priority survivor cluster coverage and target detection rate.
* **Minimizing** total battery consumption, flight makespan (time), and radiation/wind hazard exposure.
* **Respecting hard constraints** including restricted No-Fly Zones (NFZs) and UAV battery capacities.

This project implements an **Object-Oriented Strategy Engine** powered by a tailored **Genetic Algorithm (GA)** featuring dynamic-length chromosomes, multi-segment spatial operators, and dynamic penalty functions.

---

## 🏛️ System Architecture & Repository Structure

```
├── src/
│   ├── __init__.py           # Package marker
│   ├── environment.py        # 50x50 HazardGrid (targets, hazards, NFZ obstacles, Bresenham line rasterization)
│   ├── chromosome.py         # DroneGene & FleetChromosome (dynamic-length spatial encoding)
│   ├── fitness.py            # MultiObjectiveFitnessEvaluator (power physics, risk, dynamic penalties)
│   ├── operators.py          # GeneticOperators (tournament selection, cut-and-splice crossover, spatial jitter)
│   ├── ga_engine.py          # GeneticAlgorithmEngine & RandomSearchBaseline runner
│   └── visualizer.py         # Matplotlib rendering engine for trajectories, heatmaps, and convergence
├── results/                  # Auto-generated high-resolution PNG plots
│   ├── convergence_curve.png
│   ├── uav_flight_paths.png
│   ├── coverage_heatmap.png
│   ├── evolution_progression.png
│   ├── sensitivity_population.png
│   ├── sensitivity_mutation.png
│   ├── sensitivity_tournament.png
│   └── baseline_comparison.png
├── main.py                   # Main end-to-end simulation entry point
├── experiments.py            # Hyperparameter sensitivity analysis & baseline benchmark
├── requirements.txt          # Python dependencies (numpy, matplotlib)
└── README.md                 # Project documentation and execution instructions
```

---

## 🧬 Chromosome Design (Dynamic Multi-Segment Hybrid Encoding)

Unlike conventional fixed-length binary encodings, this engine utilizes a **hierarchical, variable-length hybrid chromosome** to adaptively discover optimal waypoint counts:

$$\text{Individual} = [\text{Drone}_1, \text{Drone}_2, \dots, \text{Drone}_K]$$

For each drone $d \in \{1, \dots, K\}$, its chromosome segment comprises:
1. **Spatial Trajectory Gene**: Variable-length coordinate sequence $[(x_1, y_1), \dots, (x_L, y_L)]$ bounded in $[L_{\min}=4, L_{\max}=12]$.
2. **Speed Profile Gene**: Leg velocity vector $[v_1, \dots, v_L] \in \{1.0 \text{ m/s (Economy)}, 2.0 \text{ m/s (Cruise)}, 3.0 \text{ m/s (Sprint)}\}$.
3. **Sensor Schedule Gene**: Power state vector $[s_1, \dots, s_L] \in \{0 \text{ (Off)}, 1 \text{ (Standard: } R=3.5\text{)}, 2 \text{ (High-Power: } R=6.0\text{)}\}$.

---

## ⚖️ Multi-Objective Fitness & Physics Model

The scalarized fitness landscape balances competing physical criteria:

$$\text{Fitness} = w_{\text{cov}} \cdot \mathcal{S}_{\text{cov}} - w_{\text{eng}} \cdot E_{\text{total}} - w_{\text{time}} \cdot T_{\text{makespan}} - w_{\text{haz}} \cdot \mathcal{H}_{\text{exp}} - \text{Penalty}_{\text{Hard}}$$

### Physics & Power Formulation
* **Propulsion Power**: $P_{\text{flight}}(v) = 20.0 + 7.5 \cdot v^2 \text{ (Watts)}$
* **Sensor Power**: $P_{\text{sensor}}(s) \in \{0\text{W}, 25\text{W}, 60\text{W}\}$
* **Energy per Leg**: $E_k = (P_{\text{flight}}(v_k) + P_{\text{sensor}}(s_k)) \cdot \frac{d_k}{v_k}$
* **Dynamic Hard Penalty**:
  $$\text{Penalty}_{\text{Hard}} = \lambda_{\text{NFZ}} \cdot N_{\text{blocked\_cells\_crossed}} + \lambda_{\text{bat}} \cdot \max(0, E_d - B_{\max})$$

---

## 🔬 Custom Genetic Operators

1. **Dynamic Cut-and-Splice Crossover**:
   * Slices parent trajectories at arbitrary indices, dynamically recombining sub-paths while respecting min/max waypoint bounds.
2. **Dynamic Insertion & Deletion Mutations**:
   * *Insertion*: Synthesizes an interpolated waypoint with Gaussian noise to explore obstacle detours.
   * *Deletion*: Prunes redundant waypoints to streamline flight lines and conserve battery.
3. **Spatial Gaussian Jitter**:
   * Perturbs coordinates $(x_i, y_i) \leftarrow (x_i + \mathcal{N}(0, \sigma^2), y_i + \mathcal{N}(0, \sigma^2))$.
4. **Tournament Selection**:
   * Samples $k$ individuals and selects the fittest to govern selection pressure.

---

## 🚀 Installation & Execution

### 1. Prerequisites
Ensure Python 3.9+ is installed. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run Main Simulation
To execute the end-to-end GA mission planning simulation and generate core plots:

```bash
python main.py
```

### 3. Run Hyperparameter Sensitivity Analysis & Baseline Benchmark
To execute the empirical experiments (population sizes, mutation rates, tournament sizes, and Random Search comparison):

```bash
python experiments.py
```

All plots will be rendered and saved to the `results/` directory in high resolution (300 DPI).
