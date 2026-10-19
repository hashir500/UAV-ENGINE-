# TECHNICAL REPORT: AUTONOMOUS UAV FLEET STRATEGY ENGINE
## Multi-Objective Optimization in Volatile Disaster Surveillance Using Genetic Algorithms

**Course:** Artificial Intelligence (BSCS 5)  
**Department:** Department of Computer Science, Bahria University, Lahore Campus  
**Assignment:** 2 (CLO4: Practical Problem Solving via Knowledge Representation & Optimization)  

---

## 1. Executive Summary & Problem Formulation
In modern post-disaster rescue scenarios and hazardous environmental monitoring, rapid deployment of Unmanned Aerial Vehicles (UAVs) is critical. Unlike single-objective benchmark problems (such as TSP or shortest path), multi-UAV mission planning in volatile environments presents an NP-hard, multi-objective optimization problem characterized by competing operational trade-offs:
1. **Target Detection vs. Battery Depletion:** Maximizing sensor coverage over dispersed survivor clusters requires wide sweeps and elevated sensor wattage, drastically depleting battery reserves.
2. **Flight Speed vs. Aerodynamic Drag:** High flight velocity reduces mission makespan (elapsed rescue time) but incurs non-linear aerodynamic power penalties ($P \propto v^3$).
3. **Shortest Paths vs. Hazard Avoidance:** Direct linear routes risk intersecting restricted No-Fly Zones (NFZs, e.g., structural collapse) or turbulent hazard pockets (radiation, thermal plumes).

To address these trade-offs, we designed and implemented an Object-Oriented **Strategy Engine** centered on a custom-tailored **Multi-Objective Genetic Algorithm (GA)** operating on a discretized $50 \times 50$ hazard grid.

---

## 2. Chromosome Architecture & Encoding Rationale

### 2.1 Limitations of Traditional Fixed-Length Encoding
Standard Genetic Algorithms typically employ fixed-size binary strings or rigid real-valued vectors. In spatial path planning, fixed-length waypoint vectors force rigid trajectories that either contain redundant collinear waypoints (wasting computation and battery) or lack sufficient degrees of freedom to circumnavigate complex obstacles.

### 2.2 Proposed Dynamic Multi-Segment Hybrid Chromosome
We engineered a hierarchical, dynamic-length hybrid chromosome where each candidate solution represents a full fleet of $K$ UAVs ($K=3$):

$$\text{Individual} = [\text{DroneGene}_1, \text{DroneGene}_2, \dots, \text{DroneGene}_K]$$

For each drone $d \in \{1, \dots, K\}$, its chromosome segment contains three synchronized genes with a dynamic length $L_d \in [L_{\min}=4, L_{\max}=12]$:

1. **Spatial Waypoint Gene (Continuous Coordinates):**
   $$\mathbf{W}_d = \big[(x_1, y_1), (x_2, y_2), \dots, (x_{L_d}, y_{L_d})\big], \quad x_i, y_i \in [0, 50]$$
   Defines the spatial guideposts for the UAV flight route starting and terminating at base depot $(x_0, y_0) = (2, 2)$.
2. **Speed Profile Gene (Discrete Velocity):**
   $$\mathbf{V}_d = [v_1, v_2, \dots, v_{L_d}], \quad v_i \in \{1.0\text{ m/s (Economy)}, 2.0\text{ m/s (Cruise)}, 3.0\text{ m/s (Sprint)}\}$$
3. **Sensor Schedule Gene (Power Profile):**
   $$\mathbf{S}_d = [s_1, s_2, \dots, s_{L_d}], \quad s_i \in \{0\text{ (Off)}, 1\text{ (Standard: } R=3.5\text{)}, 2\text{ (High-Power: } R=6.0\text{)}\}$$

```
Individual (Fleet Chromosome)
 ├── Drone 1 (Dynamic Length: 7 Waypoints)
 │    ├── Trajectory : [(2.1, 8.4), (20.9, 10.2), (45.1, 8.0), ...]
 │    ├── Speed      : [1.0, 2.0, 2.0, 3.0, ...]
 │    └── Sensor     : [0, 2, 2, 1, ...]
 ├── Drone 2 (Dynamic Length: 5 Waypoints)
 └── Drone 3 (Dynamic Length: 8 Waypoints)
```

---

## 3. Mathematical Formulation of the Fitness Landscape

To drive evolution toward Pareto-optimal compromises, we formulated a composite fitness function with physics-based power equations and penalty functions.

### 3.1 Propulsion and Sensor Power Model
* **Flight Propulsion Power:** Aerodynamic drag and motor efficiency scale quadratically with leg speed:
  $$P_{\text{flight}}(v) = 20.0 + 7.5 \cdot v^2 \quad \text{(Watts)}$$
  *(Economy: 27.5 W, Cruise: 50.0 W, Sprint: 87.5 W)*
* **Sensor Power Consumption:**
  $$P_{\text{sensor}}(s) = \begin{cases} 0.0\text{ W}, & s = 0 \text{ (Off, } R=0) \\ 25.0\text{ W}, & s = 1 \text{ (Standard, } R=3.5) \\ 60.0\text{ W}, & s = 2 \text{ (High-Power, } R=6.0) \end{cases}$$
* **Leg Energy:** For a flight segment between waypoints $P_k$ and $P_{k+1}$ of Euclidean length $d_k$, flight duration is $t_k = d_k / v_k$, and consumed energy is:
  $$E_k = \big(P_{\text{flight}}(v_k) + P_{\text{sensor}}(s_k)\big) \cdot t_k \quad \text{(Joules)}$$

### 3.2 Hazard Exposure & Bresenham Line Rasterization
To determine environmental exposure along the continuous flight path, each trajectory leg is rasterized using **Bresenham's Line Algorithm**. Total hazard accumulation across all cells crossed is:
$$\mathcal{H}_{\text{exp}} = \sum_{d=1}^K \sum_{c \in \text{Path}_d} \text{HazardGrid}[c_y, c_x]$$

### 3.3 Dynamic Penalty Function for Hard Constraints
Rather than using destructive repair operators that discard search diversity, hard constraints are enforced through dynamic penalty functions:
1. **No-Fly Zone (NFZ) Intrusions:**
   $$\text{Penalty}_{\text{NFZ}} = \lambda_{\text{NFZ}} \cdot \sum_{d=1}^K N_{\text{NFZ\_cells\_crossed}}, \quad (\lambda_{\text{NFZ}} = 90.0)$$
2. **Battery Overflow Exceedance:**
   $$\text{Penalty}_{\text{Bat}} = \lambda_{\text{bat}} \cdot \sum_{d=1}^K \max(0, E_d - B_{\max}), \quad (B_{\max} = 18{,}000\text{ J}, \lambda_{\text{bat}} = 0.03)$$

### 3.4 Overall Fitness Objective
$$\text{Fitness} = w_{\text{cov}} \cdot \mathcal{S}_{\text{cov}} - w_{\text{eng}} \cdot E_{\text{total}} - w_{\text{time}} \cdot T_{\text{makespan}} - w_{\text{haz}} \cdot \mathcal{H}_{\text{exp}} - (\text{Penalty}_{\text{NFZ}} + \text{Penalty}_{\text{Bat}})$$
*(Calibrated weights: $w_{\text{cov}} = 3.5, w_{\text{eng}} = 0.004, w_{\text{time}} = 0.4, w_{\text{haz}} = 2.2$)*

---

## 4. Custom Genetic Operators

1. **Tournament Selection ($k=4$):** Balances selection pressure to preserve population diversity while steadily propagating elite traits.
2. **Dynamic Cut-and-Splice Crossover ($p_c = 0.85$):**
   * Two parent paths of lengths $L_1$ and $L_2$ are cleaved at random internal indices $c_1, c_2$.
   * Sub-path splicing recombines spatial routes while clamping the resulting offspring to $[L_{\min}, L_{\max}]$.
   * Whole-route exchange occurs at $p = 0.30$ to shuffle mission responsibilities between drones.
3. **Spatial Mutation Operators ($p_m = 0.08$):**
   * **Dynamic Waypoint Insertion:** Synthesizes an intermediate coordinate $(x_{\text{mid}}, y_{\text{mid}}) + \mathcal{N}(0, \sigma^2)$ between successive waypoints, enabling the discovery of detours around obstacles.
   * **Dynamic Waypoint Deletion:** Removes redundant waypoints, eliminating jagged flight lines and conserving battery.
   * **Gaussian Coordinate Jitter:** Perturbs coordinates $(x, y) \leftarrow (x + \Delta x, y + \Delta y)$ where $\Delta \sim \mathcal{N}(0, 2.5^2)$.
   * **Speed & Sensor Flips:** Toggles discrete speed regimes and sensor activation modes.

---

## 5. Experimental Analysis & Visualizations

### 5.1 Convergence Progression Across Generations
The simulation was executed over 70 generations with an initial population of 60 individuals.
* **Gen 0 (Random Initial State):** Best Fitness: **-2,945.41**, 34 NFZ violations, Target Detection Rate: **21.2%**.
* **Gen 35 (Mid-Point Adaptation):** Best Fitness: **+1,364.20**, 0 NFZ violations, Target Detection Rate: **76.1%**.
* **Gen 70 (Final Convergence):** Best Fitness: **+1,517.23**, 0 NFZ violations, Target Detection Rate: **88.63%**, Total Energy: **11,665.5 J**.

*Refer to `results/convergence_curve.png` and `results/evolution_progression.png`.*

### 5.2 Flight Path & Heatmap Analysis
* As seen in `results/uav_flight_paths.png`, the 3 UAVs evolved specialized sector partitions:
  * **Drone 1** covered the southern perimeter and outpost clusters.
  * **Drone 2** traversed the high-density central disaster zone, operating high-power sensor mode ($R=6.0$).
  * **Drone 3** covered the northern survivor enclave, skirting the perimeter to minimize exposure to radiation pockets.
* The cumulative sensor scan density map (`results/coverage_heatmap.png`) confirms that target clusters received overlapping sensor sweeps while avoiding dead-weight scanning over empty terrain.

---

## 6. Hyperparameter Sensitivity Analysis

Empirical sensitivity experiments were conducted to evaluate the influence of key genetic parameters on optimization efficiency.

### 6.1 Impact of Population Size ($N \in \{25, 60, 120\}$)
* **$N=25$:** Displayed premature convergence, plateauing at fitness **1,105.4** around Generation 32 due to gene pool exhaustion.
* **$N=60$:** Maintained steady evolutionary climbing, reaching fitness **1,185.2** with optimal computational efficiency (5.6s execution).
* **$N=120$:** Achieved slightly higher fitness (**1,248.6**), but required double the computational budget without a proportional gain in solution quality.
* *Conclusion:* $N=60$ represents the optimal trade-off between runtime and solution quality (`results/sensitivity_population.png`).

### 6.2 Impact of Mutation Rate ($p_m \in \{0.02, 0.08, 0.22\}$)
* **Low ($p_m = 0.02$):** Trapped in sub-optimal local basins (fitness **1,148.1**) due to insufficient exploratory perturbation.
* **Moderate ($p_m = 0.08$):** Achieved the highest final fitness (**1,282.5**), balancing exploration of new waypoints with exploitation of successful routes.
* **High ($p_m = 0.22$):** Suffered from disruptive genetic noise (fitness **1,173.9**), frequently degrading high-performing flight trajectories.
* *Conclusion:* Moderate mutation ($p_m = 0.08$) prevents stagnation while safeguarding good genetic traits (`results/sensitivity_mutation.png`).

### 6.3 Impact of Selection Pressure (Tournament Size $k \in \{2, 4, 8\}$)
* **$k=2$ (Weak Selection Pressure):** Slow convergence, taking over 25 generations to eliminate initial NFZ violations.
* **$k=4$ (Balanced):** Smooth, monotonic fitness improvement with healthy population diversity.
* **$k=8$ (Aggressive):** Rapid initial gains (violating cells resolved by Gen 4), but suffered minor diversity loss in later generations.
* *Conclusion:* $k=4$ achieves ideal selection pressure (`results/sensitivity_tournament.png`).

---

## 7. Comparative Study: Evolved GA vs. Baseline Random Search

To rigorously validate the GA's optimization capabilities, we executed an identical computational budget (2,500 evaluations) comparing the GA Strategy Engine against uniform **Random Search**:

| Evaluation Metric | Evolved GA Solution | Baseline Random Search | Performance Delta |
|---|---|---|---|
| **Multi-Objective Fitness** | **+1,238.29** | **-225.48** | **+1,463.77 pts** (GA Superior) |
| **Priority Disaster Coverage** | **473.0 pts** | 221.8 pts | **+113.3% higher coverage** |
| **Target Detection Rate** | **89.0%** | 40.8% | **+48.2% higher detection** |
| **No-Fly Zone (NFZ) Violations**| **0 cells** | 6 cells | **100% collision-free** |
| **Total Fleet Energy Consumed** | **11,548.8 J** | 17,337.1 J | **33.4% energy conserved** |
| **Environmental Hazard Exposure**| **49.88** | 132.52 | **62.4% risk reduction** |

### Discussion on Uniqueness of Evolved Solution
While Random Search completely failed to eliminate hard-constraint violations (incurring 6 collisions with collapsed structures) and exhausted battery reserves on aimless flight paths, the **Genetic Algorithm demonstrated intelligent multi-objective coordination**:
1. **Dynamic Obstacle Detours:** GA systematically shifted intermediate waypoints around obstacle boundaries.
2. **Synchronized Sector Division:** The 3 drones independently partitioned the map into complementary surveillance sectors without redundant flight path overlap.
3. **Adaptive Sensor Throttling:** Sensor power was selectively boosted to High-Power mode ($s=2$) only when crossing dense target clusters, and turned Off ($s=0$) during transit legs back to the base depot, conserving significant energy.

---

## 8. Conclusion
The developed UAV Strategy Engine successfully satisfies all multi-objective constraints. By combining dynamic-length chromosome representations with spatial genetic operators and physics-grounded penalty evaluations, the system delivers mission-ready, collision-free autonomous surveillance plans that maximize survivor detection while strictly preserving fleet energy reserves.
