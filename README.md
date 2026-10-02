# Factorio Decision Support System (DSS)
## Multi-Objective Optimization of an Automated Production & Logistics System

**Student:** Carlo Barnardo  
**Student Number:** 43870449  
**Proposal Document:** `Barnardo_DSS_Problem_Proposal(1).docx`  
**Primary Dataset:** `blueprints/starter.txt` (*ElderAxe's Quick Start Base v11.2.5*)

---

### Project Architecture & Methodologies Implemented

This project implements a complete Decision Support System (DSS) integrating 6 management science and operations research techniques to solve a multi-objective factory optimization problem using Factorio as an industrial analogue:

1. **Blueprint Book Parsing & Industrial Instrumentation (`src/factorioproject/blueprint.py`)**:
   - Decodes Factorio single blueprints and multi-blueprint books (Base64 + zlib JSON decompression).
   - Analyzes stage progression across all 24 development stages of *ElderAxe's Quick Start Base*.
   - Instruments **Full Base Default** (7,162 entities):
     - 374 Smelting Furnaces (360 Stone + 14 Steel) yielding $121.25\text{ plates/sec}$ ($8.1$ full yellow belts).
     - 260 Manufacturing Units (89 AM1, 142 AM2, 23 Chem Plants, 6 Refineries, 56 Labs).
     - 54.90 MW Steam Power Generation Plant (31 Boilers + 61 Steam Engines) facing a $65.35\text{ MW}$ peak demand.
     - 51 Heavy Steel Storage Chests.

2. **Linear & Integer Programming (`src/factorioproject/optimization.py`)**:
   - Continuous LP & Mixed-Integer Programming (PuLP) formulating production-mix and machine-allocation decisions.
   - Initialized dynamically from blueprint metrics via `ProductionProblemConfig.from_blueprint_metrics(metrics)`.
   - Post-optimal sensitivity analysis (dual variables / shadow prices and reduced costs).
   - Parametric power sweep constructing the Pareto frontier between power limits and throughput value.

3. **Network Flow & Bottleneck Analysis (`src/factorioproject/network_flow.py`)**:
   - Directed network graphs modeling both solid conveyor logistics and petrochemical fluid networks in NetworkX.
   - Computes maximum network flow using Edmonds-Karp / Dinic's algorithm and extracts binding bottlenecks via the Max-Flow Min-Cut theorem.
   - **Diagram 1 (Solid Conveyor Logistics & 8-Lane Smelting Bus)**: Visualizes ore delivery, smelting lines, and main bus distribution to assembly lines ($121.25\text{ plates/sec}$ capacity).
   - **Diagram 2 (Offshore Crude Oil Rail Supply & Petrochemical Refining - Baseline)**: Models offshore oil rigs delivering crude oil via a single fluid-train shuttle ($60.0\text{ fluid/sec}$) to the 6 oil refineries and chemical plants. Proves Edmonds-Karp min-cut saturation directly on the rail delivery leg (50% refinery starvation).
   - **Diagram 3 (Infrastructure Debottlenecking - Dual Fluid-Train Shuttle)**: Demonstrates system behavior after upgrading to dual fluid-train shuttles ($140.0\text{ fluid/sec}$). Unlocks 100% refinery capacity, drives petrochemical throughput up by +70% ($102.0\text{ fluid/sec}$), and visually illustrates the shift of binding min-cut bottlenecks downstream to high-tech product synthesis sinks.
   - Zero visual element overlaps achieved via custom deterministic coordinate grids, directional arcs, and offset text badges.

4. **Transportation & Logistics Assignment (`src/factorioproject/transportation.py`)**:
   - Transportation LP in PuLP assigning distributed mining complexes (Iron, Copper, Coal) to factory delivery destinations and power plant boilers at minimal freight cost.
   - Computes location shadow prices (marginal costs of supply and demand).

5. **Multi-Criteria Decision Making (`src/factorioproject/mcdm.py`)**:
   - **Goal Programming (GP)**: Reconciles competing goals (throughput target vs. power budget ceiling) by minimizing normalized penalty deviations.
   - **Analytic Hierarchy Process (AHP)**: Saaty's eigenvector method calculating criteria priority weights with consistency verification ($CR < 0.10$), evaluating and ranking 4 operational strategy alternatives.

6. **Queuing Theory & Buffer Inventory Policy (`src/factorioproject/queuing.py`)**:
   - Models unloader stations as $M/M/1$ and multi-server $M/M/16$ queues (Erlang-C delay probability, traffic intensity $\rho$, expected queue length $L_q$, and wait time $W_q$).
   - Derives analytical Safety Stock ($SS$) and Reorder Point ($ROP$) for 95% service level, mapped to the blueprint's 51 Steel Chests.

7. **Discrete-Event Simulation (`src/factorioproject/simulation.py`)**:
   - SimPy simulation model replicating factory operations under stochastic Poisson arrivals, processing variance, and machine micro-stoppages/breakdowns.
   - Tracks dynamic inventory trajectories and validates analytical LP predictions against real stochastic operational performance.

---

### Interactive Notebook

All models, data visualizations, interactive sensitivity trade-off plots, network graphs, and comparative validation tables are fully executed and documented in:
* **[`main.ipynb`](file:///home/zuess/Documents/code/python/dss/FactorioProject/main.ipynb)**

### Running the Project
Within the NixOS devenv:
```bash
# Run Jupyter / VSCode notebook
jupyter lab main.ipynb

# Or execute CLI
factorioproject
```
