# Factorio Decision Support System (DSS)
## Multi-Objective Optimization of an Automated Production & Logistics System

**Student:** Carlo Barnardo  
**Student Number:** 43870449  
**Proposal Document:** `Barnardo_DSS_Problem_Proposal(1).docx`

---

### Project Architecture & Methodologies Implemented

This project implements a complete Decision Support System (DSS) integrating 6 management science and operations research techniques to solve a multi-objective factory optimization problem using Factorio as an industrial analogue:

1. **Blueprint Parsing & Instrumentation (`src/factorioproject/blueprint.py`)**:
   - Decodes Factorio blueprint strings (Base64 + zlib JSON decompression).
   - Extracts machine counts (24 stone furnaces, 48 inserters, belts, poles).
   - Computes nominal throughput (7.5 plates/s = 50% yellow belt saturation), active electric load (639 kW), and chemical fuel requirements (2160 kW).

2. **Linear & Integer Programming (`src/factorioproject/optimization.py`)**:
   - Continuous LP & Mixed-Integer Programming (PuLP) formulating production-mix and machine-allocation decisions.
   - Material balance constraints across Intermediate and Finished goods (Iron/Copper plates, Steel, Electronic Circuits, Automation Science).
   - Post-optimal sensitivity analysis (dual variables / shadow prices and reduced costs).
   - Parametric power sweep constructing the Pareto frontier between power limits and throughput value.

3. **Network Flow & Bottleneck Analysis (`src/factorioproject/network_flow.py`)**:
   - Directed network graph modeling the conveyor belt and rail logistics topology in NetworkX.
   - Computes maximum network flow using Edmonds-Karp / Dinic's algorithm.
   - Identifies binding physical logistics bottlenecks via the Max-Flow Min-Cut theorem.
   - Generates network flow visualizations highlighting min-cut bottlenecks in red.

4. **Transportation & Logistics Assignment (`src/factorioproject/transportation.py`)**:
   - Transportation LP in PuLP assigning distributed extraction patches (Iron, Copper, Coal) to factory delivery destinations at minimal freight cost.
   - Computes location shadow prices (marginal costs of supply and demand).

5. **Multi-Criteria Decision Making (`src/factorioproject/mcdm.py`)**:
   - **Goal Programming (GP)**: Reconciles competing goals (throughput target vs. power budget ceiling) by minimizing normalized penalty deviations.
   - **Analytic Hierarchy Process (AHP)**: Saaty's eigenvector method calculating criteria priority weights with consistency verification ($CR < 0.10$), evaluating and ranking 4 operational strategy alternatives.

6. **Queuing Theory & Buffer Inventory Policy (`src/factorioproject/queuing.py`)**:
   - Models unloader stations as $M/M/1$ and $M/M/s$ queues (Erlang-C delay probability, traffic intensity $\rho$, expected queue length $L_q$, and wait time $W_q$).
   - Derives analytical Safety Stock ($SS$) and Reorder Point ($ROP$) for 95% and 99% service levels, recommending physical chest types (Wooden vs. Iron vs. Steel).

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
