# Factorio Decision Support System (DSS)
## Multi-Objective Optimization of an Automated Production and Logistics System

* **Student:** Carlo Barnardo
* **Student Number:** 43870449
* **Proposal Document:** `Barnardo_DSS_Problem_Proposal(1).docx`
* **Primary Dataset:** `blueprints/starter.txt` (*ElderAxe's Quick Start Base v11.2.5*)
* **Standards Compliance:** ASD-STE100 (Simplified Technical English, Issue 8 and Issue 9)

---

### Project Architecture and Methods

This project implements a Decision Support System (DSS) for factory operations. The system uses Factorio as an industrial simulation model. It integrates six management science and operations research methods:

1. **Blueprint Parsing and Entity Instrumentation (`src/factorioproject/blueprint.py`)**:
   - Decodes single blueprints and blueprint books using Base64 and zlib decompression.
   - Tracks factory expansion across all 24 development stages of the quick start base.
   - Instruments **Full Base Default** (7,162 entities): 374 furnaces, 260 production machines, 54.90 MW power plant, and 115 storage chests.
   - Creates dynamic engineering specification tables with `generate_metrics_table()`.

2. **Linear and Integer Programming (`src/factorioproject/optimization.py`)**:
   - Formulates product-mix and machine allocation decisions in PuLP.
   - Reads equipment limits directly from blueprint layout metrics.
   - Calculates shadow prices and reduced costs for dual sensitivity analysis.
   - Sweeps power limits to build the Pareto tradeoff curve.

3. **Network Flow and Bottleneck Analysis (`src/factorioproject/network_flow.py`)**:
   - Models conveyor logistics and petrochemical pipelines as directed flow networks.
   - Calculates maximum network flow and binding bottlenecks with Edmonds-Karp and Min-Cut algorithms.
   - Generates three non-overlapping network flow diagrams:
     - **Diagram 1**: Solid logistics and 8-lane smelting bus ($121.25\text{ plates/sec}$).
     - **Diagrams 2 and 3**: Baseline rail delivery ($60.0\text{ fluid/sec}$) compared with upgraded dual-train delivery ($140.0\text{ fluid/sec}$, +70% throughput).

4. **Transportation and Logistics Assignment (`src/factorioproject/transportation.py`)**:
   - Assigns mining outposts to factory destinations and boiler plants at minimum freight cost.
   - Calculates location shadow prices for supply and demand nodes.

5. **Multi-Criteria Decision Making (`src/factorioproject/mcdm.py`)**:
   - **Goal Programming (GP)**: Balances throughput targets against energy limits by minimizing weighted penalties.
   - **Analytic Hierarchy Process (AHP)**: Calculates priority weights with consistency checks ($CR < 0.10$) to rank four factory policies.

6. **Queuing Theory and Buffer Inventory Policy (`src/factorioproject/queuing.py`)**:
   - Models train unloader stations as $M/M/1$ and $M/M/16$ multi-server queues.
   - Calculates Erlang-C delay probabilities, traffic intensity, and waiting times.
   - Calculates Safety Stock ($SS$) and Reorder Point ($ROP$) for a 95% service level.

7. **Discrete-Event Simulation (`src/factorioproject/simulation.py`)**:
   - Simulates stochastic factory operations in SimPy.
   - Models Poisson arrivals, processing variance, and machine micro-stoppages.
   - Validates linear programming predictions against stochastic operating results.

---

### Interactive Notebook

The complete model implementation, trade-off charts, network flow diagrams, and validation tables are in:
* **[`main.ipynb`](file:///home/zuess/Documents/code/python/dss/FactorioProject/main.ipynb)**

---

### Procedure to Run the Project

Follow these steps in the NixOS environment:

1. Open the interactive Jupyter notebook:
   ```bash
   jupyter lab main.ipynb
   ```

2. Or start the command line interface:
   ```bash
   factorioproject
   ```

3. Run the automated test suite:
   ```bash
   pytest tests
   ```
