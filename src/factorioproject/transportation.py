"""Transportation and Logistics Assignment Model.

Optimizes the routing of raw materials from remote extraction patches (iron, copper, coal)
to factory unloading and smelting stations at minimum transportation cost.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import pulp


def solve_transportation_model(
    supply_dict: Optional[Dict[str, float]] = None,
    demand_dict: Optional[Dict[str, float]] = None,
    cost_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Solve the transportation/assignment problem using PuLP.

    Args:
        supply_dict: Maximum extraction/supply rates at each remote patch (items/s)
        demand_dict: Material demand rates at each factory destination (items/s)
        cost_matrix: Unit transport/logistics cost ($ per item) from patch i to destination j
    """
    if supply_dict is None:
        supply_dict = {
            "Iron_Patch_Alpha": 20.0,  # Near mine
            "Iron_Patch_Beta": 15.0,  # Distant high-yield mine
            "Copper_Patch_1": 18.0,  # Medium distance
            "Copper_Patch_2": 12.0,  # Distant mine
            "Coal_Patch_Central": 15.0,  # Central coal deposit
        }

    if demand_dict is None:
        demand_dict = {
            "Smelter_Iron_West": 7.5,  # Consumes 7.5 ore/s (half yellow belt)
            "Smelter_Iron_East": 7.5,  # Consumes 7.5 ore/s
            "Smelter_Copper_South": 7.5,  # Consumes 7.5 ore/s
            "Smelter_Copper_North": 7.5,  # Consumes 7.5 ore/s
            "Power_Fuel_Depot": 5.0,  # Consumes 5.0 coal/s
        }

    if cost_matrix is None:
        # Distance-based freight/train logistics unit costs ($/unit transported)
        cost_matrix = {
            "Iron_Patch_Alpha": {
                "Smelter_Iron_West": 1.20,
                "Smelter_Iron_East": 2.10,
                "Smelter_Copper_South": 999.0,  # Incompatible material routing
                "Smelter_Copper_North": 999.0,
                "Power_Fuel_Depot": 999.0,
            },
            "Iron_Patch_Beta": {
                "Smelter_Iron_West": 2.80,
                "Smelter_Iron_East": 1.60,
                "Smelter_Copper_South": 999.0,
                "Smelter_Copper_North": 999.0,
                "Power_Fuel_Depot": 999.0,
            },
            "Copper_Patch_1": {
                "Smelter_Iron_West": 999.0,
                "Smelter_Iron_East": 999.0,
                "Smelter_Copper_South": 1.40,
                "Smelter_Copper_North": 2.30,
                "Power_Fuel_Depot": 999.0,
            },
            "Copper_Patch_2": {
                "Smelter_Iron_West": 999.0,
                "Smelter_Iron_East": 999.0,
                "Smelter_Copper_South": 2.60,
                "Smelter_Copper_North": 1.50,
                "Power_Fuel_Depot": 999.0,
            },
            "Coal_Patch_Central": {
                "Smelter_Iron_West": 999.0,
                "Smelter_Iron_East": 999.0,
                "Smelter_Copper_South": 999.0,
                "Smelter_Copper_North": 999.0,
                "Power_Fuel_Depot": 0.90,
            },
        }

    sources = list(supply_dict.keys())
    destinations = list(demand_dict.keys())

    # Build LP
    prob = pulp.LpProblem("Factorio_Logistics_Transportation", pulp.LpMinimize)

    # Decision variables x[i, j] >= 0
    x = {}
    for i in sources:
        for j in destinations:
            x[i, j] = pulp.LpVariable(f"ship_{i}_to_{j}", lowBound=0, cat=pulp.LpContinuous)

    # Objective: Min total transport cost
    prob += (
        pulp.lpSum(cost_matrix[i][j] * x[i, j] for i in sources for j in destinations),
        "Total_Transportation_Cost",
    )

    # Supply Constraints
    supply_constraints = {}
    for i in sources:
        c = (pulp.lpSum(x[i, j] for j in destinations) <= supply_dict[i], f"Supply_{i}")
        prob += c
        supply_constraints[i] = f"Supply_{i}"

    # Demand Constraints
    demand_constraints = {}
    for j in destinations:
        c = (pulp.lpSum(x[i, j] for i in sources) >= demand_dict[j], f"Demand_{j}")
        prob += c
        demand_constraints[j] = f"Demand_{j}"

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    status_str = pulp.LpStatus[status]
    total_cost = pulp.value(prob.objective)

    # Construct allocation matrix DataFrame
    alloc_data = {j: {} for j in destinations}
    for i in sources:
        for j in destinations:
            val = round(x[i, j].varValue or 0.0, 3)
            alloc_data[j][i] = val
    alloc_df = pd.DataFrame(alloc_data)

    # Extract dual values (shadow prices)
    dual_supply = {}
    for i in sources:
        c_name = supply_constraints[i]
        c = prob.constraints[c_name]
        dual_supply[i] = round(c.pi if c.pi is not None else 0.0, 4)

    dual_demand = {}
    for j in destinations:
        c_name = demand_constraints[j]
        c = prob.constraints[c_name]
        dual_demand[j] = round(c.pi if c.pi is not None else 0.0, 4)

    return {
        "status": status_str,
        "total_transport_cost": round(total_cost or 0.0, 4),
        "shipment_matrix": alloc_df,
        "dual_supply_values": dual_supply,
        "dual_demand_values": dual_demand,
    }
