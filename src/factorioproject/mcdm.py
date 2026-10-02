"""Multicriteria Decision Making (MCDM) for Factorio Operations.

Implements:
1. Goal Programming (PuLP) to reconcile competing goals: Throughput vs. Power vs. Logistics Cost.
2. Analytic Hierarchy Process (AHP) to determine objective weights, check consistency, and rank operational alternatives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pulp
from factorioproject.optimization import ProductionProblemConfig


def solve_goal_programming(
    target_throughput: float = 12.0,
    target_power_kw: float = 1100.0,
    weights: Optional[Dict[str, float]] = None,
    config: Optional[ProductionProblemConfig] = None,
) -> Dict[str, Any]:
    """Reconcile conflicting goals using Weighted Goal Programming.

    Goal 1: Factory Throughput Value >= target_throughput (penalize d_throughput_minus)
    Goal 2: Electric Power Draw <= target_power_kw (penalize d_power_plus)
    """
    if config is None:
        config = ProductionProblemConfig()

    if weights is None:
        weights = {
            "throughput_weight": 1.0,
            "power_weight": 0.5,
        }

    prob = pulp.LpProblem("Factorio_Goal_Programming", pulp.LpMinimize)

    # Machine allocation variables
    m_iron = pulp.LpVariable("m_iron_furnace", lowBound=0, upBound=config.num_stone_furnaces)
    m_copper = pulp.LpVariable("m_copper_furnace", lowBound=0, upBound=config.num_stone_furnaces)
    m_steel = pulp.LpVariable("m_steel_furnace", lowBound=0, upBound=config.num_stone_furnaces)

    m_cable = pulp.LpVariable("m_cable_assembler", lowBound=0, upBound=config.num_assemblers)
    m_circuit = pulp.LpVariable("m_circuit_assembler", lowBound=0, upBound=config.num_assemblers)
    m_gear = pulp.LpVariable("m_gear_assembler", lowBound=0, upBound=config.num_assemblers)
    m_science = pulp.LpVariable("m_science_assembler", lowBound=0, upBound=config.num_assemblers)

    # Net output production rates
    r_iron = pulp.LpVariable("r_iron_plate", lowBound=0)
    r_copper = pulp.LpVariable("r_copper_plate", lowBound=0)
    r_steel = pulp.LpVariable("r_steel_plate", lowBound=0)
    r_circuit = pulp.LpVariable("r_electronic_circuit", lowBound=0)
    r_science = pulp.LpVariable("r_automation_science", lowBound=0)

    # Goal deviation variables (all >= 0)
    d_thru_minus = pulp.LpVariable("d_throughput_minus", lowBound=0)
    d_thru_plus = pulp.LpVariable("d_throughput_plus", lowBound=0)

    d_pow_minus = pulp.LpVariable("d_power_minus", lowBound=0)
    d_pow_plus = pulp.LpVariable("d_power_plus", lowBound=0)

    # Objective: Minimize weighted normalized deviations from aspiration levels
    # Normalize by target levels to prevent scale dominance
    w_thru = weights.get("throughput_weight", 1.0) / target_throughput
    w_pow = weights.get("power_weight", 0.5) / target_power_kw

    prob += (w_thru * d_thru_minus + w_pow * d_pow_plus, "Weighted_Deviation_Objective")

    # Hard physical / material constraints
    prob += (m_iron + m_copper + m_steel <= config.num_stone_furnaces, "Furnace_Limit")
    prob += (m_cable + m_circuit + m_gear + m_science <= config.num_assemblers, "Assembler_Limit")

    prob += (
        r_iron + 5.0 * r_steel + 1.0 * r_circuit + 2.0 * r_science <= m_iron * (1.0 / 3.2),
        "Iron_Balance",
    )
    prob += (r_copper + 1.5 * r_circuit + 1.0 * r_science <= m_copper * (1.0 / 3.2), "Copper_Balance")
    prob += (r_steel <= m_steel * (1.0 / 16.0), "Steel_Capacity")
    prob += (r_circuit <= m_circuit * 1.0, "Circuit_Capacity")
    prob += (3.0 * r_circuit <= m_cable * 2.0, "Cable_Capacity")
    prob += (r_science <= m_gear * 1.0, "Gear_Capacity")
    prob += (r_science <= m_science * 0.1, "Science_Capacity")

    prob += (m_iron * (1.0 / 3.2) <= config.iron_ore_capacity, "Iron_Supply")
    prob += (m_copper * (1.0 / 3.2) <= config.copper_ore_capacity, "Copper_Supply")

    # Goal Constraint 1: Throughput value
    vals = config.product_values
    throughput_expr = (
        vals["iron_plate"] * r_iron
        + vals["copper_plate"] * r_copper
        + vals["steel_plate"] * r_steel
        + vals["electronic_circuit"] * r_circuit
        + vals["automation_science"] * r_science
    )
    prob += (throughput_expr + d_thru_minus - d_thru_plus == target_throughput, "Goal_Throughput")

    # Goal Constraint 2: Power consumption
    total_assemblers = m_cable + m_circuit + m_gear + m_science
    power_expr = config.base_logistics_power_kw + total_assemblers * config.assembler_electric_kw
    prob += (power_expr + d_pow_minus - d_pow_plus == target_power_kw, "Goal_Power")

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    achieved_thru = pulp.value(throughput_expr) or 0.0
    achieved_power = pulp.value(power_expr) or 0.0

    return {
        "status": pulp.LpStatus[status],
        "target_throughput": target_throughput,
        "achieved_throughput": round(achieved_thru, 4),
        "throughput_underachievement": round(pulp.value(d_thru_minus) or 0.0, 4),
        "target_power_kw": target_power_kw,
        "achieved_power_kw": round(achieved_power, 2),
        "power_overachievement": round(pulp.value(d_pow_plus) or 0.0, 2),
        "penalty_score": round(pulp.value(prob.objective) or 0.0, 5),
        "machines": {
            "m_iron_furnace": round(m_iron.varValue or 0.0, 2),
            "m_copper_furnace": round(m_copper.varValue or 0.0, 2),
            "m_assemblers_total": round(pulp.value(total_assemblers) or 0.0, 2),
        },
        "rates": {
            "r_iron": round(r_iron.varValue or 0.0, 3),
            "r_copper": round(r_copper.varValue or 0.0, 3),
            "r_circuit": round(r_circuit.varValue or 0.0, 3),
            "r_science": round(r_science.varValue or 0.0, 3),
        },
    }


class AHPModel:
    """Analytic Hierarchy Process (AHP) for evaluating operational factory alternatives."""

    # Saaty's standard Random Consistency Index for n=1..10
    RI_DICT = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def __init__(self, criteria: List[str]):
        self.criteria = criteria
        self.n = len(criteria)

    def calculate_priority_vector(self, matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Compute the priority weights using the principal eigenvector method,

        along with the principal eigenvalue and Consistency Ratio (CR).
        """
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        max_idx = int(np.argmax(np.real(eigenvalues)))
        lambda_max = float(np.real(eigenvalues[max_idx]))

        # Principal eigenvector normalized
        weights = np.real(eigenvectors[:, max_idx])
        weights = weights / np.sum(weights)

        # Consistency Index and Ratio
        ci = (lambda_max - self.n) / (self.n - 1) if self.n > 1 else 0.0
        ri = self.RI_DICT.get(self.n, 1.49)
        cr = ci / ri if ri > 0 else 0.0

        return weights, lambda_max, cr

    def evaluate_alternatives(
        self,
        criteria_matrix: np.ndarray,
        alternative_names: List[str],
        alt_matrices_per_criterion: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Perform full AHP synthesis across criteria and alternatives."""
        crit_weights, lambda_max, cr = self.calculate_priority_vector(criteria_matrix)

        alt_priority_matrix = np.zeros((len(alternative_names), len(self.criteria)))
        alt_crs = {}

        for c_idx, crit in enumerate(self.criteria):
            mat = alt_matrices_per_criterion[crit]
            a_weights, _, a_cr = self.calculate_priority_vector(mat)
            alt_priority_matrix[:, c_idx] = a_weights
            alt_crs[crit] = round(a_cr, 4)

        # Composite global scores = Alt_Matrix * Criteria_Weights
        global_scores = np.dot(alt_priority_matrix, crit_weights)

        ranking_df = pd.DataFrame(
            {
                "Alternative": alternative_names,
                "Global_Score": np.round(global_scores, 4),
                "Rank": pd.Series(global_scores).rank(ascending=False).astype(int),
            }
        ).sort_values("Rank")

        weights_df = pd.DataFrame(
            {
                "Criterion": self.criteria,
                "Weight": np.round(crit_weights, 4),
            }
        )

        return {
            "criteria_weights": weights_df,
            "criteria_cr": round(cr, 4),
            "is_consistent": cr < 0.10,
            "alternative_scores_per_criterion": pd.DataFrame(
                alt_priority_matrix, index=alternative_names, columns=self.criteria
            ),
            "final_ranking": ranking_df,
        }
