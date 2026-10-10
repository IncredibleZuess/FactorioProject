"""Queuing Theory and Buffer-Stock Inventory Policy Module.

Models factory logistics stations as M/M/1 and M/M/s queuing systems.
Evaluates congestion delays, server utilization, safety stocks, and reorder points.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def solve_mm1_queue(lambda_rate: float, mu_rate: float) -> Dict[str, float]:
    """Calculate steady-state metrics for an M/M/1 queuing system.

    Args:
        lambda_rate: Arrival rate (deliveries/sec)
        mu_rate: Service rate of the single unloader (deliveries/sec)
    """
    if lambda_rate >= mu_rate:
        raise ValueError(
            f"Unstable queue: arrival rate ({lambda_rate}) must be strictly less than service rate ({mu_rate})."
        )

    rho = lambda_rate / mu_rate
    p0 = 1.0 - rho
    l_sys = rho / (1.0 - rho)
    l_q = (rho**2) / (1.0 - rho)
    w_sys = 1.0 / (mu_rate - lambda_rate)
    w_q = rho / (mu_rate - lambda_rate)

    return {
        "lambda": lambda_rate,
        "mu": mu_rate,
        "servers": 1,
        "utilization_rho": round(rho, 4),
        "prob_idle_p0": round(p0, 4),
        "avg_in_system_L": round(l_sys, 4),
        "avg_in_queue_Lq": round(l_q, 4),
        "avg_time_system_W": round(w_sys, 4),
        "avg_wait_queue_Wq": round(w_q, 4),
    }


def solve_mms_queue(lambda_rate: float, mu_rate: float, s: int) -> Dict[str, float]:
    """Calculate steady-state metrics for an M/M/s multi-server queuing system (e.g.

    s parallel unloader inserters or s train unloading bays).
    """
    if s < 1:
        raise ValueError("Server count s must be >= 1.")

    rho = lambda_rate / (s * mu_rate)
    if rho >= 1.0:
        raise ValueError(
            f"Unstable queue: arrival rate ({lambda_rate}) exceeds total service capacity ({s * mu_rate})."
        )

    # Compute P0 (probability of 0 items in system)
    sum_terms = sum((s * rho) ** n / math.factorial(n) for n in range(s))
    last_term = ((s * rho) ** s) / (math.factorial(s) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + last_term)

    # Erlang-C formula for probability that an arrival must wait in queue
    p_wait = last_term * p0

    # Average queue length Lq and system length L
    l_q = (p_wait * rho) / (1.0 - rho)
    l_sys = l_q + (lambda_rate / mu_rate)

    # Average wait time in queue Wq and in system W (Little's Law)
    w_q = l_q / lambda_rate
    w_sys = w_q + (1.0 / mu_rate)

    return {
        "lambda": lambda_rate,
        "mu": mu_rate,
        "servers": s,
        "utilization_rho": round(rho, 4),
        "prob_idle_p0": round(p0, 4),
        "prob_delay_Pw": round(p_wait, 4),
        "avg_in_system_L": round(l_sys, 4),
        "avg_in_queue_Lq": round(l_q, 4),
        "avg_time_system_W": round(w_sys, 4),
        "avg_wait_queue_Wq": round(w_q, 4),
    }


def calculate_buffer_inventory_policy(
    consumption_rate: float,
    queue_metrics: Dict[str, float],
    service_level: float = 0.95,
    unit_lead_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute buffer stock, safety stock, and reorder point (ROP) policy for the smelting buffer.

    Args:
        consumption_rate: Factory furnace demand (items/sec, e.g. 7.5 for 24 furnaces)
        queue_metrics: Output from solve_mms_queue or solve_mm1_queue
        service_level: Target non-stockout probability (e.g. 0.95 or 0.99)
        unit_lead_time_sec: Average transit lead time before unloading (seconds)
    """
    w_q = queue_metrics["avg_wait_queue_Wq"]
    w_sys = queue_metrics["avg_time_system_W"]
    transit_time = unit_lead_time_sec if unit_lead_time_sec is not None else 10.0

    total_lead_time = transit_time + w_sys

    # Standard normal quantile Z for service level
    z = float(stats.norm.ppf(service_level))

    # Variance in lead time induced by queue delay variability
    # In M/M/s, conditional wait time is exponentially distributed with rate s*mu*(1 - rho)
    s = queue_metrics["servers"]
    mu = queue_metrics["mu"]
    rho = queue_metrics["utilization_rho"]

    std_lead_time = 1.0 / (s * mu * (1.0 - rho)) if (1.0 - rho) > 0 else 1.0

    # Safety Stock SS = Z * sigma_L * consumption_rate
    safety_stock = z * std_lead_time * consumption_rate

    # Reorder Point ROP = (mean lead time * consumption rate) + SS
    expected_lead_demand = total_lead_time * consumption_rate
    reorder_point = expected_lead_demand + safety_stock

    # Recommended Factorio chest sizes (50 ore per stack)
    wooden_chest_cap = 16 * 50  # 800 items
    iron_chest_cap = 32 * 50  # 1600 items
    steel_chest_cap = 48 * 50  # 2400 items

    rec_chest = "wooden-chest"
    if reorder_point > iron_chest_cap:
        rec_chest = "steel-chest"
    elif reorder_point > wooden_chest_cap:
        rec_chest = "iron-chest"

    return {
        "consumption_rate_items_per_sec": round(consumption_rate, 2),
        "service_level": service_level,
        "z_score": round(z, 3),
        "mean_lead_time_sec": round(total_lead_time, 2),
        "std_lead_time_sec": round(std_lead_time, 2),
        "safety_stock_items": math.ceil(safety_stock),
        "reorder_point_items": math.ceil(reorder_point),
        "recommended_chest_type": rec_chest,
        "chest_capacities": {
            "wooden-chest": wooden_chest_cap,
            "iron-chest": iron_chest_cap,
            "steel-chest": steel_chest_cap,
        },
    }
