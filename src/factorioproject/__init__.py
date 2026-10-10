"""Factorio Decision Support System (DSS) Library.

Multi-Objective Optimization of an Automated Production and Logistics System.
"""

from __future__ import annotations

from factorioproject.blueprint import (
    FACTORIO_SPECS,
    analyze_blueprint_layout,
    blueprint_to_dataframe,
    decode_blueprint,
    encode_blueprint,
    extract_blueprint,
    generate_metrics_table,
    get_book_blueprints_summary,
    is_blueprint_book,
)
from factorioproject.mcdm import (
    AHPModel,
    solve_goal_programming,
)
from factorioproject.network_flow import (
    analyze_bottlenecks,
    create_factory_network,
    create_petrochemical_network,
    plot_network_bottlenecks,
    plot_petrochemical_network,
    solve_max_network_flow,
)
from factorioproject.optimization import (
    ProductionProblemConfig,
    parametric_power_sweep,
    solve_production_allocation,
)
from factorioproject.queuing import (
    calculate_buffer_inventory_policy,
    solve_mm1_queue,
    solve_mms_queue,
)
from factorioproject.simulation import (
    FactorioSmeltingSimulation,
)
from factorioproject.transportation import (
    solve_transportation_model,
)

__all__ = [
    "decode_blueprint",
    "encode_blueprint",
    "blueprint_to_dataframe",
    "analyze_blueprint_layout",
    "generate_metrics_table",
    "FACTORIO_SPECS",
    "ProductionProblemConfig",
    "solve_production_allocation",
    "parametric_power_sweep",
    "create_factory_network",
    "create_petrochemical_network",
    "solve_max_network_flow",
    "analyze_bottlenecks",
    "plot_network_bottlenecks",
    "plot_petrochemical_network",
    "solve_transportation_model",
    "solve_goal_programming",
    "AHPModel",
    "solve_mm1_queue",
    "solve_mms_queue",
    "calculate_buffer_inventory_policy",
    "FactorioSmeltingSimulation",
    "main",
]


def main() -> None:
    """CLI entrypoint for factorioproject."""
    print("Factorio Decision Support System (DSS) CLI")
    print("Multi-Objective Optimization of an Automated Production and Logistics System")
    print("Ready. See main.ipynb for full interactive models and visualizations.")
