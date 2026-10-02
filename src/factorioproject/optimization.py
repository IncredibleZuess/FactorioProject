"""Linear Programming & Integer Programming for Factory Production Mix and Machine Allocation.

Includes post-optimal sensitivity analysis (shadow prices, reduced costs, power budget sweeps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pulp


@dataclass
class ProductionProblemConfig:
    """Parameters and capacities for the Factorio production facility."""

    # Available machinery (from blueprint and layout)
    num_stone_furnaces: int = 24
    num_assemblers: int = 10

    # Raw material supply capacities (units per second, e.g. yellow belt = 15/s)
    iron_ore_capacity: float = 15.0
    copper_ore_capacity: float = 15.0
    coal_capacity: float = 10.0

    # Electrical power grid maximum limit (kW)
    power_limit_kw: float = 1500.0

    # Unit values / selling prices for finished intermediate goods ($/unit)
    product_values: Dict[str, float] = field(
        default_factory=lambda: {
            "iron_plate": 1.0,
            "copper_plate": 1.0,
            "steel_plate": 4.5,
            "electronic_circuit": 3.8,
            "automation_science": 5.0,
        }
    )

    # Machine specs
    furnace_crafting_speed: float = 1.0
    assembler_crafting_speed: float = 0.5  # Assembling Machine 1

    # Power draws (kW)
    furnace_fuel_kw: float = 90.0  # Burner power
    assembler_electric_kw: float = 75.0  # Assembling Machine 1 electric
    inserter_electric_kw: float = 13.0  # Regular inserter
    base_logistics_power_kw: float = 48 * 13.0  # 48 inserters from blueprint = 624 kW


def solve_production_allocation(
    config: Optional[ProductionProblemConfig] = None,
    integer_machines: bool = False,
    power_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """Solve the production mix and machine allocation problem using PuLP.

    Decision Variables:
      r_iron: Net rate of iron plates produced for export (items/sec)
      r_copper: Net rate of copper plates produced for export (items/sec)
      r_steel: Net rate of steel plates produced (items/sec)
      r_circuits: Net rate of electronic circuits produced (items/sec)
      r_science: Net rate of automation science packs produced (items/sec)

      m_iron_smelt: Furnaces dedicated to iron plates
      m_copper_smelt: Furnaces dedicated to copper plates
      m_steel_smelt: Furnaces dedicated to steel plates
      m_circuit_cable: Assemblers dedicated to copper cable
      m_circuits: Assemblers dedicated to electronic circuits
      m_gear: Assemblers dedicated to iron gear wheels
      m_science: Assemblers dedicated to automation science

    Constraints:
      1. Machine allocation bounds (total furnaces <= 24, total assemblers <= 10)
      2. Machine capacity constraints linking production rates to allocated machines
      3. Raw material supply limits (iron ore, copper ore, coal)
      4. Power grid ceiling: total electric power <= power_limit_kw
    """
    if config is None:
        config = ProductionProblemConfig()

    p_limit = power_limit if power_limit is not None else config.power_limit_kw

    # Model definition
    model = pulp.LpProblem("Factorio_Production_Allocation", pulp.LpMaximize)

    var_cat = pulp.LpInteger if integer_machines else pulp.LpContinuous

    # Machine allocation variables
    m_iron = pulp.LpVariable("m_iron_furnace", lowBound=0, upBound=config.num_stone_furnaces, cat=var_cat)
    m_copper = pulp.LpVariable("m_copper_furnace", lowBound=0, upBound=config.num_stone_furnaces, cat=var_cat)
    m_steel = pulp.LpVariable("m_steel_furnace", lowBound=0, upBound=config.num_stone_furnaces, cat=var_cat)

    m_cable = pulp.LpVariable("m_cable_assembler", lowBound=0, upBound=config.num_assemblers, cat=var_cat)
    m_circuit = pulp.LpVariable("m_circuit_assembler", lowBound=0, upBound=config.num_assemblers, cat=var_cat)
    m_gear = pulp.LpVariable("m_gear_assembler", lowBound=0, upBound=config.num_assemblers, cat=var_cat)
    m_science = pulp.LpVariable("m_science_assembler", lowBound=0, upBound=config.num_assemblers, cat=var_cat)

    # Net output production rate variables (continuous items/sec)
    r_iron = pulp.LpVariable("r_iron_plate", lowBound=0, cat=pulp.LpContinuous)
    r_copper = pulp.LpVariable("r_copper_plate", lowBound=0, cat=pulp.LpContinuous)
    r_steel = pulp.LpVariable("r_steel_plate", lowBound=0, cat=pulp.LpContinuous)
    r_circuit = pulp.LpVariable("r_electronic_circuit", lowBound=0, cat=pulp.LpContinuous)
    r_science = pulp.LpVariable("r_automation_science", lowBound=0, cat=pulp.LpContinuous)

    # Objective Function: Maximize total economic value of net output
    vals = config.product_values
    model += (
        vals["iron_plate"] * r_iron
        + vals["copper_plate"] * r_copper
        + vals["steel_plate"] * r_steel
        + vals["electronic_circuit"] * r_circuit
        + vals["automation_science"] * r_science,
        "Total_Value_Throughput",
    )

    # Total Machine Limits
    furnace_limit_con = (m_iron + m_copper + m_steel <= config.num_stone_furnaces, "Furnace_Limit")
    model += furnace_limit_con

    assembler_limit_con = (
        m_cable + m_circuit + m_gear + m_science <= config.num_assemblers,
        "Assembler_Limit",
    )
    model += assembler_limit_con

    # Intermediate Production Rates (gross) based on allocated machines:
    # Stone furnace crafting speed = 1.0
    # Iron/copper plate recipe: 3.2 sec -> 0.3125 plates/sec per furnace
    # Steel plate recipe: 5 iron plates + 16 sec -> 1 / 16 = 0.0625 steel/sec per furnace
    # Assembling Machine 1 crafting speed = 0.5
    # Copper cable: 1 copper plate -> 2 cables in 0.5s base (0.5/0.5 = 1.0s effective -> 2 cables/s per assembler)
    # Electronic circuit: 1 iron plate + 3 copper cables in 0.5s base (1.0s effective -> 1 circuit/s per assembler)
    # Iron gear wheel: 2 iron plates in 0.5s base (1.0s effective -> 1 gear/s per assembler)
    # Automation science: 1 copper plate + 1 gear in 5.0s base (10.0s effective -> 0.1 science/s per assembler)

    # Maximum capacities determined by machine count:
    # gross_iron_plates = m_iron * (1.0 / 3.2)
    # gross_copper_plates = m_copper * (1.0 / 3.2)
    # gross_steel_plates = m_steel * (1.0 / 16.0)
    # gross_circuits = m_circuit * 1.0
    # gross_cables = m_cable * 2.0
    # gross_gears = m_gear * 1.0
    # gross_science = m_science * 0.1

    # Balancing constraints:
    # 1. Steel requires 5 iron plates per steel plate:
    # Gross iron plates required = r_iron + 5 * r_steel + 1.0 * r_circuit + 2.0 * gross_gears
    # Where gross_gears = r_science (1 gear per science pack)
    model += (
        r_iron + 5.0 * r_steel + 1.0 * r_circuit + 2.0 * r_science <= m_iron * (1.0 / 3.2),
        "Iron_Plate_Balance",
    )

    # 2. Copper plates required:
    # Gross copper plates = r_copper + (3.0 / 2.0) * r_circuit (from cable) + 1.0 * r_science
    model += (
        r_copper + 1.5 * r_circuit + 1.0 * r_science <= m_copper * (1.0 / 3.2),
        "Copper_Plate_Balance",
    )

    # 3. Steel production machine capacity:
    model += (r_steel <= m_steel * (1.0 / 16.0), "Steel_Furnace_Capacity")

    # 4. Electronic circuit assembler capacity:
    model += (r_circuit <= m_circuit * 1.0, "Circuit_Assembler_Capacity")

    # 5. Cable assembler capacity: 3 cables per circuit -> 1.5 cable assemblers per circuit assembler
    model += (3.0 * r_circuit <= m_cable * 2.0, "Cable_Assembler_Capacity")

    # 6. Gear assembler capacity: 1 gear per science -> 1.0 gear/s per assembler
    model += (r_science <= m_gear * 1.0, "Gear_Assembler_Capacity")

    # 7. Science pack assembler capacity:
    model += (r_science <= m_science * 0.1, "Science_Assembler_Capacity")

    # Raw material extraction supply constraints:
    # Iron ore: 1 ore per iron plate
    model += (m_iron * (1.0 / 3.2) <= config.iron_ore_capacity, "Iron_Ore_Supply")
    # Copper ore: 1 ore per copper plate
    model += (m_copper * (1.0 / 3.2) <= config.copper_ore_capacity, "Copper_Ore_Supply")

    # Power grid constraint:
    # Total electric draw = baseline logistics (48 inserters) + assemblers * 75 kW
    total_assemblers = m_cable + m_circuit + m_gear + m_science
    model += (
        config.base_logistics_power_kw + total_assemblers * config.assembler_electric_kw <= p_limit,
        "Power_Grid_Ceiling",
    )

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)

    status_str = pulp.LpStatus[status]

    # Collect results
    machines = {
        "m_iron_furnace": round(m_iron.varValue or 0.0, 3),
        "m_copper_furnace": round(m_copper.varValue or 0.0, 3),
        "m_steel_furnace": round(m_steel.varValue or 0.0, 3),
        "m_cable_assembler": round(m_cable.varValue or 0.0, 3),
        "m_circuit_assembler": round(m_circuit.varValue or 0.0, 3),
        "m_gear_assembler": round(m_gear.varValue or 0.0, 3),
        "m_science_assembler": round(m_science.varValue or 0.0, 3),
    }

    rates = {
        "r_iron_plate": round(r_iron.varValue or 0.0, 4),
        "r_copper_plate": round(r_copper.varValue or 0.0, 4),
        "r_steel_plate": round(r_steel.varValue or 0.0, 4),
        "r_electronic_circuit": round(r_circuit.varValue or 0.0, 4),
        "r_automation_science": round(r_science.varValue or 0.0, 4),
    }

    total_electric_power = config.base_logistics_power_kw + (
        sum(
            [
                machines["m_cable_assembler"],
                machines["m_circuit_assembler"],
                machines["m_gear_assembler"],
                machines["m_science_assembler"],
            ]
        )
        * config.assembler_electric_kw
    )

    total_furnace_fuel = (
        machines["m_iron_furnace"] + machines["m_copper_furnace"] + machines["m_steel_furnace"]
    ) * config.furnace_fuel_kw

    # Sensitivity analysis / Dual values (shadow prices and slack) if continuous LP
    sensitivity: Dict[str, Dict[str, float]] = {}
    if not integer_machines:
        for name, constraint in model.constraints.items():
            sensitivity[name] = {
                "shadow_price": round(constraint.pi if constraint.pi is not None else 0.0, 4),
                "slack": round(constraint.slack if constraint.slack is not None else 0.0, 4),
            }

    reduced_costs = {}
    if not integer_machines:
        for var in model.variables():
            reduced_costs[var.name] = round(var.dj if var.dj is not None else 0.0, 4)

    return {
        "status": status_str,
        "objective_value": round(pulp.value(model.objective) or 0.0, 4),
        "machine_allocations": machines,
        "production_rates": rates,
        "power_metrics": {
            "electric_power_kw": round(total_electric_power, 2),
            "power_limit_kw": p_limit,
            "furnace_fuel_kw": round(total_furnace_fuel, 2),
            "total_power_kw": round(total_electric_power + total_furnace_fuel, 2),
        },
        "shadow_prices": sensitivity,
        "reduced_costs": reduced_costs,
    }


def parametric_power_sweep(
    config: Optional[ProductionProblemConfig] = None,
    power_range: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Perform parametric sweep over power grid ceilings to construct the Pareto frontier

    between energy availability and maximum factory throughput value.
    """
    if config is None:
        config = ProductionProblemConfig()

    if power_range is None:
        power_range = list(range(650, 1600, 50))

    records = []
    for p in power_range:
        res = solve_production_allocation(config, integer_machines=False, power_limit=float(p))
        if res["status"] == "Optimal":
            rates = res["production_rates"]
            records.append(
                {
                    "power_limit_kw": p,
                    "actual_electric_kw": res["power_metrics"]["electric_power_kw"],
                    "objective_value": res["objective_value"],
                    "iron_plates_rate": rates["r_iron_plate"],
                    "copper_plates_rate": rates["r_copper_plate"],
                    "circuits_rate": rates["r_electronic_circuit"],
                    "science_rate": rates["r_automation_science"],
                    "power_shadow_price": res["shadow_prices"].get("Power_Grid_Ceiling", {}).get("shadow_price", 0.0),
                }
            )

    return pd.DataFrame(records)
