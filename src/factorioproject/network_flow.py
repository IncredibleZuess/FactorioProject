"""Network Flow Analysis for Factorio Factory Conveyor and Rail Logistics.

Models the material transport network as a directed graph, computes maximum throughput,
and identifies binding logistics bottlenecks via the max-flow min-cut theorem.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def create_factory_network(
    belt_tier: str = "yellow",
    upgrade_bottlenecks: bool = False,
    smelting_line_capacity: float = 3.75,
) -> nx.DiGraph:
    """Build a directed network graph representing raw material delivery, smelting,

    intermediate transport, and assembly in the Factorio facility.

    Capacities use units of items per second.
    """
    g = nx.DiGraph()

    # Scale logistics with smelting capacity (e.g. 3.75 for 24-furnace array, 30.31 for 374-furnace full base)
    scale = max(1.0, (smelting_line_capacity * 4.0) / 15.0)

    # Belt capacities (items/sec)
    belt_caps = {
        "yellow": 15.0 * scale,
        "red": 30.0 * scale,
        "blue": 45.0 * scale,
    }
    belt_speed = belt_caps.get(belt_tier, 15.0 * scale)

    # If upgrade_bottlenecks is True, specific high-traffic trunk belts get red/blue capacity
    trunk_speed = belt_caps["red"] if upgrade_bottlenecks else belt_speed

    # Super Source connects to extraction supply points
    # Capacities represent mining site delivery limits
    g.add_edge("Super_Source", "Iron_Site", capacity=30.0 * scale)
    g.add_edge("Super_Source", "Copper_Site", capacity=30.0 * scale)
    g.add_edge("Super_Source", "Coal_Site", capacity=15.0 * scale)

    # Transport from extraction sites to factory unloading stations (Rail / Long Belts)
    g.add_edge("Iron_Site", "Iron_Unloader", capacity=25.0 * scale)
    g.add_edge("Copper_Site", "Copper_Unloader", capacity=25.0 * scale)
    g.add_edge("Coal_Site", "Coal_Depot", capacity=15.0 * scale)

    # Unloader to feeder belts (limited by belt tier or inserter array)
    g.add_edge("Iron_Unloader", "Iron_Feeder_1", capacity=belt_speed)
    g.add_edge("Iron_Unloader", "Iron_Feeder_2", capacity=belt_speed)
    g.add_edge("Copper_Unloader", "Copper_Feeder_1", capacity=belt_speed)
    g.add_edge("Copper_Unloader", "Copper_Feeder_2", capacity=belt_speed)

    # Coal distribution to smelting lines
    g.add_edge("Coal_Depot", "Iron_Smelt_Line_1", capacity=3.0 * scale)
    g.add_edge("Coal_Depot", "Iron_Smelt_Line_2", capacity=3.0 * scale)
    g.add_edge("Coal_Depot", "Copper_Smelt_Line_1", capacity=3.0 * scale)
    g.add_edge("Coal_Depot", "Copper_Smelt_Line_2", capacity=3.0 * scale)

    # Smelting furnace lines
    smelt_capacity = smelting_line_capacity
    g.add_edge("Iron_Feeder_1", "Iron_Smelt_Line_1", capacity=smelt_capacity)
    g.add_edge("Iron_Feeder_2", "Iron_Smelt_Line_2", capacity=smelt_capacity)
    g.add_edge("Copper_Feeder_1", "Copper_Smelt_Line_1", capacity=smelt_capacity)
    g.add_edge("Copper_Feeder_2", "Copper_Smelt_Line_2", capacity=smelt_capacity)

    # Smelting output to central main bus
    g.add_edge("Iron_Smelt_Line_1", "Iron_Main_Bus", capacity=smelt_capacity)
    g.add_edge("Iron_Smelt_Line_2", "Iron_Main_Bus", capacity=smelt_capacity)
    g.add_edge("Copper_Smelt_Line_1", "Copper_Main_Bus", capacity=smelt_capacity)
    g.add_edge("Copper_Smelt_Line_2", "Copper_Main_Bus", capacity=smelt_capacity)

    # Main bus routing to intermediate assembly
    # Iron bus feeds Gear Assembly, Circuit Assembly, and Direct Export
    g.add_edge("Iron_Main_Bus", "Gear_Assembly", capacity=trunk_speed)
    g.add_edge("Iron_Main_Bus", "Circuit_Assembly", capacity=trunk_speed)
    g.add_edge("Iron_Main_Bus", "Direct_Plate_Export", capacity=10.0)

    # Copper bus feeds Cable Assembly and Science Assembly
    g.add_edge("Copper_Main_Bus", "Cable_Assembly", capacity=trunk_speed)
    g.add_edge("Copper_Main_Bus", "Science_Assembly", capacity=10.0)

    # Intermediate assemblies feed final stages
    g.add_edge("Cable_Assembly", "Circuit_Assembly", capacity=15.0)
    g.add_edge("Gear_Assembly", "Science_Assembly", capacity=10.0)

    # Finished outputs to Super Sink
    g.add_edge("Direct_Plate_Export", "Super_Sink", capacity=10.0)
    g.add_edge("Circuit_Assembly", "Super_Sink", capacity=12.0)
    g.add_edge("Science_Assembly", "Super_Sink", capacity=5.0)

    return g


def solve_max_network_flow(
    g: nx.DiGraph,
    source: str = "Super_Source",
    sink: str = "Super_Sink",
) -> Tuple[float, Dict[str, Dict[str, float]], Set[str], Set[str]]:
    """Compute maximum flow and minimum cut partition using the Edmonds-Karp algorithm."""
    flow_val, flow_dict = nx.maximum_flow(g, source, sink)
    cut_val, partition = nx.minimum_cut(g, source, sink)
    reachable_s, reachable_t = partition
    return flow_val, flow_dict, reachable_s, reachable_t


def analyze_bottlenecks(
    g: nx.DiGraph,
    flow_dict: Dict[str, Dict[str, float]],
    reachable_s: Set[str],
    reachable_t: Set[str],
) -> pd.DataFrame:
    """Identify binding bottleneck edges crossing the minimum cut partition.

    These arcs have flow == capacity and cross from S to T.
    """
    records = []
    for u, v, data in g.edges(data=True):
        cap = data["capacity"]
        actual_flow = flow_dict[u][v]
        slack = cap - actual_flow
        is_cut_edge = (u in reachable_s) and (v in reachable_t)
        is_saturated = abs(slack) < 1e-5

        records.append(
            {
                "from_node": u,
                "to_node": v,
                "capacity": cap,
                "actual_flow": actual_flow,
                "utilization_pct": round((actual_flow / cap) * 100.0, 1) if cap > 0 else 0.0,
                "slack": round(slack, 3),
                "is_saturated": is_saturated,
                "is_binding_bottleneck": is_cut_edge and is_saturated,
            }
        )

    df = pd.DataFrame(records)
    return df.sort_values(by=["is_binding_bottleneck", "utilization_pct"], ascending=[False, False])


def plot_network_bottlenecks(
    g: nx.DiGraph,
    flow_dict: Dict[str, Dict[str, float]],
    bottlenecks_df: pd.DataFrame,
    figsize: Tuple[int, int] = (20, 9.5),
) -> plt.Figure:
    """Plot network flow graph with custom clean layout and styled nodes,

    completely eliminating node, edge, and label overlaps while highlighting
    the binding min-cut bottlenecks in red.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Deterministic, non-overlapping spatial layout
    pos = {
        "Super_Source": (0.0, 4.0),
        "Iron_Site": (2.2, 7.4),
        "Coal_Site": (2.2, 4.0),
        "Copper_Site": (2.2, 0.6),
        "Iron_Unloader": (4.4, 7.4),
        "Coal_Depot": (4.4, 4.0),
        "Copper_Unloader": (4.4, 0.6),
        "Iron_Feeder_1": (6.6, 8.4),
        "Iron_Feeder_2": (6.6, 6.4),
        "Copper_Feeder_1": (6.6, 1.6),
        "Copper_Feeder_2": (6.6, -0.4),
        "Iron_Smelt_Line_1": (9.0, 8.4),
        "Iron_Smelt_Line_2": (9.0, 6.4),
        "Copper_Smelt_Line_1": (9.0, 1.6),
        "Copper_Smelt_Line_2": (9.0, -0.4),
        "Iron_Main_Bus": (11.4, 7.4),
        "Copper_Main_Bus": (11.4, 0.6),
        "Gear_Assembly": (13.8, 6.2),
        "Cable_Assembly": (13.8, 1.8),
        "Direct_Plate_Export": (16.2, 8.4),
        "Science_Assembly": (16.2, 6.2),
        "Circuit_Assembly": (16.2, 1.8),
        "Super_Sink": (18.8, 4.0),
    }

    # Fallback for any unexpected nodes
    for n in g.nodes():
        if n not in pos:
            pos[n] = (10.0, 4.0)

    binding_edges = set(
        zip(
            bottlenecks_df[bottlenecks_df["is_binding_bottleneck"]]["from_node"],
            bottlenecks_df[bottlenecks_df["is_binding_bottleneck"]]["to_node"],
        )
    )

    # Color code nodes by functional category
    node_colors = {}
    for n in g.nodes():
        if "Source" in n or "Site" in n:
            node_colors[n] = "#ffe6cc"  # Orange/mine
        elif "Unloader" in n or "Depot" in n or "Feeder" in n:
            node_colors[n] = "#d5e8d4"  # Green/logistics
        elif "Smelt" in n:
            node_colors[n] = "#dae8fc"  # Blue/smelter
        elif "Bus" in n:
            node_colors[n] = "#fff2cc"  # Yellow/bus
        elif "Assembly" in n or "Export" in n:
            node_colors[n] = "#e1d5e7"  # Purple/manufacturing
        else:
            node_colors[n] = "#f5f5f5"

    edge_colors = ["#d9534f" if (u, v) in binding_edges else "#4a90e2" for u, v in g.edges()]
    edge_widths = [3.5 if (u, v) in binding_edges else 1.3 for u, v in g.edges()]
    edge_styles = ["solid" if flow_dict[u][v] > 0 else "dashed" for u, v in g.edges()]

    # Draw routed edges
    for (u, v), col, w, sty in zip(g.edges(), edge_colors, edge_widths, edge_styles):
        rad = 0.0
        if u == "Coal_Depot":
            if "Iron" in v:
                rad = -0.06
            elif "Copper" in v:
                rad = 0.06
        elif u == "Direct_Plate_Export" and v == "Super_Sink":
            rad = -0.08
        elif u == "Iron_Main_Bus" and v == "Circuit_Assembly":
            rad = -0.03
        elif u == "Copper_Main_Bus" and v == "Science_Assembly":
            rad = 0.03

        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=w,
                linestyle=sty,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=26,
                shrinkB=26,
                mutation_scale=14,
            ),
        )

    # Draw nodes as formatted boxes with generous padding
    labels = {n: n.replace("_", "\n") for n in g.nodes()}
    for n, (x, y) in pos.items():
        if n not in g:
            continue
        is_bottleneck_node = any(n == u or n == v for u, v in binding_edges)
        border_col = "#d9534f" if is_bottleneck_node and "Smelt" in n else "#444444"
        border_w = 2.2 if is_bottleneck_node and "Smelt" in n else 1.0
        ax.text(
            x,
            y,
            labels[n],
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=node_colors.get(n, "#ffffff"),
                edgecolor=border_col,
                linewidth=border_w,
            ),
        )

    # Edge labels: placed cleanly to prevent overlaps
    for u, v in g.edges():
        f_val = flow_dict[u][v]
        cap = g[u][v]["capacity"]
        is_bot = (u, v) in binding_edges

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        frac = 0.50
        if u == "Coal_Depot":
            frac = 0.35
        elif is_bot:
            frac = 0.48
        elif u in ("Iron_Main_Bus", "Copper_Main_Bus"):
            frac = 0.36
        elif v == "Super_Sink":
            frac = 0.60

        mx = x1 + frac * (x2 - x1)
        my = y1 + frac * (y2 - y1)

        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        if dist > 0:
            offset = 0.20
            ox = -dy / dist * offset
            oy = dx / dist * offset
        else:
            ox, oy = 0, 0

        if is_bot:
            ax.text(
                mx + ox,
                my + oy,
                f"{f_val:.1f}/{cap:.1f}\n[BOTTLENECK]",
                color="#b30000",
                fontsize=7.5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#ffe6e6",
                    edgecolor="#b30000",
                    lw=1.0,
                ),
            )
        elif f_val > 0.01:
            ax.text(
                mx + ox,
                my + oy,
                f"{f_val:.1f}/{cap:.1f}",
                color="#1a5276",
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="#ffffff",
                    edgecolor="#aed6f1",
                    lw=0.7,
                    alpha=0.95,
                ),
            )

    ax.set_xlim(-1.5, 20.2)
    ax.set_ylim(-1.5, 9.8)
    ax.set_title(
        "Factorio Factory Logistics Network Flow (Edmonds-Karp)\nRed = Saturated Binding Min-Cut Bottlenecks (15.0 items/sec Max Flow)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


def create_petrochemical_network(
    train_capacity: float = 60.0,
    num_refineries: int = 6,
) -> nx.DiGraph:
    """Build a directed network graph representing the offshore crude oil rail logistics,

    oil refineries array, chemical cracking plants, and advanced product synthesis.

    Args:
        train_capacity: Fluid throughput capacity of the rail train shuttle link (fluid/sec).
                       E.g., 60.0 for a single baseline train, 140.0 for dual fluid trains.
        num_refineries: Number of operating oil refineries (default 6 from blueprint).
                       Each refinery consumes 20.0 crude/s.
    """
    g = nx.DiGraph()

    refinery_crude_demand = num_refineries * 20.0  # 120.0 for 6 refineries
    refinery_heavy_out = num_refineries * 5.0     # 30.0
    refinery_light_out = num_refineries * 9.0     # 54.0
    refinery_petro_out = num_refineries * 11.0    # 66.0

    # 1. Offshore Extraction and Rail Shovelling Chain
    g.add_edge("Offshore_Oil_Rigs", "Rail_Loading_Terminal", capacity=180.0)
    g.add_edge("Rail_Loading_Terminal", "Oil_Train_Shuttle", capacity=train_capacity)
    g.add_edge("Oil_Train_Shuttle", "Train_Unloading_Terminal", capacity=train_capacity)
    g.add_edge("Train_Unloading_Terminal", "Crude_Buffer_Tanks", capacity=train_capacity * 1.5)

    # 2. Crude feed into Refineries
    g.add_edge("Crude_Buffer_Tanks", "Oil_Refineries_Array", capacity=refinery_crude_demand)

    # 3. Fractionation yields
    g.add_edge("Oil_Refineries_Array", "Heavy_Oil_Bus", capacity=refinery_heavy_out)
    g.add_edge("Oil_Refineries_Array", "Light_Oil_Bus", capacity=refinery_light_out)
    g.add_edge("Oil_Refineries_Array", "Petroleum_Gas_Bus", capacity=refinery_petro_out)

    # 4. Heavy Oil routing: Lubricant + Heavy Cracking to Light Oil
    g.add_edge("Heavy_Oil_Bus", "Lubricant_Plant", capacity=12.0)
    g.add_edge("Heavy_Oil_Bus", "Heavy_Cracking_Plant", capacity=20.0)
    g.add_edge("Heavy_Cracking_Plant", "Light_Oil_Bus", capacity=20.0)

    # 5. Light Oil routing: Rocket Fuel + Light Cracking to Petroleum Gas
    g.add_edge("Light_Oil_Bus", "Rocket_Fuel_Plant", capacity=30.0)
    g.add_edge("Light_Oil_Bus", "Light_Cracking_Plant", capacity=40.0)
    g.add_edge("Light_Cracking_Plant", "Petroleum_Gas_Bus", capacity=36.0)

    # 6. Petroleum Gas routing: Plastic & Sulfur synthesis
    g.add_edge("Petroleum_Gas_Bus", "Plastic_Chemical_Plant", capacity=60.0)
    g.add_edge("Petroleum_Gas_Bus", "Sulfur_Chemical_Plant", capacity=30.0)

    # 7. Advanced Downstream Assembly
    g.add_edge("Plastic_Chemical_Plant", "Red_Circuit_Assembly", capacity=40.0)
    g.add_edge("Sulfur_Chemical_Plant", "Blue_Science_Assembly", capacity=25.0)
    g.add_edge("Lubricant_Plant", "Robot_Frame_Assembly", capacity=12.0)

    # 8. Sinks to Petrochem Sink
    g.add_edge("Red_Circuit_Assembly", "Petrochem_Sink", capacity=35.0)
    g.add_edge("Blue_Science_Assembly", "Petrochem_Sink", capacity=25.0)
    g.add_edge("Rocket_Fuel_Plant", "Petrochem_Sink", capacity=30.0)
    g.add_edge("Robot_Frame_Assembly", "Petrochem_Sink", capacity=12.0)

    return g


def plot_petrochemical_network(
    g: nx.DiGraph,
    flow_dict: Dict[str, Dict[str, float]],
    bottlenecks_df: pd.DataFrame,
    title_suffix: str = "",
    figsize: Tuple[int, int] = (22, 10),
) -> plt.Figure:
    """Plot the Offshore Oil Rail Supply and Petrochemical Refining Network."""
    fig, ax = plt.subplots(figsize=figsize)

    binding_edges = set(
        zip(
            bottlenecks_df[bottlenecks_df["is_binding_bottleneck"]]["from_node"],
            bottlenecks_df[bottlenecks_df["is_binding_bottleneck"]]["to_node"],
        )
    )

    pos = {
        "Offshore_Oil_Rigs": (0.0, 4.0),
        "Rail_Loading_Terminal": (2.2, 4.0),
        "Oil_Train_Shuttle": (4.4, 4.0),
        "Train_Unloading_Terminal": (6.6, 4.0),
        "Crude_Buffer_Tanks": (8.8, 4.0),
        "Oil_Refineries_Array": (11.0, 4.0),
        # 3 Fraction Buses
        "Heavy_Oil_Bus": (13.2, 6.8),
        "Light_Oil_Bus": (13.2, 4.0),
        "Petroleum_Gas_Bus": (13.2, 1.2),
        # Intermediate Processing
        "Lubricant_Plant": (15.4, 7.8),
        "Heavy_Cracking_Plant": (15.4, 6.0),
        "Rocket_Fuel_Plant": (15.4, 4.8),
        "Light_Cracking_Plant": (15.4, 3.2),
        "Plastic_Chemical_Plant": (15.4, 1.8),
        "Sulfur_Chemical_Plant": (15.4, 0.4),
        # Downstream
        "Robot_Frame_Assembly": (17.8, 7.4),
        "Red_Circuit_Assembly": (17.8, 2.4),
        "Blue_Science_Assembly": (17.8, 0.8),
        # Sink
        "Petrochem_Sink": (20.2, 4.0),
    }

    for n in g.nodes():
        if n not in pos:
            pos[n] = (10.0, 4.0)

    node_colors = {
        "Offshore_Oil_Rigs": "#b2ebf2",      # Cyan/Offshore
        "Rail_Loading_Terminal": "#c8e6c9",
        "Oil_Train_Shuttle": "#ffe0b2",      # Orange train
        "Train_Unloading_Terminal": "#c8e6c9",
        "Crude_Buffer_Tanks": "#cfd8dc",      # Slate
        "Oil_Refineries_Array": "#d1c4e9",    # Purple Refineries
        "Heavy_Oil_Bus": "#d7ccc8",
        "Light_Oil_Bus": "#fff9c4",
        "Petroleum_Gas_Bus": "#e1bee7",
        "Lubricant_Plant": "#c8e6c9",
        "Heavy_Cracking_Plant": "#ffcdd2",
        "Rocket_Fuel_Plant": "#ffe0b2",
        "Light_Cracking_Plant": "#ffcdd2",
        "Plastic_Chemical_Plant": "#e0f2f1",
        "Sulfur_Chemical_Plant": "#fff59d",
        "Robot_Frame_Assembly": "#e1bee7",
        "Red_Circuit_Assembly": "#ffcdd2",
        "Blue_Science_Assembly": "#bbdefb",
        "Petrochem_Sink": "#f5f5f5",
    }

    edge_colors = ["#d9534f" if (u, v) in binding_edges else "#4a90e2" for u, v in g.edges()]
    edge_widths = [3.8 if (u, v) in binding_edges else 1.4 for u, v in g.edges()]
    edge_styles = ["solid" if flow_dict[u][v] > 0 else "dashed" for u, v in g.edges()]

    for (u, v), col, w, sty in zip(g.edges(), edge_colors, edge_widths, edge_styles):
        rad = 0.0
        if u == "Heavy_Cracking_Plant" and v == "Light_Oil_Bus":
            rad = 0.08
        elif u == "Light_Cracking_Plant" and v == "Petroleum_Gas_Bus":
            rad = 0.08
        elif u == "Rocket_Fuel_Plant" and v == "Petrochem_Sink":
            rad = -0.04

        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->",
                color=col,
                lw=w,
                linestyle=sty,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=26,
                shrinkB=26,
                mutation_scale=14,
            ),
        )

    labels = {n: n.replace("_", "\n") for n in g.nodes()}
    for n, (x, y) in pos.items():
        if n not in g:
            continue
        is_bot = any(n == u or n == v for u, v in binding_edges)
        border_col = "#d9534f" if is_bot and ("Train" in n or "Refiner" in n) else "#444444"
        border_w = 2.4 if is_bot and ("Train" in n or "Refiner" in n) else 1.0
        ax.text(
            x,
            y,
            labels[n],
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.55",
                facecolor=node_colors.get(n, "#ffffff"),
                edgecolor=border_col,
                linewidth=border_w,
            ),
        )

    for u, v in g.edges():
        f_val = flow_dict[u][v]
        cap = g[u][v]["capacity"]
        is_bot = (u, v) in binding_edges

        x1, y1 = pos[u]
        x2, y2 = pos[v]
        frac = 0.50
        if v == "Petrochem_Sink":
            frac = 0.60
        elif u == "Heavy_Cracking_Plant":
            frac = 0.40
        elif u == "Light_Cracking_Plant":
            frac = 0.40

        mx = x1 + frac * (x2 - x1)
        my = y1 + frac * (y2 - y1)

        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        ox, oy = (-dy / dist * 0.20, dx / dist * 0.20) if dist > 0 else (0, 0)

        if is_bot:
            ax.text(
                mx + ox,
                my + oy,
                f"{f_val:.1f}/{cap:.1f}\n[BOTTLENECK]",
                color="#b30000",
                fontsize=7.8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#ffe6e6",
                    edgecolor="#b30000",
                    lw=1.0,
                ),
            )
        elif f_val > 0.01:
            ax.text(
                mx + ox,
                my + oy,
                f"{f_val:.1f}/{cap:.1f}",
                color="#1a5276",
                fontsize=7.2,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="#ffffff",
                    edgecolor="#aed6f1",
                    lw=0.7,
                    alpha=0.95,
                ),
            )

    max_flow = sum(flow_dict[u]["Petrochem_Sink"] for u in g.predecessors("Petrochem_Sink"))
    title = f"Offshore Oil Rail Supply & Petrochemical Refining Network Flow (Edmonds-Karp)\n{title_suffix} (Max Flow = {max_flow:.1f} fluid/sec)"
    ax.set_xlim(-1.5, 21.6)
    ax.set_ylim(-1.0, 9.0)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    return fig


