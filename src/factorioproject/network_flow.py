"""Network Flow Analysis for Factorio Factory Conveyor and Rail Logistics.

Models the material transport network as a directed graph, computes maximum throughput,
and identifies binding logistics bottlenecks via the max-flow min-cut theorem.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def create_factory_network(belt_tier: str = "yellow", upgrade_bottlenecks: bool = False) -> nx.DiGraph:
    """Build a directed network graph representing raw material delivery, smelting,

    intermediate transport, and assembly in the Factorio facility.

    Capacities are specified in items per second.
    """
    g = nx.DiGraph()

    # Belt capacities (items/sec)
    belt_caps = {
        "yellow": 15.0,
        "red": 30.0,
        "blue": 45.0,
    }
    belt_speed = belt_caps.get(belt_tier, 15.0)

    # If upgrade_bottlenecks is True, specific high-traffic trunk belts get red/blue capacity
    trunk_speed = belt_caps["red"] if upgrade_bottlenecks else belt_speed

    # Super Source connects to extraction supply points
    # Capacities represent mining site delivery limits
    g.add_edge("Super_Source", "Iron_Site", capacity=30.0)
    g.add_edge("Super_Source", "Copper_Site", capacity=30.0)
    g.add_edge("Super_Source", "Coal_Site", capacity=15.0)

    # Transport from extraction sites to factory unloading stations (Rail / Long Belts)
    g.add_edge("Iron_Site", "Iron_Unloader", capacity=25.0)
    g.add_edge("Copper_Site", "Copper_Unloader", capacity=25.0)
    g.add_edge("Coal_Site", "Coal_Depot", capacity=15.0)

    # Unloader to feeder belts (limited by belt tier or inserter array)
    g.add_edge("Iron_Unloader", "Iron_Feeder_1", capacity=belt_speed)
    g.add_edge("Iron_Unloader", "Iron_Feeder_2", capacity=belt_speed)
    g.add_edge("Copper_Unloader", "Copper_Feeder_1", capacity=belt_speed)
    g.add_edge("Copper_Unloader", "Copper_Feeder_2", capacity=belt_speed)

    # Coal distribution to smelting lines
    g.add_edge("Coal_Depot", "Iron_Smelt_Line_1", capacity=3.0)
    g.add_edge("Coal_Depot", "Iron_Smelt_Line_2", capacity=3.0)
    g.add_edge("Coal_Depot", "Copper_Smelt_Line_1", capacity=3.0)
    g.add_edge("Coal_Depot", "Copper_Smelt_Line_2", capacity=3.0)

    # Smelting furnace lines:
    # 24 furnaces from blueprint: 12 in Line 1, 12 in Line 2
    # 12 stone furnaces * 0.3125 plates/sec = 3.75 plates/sec capacity
    smelt_capacity = 3.75
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
    """Compute maximum flow and minimum cut partition using Edmonds-Karp/preflow-push."""
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
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Plot network flow graph highlighting flow rates and binding bottleneck edges."""
    fig, ax = plt.subplots(figsize=figsize)

    # Use a hierarchical multipartite layout
    # Assign layer based on distance from Super_Source
    layers = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        layers[node] = 0 if not preds else max(layers[p] for p in preds) + 1

    nx.set_node_attributes(g, layers, "subset")
    pos = nx.multipartite_layout(g, subset_key="subset")

    # Find bottleneck edge pairs
    binding_edges = set(
        zip(
            bottlenecks_df[bottlenecks_df["is_binding_bottleneck"]]["from_node"],
            bottlenecks_df[bottlenecks_df["is_binding_bottleneck"]]["to_node"],
        )
    )

    edge_colors = ["#d9534f" if (u, v) in binding_edges else "#428bca" for u, v in g.edges()]
    edge_widths = [2.8 if (u, v) in binding_edges else 1.2 for u, v in g.edges()]

    nx.draw_networkx_nodes(g, pos, node_size=1800, node_color="#eef2f7", edgecolors="#333333", ax=ax)
    nx.draw_networkx_labels(g, pos, font_size=8, font_family="sans-serif", font_weight="bold", ax=ax)

    nx.draw_networkx_edges(
        g,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    # Edge labels showing flow / capacity
    edge_labels = {
        (u, v): f"{round(flow_dict[u][v], 1)}/{round(g[u][v]['capacity'], 1)}"
        for u, v in g.edges()
    }
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.set_title(
        "Factorio Factory Logistics Network Flow (Edmonds-Karp)\nRed = Saturated Binding Min-Cut Bottlenecks",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig
