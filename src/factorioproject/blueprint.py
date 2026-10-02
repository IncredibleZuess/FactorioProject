"""Factorio Blueprint and Blueprint Book Parsing, Decoding, and Entity Instrumentation Module."""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


def decode_blueprint(blueprint_string: str) -> Dict[str, Any]:
    """Decode a Factorio blueprint or blueprint-book string into a Python dictionary.

    Factorio blueprints are encoded as:
    1. A single-byte version prefix ('0' in Factorio 1.0+)
    2. Base64-encoded string
    3. zlib-compressed JSON payload
    """
    clean_str = blueprint_string.strip()
    if clean_str.startswith("0"):
        raw_b64 = clean_str[1:]
    else:
        raw_b64 = clean_str

    compressed = base64.b64decode(raw_b64)
    decompressed = zlib.decompress(compressed)
    return json.loads(decompressed.decode("utf-8"))


def encode_blueprint(data: Dict[str, Any]) -> str:
    """Encode a Python dictionary into a Factorio blueprint string."""
    json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(json_bytes, level=9)
    b64 = base64.b64encode(compressed).decode("utf-8")
    return "0" + b64


def is_blueprint_book(data: Dict[str, Any]) -> bool:
    """Check whether decoded dictionary represents a blueprint book."""
    return "blueprint_book" in data


def get_book_blueprints_summary(book_dict: Dict[str, Any]) -> pd.DataFrame:
    """Extract a summary table of all stages/blueprints contained within a blueprint book."""
    book = book_dict.get("blueprint_book", book_dict)
    blueprints = book.get("blueprints", [])
    records = []

    for i, item in enumerate(blueprints):
        bp = item.get("blueprint", item)
        lbl = bp.get("label", f"Stage {i+1}")
        entities = bp.get("entities", [])
        counts = pd.Series([e.get("name") for e in entities]).value_counts() if entities else pd.Series(dtype=int)

        furnaces = sum(counts.get(ft, 0) for ft in ["stone-furnace", "steel-furnace", "electric-furnace"])
        assemblers = sum(counts.get(at, 0) for at in ["assembling-machine-1", "assembling-machine-2", "assembling-machine-3"])
        power_gen = counts.get("steam-engine", 0) * 0.9  # MW
        belts = sum(counts.get(bt, 0) for bt in ["transport-belt", "fast-transport-belt", "express-transport-belt"])

        records.append({
            "stage_index": i,
            "label": lbl,
            "total_entities": len(entities),
            "furnaces": furnaces,
            "assemblers": assemblers,
            "power_generation_mw": round(power_gen, 2),
            "belts": belts,
        })

    return pd.DataFrame(records)


def extract_blueprint(data: Dict[str, Any], selection: Union[int, str] = "Full Base Default") -> Dict[str, Any]:
    """Extract a specific blueprint from a blueprint book by label or index.

    If data is already a single blueprint, returns it directly.
    """
    if not is_blueprint_book(data):
        return data

    book = data["blueprint_book"]
    blueprints = book.get("blueprints", [])

    if isinstance(selection, int):
        if 0 <= selection < len(blueprints):
            return blueprints[selection]
        raise IndexError(f"Blueprint index {selection} out of range (0-{len(blueprints)-1})")

    # Match by label
    for item in blueprints:
        bp = item.get("blueprint", item)
        if bp.get("label") == selection:
            return item

    # Partial match
    for item in blueprints:
        bp = item.get("blueprint", item)
        lbl = bp.get("label", "")
        if selection.lower() in lbl.lower():
            return item

    # Fallback to the largest blueprint by entity count
    sorted_bps = sorted(blueprints, key=lambda b: len(b.get("blueprint", b).get("entities", [])), reverse=True)
    if sorted_bps:
        return sorted_bps[0]

    return blueprints[0] if blueprints else data


def blueprint_to_dataframe(blueprint_dict: Dict[str, Any], selection: Optional[Union[int, str]] = None) -> pd.DataFrame:
    """Extract entities from a blueprint or blueprint-book dictionary into a normalized pandas DataFrame."""
    if is_blueprint_book(blueprint_dict):
        sel = selection if selection is not None else "Full Base Default"
        target_bp = extract_blueprint(blueprint_dict, selection=sel)
    else:
        target_bp = blueprint_dict

    bp = target_bp.get("blueprint", target_bp)
    entities = bp.get("entities", [])
    if not entities:
        return pd.DataFrame()

    df = pd.json_normalize(entities)
    # Ensure standard position columns
    if "position.x" in df.columns and "x" not in df.columns:
        df["x"] = df["position.x"]
        df["y"] = df["position.y"]
    elif "position" in df.columns and "x" not in df.columns:
        df["x"] = df["position"].apply(lambda p: p.get("x") if isinstance(p, dict) else None)
        df["y"] = df["position"].apply(lambda p: p.get("y") if isinstance(p, dict) else None)

    return df


# Standard Factorio baseline equipment constants (vanilla Factorio)
FACTORIO_SPECS = {
    "stone-furnace": {
        "crafting_speed": 1.0,
        "energy_type": "burner",
        "power_draw_kw": 90.0,
        "pollution": 2.0,
        "base_recipe_time": 3.2,
    },
    "steel-furnace": {
        "crafting_speed": 2.0,
        "energy_type": "burner",
        "power_draw_kw": 90.0,
        "pollution": 4.0,
        "base_recipe_time": 3.2,
    },
    "electric-furnace": {
        "crafting_speed": 2.0,
        "energy_type": "electric",
        "power_draw_kw": 180.0,
        "pollution": 1.0,
        "base_recipe_time": 3.2,
    },
    "assembling-machine-1": {
        "crafting_speed": 0.5,
        "energy_type": "electric",
        "power_draw_kw": 75.0,
    },
    "assembling-machine-2": {
        "crafting_speed": 0.75,
        "energy_type": "electric",
        "power_draw_kw": 150.0,
    },
    "assembling-machine-3": {
        "crafting_speed": 1.25,
        "energy_type": "electric",
        "power_draw_kw": 375.0,
    },
    "chemical-plant": {
        "crafting_speed": 1.0,
        "energy_type": "electric",
        "power_draw_kw": 210.0,
    },
    "oil-refinery": {
        "crafting_speed": 1.0,
        "energy_type": "electric",
        "power_draw_kw": 420.0,
    },
    "lab": {
        "energy_type": "electric",
        "power_draw_kw": 60.0,
    },
    "boiler": {
        "energy_type": "burner",
        "power_output_kw": 1800.0,  # 1.8 MW steam output
    },
    "steam-engine": {
        "power_output_kw": 900.0,   # 900 kW electric output
    },
    "inserter": {
        "energy_type": "electric",
        "active_power_kw": 13.0,
        "drain_power_kw": 0.4,
    },
    "fast-inserter": {
        "energy_type": "electric",
        "active_power_kw": 46.0,
        "drain_power_kw": 0.5,
    },
    "long-handed-inserter": {
        "energy_type": "electric",
        "active_power_kw": 20.0,
        "drain_power_kw": 0.4,
    },
    "stack-inserter": {
        "energy_type": "electric",
        "active_power_kw": 132.0,
        "drain_power_kw": 1.0,
    },
    "transport-belt": {
        "throughput_items_per_sec": 15.0,
    },
    "fast-transport-belt": {
        "throughput_items_per_sec": 30.0,
    },
    "express-transport-belt": {
        "throughput_items_per_sec": 45.0,
    },
    "splitter": {
        "throughput_items_per_sec": 15.0,
    },
    "underground-belt": {
        "throughput_items_per_sec": 15.0,
    },
    "small-lamp": {
        "power_draw_kw": 5.0,
    },
}


def analyze_blueprint_layout(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate equipment metrics, theoretical production capacities,

    and power requirements based on the blueprint entity dataset.
    """
    if df.empty or "name" not in df.columns:
        return {"error": "Empty or invalid blueprint dataframe"}

    counts = df["name"].value_counts().to_dict()

    # Furnace stats
    stone_furnaces = counts.get("stone-furnace", 0)
    steel_furnaces = counts.get("steel-furnace", 0)
    electric_furnaces = counts.get("electric-furnace", 0)
    furnace_count = stone_furnaces + steel_furnaces + electric_furnaces

    # Throughput capacity (plates/sec)
    stone_rate = (1.0 / 3.2) * stone_furnaces
    steel_rate = (2.0 / 3.2) * steel_furnaces
    electric_rate = (2.0 / 3.2) * electric_furnaces
    total_smelting_rate = stone_rate + steel_rate + electric_rate

    # Yellow belts equivalent (15 items/sec per belt)
    yellow_belt_cap = FACTORIO_SPECS["transport-belt"]["throughput_items_per_sec"]
    equivalent_yellow_belts = total_smelting_rate / yellow_belt_cap

    # Assembling Machines
    am1 = counts.get("assembling-machine-1", 0)
    am2 = counts.get("assembling-machine-2", 0)
    am3 = counts.get("assembling-machine-3", 0)
    chem_plants = counts.get("chemical-plant", 0)
    oil_refineries = counts.get("oil-refinery", 0)
    labs = counts.get("lab", 0)
    total_manufacturing_units = am1 + am2 + am3 + chem_plants + oil_refineries

    # Power Generation Capacity
    boilers = counts.get("boiler", 0)
    steam_engines = counts.get("steam-engine", 0)
    power_generation_kw = steam_engines * FACTORIO_SPECS["steam-engine"]["power_output_kw"]
    power_generation_mw = power_generation_kw / 1000.0

    # Power Consumption (active electric draw)
    inserters = counts.get("inserter", 0)
    fast_inserters = counts.get("fast-inserter", 0)
    long_inserters = counts.get("long-handed-inserter", 0)
    stack_inserters = counts.get("stack-inserter", 0)
    lamps = counts.get("small-lamp", 0)

    electric_consumers_kw = (
        am1 * FACTORIO_SPECS["assembling-machine-1"]["power_draw_kw"]
        + am2 * FACTORIO_SPECS["assembling-machine-2"]["power_draw_kw"]
        + am3 * FACTORIO_SPECS["assembling-machine-3"]["power_draw_kw"]
        + chem_plants * FACTORIO_SPECS["chemical-plant"]["power_draw_kw"]
        + oil_refineries * FACTORIO_SPECS["oil-refinery"]["power_draw_kw"]
        + labs * FACTORIO_SPECS["lab"]["power_draw_kw"]
        + inserters * FACTORIO_SPECS["inserter"]["active_power_kw"]
        + fast_inserters * FACTORIO_SPECS["fast-inserter"]["active_power_kw"]
        + long_inserters * FACTORIO_SPECS["long-handed-inserter"]["active_power_kw"]
        + stack_inserters * FACTORIO_SPECS["stack-inserter"]["active_power_kw"]
        + lamps * FACTORIO_SPECS["small-lamp"]["power_draw_kw"]
    )
    electric_consumers_mw = electric_consumers_kw / 1000.0

    # Chemical fuel draw for burner equipment (furnaces + boilers)
    furnace_fuel_kw = (stone_furnaces + steel_furnaces) * 90.0
    boiler_fuel_kw = boilers * 1800.0
    total_burner_fuel_mw = (furnace_fuel_kw + boiler_fuel_kw) / 1000.0

    # Logistics infrastructure counts
    belts_count = sum(counts.get(b, 0) for b in ["transport-belt", "fast-transport-belt", "express-transport-belt"])
    total_inserters = inserters + fast_inserters + long_inserters + stack_inserters
    storage_chests = sum(counts.get(c, 0) for c in ["wooden-chest", "iron-chest", "steel-chest", "passive-provider-chest", "storage-chest"])

    # Recipe breakdown if present
    recipe_counts = {}
    if "recipe" in df.columns:
        recipe_counts = df["recipe"].dropna().value_counts().to_dict()

    # Spatial bounding box
    bbox = {
        "min_x": float(df["x"].min()) if "x" in df.columns and df["x"].notna().any() else 0.0,
        "max_x": float(df["x"].max()) if "x" in df.columns and df["x"].notna().any() else 0.0,
        "min_y": float(df["y"].min()) if "y" in df.columns and df["y"].notna().any() else 0.0,
        "max_y": float(df["y"].max()) if "y" in df.columns and df["y"].notna().any() else 0.0,
    }
    bbox["width"] = round(bbox["max_x"] - bbox["min_x"], 2)
    bbox["height"] = round(bbox["max_y"] - bbox["min_y"], 2)

    return {
        "total_entities": len(df),
        "furnaces": {
            "total_count": furnace_count,
            "stone_furnaces": stone_furnaces,
            "steel_furnaces": steel_furnaces,
            "electric_furnaces": electric_furnaces,
            "plate_production_rate_per_sec": round(total_smelting_rate, 2),
            "equivalent_full_yellow_belts": round(equivalent_yellow_belts, 2),
        },
        "manufacturing": {
            "total_units": total_manufacturing_units,
            "assembling_machine_1": am1,
            "assembling_machine_2": am2,
            "assembling_machine_3": am3,
            "chemical_plants": chem_plants,
            "oil_refineries": oil_refineries,
            "labs": labs,
        },
        "power_grid": {
            "boilers_count": boilers,
            "steam_engines_count": steam_engines,
            "power_generation_mw": round(power_generation_mw, 2),
            "peak_electric_demand_mw": round(electric_consumers_mw, 2),
            "peak_electric_demand_kw": round(electric_consumers_kw, 2),
            "grid_margin_mw": round(power_generation_mw - electric_consumers_mw, 2),
            "burner_fuel_mw": round(total_burner_fuel_mw, 2),
        },
        "logistics": {
            "belts_count": belts_count,
            "inserters_count": total_inserters,
            "splitters_count": counts.get("splitter", 0),
            "underground_belts_count": counts.get("underground-belt", 0),
            "electric_poles_count": sum(counts.get(p, 0) for p in ["small-electric-pole", "medium-electric-pole", "big-electric-pole", "substation"]),
            "storage_chests_count": storage_chests,
        },
        "recipe_breakdown": recipe_counts,
        "bounding_box": bbox,
    }
