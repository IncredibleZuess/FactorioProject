"""Factorio Blueprint Parsing, Decoding, and Entity Instrumentation Module."""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, Optional, Tuple
import pandas as pd


def decode_blueprint(blueprint_string: str) -> Dict[str, Any]:
    """Decode a Factorio blueprint string into a Python dictionary.

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


def blueprint_to_dataframe(blueprint_dict: Dict[str, Any]) -> pd.DataFrame:
    """Extract entities from blueprint dictionary into a normalized pandas DataFrame."""
    bp = blueprint_dict.get("blueprint", blueprint_dict)
    entities = bp.get("entities", [])
    if not entities:
        return pd.DataFrame()

    df = pd.json_normalize(entities)
    # Ensure standard column names if position is nested
    if "position.x" in df.columns and "x" not in df.columns:
        df["x"] = df["position.x"]
        df["y"] = df["position.y"]
    elif "position" in df.columns and "x" not in df.columns:
        df["x"] = df["position"].apply(lambda p: p.get("x") if isinstance(p, dict) else None)
        df["y"] = df["position"].apply(lambda p: p.get("y") if isinstance(p, dict) else None)

    return df


# Standard Factorio baseline machine constants (vanilla Factorio)
FACTORIO_SPECS = {
    "stone-furnace": {
        "crafting_speed": 1.0,
        "energy_type": "burner",
        "power_draw_kw": 90.0,  # 90 kW chemical fuel
        "pollution": 2.0,
        "width": 2,
        "height": 2,
        "base_recipe_time": 3.2,  # seconds per iron/copper plate
        "output_per_cycle": 1,
    },
    "steel-furnace": {
        "crafting_speed": 2.0,
        "energy_type": "burner",
        "power_draw_kw": 90.0,
        "pollution": 4.0,
        "width": 2,
        "height": 2,
        "base_recipe_time": 3.2,
        "output_per_cycle": 1,
    },
    "electric-furnace": {
        "crafting_speed": 2.0,
        "energy_type": "electric",
        "power_draw_kw": 180.0,  # 180 kW electrical
        "pollution": 1.0,
        "width": 3,
        "height": 3,
        "base_recipe_time": 3.2,
        "output_per_cycle": 1,
    },
    "assembling-machine-1": {
        "crafting_speed": 0.5,
        "energy_type": "electric",
        "power_draw_kw": 75.0,
        "pollution": 4.0,
        "width": 3,
        "height": 3,
    },
    "assembling-machine-2": {
        "crafting_speed": 0.75,
        "energy_type": "electric",
        "power_draw_kw": 150.0,
        "pollution": 3.0,
        "width": 3,
        "height": 3,
    },
    "inserter": {
        "energy_type": "electric",
        "active_power_kw": 13.0,
        "drain_power_kw": 0.4,
        "throughput_items_per_sec": 1.2,  # single-item swing
    },
    "fast-inserter": {
        "energy_type": "electric",
        "active_power_kw": 46.0,
        "drain_power_kw": 0.5,
        "throughput_items_per_sec": 2.3,
    },
    "transport-belt": {
        "energy_type": "none",
        "power_draw_kw": 0.0,
        "throughput_items_per_sec": 15.0,  # 1 yellow belt = 15 items/sec
    },
    "fast-transport-belt": {
        "energy_type": "none",
        "power_draw_kw": 0.0,
        "throughput_items_per_sec": 30.0,  # red belt = 30 items/sec
    },
    "express-transport-belt": {
        "energy_type": "none",
        "power_draw_kw": 0.0,
        "throughput_items_per_sec": 45.0,  # blue belt = 45 items/sec
    },
    "splitter": {
        "energy_type": "none",
        "power_draw_kw": 0.0,
        "throughput_items_per_sec": 15.0,
    },
    "underground-belt": {
        "energy_type": "none",
        "power_draw_kw": 0.0,
        "throughput_items_per_sec": 15.0,
    },
    "small-electric-pole": {
        "supply_area": 5.0,
        "wire_reach": 7.5,
    },
    "small-lamp": {
        "energy_type": "electric",
        "active_power_kw": 5.0,
        "drain_power_kw": 5.0,
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
    furnace_types = ["stone-furnace", "steel-furnace", "electric-furnace"]
    furnace_count = sum(counts.get(ft, 0) for ft in furnace_types)

    # Calculate throughput capacity for stone furnaces
    stone_furnaces = counts.get("stone-furnace", 0)
    furnace_rate_per_sec = (1.0 / 3.2) * stone_furnaces  # 0.3125 plates/sec per stone furnace
    belt_speed = FACTORIO_SPECS["transport-belt"]["throughput_items_per_sec"]
    belt_saturation_pct = (furnace_rate_per_sec / belt_speed) * 100.0

    # Power requirements
    inserters = counts.get("inserter", 0)
    fast_inserters = counts.get("fast-inserter", 0)
    lamps = counts.get("small-lamp", 0)

    electric_active_kw = (
        inserters * FACTORIO_SPECS["inserter"]["active_power_kw"]
        + fast_inserters * FACTORIO_SPECS["fast-inserter"]["active_power_kw"]
        + lamps * FACTORIO_SPECS["small-lamp"]["active_power_kw"]
    )
    electric_idle_kw = (
        inserters * FACTORIO_SPECS["inserter"]["drain_power_kw"]
        + fast_inserters * FACTORIO_SPECS["fast-inserter"]["drain_power_kw"]
        + lamps * FACTORIO_SPECS["small-lamp"]["drain_power_kw"]
    )

    burner_fuel_kw = stone_furnaces * FACTORIO_SPECS["stone-furnace"]["power_draw_kw"]

    # Spatial bounding box
    bbox = {
        "min_x": float(df["x"].min()) if "x" in df.columns else 0.0,
        "max_x": float(df["x"].max()) if "x" in df.columns else 0.0,
        "min_y": float(df["y"].min()) if "y" in df.columns else 0.0,
        "max_y": float(df["y"].max()) if "y" in df.columns else 0.0,
    }
    bbox["width"] = bbox["max_x"] - bbox["min_x"]
    bbox["height"] = bbox["max_y"] - bbox["min_y"]

    return {
        "entity_counts": counts,
        "furnaces": {
            "total_count": furnace_count,
            "stone_furnaces": stone_furnaces,
            "plate_production_rate_per_sec": round(furnace_rate_per_sec, 4),
            "ore_consumption_rate_per_sec": round(furnace_rate_per_sec, 4),
            "belt_saturation_percentage": round(belt_saturation_pct, 2),
        },
        "power_analysis": {
            "electric_active_kw": round(electric_active_kw, 2),
            "electric_idle_kw": round(electric_idle_kw, 2),
            "burner_fuel_draw_kw": round(burner_fuel_kw, 2),
            "total_energy_draw_kw": round(electric_active_kw + burner_fuel_kw, 2),
        },
        "logistics": {
            "belts_count": counts.get("transport-belt", 0),
            "inserters_count": inserters,
            "splitters_count": counts.get("splitter", 0),
            "underground_belts_count": counts.get("underground-belt", 0),
            "electric_poles_count": counts.get("small-electric-pole", 0),
        },
        "bounding_box": bbox,
    }
