"""Factorio Blueprint and Blueprint Book Parsing, Decoding, and Entity Instrumentation Module."""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd


def decode_blueprint(blueprint_string: str) -> Dict[str, Any]:
    """Decode a Factorio blueprint or blueprint-book string into a Python dictionary.

    Factorio encodes blueprints as:
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
    """Extract a summary table of all stages and blueprints in a blueprint book."""
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
    # Smelting
    "stone-furnace": {
        "domain": "Smelting Columns",
        "display_name": "Stone Furnace",
        "crafting_speed": 1.0,
        "energy_type": "burner",
        "power_draw_kw": 90.0,
        "pollution": 2.0,
        "base_recipe_time": 3.2,
    },
    "steel-furnace": {
        "domain": "Smelting Columns",
        "display_name": "Steel Furnace",
        "crafting_speed": 2.0,
        "energy_type": "burner",
        "power_draw_kw": 90.0,
        "pollution": 4.0,
        "base_recipe_time": 3.2,
    },
    "electric-furnace": {
        "domain": "Smelting Columns",
        "display_name": "Electric Furnace",
        "crafting_speed": 2.0,
        "energy_type": "electric",
        "power_draw_kw": 180.0,
        "pollution": 1.0,
        "base_recipe_time": 3.2,
    },
    # Assembling & Manufacturing
    "assembling-machine-1": {
        "domain": "Manufacturing",
        "display_name": "Assembling Machine 1",
        "crafting_speed": 0.5,
        "energy_type": "electric",
        "power_draw_kw": 75.0,
    },
    "assembling-machine-2": {
        "domain": "Manufacturing",
        "display_name": "Assembling Machine 2",
        "crafting_speed": 0.75,
        "energy_type": "electric",
        "power_draw_kw": 150.0,
    },
    "assembling-machine-3": {
        "domain": "Manufacturing",
        "display_name": "Assembling Machine 3",
        "crafting_speed": 1.25,
        "energy_type": "electric",
        "power_draw_kw": 375.0,
    },
    "chemical-plant": {
        "domain": "Manufacturing",
        "display_name": "Chemical Plant",
        "crafting_speed": 1.0,
        "energy_type": "electric",
        "power_draw_kw": 210.0,
    },
    "oil-refinery": {
        "domain": "Manufacturing",
        "display_name": "Oil Refinery",
        "crafting_speed": 1.0,
        "energy_type": "electric",
        "power_draw_kw": 420.0,
    },
    "lab": {
        "domain": "Manufacturing",
        "display_name": "Research Lab",
        "energy_type": "electric",
        "power_draw_kw": 60.0,
    },
    "rocket-silo": {
        "domain": "Advanced Aerospace",
        "display_name": "Rocket Silo",
        "energy_type": "electric",
        "power_draw_kw": 250.0,
    },
    "cargo-landing-pad": {
        "domain": "Advanced Aerospace",
        "display_name": "Cargo Landing Pad",
        "energy_type": "electric",
        "power_draw_kw": 50.0,
    },
    # Power Generation
    "boiler": {
        "domain": "Power Grid",
        "display_name": "Steam Boiler",
        "energy_type": "burner",
        "power_output_kw": 1800.0,  # 1.8 MW steam output
        "fuel_consumption_kw": 1800.0,
    },
    "steam-engine": {
        "domain": "Power Grid",
        "display_name": "Steam Engine",
        "power_output_kw": 900.0,   # 900 kW electric output
    },
    "steam-turbine": {
        "domain": "Power Grid",
        "display_name": "Steam Turbine",
        "power_output_kw": 5820.0,
    },
    "solar-panel": {
        "domain": "Power Grid",
        "display_name": "Solar Panel",
        "power_output_kw": 60.0,
    },
    "accumulator": {
        "domain": "Power Grid",
        "display_name": "Accumulator",
        "capacity_mj": 5.0,
        "max_power_kw": 300.0,
    },
    # Inserters
    "burner-inserter": {
        "domain": "Logistics Array",
        "display_name": "Burner Inserter",
        "energy_type": "burner",
        "active_power_kw": 0.0,
    },
    "inserter": {
        "domain": "Logistics Array",
        "display_name": "Standard Inserter",
        "energy_type": "electric",
        "active_power_kw": 13.0,
        "drain_power_kw": 0.4,
    },
    "fast-inserter": {
        "domain": "Logistics Array",
        "display_name": "Fast Inserter",
        "energy_type": "electric",
        "active_power_kw": 46.0,
        "drain_power_kw": 0.5,
    },
    "long-handed-inserter": {
        "domain": "Logistics Array",
        "display_name": "Long-Handed Inserter",
        "energy_type": "electric",
        "active_power_kw": 20.0,
        "drain_power_kw": 0.4,
    },
    "filter-inserter": {
        "domain": "Logistics Array",
        "display_name": "Filter Inserter",
        "energy_type": "electric",
        "active_power_kw": 46.0,
        "drain_power_kw": 0.5,
    },
    "stack-inserter": {
        "domain": "Logistics Array",
        "display_name": "Stack Inserter",
        "energy_type": "electric",
        "active_power_kw": 132.0,
        "drain_power_kw": 1.0,
    },
    "stack-filter-inserter": {
        "domain": "Logistics Array",
        "display_name": "Stack Filter Inserter",
        "energy_type": "electric",
        "active_power_kw": 132.0,
        "drain_power_kw": 1.0,
    },
    # Belts & Splitters
    "transport-belt": {
        "domain": "Logistics Array",
        "display_name": "Transport Belt",
        "throughput_items_per_sec": 15.0,
    },
    "fast-transport-belt": {
        "domain": "Logistics Array",
        "display_name": "Fast Transport Belt",
        "throughput_items_per_sec": 30.0,
    },
    "express-transport-belt": {
        "domain": "Logistics Array",
        "display_name": "Express Transport Belt",
        "throughput_items_per_sec": 45.0,
    },
    "underground-belt": {
        "domain": "Logistics Array",
        "display_name": "Underground Belt",
        "throughput_items_per_sec": 15.0,
    },
    "fast-underground-belt": {
        "domain": "Logistics Array",
        "display_name": "Fast Underground Belt",
        "throughput_items_per_sec": 30.0,
    },
    "express-underground-belt": {
        "domain": "Logistics Array",
        "display_name": "Express Underground Belt",
        "throughput_items_per_sec": 45.0,
    },
    "splitter": {
        "domain": "Logistics Array",
        "display_name": "Splitter",
        "throughput_items_per_sec": 15.0,
    },
    "fast-splitter": {
        "domain": "Logistics Array",
        "display_name": "Fast Splitter",
        "throughput_items_per_sec": 30.0,
    },
    "express-splitter": {
        "domain": "Logistics Array",
        "display_name": "Express Splitter",
        "throughput_items_per_sec": 45.0,
    },
    # Storage Chests
    "wooden-chest": {
        "domain": "Logistics Array",
        "display_name": "Wooden Chest",
        "capacity_slots": 16,
    },
    "iron-chest": {
        "domain": "Logistics Array",
        "display_name": "Iron Chest",
        "capacity_slots": 32,
    },
    "steel-chest": {
        "domain": "Logistics Array",
        "display_name": "Steel Chest",
        "capacity_slots": 48,
    },
    "passive-provider-chest": {
        "domain": "Logistics Array",
        "display_name": "Passive Provider Chest",
        "capacity_slots": 48,
    },
    "active-provider-chest": {
        "domain": "Logistics Array",
        "display_name": "Active Provider Chest",
        "capacity_slots": 48,
    },
    "storage-chest": {
        "domain": "Logistics Array",
        "display_name": "Storage Chest",
        "capacity_slots": 48,
    },
    "buffer-chest": {
        "domain": "Logistics Array",
        "display_name": "Buffer Chest",
        "capacity_slots": 48,
    },
    "requester-chest": {
        "domain": "Logistics Array",
        "display_name": "Requester Chest",
        "capacity_slots": 48,
    },
    # Fluid Handling
    "pipe": {
        "domain": "Fluid Network",
        "display_name": "Pipe",
    },
    "pipe-to-ground": {
        "domain": "Fluid Network",
        "display_name": "Pipe-to-Ground",
    },
    "storage-tank": {
        "domain": "Fluid Network",
        "display_name": "Storage Tank",
        "capacity_fluid": 25000.0,
    },
    "pump": {
        "domain": "Fluid Network",
        "display_name": "Fluid Pump",
        "power_draw_kw": 29.0,
        "throughput_fluid_per_sec": 1200.0,
    },
    # Electrical Poles
    "small-electric-pole": {
        "domain": "Power Grid",
        "display_name": "Small Electric Pole",
    },
    "medium-electric-pole": {
        "domain": "Power Grid",
        "display_name": "Medium Electric Pole",
    },
    "big-electric-pole": {
        "domain": "Power Grid",
        "display_name": "Big Electric Pole",
    },
    "substation": {
        "domain": "Power Grid",
        "display_name": "Substation",
    },
    # Automation & Auxiliary
    "small-lamp": {
        "domain": "Auxiliary",
        "display_name": "Small Lamp",
        "power_draw_kw": 5.0,
    },
    "roboport": {
        "domain": "Advanced Systems",
        "display_name": "Roboport",
        "power_draw_kw": 50.0,
        "charging_power_kw": 1000.0,
    },
    "radar": {
        "domain": "Advanced Systems",
        "display_name": "Radar",
        "power_draw_kw": 300.0,
    },
    "constant-combinator": {
        "domain": "Circuit Network",
        "display_name": "Constant Combinator",
    },
    "arithmetic-combinator": {
        "domain": "Circuit Network",
        "display_name": "Arithmetic Combinator",
        "power_draw_kw": 1.0,
    },
    "decider-combinator": {
        "domain": "Circuit Network",
        "display_name": "Decider Combinator",
        "power_draw_kw": 1.0,
    },
}


def analyze_blueprint_layout(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate equipment metrics, theoretical production capacities,
    and power requirements based on the blueprint entity dataset.
    """
    if df.empty or "name" not in df.columns:
        return {"error": "Empty or invalid blueprint dataframe"}

    counts = df["name"].value_counts().to_dict()

    # 1. Smelting stats & throughput capacity
    stone_furnaces = counts.get("stone-furnace", 0)
    steel_furnaces = counts.get("steel-furnace", 0)
    electric_furnaces = counts.get("electric-furnace", 0)
    furnace_count = stone_furnaces + steel_furnaces + electric_furnaces

    stone_speed = FACTORIO_SPECS["stone-furnace"]["crafting_speed"]
    steel_speed = FACTORIO_SPECS["steel-furnace"]["crafting_speed"]
    electric_speed = FACTORIO_SPECS["electric-furnace"]["crafting_speed"]
    base_time = FACTORIO_SPECS["stone-furnace"]["base_recipe_time"]

    stone_rate = (stone_speed / base_time) * stone_furnaces
    steel_rate = (steel_speed / base_time) * steel_furnaces
    electric_rate = (electric_speed / base_time) * electric_furnaces
    total_smelting_rate = stone_rate + steel_rate + electric_rate

    yellow_belt_cap = FACTORIO_SPECS["transport-belt"]["throughput_items_per_sec"]
    equivalent_yellow_belts = total_smelting_rate / yellow_belt_cap

    furnace_breakdown = {
        k: counts.get(k, 0)
        for k in ["stone-furnace", "steel-furnace", "electric-furnace"]
        if counts.get(k, 0) > 0
    }

    # 2. Manufacturing units
    am1 = counts.get("assembling-machine-1", 0)
    am2 = counts.get("assembling-machine-2", 0)
    am3 = counts.get("assembling-machine-3", 0)
    chem_plants = counts.get("chemical-plant", 0)
    oil_refineries = counts.get("oil-refinery", 0)
    labs = counts.get("lab", 0)
    total_manufacturing_units = am1 + am2 + am3 + chem_plants + oil_refineries

    mfg_breakdown = {
        k: counts.get(k, 0)
        for k in [
            "assembling-machine-1",
            "assembling-machine-2",
            "assembling-machine-3",
            "chemical-plant",
            "oil-refinery",
            "lab",
        ]
        if counts.get(k, 0) > 0
    }

    # 3. Power Generation Capacity
    boilers = counts.get("boiler", 0)
    steam_engines = counts.get("steam-engine", 0)
    steam_turbines = counts.get("steam-turbine", 0)
    solar_panels = counts.get("solar-panel", 0)
    power_generation_kw = (
        steam_engines * FACTORIO_SPECS["steam-engine"]["power_output_kw"]
        + steam_turbines * FACTORIO_SPECS.get("steam-turbine", {}).get("power_output_kw", 5820.0)
        + solar_panels * FACTORIO_SPECS.get("solar-panel", {}).get("power_output_kw", 60.0)
    )
    power_generation_mw = power_generation_kw / 1000.0

    # 4. Electric Consumers
    inserters = counts.get("inserter", 0)
    fast_inserters = counts.get("fast-inserter", 0)
    long_inserters = counts.get("long-handed-inserter", 0)
    stack_inserters = counts.get("stack-inserter", 0)
    filter_inserters = counts.get("filter-inserter", 0)
    stack_filter_inserters = counts.get("stack-filter-inserter", 0)
    burner_inserters = counts.get("burner-inserter", 0)
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
        + filter_inserters * FACTORIO_SPECS.get("filter-inserter", {}).get("active_power_kw", 46.0)
        + stack_filter_inserters * FACTORIO_SPECS.get("stack-filter-inserter", {}).get("active_power_kw", 132.0)
        + lamps * FACTORIO_SPECS["small-lamp"]["power_draw_kw"]
    )
    electric_consumers_mw = electric_consumers_kw / 1000.0

    electrified_consumer_count = (
        am1 + am2 + am3 + chem_plants + oil_refineries + labs
        + inserters + fast_inserters + long_inserters + stack_inserters
        + filter_inserters + stack_filter_inserters + lamps
    )

    # Chemical fuel draw for burner equipment (furnaces + boilers)
    furnace_fuel_kw = (stone_furnaces + steel_furnaces) * FACTORIO_SPECS["stone-furnace"]["power_draw_kw"]
    boiler_fuel_kw = boilers * FACTORIO_SPECS["boiler"]["power_output_kw"]
    total_burner_fuel_mw = (furnace_fuel_kw + boiler_fuel_kw) / 1000.0

    # 5. Logistics infrastructure
    belt_types = ["transport-belt", "fast-transport-belt", "express-transport-belt"]
    underground_types = ["underground-belt", "fast-underground-belt", "express-underground-belt"]
    splitter_types = ["splitter", "fast-splitter", "express-splitter"]

    transport_belts = sum(counts.get(b, 0) for b in belt_types)
    underground_belts = sum(counts.get(u, 0) for u in underground_types)
    splitters = sum(counts.get(s, 0) for s in splitter_types)
    total_belt_segments = transport_belts + underground_belts + splitters

    belt_breakdown = {
        k: counts.get(k, 0)
        for k in belt_types + underground_types + splitter_types
        if counts.get(k, 0) > 0
    }

    total_inserters = (
        inserters + fast_inserters + long_inserters + stack_inserters
        + filter_inserters + stack_filter_inserters + burner_inserters
    )
    inserter_breakdown = {
        k: counts.get(k, 0)
        for k in [
            "inserter",
            "fast-inserter",
            "long-handed-inserter",
            "stack-inserter",
            "filter-inserter",
            "stack-filter-inserter",
            "burner-inserter",
        ]
        if counts.get(k, 0) > 0
    }

    chest_types = [
        "wooden-chest",
        "iron-chest",
        "steel-chest",
        "passive-provider-chest",
        "active-provider-chest",
        "storage-chest",
        "buffer-chest",
        "requester-chest",
    ]
    chest_breakdown = {c: counts.get(c, 0) for c in chest_types if counts.get(c, 0) > 0}
    storage_chests = sum(chest_breakdown.values())
    total_storage_slots = sum(
        count * FACTORIO_SPECS.get(c, {}).get("capacity_slots", 48)
        for c, count in chest_breakdown.items()
    )

    pole_types = [
        "small-electric-pole",
        "medium-electric-pole",
        "big-electric-pole",
        "substation",
    ]
    pole_breakdown = {p: counts.get(p, 0) for p in pole_types if counts.get(p, 0) > 0}
    total_poles = sum(pole_breakdown.values())

    # 6. Fluids
    fluid_types = ["pipe", "pipe-to-ground", "storage-tank", "pump"]
    fluid_breakdown = {f: counts.get(f, 0) for f in fluid_types if counts.get(f, 0) > 0}
    total_fluids = sum(fluid_breakdown.values())

    # 7. Advanced
    adv_types = [
        "roboport",
        "rocket-silo",
        "cargo-landing-pad",
        "radar",
        "constant-combinator",
        "arithmetic-combinator",
        "decider-combinator",
    ]
    adv_breakdown = {a: counts.get(a, 0) for a in adv_types if counts.get(a, 0) > 0}
    total_adv = sum(adv_breakdown.values())

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

    grid_margin_mw = round(power_generation_mw - electric_consumers_mw, 2)

    return {
        "total_entities": len(df),
        "furnaces": {
            "total_count": furnace_count,
            "stone_furnaces": stone_furnaces,
            "steel_furnaces": steel_furnaces,
            "electric_furnaces": electric_furnaces,
            "plate_production_rate_per_sec": round(total_smelting_rate, 2),
            "equivalent_full_yellow_belts": round(equivalent_yellow_belts, 2),
            "breakdown": furnace_breakdown,
        },
        "manufacturing": {
            "total_units": total_manufacturing_units,
            "assembling_machine_1": am1,
            "assembling_machine_2": am2,
            "assembling_machine_3": am3,
            "chemical_plants": chem_plants,
            "oil_refineries": oil_refineries,
            "labs": labs,
            "breakdown": mfg_breakdown,
        },
        "power_grid": {
            "boilers_count": boilers,
            "steam_engines_count": steam_engines,
            "steam_turbines_count": steam_turbines,
            "solar_panels_count": solar_panels,
            "power_generation_mw": round(power_generation_mw, 2),
            "peak_electric_demand_mw": round(electric_consumers_mw, 2),
            "peak_electric_demand_kw": round(electric_consumers_kw, 2),
            "grid_margin_mw": grid_margin_mw,
            "burner_fuel_mw": round(total_burner_fuel_mw, 2),
            "electrified_consumers_count": electrified_consumer_count,
        },
        "logistics": {
            "belts_count": transport_belts,
            "transport_belts_count": transport_belts,
            "underground_belts_count": underground_belts,
            "splitters_count": splitters,
            "total_belt_segments": total_belt_segments,
            "inserters_count": total_inserters,
            "storage_chests_count": storage_chests,
            "total_storage_slots": total_storage_slots,
            "electric_poles_count": total_poles,
            "belt_breakdown": belt_breakdown,
            "inserter_breakdown": inserter_breakdown,
            "chest_breakdown": chest_breakdown,
            "pole_breakdown": pole_breakdown,
        },
        "fluids": {
            "total_count": total_fluids,
            "breakdown": fluid_breakdown,
        },
        "advanced": {
            "total_count": total_adv,
            "breakdown": adv_breakdown,
        },
        "recipe_breakdown": recipe_counts,
        "bounding_box": bbox,
    }


def generate_metrics_table(
    data: Union[pd.DataFrame, Dict[str, Any]],
    include_empty: bool = False,
    include_auxiliary: bool = False,
) -> pd.DataFrame:
    """Transform analyzed blueprint metrics into a structured DataFrame.

    The function computes all metrics, units, and notes directly from parsed entity data.

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, Any]]
        Normalized entity DataFrame or metrics dictionary from analyze_blueprint_layout().
    include_empty : bool, default False
        If True, include domains with zero equipment.
    include_auxiliary : bool, default False
        If True, include fluid, power distribution, and advanced infrastructure domains.

    Returns
    -------
    pd.DataFrame
        Formatted table with columns: 'System Domain', 'Specification Metric', 'Value', 'Notes'.
    """
    if isinstance(data, pd.DataFrame):
        metrics = analyze_blueprint_layout(data)
    elif isinstance(data, dict):
        if "furnaces" in data and "power_grid" in data:
            metrics = data
        else:
            metrics = analyze_blueprint_layout(pd.DataFrame())
    else:
        raise TypeError(f"Expected pandas DataFrame or metrics dict, got {type(data)}")

    furnaces = metrics.get("furnaces", {})
    mfg = metrics.get("manufacturing", {})
    power = metrics.get("power_grid", {})
    logistics = metrics.get("logistics", {})
    fluids = metrics.get("fluids", {})
    advanced = metrics.get("advanced", {})

    rows: List[Dict[str, Any]] = []

    def format_breakdown(breakdown: Dict[str, int], default_name: str = "elements") -> str:
        parts = []
        for k, v in breakdown.items():
            if v > 0:
                disp = FACTORIO_SPECS.get(k, {}).get("display_name", k)
                parts.append(f"{v:,} {disp}")
        return " + ".join(parts) if parts else f"No active {default_name}"

    # 1. Smelting: Total Furnaces
    total_furnaces = furnaces.get("total_count", 0)
    if total_furnaces > 0 or include_empty:
        f_breakdown = furnaces.get("breakdown", {})
        rows.append({
            "System Domain": "Smelting Columns",
            "Specification Metric": "Total Furnaces",
            "Value": total_furnaces,
            "Notes": format_breakdown(f_breakdown, "furnaces"),
        })

        # 2. Smelting: Nominal Plate Output
        rate = furnaces.get("plate_production_rate_per_sec", 0.0)
        belts_eq = furnaces.get("equivalent_full_yellow_belts", 0.0)
        belt_speed = FACTORIO_SPECS.get("transport-belt", {}).get("throughput_items_per_sec", 15.0)
        rows.append({
            "System Domain": "Smelting Columns",
            "Specification Metric": "Nominal Plate Output",
            "Value": f"{rate:.2f} plates/s",
            "Notes": f"{belts_eq:.2f} full yellow belts ({belt_speed:.0f}/s each)",
        })

    # 3. Manufacturing: Assembling Machines
    am1 = mfg.get("assembling_machine_1", 0)
    am2 = mfg.get("assembling_machine_2", 0)
    am3 = mfg.get("assembling_machine_3", 0)
    total_assemblers = am1 + am2 + am3
    if total_assemblers > 0 or include_empty:
        am_parts = []
        if am1 > 0:
            am_parts.append(f"{am1} AM1")
        if am2 > 0:
            am_parts.append(f"{am2} AM2")
        if am3 > 0:
            am_parts.append(f"{am3} AM3")
        rows.append({
            "System Domain": "Manufacturing",
            "Specification Metric": "Assembling Machines",
            "Value": total_assemblers,
            "Notes": " + ".join(am_parts) if am_parts else "No assembling machines",
        })

    # 4. Manufacturing: Chemical & Refining
    chem_plants = mfg.get("chemical_plants", 0)
    oil_refineries = mfg.get("oil_refineries", 0)
    total_chem_ref = chem_plants + oil_refineries
    if total_chem_ref > 0 or include_empty:
        cr_parts = []
        if chem_plants > 0:
            cr_parts.append(f"{chem_plants} Chem Plants")
        if oil_refineries > 0:
            cr_parts.append(f"{oil_refineries} Refineries")
        rows.append({
            "System Domain": "Manufacturing",
            "Specification Metric": "Chemical & Refining",
            "Value": total_chem_ref,
            "Notes": ", ".join(cr_parts) if cr_parts else "No chemical or refining units",
        })

    # 5. Manufacturing: Research Laboratories
    labs = mfg.get("labs", 0)
    if labs > 0 or include_empty:
        lab_kw = FACTORIO_SPECS.get("lab", {}).get("power_draw_kw", 60.0)
        total_lab_mw = (labs * lab_kw) / 1000.0
        rows.append({
            "System Domain": "Manufacturing",
            "Specification Metric": "Research Laboratories",
            "Value": labs,
            "Notes": f"{lab_kw:.0f} kW power draw per lab ({total_lab_mw:.2f} MW total)",
        })

    # 6. Power Grid: Power Plant Generation
    p_gen_mw = power.get("power_generation_mw", 0.0)
    boilers = power.get("boilers_count", 0)
    engines = power.get("steam_engines_count", 0)
    turbines = power.get("steam_turbines_count", 0)
    solar = power.get("solar_panels_count", 0)
    if p_gen_mw > 0 or boilers > 0 or engines > 0 or include_empty:
        if boilers > 0:
            ratio = (engines / boilers) if boilers > 0 else 0.0
            p_notes = f"{boilers} Boilers + {engines} Steam Engines (1:{ratio:.1f} ratio, ideal 1:2)"
        elif solar > 0:
            p_notes = f"{solar} Solar Panels ({p_gen_mw:.2f} MW)"
        else:
            p_notes = f"{engines} Steam Engines ({p_gen_mw:.2f} MW)"
        rows.append({
            "System Domain": "Power Grid",
            "Specification Metric": "Power Plant Generation",
            "Value": f"{p_gen_mw:.2f} MW",
            "Notes": p_notes,
        })

    # 7. Power Grid: Peak Electrical Demand
    p_demand_mw = power.get("peak_electric_demand_mw", 0.0)
    consumers_count = power.get("electrified_consumers_count", 0)
    if p_demand_mw > 0 or include_empty:
        c_note = f"At 100% simultaneous equipment duty across {consumers_count:,} active units" if consumers_count > 0 else "At 100% simultaneous equipment duty"
        rows.append({
            "System Domain": "Power Grid",
            "Specification Metric": "Peak Electrical Demand",
            "Value": f"{p_demand_mw:.2f} MW",
            "Notes": c_note,
        })

    # 8. Power Grid: Grid Reserve Margin
    margin_mw = power.get("grid_margin_mw", 0.0)
    if p_demand_mw > 0 or p_gen_mw > 0 or include_empty:
        if margin_mw < 0:
            overload_pct = (abs(margin_mw) / p_demand_mw * 100.0) if p_demand_mw > 0 else 0.0
            margin_note = f"Deficit: {margin_mw:.2f} MW ({overload_pct:.1f}% overload under full duty)"
        else:
            surplus_pct = (margin_mw / p_gen_mw * 100.0) if p_gen_mw > 0 else 0.0
            margin_note = f"Surplus: +{margin_mw:.2f} MW ({surplus_pct:.1f}% operational reserve buffer)"
        rows.append({
            "System Domain": "Power Grid",
            "Specification Metric": "Grid Reserve Margin",
            "Value": f"{margin_mw:+.2f} MW",
            "Notes": margin_note,
        })

    # 9. Logistics: Conveyor Belts
    belts_count = logistics.get("belts_count", 0)
    total_belt_segments = logistics.get("total_belt_segments", belts_count)
    if total_belt_segments > 0 or include_empty:
        b_breakdown = logistics.get("belt_breakdown", {})
        b_notes = format_breakdown(b_breakdown, "belts") if b_breakdown else f"{belts_count:,} yellow / fast belts"
        rows.append({
            "System Domain": "Logistics Array",
            "Specification Metric": "Conveyor Belts",
            "Value": total_belt_segments,
            "Notes": b_notes,
        })

    # 10. Logistics: Inserter Arms
    inserters_count = logistics.get("inserters_count", 0)
    if inserters_count > 0 or include_empty:
        i_breakdown = logistics.get("inserter_breakdown", {})
        i_notes = format_breakdown(i_breakdown, "inserters") if i_breakdown else "Standard, fast, and long-handed material manipulators"
        rows.append({
            "System Domain": "Logistics Array",
            "Specification Metric": "Inserter Arms",
            "Value": inserters_count,
            "Notes": i_notes,
        })

    # 11. Logistics: Storage Chests
    chests_count = logistics.get("storage_chests_count", 0)
    if chests_count > 0 or include_empty:
        c_breakdown = logistics.get("chest_breakdown", {})
        slots = logistics.get("total_storage_slots", 0)
        slot_suffix = f" ({slots:,} buffer slots total)" if slots > 0 else ""
        c_notes = (format_breakdown(c_breakdown, "chests") + slot_suffix) if c_breakdown else f"{chests_count:,} storage chests"
        rows.append({
            "System Domain": "Logistics Array",
            "Specification Metric": "Bulk Storage Chests",
            "Value": chests_count,
            "Notes": c_notes,
        })

    # Optional Auxiliary Infrastructure
    if include_auxiliary:
        # 12. Fluid Piping Network
        fluid_total = fluids.get("total_count", 0)
        if fluid_total > 0:
            fl_breakdown = fluids.get("breakdown", {})
            rows.append({
                "System Domain": "Fluid Network",
                "Specification Metric": "Piping & Fluid Buffers",
                "Value": fluid_total,
                "Notes": format_breakdown(fl_breakdown, "fluid elements"),
            })

        # 13. Electrical Transmission Poles
        poles_total = logistics.get("electric_poles_count", 0)
        if poles_total > 0:
            p_breakdown = logistics.get("pole_breakdown", {})
            rows.append({
                "System Domain": "Power Grid",
                "Specification Metric": "Transmission Poles",
                "Value": poles_total,
                "Notes": format_breakdown(p_breakdown, "poles and substations"),
            })

        # 14. Advanced Systems
        adv_total = advanced.get("total_count", 0)
        if adv_total > 0:
            a_breakdown = advanced.get("breakdown", {})
            rows.append({
                "System Domain": "Advanced Systems",
                "Specification Metric": "Robotics & Space Facilities",
                "Value": adv_total,
                "Notes": format_breakdown(a_breakdown, "advanced facilities"),
            })

    return pd.DataFrame(rows)
