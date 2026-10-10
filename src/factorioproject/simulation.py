"""Discrete-Event Simulation Module using SimPy.

Validates analytical LP and queuing solutions under stochastic operating conditions:
- Poisson and exponential arrival intervals for raw materials
- Stochastic inserter unloading times
- Smelting production with machine downtime disruptions
- Dynamic buffer tracking and starvation risk evaluation
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import simpy


class FactorioSmeltingSimulation:
    """Discrete-Event Simulation of a Factorio smelting and logistics line."""

    def __init__(
        self,
        sim_duration: float = 3600.0,  # 1 hour simulation (seconds)
        num_unloader_inserters: int = 4,
        unloader_service_rate: float = 2.5,  # items/sec per unloader
        batch_arrival_rate: float = 0.5,  # delivery batches per sec
        batch_size_mean: float = 15.0,  # items per batch
        batch_size_std: float = 2.0,
        buffer_capacity: float = 2400.0,  # Steel chest capacity (items)
        initial_buffer: float = 400.0,
        furnace_count: int = 24,
        furnace_craft_time_mean: float = 3.2,  # seconds per item
        furnace_craft_time_std: float = 0.3,
        mtbf: float = 600.0,  # Mean Time Between Failures / belt jams (seconds)
        mttr: float = 30.0,  # Mean Time To Repair (seconds)
        seed: int = 42,
    ):
        self.sim_duration = sim_duration
        self.num_unloaders = num_unloader_inserters
        self.unloader_service_rate = unloader_service_rate
        self.batch_arrival_rate = batch_arrival_rate
        self.batch_size_mean = batch_size_mean
        self.batch_size_std = batch_size_std
        self.buffer_capacity = buffer_capacity
        self.initial_buffer = initial_buffer
        self.furnace_count = furnace_count
        self.furnace_craft_time_mean = furnace_craft_time_mean
        self.furnace_craft_time_std = furnace_craft_time_std
        self.mtbf = mtbf
        self.mttr = mttr
        self.seed = seed

        # State tracking
        self.time_records: List[Dict[str, float]] = []
        self.total_plates_produced: int = 0
        self.starvation_time: float = 0.0
        self.last_buffer_empty_time: Optional[float] = None

    def run(self) -> Dict[str, Any]:
        """Execute the discrete-event simulation."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        env = simpy.Environment()

        # SimPy shared resources
        unloader_resource = simpy.Resource(env, capacity=self.num_unloaders)
        buffer_chest = simpy.Container(env, capacity=self.buffer_capacity, init=self.initial_buffer)
        output_chest = simpy.Container(env, capacity=100000.0, init=0.0)

        # Machine health state: True = working, False = broken down / jammed
        machine_operational = [True]

        # Process 1: Stochastic batch arrivals from extraction sites
        def arrival_process(env: simpy.Environment):
            while True:
                # Exponential inter-arrival time
                inter_arrival = random.expovariate(self.batch_arrival_rate)
                yield env.timeout(inter_arrival)

                # Batch delivery arrives
                batch_size = max(1, int(np.random.normal(self.batch_size_mean, self.batch_size_std)))
                env.process(unload_batch(env, batch_size))

        # Process 2: Unloading batch at unloader stations
        def unload_batch(env: simpy.Environment, batch_size: int):
            with unloader_resource.request() as req:
                yield req
                # Time to unload batch
                unload_time = batch_size / self.unloader_service_rate
                yield env.timeout(unload_time)

                # Deposit into buffer container (bounded by capacity)
                space_available = buffer_chest.capacity - buffer_chest.level
                deposit_amt = min(batch_size, space_available)
                if deposit_amt > 0:
                    yield buffer_chest.put(deposit_amt)

        # Process 3: Smelting furnaces converting ore to plates
        def furnace_worker(env: simpy.Environment, furnace_id: int):
            while True:
                # Check if system is jammed / broken down
                while not machine_operational[0]:
                    yield env.timeout(1.0)

                # Try to take 1 ore from buffer
                t_req = env.now
                yield buffer_chest.get(1)
                wait_time = env.now - t_req
                if wait_time > 0.05:
                    self.starvation_time += wait_time

                # Smelting processing time with variability
                proc_time = max(0.5, np.random.normal(self.furnace_craft_time_mean, self.furnace_craft_time_std))
                yield env.timeout(proc_time)

                yield output_chest.put(1)
                self.total_plates_produced += 1

        # Process 4: Disruptions / Machine downtime
        def breakdown_process(env: simpy.Environment):
            while True:
                time_to_fail = random.expovariate(1.0 / self.mtbf)
                yield env.timeout(time_to_fail)

                # Facility experiences micro-stoppage / jam
                machine_operational[0] = False
                repair_time = random.expovariate(1.0 / self.mttr)
                yield env.timeout(repair_time)
                machine_operational[0] = True

        # Process 5: Periodic time-series logger
        def logger_process(env: simpy.Environment):
            while True:
                self.time_records.append(
                    {
                        "time_sec": env.now,
                        "ore_buffer_level": buffer_chest.level,
                        "cumulative_plates": self.total_plates_produced,
                        "active_unloaders": unloader_resource.count,
                        "is_operational": int(machine_operational[0]),
                    }
                )
                yield env.timeout(10.0)  # Log every 10 seconds

        # Spawn processes
        env.process(arrival_process(env))
        for fid in range(self.furnace_count):
            env.process(furnace_worker(env, fid))
        env.process(breakdown_process(env))
        env.process(logger_process(env))

        # Run simulation
        env.run(until=self.sim_duration)

        # Compile results
        df_log = pd.DataFrame(self.time_records)
        df_log["production_rate_mov_avg"] = (
            df_log["cumulative_plates"].diff() / df_log["time_sec"].diff()
        ).fillna(0.0)

        actual_rate = self.total_plates_produced / self.sim_duration
        theoretical_rate = (1.0 / self.furnace_craft_time_mean) * self.furnace_count
        efficiency = (actual_rate / theoretical_rate) * 100.0

        return {
            "simulation_duration_sec": self.sim_duration,
            "total_plates_produced": self.total_plates_produced,
            "simulated_throughput_per_sec": round(actual_rate, 4),
            "theoretical_throughput_per_sec": round(theoretical_rate, 4),
            "throughput_efficiency_pct": round(efficiency, 2),
            "mean_buffer_level": round(df_log["ore_buffer_level"].mean(), 2),
            "min_buffer_level": round(df_log["ore_buffer_level"].min(), 2),
            "max_buffer_level": round(df_log["ore_buffer_level"].max(), 2),
            "starvation_time_total_sec": round(self.starvation_time, 2),
            "starvation_pct": round((self.starvation_time / (self.sim_duration * self.furnace_count)) * 100.0, 3),
            "time_series_df": df_log,
        }
