from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Machine:
    machine_id: str
    input_node_id: str
    output_node_id: str
    process_time_mean_min: float
    capacity_mgz: int
    active: bool


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    rule: str
    robot_count: int
    handling_time_sec: float
    fail_prob: float
    failure_recovery_time_sec: float
    traffic_constraints_enabled: bool
    gripper_scenario: str
    replications: int
    random_seed: int
    process_time_distribution: str
    active: bool
    rejected_reason: str
    process_time_min_min: float
    process_time_mode_min: float
    process_time_max_min: float


def load_machines(path: Path) -> list[Machine]:
    machines: list[Machine] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            machines.append(
                Machine(
                    machine_id=row["machine_id"],
                    input_node_id=row["input_node_id"],
                    output_node_id=row["output_node_id"],
                    process_time_mean_min=float(row["process_time_mean_min"]),
                    capacity_mgz=int(row["capacity_mgz"]),
                    active=row["active"] == "1",
                )
            )
    return [machine for machine in machines if machine.active]


def load_scenarios(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scenarios.append(
                Scenario(
                    scenario_id=row["scenario_id"],
                    rule=row["rule"],
                    robot_count=int(row["robot_count"]),
                    handling_time_sec=float(row["handling_time_sec"]),
                    fail_prob=float(row["fail_prob"]),
                    failure_recovery_time_sec=float(row.get("failure_recovery_time_sec") or 600.0),
                    traffic_constraints_enabled=row["traffic_constraints_enabled"] == "1",
                    gripper_scenario=row["gripper_scenario"],
                    replications=int(row["replications"]),
                    random_seed=int(row["random_seed"]),
                    process_time_distribution=row.get("process_time_distribution", "fixed"),
                    active=row.get("active", "1") == "1",
                    rejected_reason=row.get("rejected_reason", ""),
                    process_time_min_min=float(row.get("process_time_min_min") or 90.0),
                    process_time_mode_min=float(row.get("process_time_mode_min") or 105.0),
                    process_time_max_min=float(row.get("process_time_max_min") or 120.0),
                )
            )
    return [scenario for scenario in scenarios if scenario.active]
