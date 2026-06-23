from __future__ import annotations

import csv
import heapq
import itertools
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from data_loader import Machine, Scenario
from graph_model import GraphModel


DEFAULT_START_NODE = "S01"
DEFAULT_INPUT_STOCKER_NODE = "S02"
DEFAULT_COMPLETED_STOCKER_NODE = "S03"
SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0
INITIAL_REMAINING_TIME_MAX_MIN = 120.0
PICK_PLACE_EVENT_COUNT = 6
INPUT_PLACE_HANDLING_EVENT_COUNT = 2


@dataclass
class Task:
    task_id: int
    machine: Machine
    request_time: float


@dataclass
class Robot:
    robot_id: str
    current_node: str
    busy: bool = False
    total_travel_distance_mm: float = 0.0
    total_travel_time_sec: float = 0.0
    total_handling_time_sec: float = 0.0
    total_busy_time_sec: float = 0.0
    task_count: int = 0


@dataclass
class MachineState:
    machine: Machine
    status: str = "PROCESSING"
    process_generation: int = 1
    initial_remaining_time_sec: float = 0.0
    starve_start_time: float | None = None
    total_starve_time_sec: float = 0.0
    starve_count: int = 0
    completed_mgz: int = 0


@dataclass(order=True)
class Event:
    time: float
    seq: int
    event_type: str = field(compare=False)
    payload: object = field(compare=False)


class BaselineSimulation:
    def __init__(
        self,
        graph: GraphModel,
        machines: list[Machine],
        scenario: Scenario,
        simulation_time_hours: float = 24.0,
        start_node: str = DEFAULT_START_NODE,
        input_stocker_node: str = DEFAULT_INPUT_STOCKER_NODE,
        completed_stocker_node: str = DEFAULT_COMPLETED_STOCKER_NODE,
        idle_wait_node: str | None = None,
    ) -> None:
        self.graph = graph
        self.machines = machines
        self.scenario = scenario
        self.simulation_end_time = simulation_time_hours * SECONDS_PER_HOUR
        self.start_node = start_node
        self.input_stocker_node = input_stocker_node
        self.completed_stocker_node = completed_stocker_node
        self.idle_wait_node = idle_wait_node or input_stocker_node

        self.now = 0.0
        self._seq = itertools.count()
        self.events: list[Event] = []
        self.pending_tasks: list[Task] = []
        self.task_id_counter = itertools.count(1)

        self.machine_states = {
            machine.machine_id: MachineState(machine=machine) for machine in machines
        }
        self.random = random.Random(scenario.random_seed)
        self.robots = [
            Robot(robot_id=f"AMR{i + 1:02d}", current_node=start_node)
            for i in range(scenario.robot_count)
        ]

        self.completed_task_wait_times_sec: list[float] = []
        self.completed_response_times_sec: list[float] = []
        self.process_time_samples_sec: list[float] = []
        self.initial_remaining_time_samples_sec: list[float] = []
        self.task_log_rows: list[dict[str, float | str | int]] = []
        self.initial_state_rows: list[dict[str, float | str | int]] = []
        self.process_sample_rows: list[dict[str, float | str | int]] = []
        self.throughput = 0
        self.completed_magazines_delivered_to_stk3 = 0
        self.total_same_position_transfer_edges = 0
        self.total_single_lane_edges = 0
        self.total_bypass_edges = 0
        self.total_two_lane_edges = 0
        self.edge_usage_counts: dict[tuple[str, str], int] = defaultdict(int)

    def schedule(self, time: float, event_type: str, payload: object) -> None:
        heapq.heappush(self.events, Event(time, next(self._seq), event_type, payload))

    def initialize(self) -> None:
        for state in self.machine_states.values():
            initial_remaining_time = self.sample_initial_remaining_time_sec()
            state.initial_remaining_time_sec = initial_remaining_time
            self.initial_state_rows.append(
                {
                    "scenario_id": self.scenario.scenario_id,
                    "machine_id": state.machine.machine_id,
                    "initial_remaining_time_sec": round(initial_remaining_time, 3),
                    "initial_remaining_time_min": round(initial_remaining_time / SECONDS_PER_MINUTE, 3),
                    "initial_distribution": "uniform_0_120_min",
                }
            )
            self.schedule(initial_remaining_time, "process_complete", state.machine.machine_id)

    def run(self) -> dict[str, float | str | int]:
        self.initialize()
        self.try_dispatch()

        while self.events:
            event = heapq.heappop(self.events)
            if event.time > self.simulation_end_time:
                break
            self.now = event.time

            if event.event_type == "process_complete":
                self.handle_process_complete(str(event.payload))
            elif event.event_type == "input_place_complete":
                self.handle_input_place_complete(event.payload)  # type: ignore[arg-type]
            elif event.event_type == "robot_complete":
                self.handle_robot_complete(event.payload)  # type: ignore[arg-type]
            elif event.event_type == "robot_reposition_complete":
                self.handle_robot_reposition_complete(event.payload)  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown event type: {event.event_type}")

            self.try_dispatch()

        self.now = self.simulation_end_time
        self.close_open_starves()
        return self.kpis()

    def handle_process_complete(self, machine_id: str) -> None:
        state = self.machine_states[machine_id]
        if state.status != "PROCESSING":
            return

        state.status = "WAITING_PICKUP"
        state.starve_start_time = self.now
        state.starve_count += 1
        task = Task(
            task_id=next(self.task_id_counter),
            machine=state.machine,
            request_time=self.now,
        )
        self.pending_tasks.append(task)

    def handle_input_place_complete(self, payload: dict[str, object]) -> None:
        task = payload["task"]
        if not isinstance(task, Task):
            raise TypeError("Invalid input-place completion payload")

        state = self.machine_states[task.machine.machine_id]
        if state.starve_start_time is not None:
            wb_starve_time = self.now - state.starve_start_time
            state.total_starve_time_sec += wb_starve_time
            self.completed_response_times_sec.append(wb_starve_time)
        state.starve_start_time = None
        state.status = "PROCESSING"

        state.process_generation += 1
        process_time = self.sample_process_time_sec(
            state.machine,
            process_generation=state.process_generation,
        )
        self.schedule(self.now + process_time, "process_complete", state.machine.machine_id)

    def handle_robot_complete(self, payload: dict[str, object]) -> None:
        robot = payload["robot"]
        task = payload["task"]
        if not isinstance(robot, Robot) or not isinstance(task, Task):
            raise TypeError("Invalid robot completion payload")

        robot.current_node = self.completed_stocker_node
        robot.task_count += 1

        state = self.machine_states[task.machine.machine_id]
        state.completed_mgz += 1
        self.throughput += 1
        self.completed_magazines_delivered_to_stk3 += 1

        if self.idle_wait_node != self.completed_stocker_node:
            _, relocation_distance, relocation_time, relocation_edges = self.graph.shortest_path(
                self.completed_stocker_node,
                self.idle_wait_node,
            )
            robot.total_travel_distance_mm += relocation_distance
            robot.total_travel_time_sec += relocation_time
            robot.total_busy_time_sec += relocation_time
            self.count_path_edges(relocation_edges)
            self.schedule(
                self.now + relocation_time,
                "robot_reposition_complete",
                {
                    "robot": robot,
                    "from_node": self.completed_stocker_node,
                    "to_node": self.idle_wait_node,
                    "relocation_time_sec": relocation_time,
                    "relocation_distance_mm": relocation_distance,
                },
            )
        else:
            robot.busy = False

    def handle_robot_reposition_complete(self, payload: dict[str, object]) -> None:
        robot = payload["robot"]
        if not isinstance(robot, Robot):
            raise TypeError("Invalid robot reposition completion payload")

        robot.current_node = str(payload["to_node"])
        robot.busy = False

    def sample_initial_remaining_time_sec(self) -> float:
        sample = self.random.uniform(
            0.0,
            INITIAL_REMAINING_TIME_MAX_MIN * SECONDS_PER_MINUTE,
        )
        self.initial_remaining_time_samples_sec.append(sample)
        return sample

    def sample_process_time_sec(self, machine: Machine, process_generation: int | None = None) -> float:
        mean_sec = machine.process_time_mean_min * SECONDS_PER_MINUTE
        distribution = self.scenario.process_time_distribution
        if distribution == "fixed":
            sample = mean_sec
        elif distribution == "exponential":
            sample = self.random.expovariate(1.0 / mean_sec)
        elif distribution == "triangular":
            sample = self.random.triangular(
                self.scenario.process_time_min_min * SECONDS_PER_MINUTE,
                self.scenario.process_time_max_min * SECONDS_PER_MINUTE,
                self.scenario.process_time_mode_min * SECONDS_PER_MINUTE,
            )
        else:
            raise ValueError(f"Unsupported process_time_distribution: {distribution}")
        self.process_time_samples_sec.append(sample)
        self.process_sample_rows.append(
            {
                "scenario_id": self.scenario.scenario_id,
                "machine_id": machine.machine_id,
                "process_generation": process_generation if process_generation is not None else "",
                "sample_context": "subsequent_processing_time",
                "process_time_distribution": distribution,
                "process_time_sec": round(sample, 3),
                "process_time_min": round(sample / SECONDS_PER_MINUTE, 3),
            }
        )
        return sample

    def try_dispatch(self) -> None:
        while self.pending_tasks:
            pair = self.select_dispatch_pair()
            if pair is None:
                return
            robot, task = pair
            self.pending_tasks.remove(task)
            dispatched = self.dispatch(robot, task)
            if not dispatched:
                self.pending_tasks.insert(0, task)
                return

    def first_idle_robot(self) -> Robot | None:
        for robot in self.robots:
            if not robot.busy:
                return robot
        return None

    def select_task(self, robot: Robot) -> Task:
        if self.scenario.rule != "FIFO":
            raise NotImplementedError(
                f"Only FIFO is implemented in this baseline. Got {self.scenario.rule}"
            )
        return min(self.pending_tasks, key=lambda task: (task.request_time, task.task_id))

    def select_dispatch_pair(self) -> tuple[Robot, Task] | None:
        idle_robots = [robot for robot in self.robots if not robot.busy]
        if not idle_robots or not self.pending_tasks:
            return None

        if self.scenario.rule == "FIFO":
            return idle_robots[0], min(
                self.pending_tasks,
                key=lambda task: (task.request_time, task.task_id),
            )

        if self.scenario.rule == "nearest_edge_distance":
            task = min(self.pending_tasks, key=lambda item: (item.request_time, item.task_id))
            robot = min(
                idle_robots,
                key=lambda item: (
                    self.graph.shortest_path(item.current_node, task.machine.input_node_id)[1],
                    item.robot_id,
                ),
            )
            return robot, task

        raise NotImplementedError(f"Unsupported dispatch rule: {self.scenario.rule}")

    def effective_handling_time_per_event_sec(self) -> float:
        return self.scenario.handling_time_sec + (
            self.scenario.fail_prob * self.scenario.failure_recovery_time_sec
        )

    @staticmethod
    def join_path_nodes(paths: list[list[str]]) -> list[str]:
        sequence: list[str] = []
        for path in paths:
            for node_id in path:
                if not sequence or sequence[-1] != node_id:
                    sequence.append(node_id)
        return sequence

    def dispatch(self, robot: Robot, task: Task) -> bool:
        wait_time = self.now - task.request_time

        assigned_time = self.now
        start_node = robot.current_node
        wb_l_node = task.machine.input_node_id
        wb_r_node = task.machine.output_node_id

        path_to_stk2, dist_to_stk2, travel_to_stk2, edges_to_stk2 = self.graph.shortest_path(
            start_node, self.input_stocker_node
        )
        path_stk2_to_wbl, dist_stk2_to_wbl, travel_stk2_to_wbl, edges_stk2_to_wbl = self.graph.shortest_path(
            self.input_stocker_node, wb_l_node
        )
        path_wbl_to_wbr, dist_wbl_to_wbr, travel_wbl_to_wbr, edges_wbl_to_wbr = self.graph.shortest_path(
            wb_l_node, wb_r_node
        )
        path_wbr_to_stk3, dist_wbr_to_stk3, travel_wbr_to_stk3, edges_wbr_to_stk3 = self.graph.shortest_path(
            wb_r_node, self.completed_stocker_node
        )
        path_node_sequence = self.join_path_nodes(
            [path_to_stk2, path_stk2_to_wbl, path_wbl_to_wbr, path_wbr_to_stk3]
        )

        handling_time_per_event = self.effective_handling_time_per_event_sec()
        handling_events = PICK_PLACE_EVENT_COUNT
        handling_time = handling_events * handling_time_per_event
        input_place_complete_offset = (
            travel_to_stk2
            + (INPUT_PLACE_HANDLING_EVENT_COUNT - 1) * handling_time_per_event
            + travel_stk2_to_wbl
            + handling_time_per_event
        )
        input_place_complete_time = assigned_time + input_place_complete_offset
        travel_time = (
            travel_to_stk2
            + travel_stk2_to_wbl
            + travel_wbl_to_wbr
            + travel_wbr_to_stk3
        )
        busy_time = travel_time + handling_time
        travel_distance = dist_to_stk2 + dist_stk2_to_wbl + dist_wbl_to_wbr + dist_wbr_to_stk3

        if assigned_time + busy_time > self.simulation_end_time:
            return False

        robot.busy = True
        self.completed_task_wait_times_sec.append(wait_time)
        robot.total_travel_distance_mm += travel_distance
        robot.total_travel_time_sec += travel_time
        robot.total_handling_time_sec += handling_time
        robot.total_busy_time_sec += busy_time
        self.count_path_edges(
            edges_to_stk2 + edges_stk2_to_wbl + edges_wbl_to_wbr + edges_wbr_to_stk3
        )

        self.task_log_rows.append(
            {
                "scenario_id": self.scenario.scenario_id,
                "task_id": task.task_id,
                "robot_id": robot.robot_id,
                "machine_id": task.machine.machine_id,
                "request_time_sec": round(task.request_time, 3),
                "assigned_time_sec": round(assigned_time, 3),
                "dispatch_wait_time_sec": round(wait_time, 3),
                "start_node": start_node,
                "input_stocker_node": self.input_stocker_node,
                "wb_l_node": wb_l_node,
                "wb_r_node": wb_r_node,
                "completed_stocker_node": self.completed_stocker_node,
                "task_flow": f"{self.input_stocker_node}->{wb_l_node}->{wb_r_node}->{self.completed_stocker_node}",
                "path_node_sequence": "->".join(path_node_sequence),
                "path_current_to_stk2": "->".join(path_to_stk2),
                "path_stk2_to_wbl": "->".join(path_stk2_to_wbl),
                "path_wbl_to_wbr": "->".join(path_wbl_to_wbr),
                "path_wbr_to_stk3": "->".join(path_wbr_to_stk3),
                "travel_to_stk2_sec": round(travel_to_stk2, 3),
                "travel_stk2_to_wbl_sec": round(travel_stk2_to_wbl, 3),
                "travel_wbl_to_wbr_sec": round(travel_wbl_to_wbr, 3),
                "travel_wbr_to_stk3_sec": round(travel_wbr_to_stk3, 3),
                "total_travel_time_sec": round(travel_time, 3),
                "handling_event_count": handling_events,
                "input_place_handling_event_count": INPUT_PLACE_HANDLING_EVENT_COUNT,
                "base_handling_time_per_event_sec": round(self.scenario.handling_time_sec, 3),
                "failure_rate": round(self.scenario.fail_prob, 6),
                "failure_recovery_time_sec": round(self.scenario.failure_recovery_time_sec, 3),
                "expected_failure_delay_per_event_sec": round(
                    self.scenario.fail_prob * self.scenario.failure_recovery_time_sec,
                    3,
                ),
                "effective_handling_time_per_event_sec": round(handling_time_per_event, 3),
                "total_handling_time_sec": round(handling_time, 3),
                "input_place_complete_time_sec": round(input_place_complete_time, 3),
                "wb_starve_time_sec": round(input_place_complete_time - task.request_time, 3),
                "task_duration_sec": round(busy_time, 3),
                "task_complete_time_sec": round(assigned_time + busy_time, 3),
            }
        )

        self.schedule(
            input_place_complete_time,
            "input_place_complete",
            {"robot": robot, "task": task},
        )
        self.schedule(self.now + busy_time, "robot_complete", {"robot": robot, "task": task})
        return True

    def count_path_edges(self, edges) -> None:
        for edge in edges:
            key = tuple(sorted((edge.from_node, edge.to_node)))
            self.edge_usage_counts[key] += 1
            if edge.segment_type == "same_position_transfer":
                self.total_same_position_transfer_edges += 1
            elif edge.segment_type == "single_lane_bidirectional":
                self.total_single_lane_edges += 1
            elif edge.segment_type == "bypass":
                self.total_bypass_edges += 1
            elif edge.segment_type == "two_lane":
                self.total_two_lane_edges += 1

    def close_open_starves(self) -> None:
        for state in self.machine_states.values():
            if state.starve_start_time is not None:
                state.total_starve_time_sec += self.simulation_end_time - state.starve_start_time
                state.starve_start_time = None

    def kpis(self) -> dict[str, float | str | int]:
        total_starve = sum(s.total_starve_time_sec for s in self.machine_states.values())
        starve_count = sum(s.starve_count for s in self.machine_states.values())
        total_robot_busy = sum(r.total_busy_time_sec for r in self.robots)
        total_robot_travel_time = sum(r.total_travel_time_sec for r in self.robots)
        total_robot_handling_time = sum(r.total_handling_time_sec for r in self.robots)
        total_robot_distance = sum(r.total_travel_distance_mm for r in self.robots)
        avg_wait = (
            sum(self.completed_task_wait_times_sec) / len(self.completed_task_wait_times_sec)
            if self.completed_task_wait_times_sec
            else 0.0
        )
        avg_response = (
            sum(self.completed_response_times_sec) / len(self.completed_response_times_sec)
            if self.completed_response_times_sec
            else 0.0
        )
        avg_starve_per_request = total_starve / starve_count if starve_count else 0.0
        starve_rate = total_starve / (len(self.machines) * self.simulation_end_time)
        avg_process_time = (
            sum(self.process_time_samples_sec) / len(self.process_time_samples_sec)
            if self.process_time_samples_sec
            else 0.0
        )
        avg_initial_remaining_time = (
            sum(self.initial_remaining_time_samples_sec) / len(self.initial_remaining_time_samples_sec)
            if self.initial_remaining_time_samples_sec
            else 0.0
        )
        return {
            "scenario_id": self.scenario.scenario_id,
            "rule": self.scenario.rule,
            "robot_count": self.scenario.robot_count,
            "process_time_distribution": self.scenario.process_time_distribution,
            "machine_count": len(self.machines),
            "simulation_time_hours": self.simulation_end_time / SECONDS_PER_HOUR,
            "throughput_mgz": self.throughput,
            "completed_magazine_count_delivered_to_stk3": self.completed_magazines_delivered_to_stk3,
            "total_starve_time_sec": round(total_starve, 3),
            "total_starve_time_min": round(total_starve / SECONDS_PER_MINUTE, 3),
            "total_wb_idle_time_sec": round(total_starve, 3),
            "total_wb_idle_time_min": round(total_starve / SECONDS_PER_MINUTE, 3),
            "wb_starve_definition": "WL_input_place_complete_minus_WB_process_complete",
            "starve_count": starve_count,
            "average_starve_time_per_request_sec": round(avg_starve_per_request, 3),
            "starve_rate": round(starve_rate, 6),
            "average_task_waiting_time_sec": round(avg_wait, 3),
            "average_response_time_sec": round(avg_response, 3),
            "robot_utilization": round(total_robot_busy / (len(self.robots) * self.simulation_end_time), 6),
            "total_robot_working_time_sec": round(total_robot_busy, 3),
            "total_robot_travel_time_sec": round(total_robot_travel_time, 3),
            "total_robot_handling_time_sec": round(total_robot_handling_time, 3),
            "average_robot_handling_time_per_task_sec": round(
                total_robot_handling_time / self.completed_magazines_delivered_to_stk3
                if self.completed_magazines_delivered_to_stk3
                else 0.0,
                3,
            ),
            "handling_event_count_per_task": PICK_PLACE_EVENT_COUNT,
            "input_place_handling_event_count": INPUT_PLACE_HANDLING_EVENT_COUNT,
            "base_handling_time_per_event_sec": round(self.scenario.handling_time_sec, 3),
            "failure_rate": round(self.scenario.fail_prob, 6),
            "failure_recovery_time_sec": round(self.scenario.failure_recovery_time_sec, 3),
            "expected_failure_delay_per_event_sec": round(
                self.scenario.fail_prob * self.scenario.failure_recovery_time_sec,
                3,
            ),
            "effective_handling_time_per_event_sec": round(
                self.effective_handling_time_per_event_sec(),
                3,
            ),
            "idle_wait_node": self.idle_wait_node,
            "input_stocker_node": self.input_stocker_node,
            "completed_stocker_node": self.completed_stocker_node,
            "initial_remaining_distribution": "uniform_0_120_min",
            "average_initial_remaining_time_min": round(
                avg_initial_remaining_time / SECONDS_PER_MINUTE, 3
            ),
            "average_sampled_process_time_min": round(avg_process_time / SECONDS_PER_MINUTE, 3),
            "process_time_sample_count": len(self.process_time_samples_sec),
            "total_robot_travel_distance_mm": round(total_robot_distance, 3),
            "average_robot_travel_distance_mm": round(total_robot_distance / len(self.robots), 3),
            "two_lane_edges_used": self.total_two_lane_edges,
            "single_lane_edges_used": self.total_single_lane_edges,
            "bypass_edges_used": self.total_bypass_edges,
            "same_position_transfer_edges_used": self.total_same_position_transfer_edges,
        }

    def machine_starve_rows(self) -> list[dict[str, float | str | int]]:
        rows: list[dict[str, float | str | int]] = []
        for state in sorted(self.machine_states.values(), key=lambda item: item.machine.machine_id):
            rows.append(
                {
                    "scenario_id": self.scenario.scenario_id,
                    "machine_id": state.machine.machine_id,
                    "input_node_id": state.machine.input_node_id,
                    "output_node_id": state.machine.output_node_id,
                    "starve_definition": "WL_input_place_complete_minus_WB_process_complete",
                    "starve_time_sec": round(state.total_starve_time_sec, 3),
                    "starve_time_min": round(state.total_starve_time_sec / SECONDS_PER_MINUTE, 3),
                    "starve_count": state.starve_count,
                    "completed_mgz": state.completed_mgz,
                }
            )
        return rows


def write_kpis(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
