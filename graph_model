from __future__ import annotations

import csv
import heapq
import math
from dataclasses import dataclass
from pathlib import Path


TRAVEL_SPEED_MM_PER_SEC = 500.0


@dataclass(frozen=True)
class Node:
    node_id: str
    x_mm: float
    y_mm: float
    node_type: str


@dataclass(frozen=True)
class Edge:
    from_node: str
    to_node: str
    distance_mm: float
    segment_type: str
    capacity: int

    @property
    def travel_time_sec(self) -> float:
        return self.distance_mm / TRAVEL_SPEED_MM_PER_SEC


class GraphModel:
    def __init__(self, nodes: dict[str, Node], edges: list[Edge]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.adj: dict[str, list[tuple[str, Edge]]] = {node_id: [] for node_id in nodes}
        for edge in edges:
            self.adj[edge.from_node].append((edge.to_node, edge))
            self.adj[edge.to_node].append((edge.from_node, edge))

    @classmethod
    def from_csv(cls, nodes_path: Path, edges_path: Path) -> "GraphModel":
        nodes: dict[str, Node] = {}
        with nodes_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                nodes[row["node_id"]] = Node(
                    node_id=row["node_id"],
                    x_mm=float(row["x_mm"]),
                    y_mm=float(row["y_mm"]),
                    node_type=row["node_type"],
                )

        edges: list[Edge] = []
        with edges_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                edges.append(
                    Edge(
                        from_node=row["from_node"],
                        to_node=row["to_node"],
                        distance_mm=float(row["distance_mm"]),
                        segment_type=row["segment_type"],
                        capacity=int(row["capacity"]),
                    )
                )
        return cls(nodes, edges)

    def shortest_path(self, source: str, target: str) -> tuple[list[str], float, float, list[Edge]]:
        if source == target:
            return [source], 0.0, 0.0, []

        best_time: dict[str, float] = {source: 0.0}
        best_distance: dict[str, float] = {source: 0.0}
        prev: dict[str, tuple[str, Edge]] = {}
        heap: list[tuple[float, str]] = [(0.0, source)]

        while heap:
            current_time, current = heapq.heappop(heap)
            if current_time > best_time[current] + 1e-12:
                continue
            if current == target:
                break
            for nxt, edge in self.adj[current]:
                new_time = current_time + edge.travel_time_sec
                if new_time < best_time.get(nxt, math.inf) - 1e-12:
                    best_time[nxt] = new_time
                    best_distance[nxt] = best_distance[current] + edge.distance_mm
                    prev[nxt] = (current, edge)
                    heapq.heappush(heap, (new_time, nxt))

        if target not in best_time:
            raise ValueError(f"No path from {source} to {target}")

        path = [target]
        path_edges: list[Edge] = []
        current = target
        while current != source:
            parent, edge = prev[current]
            path_edges.append(edge)
            current = parent
            path.append(current)
        path.reverse()
        path_edges.reverse()
        return path, best_distance[target], best_time[target], path_edges
