"""Candidate-state HMM/Viterbi decoder for GPS-to-road matching."""
from __future__ import annotations

import heapq
import math

import numpy as np
import pandas as pd
from pyproj import Geod


class DirectedRoadNetwork:
    """Directed road graph with bounded, cached shortest-path queries."""

    def __init__(self, node_lonlat: np.ndarray, arc_endpoints: np.ndarray) -> None:
        self.node_lonlat = np.asarray(node_lonlat, dtype=float)
        raw = np.asarray(arc_endpoints, dtype=float)
        self.endpoints = np.full(raw.shape, -1, dtype=np.int64)
        self.arc_lengths_m = np.full(len(raw), np.nan, dtype=float)
        self.adjacency: list[list[tuple[int, float]]] = [[] for _ in range(len(self.node_lonlat))]
        integral = np.isfinite(raw).all(axis=1) & np.equal(raw, np.floor(np.nan_to_num(raw))).all(axis=1)
        valid = integral & (raw[:, 0] >= 0) & (raw[:, 1] >= 0)
        valid &= (raw[:, 0] < len(self.node_lonlat)) & (raw[:, 1] < len(self.node_lonlat))
        arc_ids = np.flatnonzero(valid)
        if len(arc_ids):
            self.endpoints[arc_ids] = raw[valid].astype(np.int64)
            edge_nodes = self.endpoints[arc_ids]
            finite = np.isfinite(self.node_lonlat).all(axis=1)
            keep = finite[edge_nodes].all(axis=1)
            arc_ids, edge_nodes = arc_ids[keep], edge_nodes[keep]
            geod = Geod(ellps="WGS84")
            _, _, lengths = geod.inv(
                self.node_lonlat[edge_nodes[:, 0], 0], self.node_lonlat[edge_nodes[:, 0], 1],
                self.node_lonlat[edge_nodes[:, 1], 0], self.node_lonlat[edge_nodes[:, 1], 1],
            )
            for arc_id, edge, length in zip(arc_ids, edge_nodes, lengths):
                length = float(length)
                if length > 0 and np.isfinite(length):
                    source, target = map(int, edge)
                    self.arc_lengths_m[arc_id] = length
                    self.adjacency[source].append((target, length))
        self._source_cache: dict[int, tuple[float, dict[int, float]]] = {}

    def shortest_distance(self, source: int, target: int, limit_m: float) -> float | None:
        """Return directed shortest distance when target is within the bound."""
        if source == target:
            return 0.0
        if not (0 <= source < len(self.adjacency) and 0 <= target < len(self.adjacency)):
            return None
        limit = min(5000.0, max(100.0, math.ceil(limit_m / 100.0) * 100.0))
        cached = self._source_cache.get(source)
        if cached is None or cached[0] < limit:
            distances = {source: 0.0}
            queue = [(0.0, source)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance > distances[node] or distance > limit:
                    continue
                for neighbor, edge_length in self.adjacency[node]:
                    candidate = distance + edge_length
                    if candidate <= limit and candidate < distances.get(neighbor, math.inf):
                        distances[neighbor] = candidate
                        heapq.heappush(queue, (candidate, neighbor))
            self._source_cache[source] = (limit, distances)
        return self._source_cache[source][1].get(target)


def _gps_distance_m(a: np.ndarray, b: np.ndarray) -> float | None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    _, _, distance = Geod(ellps="WGS84").inv(float(a[0]), float(a[1]), float(b[0]), float(b[1]))
    return float(distance) if np.isfinite(distance) else None


def _transition_log_probability(
    previous: dict[str, float | int],
    current: dict[str, float | int],
    network: DirectedRoadNetwork,
    gps_distance_m: float | None,
    elapsed_seconds: float,
    disconnected_penalty: float,
) -> tuple[float, bool]:
    dt = max(0.1, float(elapsed_seconds))
    displacement = max(0.0, float(gps_distance_m or 0.0))
    sigma = max(15.0, 8.0 * math.sqrt(dt) + 0.25 * displacement)
    search_limit = min(5000.0, max(100.0, displacement + 4.0 * sigma))
    arc_a, arc_b = int(previous["arc_id"]), int(current["arc_id"])
    valid = (
        0 <= arc_a < len(network.endpoints)
        and 0 <= arc_b < len(network.endpoints)
        and np.isfinite(network.arc_lengths_m[arc_a])
        and np.isfinite(network.arc_lengths_m[arc_b])
    )
    if not valid:
        return -disconnected_penalty / math.sqrt(dt), False
    length_a, length_b = network.arc_lengths_m[arc_a], network.arc_lengths_m[arc_b]
    fraction_a = float(np.clip(previous["fraction"], 0.0, 1.0))
    fraction_b = float(np.clip(current["fraction"], 0.0, 1.0))
    if arc_a == arc_b and fraction_b >= fraction_a:
        route_distance = (fraction_b - fraction_a) * length_a
    else:
        middle = network.shortest_distance(
            int(network.endpoints[arc_a, 1]), int(network.endpoints[arc_b, 0]), search_limit
        )
        if middle is None:
            return -disconnected_penalty / math.sqrt(dt), False
        route_distance = (1.0 - fraction_a) * length_a + middle + fraction_b * length_b
    residual = (route_distance - displacement) / sigma
    connection_bonus = 1.0 if arc_a == arc_b else 0.75
    return connection_bonus - 0.5 * residual * residual, True


def decode_trip_candidates(
    candidate_rows: pd.DataFrame,
    track: pd.DataFrame,
    node_lonlat: np.ndarray,
    arc_endpoints: np.ndarray,
    *,
    probability_column: str = "_model_probability",
    max_gap_seconds: float = 60.0,
    disconnected_penalty: float = 8.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Decode candidate arcs, returning one Viterbi-selected arc per candidate observation.

    Emissions are model probabilities normalized within each observation.
    Transitions compare directed road-path distance with GPS displacement.
    Missing-candidate observations are skipped and bridged when possible;
    timestamp gaps over ``max_gap_seconds`` begin a fresh Viterbi chain.
    """
    needed = {"obs_idx", "candidate_arc_id", "projected_fraction", "segment_length_m", probability_column}
    missing = needed - set(candidate_rows.columns)
    if missing:
        raise ValueError(f"Candidate rows missing columns: {sorted(missing)}")
    if max_gap_seconds <= 0 or disconnected_penalty <= 0:
        raise ValueError("Gap limit and disconnected penalty must be positive")

    track_index = track.set_index("obs_idx", drop=False)
    network = DirectedRoadNetwork(node_lonlat, arc_endpoints)
    layers: list[dict[str, object]] = []
    for obs_idx, group in candidate_rows.groupby("obs_idx", sort=True):
        group = group.drop_duplicates("candidate_arc_id", keep="first")
        probabilities = pd.to_numeric(group[probability_column], errors="coerce").fillna(0).to_numpy(float)
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        total = float(probabilities.sum())
        probabilities = probabilities / total if total > 0 else np.full(len(group), 1.0 / len(group))
        states = [
            {
                "arc_id": int(row.candidate_arc_id),
                "fraction": float(row.projected_fraction),
                "log_emission": math.log(max(float(probability), 1e-12)),
            }
            for row, probability in zip(group.itertuples(index=False), probabilities)
        ]
        if int(obs_idx) not in track_index.index:
            continue
        point = track_index.loc[int(obs_idx)]
        if isinstance(point, pd.DataFrame):
            point = point.iloc[0]
        layers.append({
            "obs_idx": int(obs_idx),
            "timestamp": float(point.timestamp) if pd.notna(point.timestamp) else np.nan,
            "lonlat": np.array([point.longitude, point.latitude], dtype=float),
            "states": states,
        })
    if not layers:
        return pd.DataFrame(columns=["obs_idx", "hmm_arc_id", "hmm_transition_connected"]), {
            "candidate_observations": 0,
            "missing_candidate_observations": int(len(track)),
            "large_gap_resets": 0,
        }

    decoded: dict[int, tuple[int, bool]] = {}
    large_gap_resets = 0

    def finish_chain(chain: list[dict[str, object]]) -> None:
        if not chain:
            return
        dynamic = np.array([state["log_emission"] for state in chain[0]["states"]], dtype=float)
        backpointers: list[np.ndarray] = []
        connected_matrices: list[np.ndarray] = []
        for layer_index in range(1, len(chain)):
            previous, current = chain[layer_index - 1], chain[layer_index]
            obs_gap = int(current["obs_idx"]) - int(previous["obs_idx"])
            elapsed = float(current["timestamp"]) - float(previous["timestamp"])
            if not np.isfinite(elapsed) or elapsed <= 0:
                elapsed = float(max(1, obs_gap))
            displacement = _gps_distance_m(previous["lonlat"], current["lonlat"])
            transition = np.zeros((len(previous["states"]), len(current["states"])), dtype=float)
            connected = np.zeros_like(transition, dtype=bool)
            for i, state_a in enumerate(previous["states"]):
                for j, state_b in enumerate(current["states"]):
                    transition[i, j], connected[i, j] = _transition_log_probability(
                        state_a, state_b, network, displacement, elapsed, disconnected_penalty
                    )
            total_scores = dynamic[:, None] + transition
            best_previous = np.argmax(total_scores, axis=0)
            best_scores = total_scores[best_previous, np.arange(len(current["states"]))]
            emissions = np.array([state["log_emission"] for state in current["states"]], dtype=float)
            dynamic = best_scores + emissions
            backpointers.append(best_previous)
            connected_matrices.append(connected)
        state_path = np.empty(len(chain), dtype=np.int64)
        state_path[-1] = int(np.argmax(dynamic))
        for layer_index in range(len(chain) - 1, 0, -1):
            state_path[layer_index - 1] = backpointers[layer_index - 1][state_path[layer_index]]
        for layer_index, state_index in enumerate(state_path):
            layer = chain[layer_index]
            state = layer["states"][state_index]
            is_connected = True
            if layer_index:
                prev_index = state_path[layer_index - 1]
                is_connected = bool(connected_matrices[layer_index - 1][prev_index, state_index])
            decoded[int(layer["obs_idx"])] = (int(state["arc_id"]), is_connected)

    chain: list[dict[str, object]] = []
    previous_layer: dict[str, object] | None = None
    for layer in layers:
        if previous_layer is not None:
            elapsed = float(layer["timestamp"]) - float(previous_layer["timestamp"])
            if not np.isfinite(elapsed) or elapsed <= 0:
                elapsed = float(max(1, int(layer["obs_idx"]) - int(previous_layer["obs_idx"])))
            if elapsed > max_gap_seconds:
                finish_chain(chain)
                chain = []
                large_gap_resets += 1
        chain.append(layer)
        previous_layer = layer
    finish_chain(chain)

    output = pd.DataFrame([
        {"obs_idx": obs_idx, "hmm_arc_id": arc, "hmm_transition_connected": connected}
        for obs_idx, (arc, connected) in sorted(decoded.items())
    ])
    return output, {
        "candidate_observations": int(len(layers)),
        "missing_candidate_observations": int(max(0, len(track) - len(layers))),
        "large_gap_resets": int(large_gap_resets),
    }
