"""Spatially align the ordered reference route to GPS observations.

The supplied route has one row per traversed road arc and is shorter than the
GPS track, so row-wise zipping is invalid.  This module uses a monotone
Viterbi-style dynamic program over route positions.  It returns a *derived
alignment* plus the GPS-to-route distance; treat these labels as an alignment
assumption and inspect misses/high-distance cases before model training.
"""
from __future__ import annotations

import numpy as np


def align_reference_route(
    track_xy_m: np.ndarray,
    node_xy_m: np.ndarray,
    arc_endpoints: np.ndarray,
    route_arc_ids: np.ndarray,
    *,
    sigma_m: float = 25.0,
    advance_penalty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each valid track point to a nondecreasing reference-route position.

    Distances to each route arc provide emission costs. The state is the
    position in the ordered ``route_arc_ids`` sequence; transitions can stay
    on an arc or advance, but cannot travel backward. A linear advance cost
    permits efficient prefix-minimum Viterbi updates.

    Returns ``(route_position, aligned_arc_id, point_to_arc_distance_m)``.
    Invalid GPS rows are kept and receive a zero-emission state, allowing the
    sequence constraint to carry the alignment through a missing observation.
    """
    points = np.asarray(track_xy_m, dtype=float)
    nodes = np.asarray(node_xy_m, dtype=float)
    endpoints = np.asarray(arc_endpoints, dtype=np.int64)
    route = np.asarray(route_arc_ids, dtype=np.int64).reshape(-1)
    if sigma_m <= 0 or advance_penalty < 0:
        raise ValueError("sigma_m must be positive and advance_penalty nonnegative")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("track_xy_m must have shape (n_observations, 2)")
    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("arc_endpoints must have shape (n_arcs, 2)")
    if len(points) == 0 or len(route) == 0:
        empty = np.full(len(points), -1, dtype=np.int64)
        return empty, empty.copy(), np.full(len(points), np.nan, dtype=float)
    if (route < 0).any() or (route >= len(endpoints)).any():
        raise ValueError("route contains an arc ID outside arc_endpoints")

    route_endpoints = endpoints[route]
    if (route_endpoints < 0).any() or (route_endpoints >= len(nodes)).any():
        raise ValueError("a reference route arc points outside node_xy_m")
    starts = nodes[route_endpoints[:, 0]]
    ends = nodes[route_endpoints[:, 1]]
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("reference route uses nodes with invalid coordinates")

    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    lengths_sq[lengths_sq == 0] = 1e-12
    n_obs, n_states = len(points), len(route)
    states = np.arange(n_states, dtype=float)
    # One int32 backpointer per observation/state. This avoids retaining a
    # second full emission matrix while keeping the decoding exact.
    back = np.zeros((n_obs, n_states), dtype=np.int32)

    def emission(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(point).all():
            return np.zeros(n_states, dtype=float), np.full(n_states, np.nan)
        delta = point[None, :] - starts
        fraction = np.clip(np.einsum("ij,ij->i", delta, vectors) / lengths_sq, 0.0, 1.0)
        closest = starts + fraction[:, None] * vectors
        distance = np.linalg.norm(point[None, :] - closest, axis=1)
        return 0.5 * np.square(distance / sigma_m), distance

    first_cost, _ = emission(points[0])
    previous = first_cost
    for obs_idx in range(1, n_obs):
        obs_cost, _ = emission(points[obs_idx])
        adjusted = previous - advance_penalty * states
        prefix_cost = np.minimum.accumulate(adjusted)
        # Store the latest state attaining each prefix minimum. Latest wins
        # ties, which avoids needlessly delaying route progress on flat costs.
        eligible = np.where(adjusted <= prefix_cost + 1e-12, np.arange(n_states), -1)
        prefix_arg = np.maximum.accumulate(eligible).astype(np.int32, copy=False)
        back[obs_idx] = prefix_arg
        previous = obs_cost + advance_penalty * states + prefix_cost

    path = np.empty(n_obs, dtype=np.int64)
    path[-1] = int(np.argmin(previous))
    for obs_idx in range(n_obs - 1, 0, -1):
        path[obs_idx - 1] = back[obs_idx, path[obs_idx]]

    distances = np.empty(n_obs, dtype=float)
    for obs_idx, state in enumerate(path):
        _, all_distances = emission(points[obs_idx])
        distances[obs_idx] = all_distances[state]
    return path, route[path], distances
