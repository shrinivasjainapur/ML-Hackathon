# Dataset inventory and quality summary

Generated from the raw trip files by `src.preprocessing.inspect_dataset`.
This is a structural audit, not a model evaluation.

## Partition counts

| partition | trip_count | trips_missing_track | trips_missing_nodes | trips_missing_arcs | trips_missing_route | total_gps_observations | median_gps_observations | total_road_nodes | total_road_arcs | total_reference_arcs | trips_with_parse_errors | invalid_node_coordinate_rows | invalid_arc_endpoints | invalid_route_arc_ids | disconnected_route_transitions | trips_with_equal_track_route_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| student_test | 10 | 0 | 0 | 0 | 0 | 19160 | 1420.0 | 90689 | 185041 | 3477 | 0 | 0 | 0 | 0 | 1 | 0 |
| train | 80 | 0 | 0 | 0 | 0 | 208417 | 1529.0 | 1933485 | 3876904 | 31336 | 0 | 0 | 0 | 0 | 13 | 0 |

## Data-quality findings

- Trips inventoried: **90**.
- Missing required trip files and parse errors are counted in the partition table.
- Missing GPS cells: **0**.
- Invalid GPS coordinate rows: **0**.
- Invalid road-node coordinate rows: **0**.
- Invalid arc endpoint references: **0**.
- Invalid reference-route arc IDs: **0**.
- Non-positive timestamp gaps: **0**.
- Timestamp gaps other than one second: **7548**, across **66** trips.
- Trips with equal `.track` and `.route` row counts: **0/90**.
- Disconnected consecutive route-arc transitions: **14/34723** transitions.
- `train/00000002`: 1 transition(s).
- `train/00000019`: 3 transition(s).
- `train/00000033`: 1 transition(s).
- `train/00000041`: 1 transition(s).
- `train/00000042`: 1 transition(s).
- `train/00000069`: 1 transition(s).
- `train/00000075`: 2 transition(s).
- `train/00000080`: 1 transition(s).
- `train/00000086`: 1 transition(s).
- `train/00000089`: 1 transition(s).
- `student_test/00000076`: 1 transition(s).

## Interpretation and limits

The `.route` file is an ordered sequence of road-arc IDs, while `.track` is a
sequence of GPS observations. Their row counts differ, so they cannot be
paired by row position. The route/observation association used later is a
derived spatial alignment and must be treated as an assumption. A few route
transitions also do not connect in the directed graph; inspect those trips
before treating every reference sequence as a continuous path.

Use `src.preprocessing.make_splits` for the deterministic trip-level holdout
and GroupKFold assignments. Do not split individual GPS observations across
training and validation.
