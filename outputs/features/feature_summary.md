# Candidate feature preparation summary

Generated from the raw trips by `src.features.candidate_features`.

## Configuration

- Candidate search radius: **80 m**.
- Maximum returned candidates per observation: **32**.
- Route-alignment confidence threshold: **50 m**.
- Distances and bearings are computed in a per-trip local azimuthal-equidistant projection.
- Candidate features are hand-designed geometry/motion features; no learned scaler is fitted here.

## Results

### train

- Trips processed: **80**; trip errors: **0**.
- Candidate rows: **2,791,358**.
- Route-aligned observations: **208,417**; confident labels: **185,146**; low-confidence observations left unlabeled: **23,271**.
- Candidate rows retained for candidate-classifier training after confidence/candidate filtering: **2,664,481**.
- Candidate recall among confident alignments: **99.9827%** (185,114 hits, 32 misses).
- Confidence coverage among route-aligned observations: **88.8344%**.
- Observations requiring nearest-road fallback: **11,769**.
### student_test

- Trips processed: **10**; trip errors: **0**.
- Candidate rows: **327,190**.
- Route-aligned observations: **19,160**; confident labels: **19,133**; low-confidence observations left unlabeled: **27**.
- Candidate rows retained for candidate-classifier training after confidence/candidate filtering: **327,048**.
- Candidate recall among confident alignments: **100.0000%** (19,133 hits, 0 misses).
- Confidence coverage among route-aligned observations: **99.8591%**.
- Observations requiring nearest-road fallback: **0**.


## Feature definitions

| Feature | Meaning |
| --- | --- |
| `distance_to_segment_m` | GPS-to-road perpendicular distance in metres |
| `projected_fraction` | position along the directed segment, from 0 to 1 |
| `segment_length_m` | candidate road-segment length in metres |
| `distance_to_start_m` | distance to candidate segment's start node in metres |
| `distance_to_end_m` | distance to candidate segment's end node in metres |
| `heading_cosine` | cosine agreement between GPS movement and directed road bearing |
| `heading_available` | whether the previous valid GPS point supports a bearing |
| `gps_speed_mps` | consecutive-point speed in metres per second |
| `time_gap_seconds` | elapsed time from the preceding GPS observation |
| `candidate_rank` | distance rank among returned candidates |
| `candidate_count` | number of returned candidates for this observation |
| `nearest_fallback_used` | 1 when no road intersects the search radius and nearest road is used |

`is_reference_candidate`, `has_reference_candidate`, arc IDs, route positions,
and trip/observation IDs are labels or metadata. Do not pass them as model
features. For training, retain confident observations with an available
positive candidate; report candidate recall separately from classifier
accuracy. The route-to-observation alignment is an assumption, not an
organizer-provided pointwise ground truth.
