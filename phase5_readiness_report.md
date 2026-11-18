# PathSnap Phase 5 – Data Generation & Augmentation Readiness

- Trips analyzed: 80
- GPS points represented: 208,417
- Median of per-trip median route distances: 6.38 m
- Median of per-trip 95th-percentile route distances: 14.75 m
- Trips flagged for alignment review: 20

## Candidate-radius diagnostic

| Radius (m) | Trips with median <= R | Trips with p95 <= R | Trips with p99 <= R |
|---:|---:|---:|---:|
| 10 | 77 | 0 | 0 |
| 15 | 78 | 42 | 1 |
| 20 | 78 | 61 | 35 |
| 30 | 78 | 74 | 56 |
| 50 | 78 | 76 | 68 |
| 75 | 78 | 77 | 71 |
| 100 | 78 | 77 | 76 |
| 150 | 80 | 78 | 77 |
| 250 | 80 | 80 | 79 |

A trip with p95 distance <= R has at least approximately 95% of its GPS points within R of the sampled reference-route geometry. This is not the same as candidate recall.

## Required next experiments

1. Build a spatial index over road segments and measure candidate counts/runtime.
2. Validate GPS-to-route label alignment and report positive-candidate recall.
3. Only then create hard negatives and train-only GPS-jitter variants.

## Guardrails

- Do not run SMOTE on raw coordinates, node IDs, arc IDs or unvalidated candidate rows.
- Do not interpolate graph identifiers or generate arbitrary synthetic routes.
- Do not augment validation or student-test data.
- Keep original observations and stable trip-local IDs.
- Report baseline versus augmented validation performance by held-out trip.

This report uses only Phase 4 diagnostics and creates no synthetic training rows.
