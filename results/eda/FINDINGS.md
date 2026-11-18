# Phase 2 EDA findings

Every number below comes from code in this repo. Each one is tagged with its source:

- **[S]**: `results/eda/trip_stats.csv`. One row per trip for all 80 train and 10 student-test trips, produced by an earlier run of `pathsnap.eda` on the raw data.
- **[T1]**: train trip `00000001`, rebuilt from the committed Folium map by `pathsnap.eda_recover`. Its GPS and route are exact, apart from timestamps rounded to 1 s. Its network is a partial 4,000-arc sample, so its network statistics are not used here.
- **[P]**: pending. The code exists (`python -m pathsnap.eda --deep`) but needs the raw files in `data/train` and `data/student_test`.

Evaluator trips were never read. Train and student-test results are always reported separately.

Regenerate everything:

```
PYTHONPATH=src python -m pathsnap.eda --deep    # needs raw data; rewrites trip_stats.csv
PYTHONPATH=src python -m pathsnap.eda_inventory # works now, from trip_stats.csv
PYTHONPATH=src python -m pathsnap.eda_trip 00000001
PYTHONPATH=src python -m pathsnap.viz --world
```

---

## 2.1 Dataset inventory [S]

| | train | student_test |
|---|---:|---:|
| trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| road nodes / arcs | 1,933,485 / 3,876,904 | 90,689 / 185,041 |
| reference-route arcs | 31,336 | 3,477 |
| median GPS points / trip | 1,529 | 1,420 |
| median route arcs / trip | 280 | 250 |
| median duration | 28.5 min | 30.2 min |
| median track length | 24.1 km | 20.9 km |
| largest sampling gap | 153,064 s (42.5 h) | 210 s |

Figures: `inventory_sizes.png`, `inventory_gps_vs_route.png`, `inventory_route_agreement.png` and `inventory_flags.png`. Tables: `partition_comparison.csv` and `trip_flags.csv`.

- **Sizes are heavy-tailed and vary 100–1000× between trips.** GPS points range from 262 to 19,313, network arcs from 649 to 307,392, and duration from 4 min to 42 h. **Implication:** if metrics are pooled over observations, a few long trips will dominate them. Report per-trip-averaged metrics as well, and use per-trip groups in CV.
- **Trips are spread worldwide.** 60 are in Europe, 18 in Asia or Russia, 11 in the Americas and 1 in South Africa. Latitudes run from −34.6 to 61.7 and longitudes from −100 to 137 (`results/viz/world_trips.html`). **Implication:** use a separate local projection for each trip, as `geo.make_projector` does, and never use degrees as a distance. Road density and driving style vary by country, so regional features can't be assumed to transfer.
- **Student-test is similar to train, but cleaner.** Its medians for size, distance-to-route and continuity are close to train's. It has none of the long gaps, implausible speeds or route/GPS mismatches. **Implication:** student-test scores will probably be optimistic compared with the dirtiest train trips. The evaluator split's quality is unknown, so the pipeline must still handle gaps and spikes.
- **There are always more GPS points than route arcs,** with a median of 5.3 GPS points per route arc and a range of 1.3 to 57. The rows don't correspond one-to-one, which confirms that labels need a real alignment step and can't come from matching row numbers.
- **Network size relative to the track:** the median is 14 network arcs per GPS point, and the maximum is 126. Thirteen train trips have more than 100k arcs, which rules out brute-force candidate search. Index only the arcs inside the track's bounding box plus a margin. `eda_network.nearest_arc_dist` already does this.

### Data-quality flags (fixed thresholds, `eda_inventory.FLAGS`)

| flag | train | student_test | trips |
|---|---:|---:|---|
| <95% of GPS points within 30 m of the route | 6 | 0 | 002, 033, 043, 054, 081, 084 |
| sampling gap > 10 min | 6 | 0 | 011, 021, 042, 069, 078, 090 |
| step speed > 80 m/s | 7 | 0 | 013, 032, 033, 043, 057, 079, 092 |
| consecutive route arcs not connected | 10 | 1 | 002, 019, 033, 041, 042, 069, 075, 080, 086, 089; student 076 |
| route revisits an arc | 9 | 1 | 008, 018, 069, 075, 084, 086, 090, 092, 095; student 056 |

31 of 80 train trips have at least one flag. 6 have three: 033, 042, 043, 069 and 086.

---

## 2.2 GPS trajectories

**All trips [S]:**
- No NaN or out-of-range coordinates, and timestamps strictly increase in every trip.
- The median interval is 1 s (1 Hz). 7,286 train intervals are longer than 1 s, compared with 264 in student-test.
- **Long gaps are rare but extreme.** Trip 090 has a 42.5 h gap and 078 has a 14.7 h gap; these are separate drives stored as one trip. 011, 021, 042 and 069 have gaps of 11–44 min. **Decision:** split trips into segments at gaps over 60 s, both for features and for HMM transitions. Never interpolate or route across a multi-hour gap. `eda_gps.segment_ids` implements this, and `gaps.csv` will list every gap with its jump distance and implied speed **[P]**.
- **Speeds:** the median of each trip's 99th-percentile speed is about 30 m/s. Seven train trips have a step faster than 80 m/s, reaching 180 m/s. These are GPS jumps, not real driving.

**Trip 00000001 in detail [T1]** (`trip_00000001/gps_timeline.png`):
- This is a 50 km motorway-then-urban drive in Germany, around 49.8–50.2°N, 8.6°E. It runs for about 600 s at 45–55 m/s (160–200 km/h), then slows to 10–20 m/s with sharper turns.
- **171 of 1,314 steps are faster than 55 m/s, and they are genuine motorway driving, not noise.** **Decision:** a fixed speed threshold can't be used to detect outliers. `eda_gps.spike_mask` requires an implausible speed in *and* out of a point plus a direction reversal, and it finds 0 spikes in this trip, which is correct.
- The largest gap is 16 s, with a 19.7 m jump: the vehicle was stopped and the GPS was not dropping out. The median heading change rises from about 5° on the motorway to 25–75° in town.

---

## 2.3 Road network

**All trips [S]:**
- No invalid node references, invalid route IDs or isolated nodes.
- 24 self-loops across 9 trips, and 1,903 duplicate arcs across 60 trips (up to 322 in one trip). **Decision:** collapse duplicate `(from, to)` pairs and drop self-loops when building candidates, but keep a map back to the original arc IDs, because the reference route uses the original IDs.
- **The networks are fragmented:** the median trip has 19 weakly connected components, and the maximum is 421, in trip 043. The networks look like clipped extracts, with roads cut at the clip boundary. **Implication:** HMM transitions need a fallback, such as a cost based on straight-line distance, for when no path exists. The full run reports the share of each trip's network in the largest strongly connected component and how many route arcs fall outside it **[P]**.

**[P] Implemented but not yet run:**
- in- and out-degree distributions
- dead-end and source-only nodes
- share of two-way roads
- strongly connected components
- arc-length distribution
- the distance from each GPS point to its **nearest network arc**, at p50, p95 and p99

The nearest-arc distance is the key input for choosing the candidate search radius. For trip 001 **[T1]**, which has complete route geometry, every GPS point is within 27.2 m of the route, with p95 = 12.4 m and p99 about 15 m. So a 30–50 m radius looks reasonable for clean trips, to be confirmed across all trips.

---

## 2.4 Reference routes

**All trips [S]:**
- **The routes match the GPS closely in most trips.** The median per-trip distance from GPS to route is 4.5 m, the median p95 is 14 m, and 99.9% of points are within 30 m in the median trip.
- **Mismatches cluster in 6 train trips.** Two kinds appear:
  - **Constant offset:** 033 (median 122 m, p95 142 m, 29% of points within 30 m) and 054 (median 139 m, p95 210 m, 11% within 30 m). The median is close to the p95, so most points sit the same distance away, which looks like a systematic shift or a parallel road rather than noise. 033 covers 47.8 km in about 5 km² over 104 min, which suggests laps, while its route has only 169 arcs. One possible explanation is that the route covers only part of the GPS recording. **This must be checked on the maps before these trips are used for training.**
  - **Local departures:** 002 (p95 242 m), 043 (p95 70 m), 081 and 084 (93% within 30 m). These are probably short detours, a missing segment, or GPS errors.
- **Route continuity:** 11 trips have at least one break where consecutive arcs don't connect, but only 1–2% of transitions in each trip. 10 trips revisit arcs; 069 revisits 29. **Implication:** the alignment step must allow loops and revisits, so "the next route arc is always new" can't be assumed.

**Trip 00000001 in detail [T1]** (`trip_00000001/route_alignment.png` and `step_vs_arc.png`):
- The route is fully continuous, with 525 arcs and no repeats or U-turns. Its length is 49.87 km, against a GPS path of 50.19 km.
- **Monotone alignment is plausible:** the nearest route position increases over time, with Spearman ρ = 0.99997. Only 0.3% of steps move backwards, in 3 big jumps at junctions where a later route arc runs close by. **Decision:** build Phase 3 labels with a monotone, sequence-constrained alignment, such as DP or Viterbi over route positions. Labelling each point with its nearest route arc is wrong at exactly these junction points.
- **67 of 525 route arcs (13%) have no GPS point within 30 m.** At motorway speed, one sample (median step 41 m) covers more than one short arc (median route arc 56 m). **Implication:** point-level labels can't produce the full route. The predicted route has to fill the gaps between matched arcs with shortest paths, and route metrics should be computed on the arc sequence, not just per point.
- The first GPS point is 32 m from route arc 0 and snaps to route arc 1; the route starts slightly before the GPS does.

---

## 2.5 Maps

- `results/viz/world_trips.html`: every trip at its location, coloured by partition, with a tooltip of key stats and flags **[S]**.
- `results/viz/00000001_eda.html`: the trip map with layers for start and end, gaps over 10 s, spikes, GPS points more than 30 m from the route, and route breaks **[T1]**.
- `results/viz/00000001.png`: a static plot in local metres for the report **[T1]**.
- **[P]** Once the raw data is present, `python -m pathsnap.viz <trip_id>` makes the same map for any trip. Do the flagged trips first: 033, 054, 002, 043, 090 and 069.

---

## 2.6 Decisions carried into Phases 3–4

1. **Segment trips at gaps over 60 s.** No features, transitions or interpolation cross a segment boundary.
2. **Detect outliers using physics plus a direction reversal,** not a speed threshold. Remove spikes before candidate generation, and keep the count for reporting.
3. **Generate candidates from a per-trip spatial index** over arcs near the track's bounding box. Start with a 50 m radius and top-k, then measure the candidate miss rate against the radius. The 50 m starting point is supported by the GPS-to-route p95 of 14 m and max of 27 m in trip 001.
4. **Build labels by monotone sequence alignment** to route positions, allowing revisits. Validate them with the distance to the assigned arc. Points with no route arc within 30 m get no label and are excluded from training, not labelled 0.
5. **Training-data policy for mismatched trips (033, 054 and others):** inspect them first. Then fix them by cropping to the part the route covers, or exclude them from *training* labels. Every exclusion must be decided by a documented rule, not by checking model scores, and must be reported. Student-test trips are never excluded.
6. **Deduplicate arcs and drop self-loops** for the graph, keeping a map back to the original arc IDs.
7. **Evaluate per trip as well as pooled,** because of the heavy-tailed trip sizes. Check that results hold when the 6–7 flagged trips are left out.

## Deep-run results (all 90 trips, `python -m pathsnap.eda --deep`)

These replace the [P] items above. Source: `trip_stats.csv`, `pooled_quantiles.csv`, `gaps.csv`.

**Candidate search radius (distance from each GPS point to the nearest network arc)**
- Per-trip medians: p50 4.1 m, p95 12.9 m, p99 18.2 m (train); p99 21.5 m (student_test).
- Pooled over all points the train tail is heavy: p99 97 m, p100 240 m. It comes from a few trips: 002 (p99 210 m), 054 (p50 84 m, so GPS is far from every road), 081, 096, 043, 084.
- 73 of 90 trips have every GPS point within 50 m of some arc; 9 trips have p99 above 50 m.
- **Decision:** start with a 50 m radius with a fallback to the k nearest arcs when fewer than a minimum number of candidates are found, and report candidate miss rate at 30, 50 and 100 m on train.

**Connectivity**
- Largest strongly connected component holds 96% of nodes (median trip); largest weakly connected component 98%. 95% of arcs are two-way.
- **All reference-route arcs lie in the main strongly connected component, and none are dead ends** (median trip, both partitions). The decoder can restrict paths to that component.

**Gaps**
- 23 of 90 trips have at least one gap over 60 s, but the median trip spends 0% of its time in such gaps, so segmenting at 60 s costs little data.

**Trips 033 and 054 (reference route disagrees with GPS)**
- 054: GPS is far from every network arc (median 84 m), so the GPS looks shifted relative to the network itself.
- 033: GPS lies near network arcs, but not near its reference route, so the route may cover only part of the recording. This is a hypothesis; confirm on the maps (`python -m pathsnap.viz 00000033`).

## Open items
- Look at the 033 and 054 maps to explain the offset, then decide whether to exclude them from training.
- Candidate miss rate at several radii needs the Phase 3 label alignment.
