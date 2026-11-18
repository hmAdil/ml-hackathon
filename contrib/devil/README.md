# Contribution: Adarsh Krishna Naik ("Devil")

Standalone exploration, validation and modelling scripts written against the raw trip files,
kept as submitted (authorship preserved in git history). They are independent of the `pathsnap`
package and are not run by the pipeline or CI; the lint config excludes this folder.

- `inspect_schema.py`, `inventory.py`, `explore_road_network.py`, `explore_reference_routes.py`:
  schema, inventory and structure checks.
- `validate_*.py`: route, route-graph and trip-link validation.
- `phase3_preprocessing_audit.py`, `phase4_feature_engineering.py`, `phase5_candidate_coverage.py`:
  preprocessing audit, feature diagnostics and the candidate-radius sweep
  (`candidate_radius_sweep.csv`, `alignment_review_trips.csv`, `phase5_readiness_*`).
- `train_catboost.py`: CatBoost candidate scorer with trip-grouped CV. It expects a candidate
  table with columns `trip_id, gps_index, arc_id, label`, not our `data/processed` schema.
- `visualize_geography.py`: geographic plots.

Its phase-5 guardrails (never augment validation or test data, no synthetic IDs or routes,
report baseline vs augmented by held-out trip) are the rules our augmentation experiment follows.
Note: `inventory.py` also lists the `evaluator_test` partition when present; the project rule is
that evaluator data is never used for training, selection or tuning.
