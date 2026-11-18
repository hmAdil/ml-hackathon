Raw dataset lives here locally and is NOT committed.
Layout: `data/train/<trip_id>/` and `data/student_test/<trip_id>/`, each with
`.track` (lon lat t), `.nodes` (lon lat), `.arcs` (from to), `.route` (arc_id).
Evaluator trips are never placed here and never used for training or tuning.
