# Experiment configs

One YAML per experiment. Each `extends: ../default.yaml` and overrides only
what changes. The loader shallow-merges per top-level key, so you must restate
a whole block (e.g. all of `features.blocks`) when you change part of it.

Naming: `<what-varies>_<value>.yaml`, e.g. `features_no_text.yaml`,
`split_train_end_2021.yaml`. Every run writes its resolved config alongside
its results in `reports/results/<experiment_name>/config.resolved.yaml`, so a
result is always traceable to the exact settings that produced it.
