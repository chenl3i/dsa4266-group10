# Scripts

Thin wrappers only. All real logic lives in `src/vulnprio/` so it can be tested;
a script that grows past argument parsing and a function call should become a
CLI command in `src/vulnprio/cli.py` instead.

Prefer the CLI:

```bash
vulnprio fetch --config configs/default.yaml
```

If the package is not installed, run it as a module from the repo root:

```bash
python -m vulnprio.cli info --config configs/default.yaml
```
