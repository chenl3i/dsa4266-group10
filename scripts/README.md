# Scripts

Thin, runnable entry points. Logic that a later step depends on belongs in
`src/ugr16ad/` where it can be tested.

| Script | Purpose |
|---|---|
| `eda_checks.py` | Read-only data checks; source of every number in `docs/eda-findings.md` |

Run from the repo root after `pip install -e .`:

```bash
python scripts/eda_checks.py
```
