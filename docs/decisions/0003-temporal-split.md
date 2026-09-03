# ADR 0003 — Temporal split with an embargo window

**Status:** Accepted
**Date:** 2026-09-02

## Context

The brief's Step 4 already calls for a time-based split: train on older CVEs,
test on newer ones, because that mirrors the real setting. Correct — but two
further effects need handling, and neither is obvious.

**Right-censoring.** Exploitation is observed with a lag. A CVE published one week
before the training cutoff appears unexploited largely because nobody has had time
to exploit it yet, or CISA has not yet catalogued it. Feeding that to a model
teaches it "recent implies safe", which is an artefact of the observation window,
not a fact about vulnerabilities. Any temporal feature (`age_days` especially)
will happily learn it.

**Label horizon vs split boundary.** These are different dates and it is easy to
conflate them. The split boundary controls which CVEs are *features*; the label
horizon controls how long we *watched* for exploitation. See
[ADR 0002](0002-label-definition.md).

## Decision

Split on `published_date` into `train` / `valid` / `test` / `excluded`, with an
**embargo**: CVEs published within `embargo_days` before `train_end` are moved to
`excluded` rather than into `train`.

Default `embargo_days: 180`, chosen because KEV additions frequently lag CVE
publication by several months. This is a judgement call, not a derived constant —
the report should include a sensitivity check at 0, 90, 180 and 365 days.

Excluded rows are **labelled `excluded`, not dropped**, so
`TemporalSplit.describe()` shows exactly what the windowing cost.

`valid` exists for all tuning. `test` is touched once, at the very end.

## Consequences

**Good.** The model cannot learn "recent implies safe" from the training set.

**Good.** `leakage.assert_temporal_order` becomes a cheap, meaningful invariant:
no training record may be newer than the oldest test record. It runs as a test.

**Good.** Row accounting stays visible. Silent row loss is how "we have 200k CVEs"
quietly becomes 140k.

**Cost.** Training data shrinks by roughly six months' worth of CVEs — order 20k
records. Acceptable given the corpus size, and the sensitivity check quantifies
what it bought.

**Cost.** Three date parameters that must stay ordered. Enforced by a validator in
`Config` and tested in `tests/test_config.py`.

## Alternatives considered

**Random stratified split.** Rejected. It lets the model see the future, and on
this task it inflates results substantially — a model can learn vendor-specific
patterns from CVEs published after the ones it is scored on.

**Rolling-origin / walk-forward evaluation.** More robust, and standard in
time-series work: several successive train/test folds moving through time. A
genuinely better design, and the right extension if time allows. Deferred because
a single split is easier to explain in a report of this length and gets the
comparison table built sooner. Noted in the spec as a stretch goal.

**No embargo.** Simpler, and defensible if the report states the censoring problem
explicitly. Rejected because the effect is large enough to change conclusions, and
the fix costs one config parameter.
