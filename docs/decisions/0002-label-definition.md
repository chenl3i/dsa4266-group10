# ADR 0002 — KEV membership as the target, with the PU-learning caveat stated

**Status:** Accepted
**Date:** 2026-09-02

## Context

The brief defines the target as

```
Exploited = 1 if known to be exploited, 0 otherwise
```

with CISA KEV as the source of "known to be exploited". This is the right choice —
KEV is authoritative, freely available, and directly meaningful. But the phrase
*known to be* is doing significant work that the report needs to acknowledge.

A CVE absent from KEV is in one of two states:

1. genuinely never exploited, or
2. exploited, but not observed, not attributed, or not yet catalogued by CISA.

The dataset cannot distinguish them. The negative class is therefore **unlabelled**,
not negative. Formally this is positive-unlabelled (PU) learning.

There is a second issue. KEV grows continuously. A pipeline run in March and one
run in September, against the same CVE window, would produce different labels
purely because the catalogue moved.

## Decision

**Target.** `exploited = 1` iff the CVE appears in KEV with
`dateAdded <= labels.observation_cutoff`.

**Freeze the label horizon.** `observation_cutoff` is a config value, not
`date.today()`. Re-running the pipeline later reproduces the same labels.

**Collect labels past the test boundary.** `observation_cutoff` must be **later**
than `split.test_end`, and `Config` enforces this. A CVE published in December
2023 and KEV-listed in March 2025 is a true positive for the 2023 test set.
Truncating label collection at the split boundary would manufacture a false
pattern that recent CVEs are never exploited.

**Train as binary, report as PU.** Use ordinary binary classification, but state
in the report that reported precision@k is a **lower bound** on true precision:
some scored-high "false positives" may be genuine exploits that never reached the
catalogue.

## Consequences

**Good.** Reproducible labels. Two group members running the pipeline a month
apart get identical numbers.

**Good.** The limitations section writes itself, and demonstrates that the team
understood the data rather than treating a public dataset as ground truth.

**Cost.** Absolute precision figures understate true performance by an unknown
amount. Mitigated by the fact that the *comparison* against CVSS and EPSS is
affected symmetrically — all rankers are scored against the same imperfect labels,
so the relative claim survives.

**Cost.** `observation_cutoff` is one more thing to get right, and getting it
wrong is silent. Hence the validator in `Config` and the test in
`tests/data/test_labels.py`.

## Alternatives considered

**KEV plus exploit-database presence as positives.** Broader coverage, but
ExploitDB records *proof-of-concept* availability, not real-world exploitation.
Those are meaningfully different things and mixing them would blur the research
question. Worth mentioning in the report as a possible extension.

**An explicit PU-learning estimator** (e.g. Elkan-Noto). Statistically more
principled, but requires estimating the label frequency `c = P(labelled | positive)`,
which is not identifiable here without an assumption nobody can defend. Overkill
for this project; naming the issue is the valuable part.
