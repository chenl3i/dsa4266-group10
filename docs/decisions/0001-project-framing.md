# ADR 0001 — Frame the task as ranking under a budget, not classification

**Status:** Accepted
**Date:** 2026-09-02

## Context

The obvious framing is binary classification: predict `exploited ∈ {0,1}`, report
accuracy / precision / recall / F1. The project brief already pushes back on this,
noting that accuracy is a poor metric when exploited vulnerabilities are rare.

The pushback can go further. The decision a security team actually faces is not
"is this CVE exploited?" — it is "we have capacity to patch 10% of our backlog
this quarter; which 10%?" That is a **ranking problem under a budget constraint**,
and the framing has consequences all the way down the codebase.

## Decision

Model the task as ranking. Concretely:

- The model interface is `score(X) -> np.ndarray`, never `predict(X)`. There is
  no decision threshold to tune; the budget sets the cut point.
- Primary metrics are precision@k and recall@k across a sweep of plausible
  budgets (0.5% … 20%), not a single-threshold F1.
- PR-AUC is the threshold-free summary. ROC-AUC is reported for comparability
  with prior work but never led with.
- Baselines are implemented against the same `Ranker` interface as models, so the
  evaluation code cannot treat them differently by accident.

## Consequences

**Good.** Every number in the results table maps to a sentence a security manager
would say out loud. "At a 10% patch budget, our ranking catches 61% of
exploited vulnerabilities versus CVSS's 38%" needs no methodological caveat to be
understood.

**Good.** Class imbalance stops being a problem to be "fixed" and becomes a
property of the data. There is no temptation to resample the test set.

**Cost.** Ranking metrics need a tie-breaking policy, which classification metrics
do not. CVSS scores are heavily tied, so this turns into a real decision — see
[ADR 0004](0004-tie-breaking.md).

**Cost.** Some standard tooling (`sklearn.metrics.classification_report`,
threshold-based model selection) does not apply and had to be replaced with
`vulnprio.evaluation.metrics`.

## Alternatives considered

**Binary classification with a tuned threshold.** Rejected: the threshold would be
chosen to optimise a metric nobody in the problem domain cares about, and it
hides the budget question the project is actually asking.

**Regression on time-to-exploitation.** Interesting, and a natural extension, but
KEV gives only `dateAdded` — the date exploitation became *known*, not when it
began. Building a survival model on that timestamp would model CISA's publication
process as much as attacker behaviour.
