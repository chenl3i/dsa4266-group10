# ADR 0004 — Tie-breaking is an explicit, reported decision

**Status:** Proposed — implementation open (see `metrics.order_by_score`)
**Date:** 2026-09-02

## Context

CVSS base scores come from a discrete formula over eight categorical metrics.
The result is not a continuous distribution: the scores clump hard, and in the
NVD corpus tens of thousands of CVEs share a base score of exactly 9.8.

For any realistic patch budget, the cut point lands **inside** one of these tied
groups. So "the top 10% by CVSS" is not a well-defined set. Which CVEs fall inside
the budget depends entirely on how ties are ordered — and in numpy, that means it
depends on the accident of row order in the input file.

This is not a rounding-error concern. Reported precision@10% for the CVSS baseline
can differ by roughly a factor of two between the most and least favourable tie
orderings. Since the project's headline claim is "our ranking beats CVSS", an
unexamined tie policy could produce that result on its own.

Model scores are near-continuous and barely tie at all. So a careless default
systematically **disadvantages the baseline** — flattering exactly the conclusion
the project hopes to reach. That asymmetry is what makes this worth an ADR.

## Decision

Tie handling is an explicit parameter of `order_by_score`, with three policies:

| Policy | Within a tied group | Reports | Use for |
|---|---|---|---|
| `pessimistic` | negatives first | worst case | **Default.** Cannot be accused of flattering a model against its baseline. |
| `optimistic` | positives first | best case | Upper bound; brackets the result with `pessimistic`. |
| `random` | seeded shuffle | expected case | Most realistic — mirrors a team drawing arbitrarily from a pile of 9.8s. Pair with the bootstrap CI. |

**All rankers are scored under the same policy in a given table.** Mixing policies
across rows makes the comparison meaningless.

**The report must state which policy produced the headline number**, and include
the pessimistic/optimistic bracket for the CVSS baseline at the primary budget.
If the model's advantage survives comparison against the *optimistic* baseline,
the finding is robust and the report should say so — that is a strong claim.

## Consequences

**Good.** The most fragile number in the project is derived from a stated rule
rather than from input row order.

**Good.** The bracket is itself a finding: "CVSS precision@10% lies between X and
Y depending purely on tie handling" is a genuinely interesting observation about
the limits of severity-based prioritisation, and one most write-ups of this task
miss entirely.

**Cost.** Every metric call carries a `tie_policy` argument.

**Cost.** Three numbers to report instead of one. Worth it.

## Open

The implementation of `order_by_score` is deliberately left to the team —
see the TODO in `src/vulnprio/evaluation/metrics.py`. The executable
specification is `tests/evaluation/test_metrics.py::TestTiePolicies`.
