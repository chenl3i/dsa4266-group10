# Risk-Based Vulnerability Prioritisation — Project Specification

**DSA4266 Group 10**
**Status:** Working specification. Supersedes the initial draft.
**Last updated:** 2026-09-02

---

## 1. Research question

> Given limited patching resources, can publicly available vulnerability
> characteristics identify vulnerabilities that are more likely to be exploited,
> and can this prioritise vulnerabilities better than ranking by CVSS severity?

Operationally: **if a company can patch only 10% of its vulnerabilities, which
10% should it patch first?**

### 1.1 Sub-questions

Each maps to a specific artefact, so nothing in the report is unmotivated.

| # | Question | Answered by |
|---|---|---|
| Q1 | Are higher-CVSS vulnerabilities actually more likely to be exploited? | EDA: exploitation rate by CVSS band |
| Q2 | Are vulnerabilities requiring less attacker effort exploited more often? | EDA: rate by attack complexity / privileges required |
| Q3 | Are network-accessible vulnerabilities exploited more often? | EDA: rate by attack vector |
| Q4 | Are certain weakness types disproportionately targeted? | EDA: rate by CWE class |
| Q5 | Does software prevalence predict exploitation beyond severity? | CPE feature ablation |
| Q6 | Does a learned ranking beat CVSS at realistic patch budgets? | precision@k / recall@k vs baseline |
| Q7 | How does it compare to EPSS, the deployed state of the art? | EPSS baseline comparison |
| Q8 | Which signals carry the improvement? | Per-block feature ablation |

Q5, Q7 and Q8 are additions to the original draft. Q7 in particular closes the
most likely reviewer objection.

---

## 2. Why this is a cybersecurity problem

A software vulnerability is a weakness attackers may exploit. Organisations
routinely discover hundreds or thousands and cannot patch them all at once. The
common heuristic is to rank by CVSS severity — but **severity is not likelihood**.
CVSS measures how bad exploitation *would be*, not how probable it *is*.

The consequence of poor prioritisation is concrete: effort spent patching severe
but unattacked vulnerabilities, while genuinely targeted ones stay open. Empirically
only a small single-digit percentage of published CVEs are ever exploited in the
wild, so the ranking problem is where nearly all the leverage sits.

---

## 3. Data

### 3.1 Sources

| Source | Role | Volume | Refresh | Access |
|---|---|---|---|---|
| **NIST NVD** | Features | ~250k CVEs | Continuous | API 2.0 (key recommended) or bulk JSON feeds |
| **CISA KEV** | Positive labels | ~1,300 entries | As added | Single JSON/CSV |
| **FIRST EPSS** | Baseline comparator | Daily scores, all CVEs | Daily | Daily CSV.GZ archive |

### 3.2 Fields used from NVD

- **CVSS v3.1 base metrics** — attack vector, attack complexity, privileges
  required, user interaction, scope, and confidentiality / integrity /
  availability impact. Plus the composite base score and severity band.
- **CWE** weakness classification.
- **CPE** applicability statements → affected vendors and products.
- **References** — URLs and NVD-assigned tags (`Exploit`, `Patch`, `Vendor Advisory`, …).
- **Dates** — `published`, `lastModified`.
- **Description** — English free text.

### 3.3 Study window

CVEs published **2016-01-01 to 2024-12-31**. Bounded to the CVSS v3.1 era so the
feature schema is consistent across the corpus; earlier CVEs carry v2 vectors with
different metrics and would need a separate, incompatible feature path.

### 3.4 Target variable

```
exploited = 1  if the CVE appears in CISA KEV with dateAdded <= observation_cutoff
            0  otherwise
```

**This is positive-unlabelled data, not clean binary data.** KEV lists *known*
exploited vulnerabilities. A CVE absent from KEV may be unexploited, or exploited
and unobserved. The report treats reported precision@k as a **lower bound** on
true precision. Full reasoning in [ADR 0002](decisions/0002-label-definition.md).

### 3.5 Class imbalance

Roughly 1,300 positives against 250k CVEs — a positive rate near **0.5%**, an
imbalance ratio around 190:1.

Direct consequences, all of which the draft correctly anticipated:
- Accuracy is meaningless (a constant-zero predictor scores 99.5%).
- ROC-AUC is misleadingly optimistic: with an enormous negative class, the false
  positive rate barely moves.
- PR-AUC and top-k metrics are the meaningful summaries.
- Absolute counts matter for uncertainty: the top 10% of a test window may contain
  only a few dozen positives, so **every headline comparison carries a bootstrap
  confidence interval**.

---

## 4. Method

### Step 1 — Understand the data (EDA)

Answer Q1–Q4 above with exploitation rate broken down by each CVSS component,
CWE class, vendor, and publication year. Deliverable: `notebooks/01_eda.ipynb`
plus the figures it exports to `reports/figures/`.

Two things to look for specifically:
- **Non-monotonicity in CVSS.** If the exploitation rate does not rise cleanly with
  the base score, that finding alone partly answers the research question.
- **Vendor concentration.** If exploited CVEs cluster in a handful of vendors, that
  is the CPE feature block earning its place before a line of model code is written.

### Step 2 — Establish baselines

| Baseline | Rule | Purpose |
|---|---|---|
| `random` | Seeded shuffle | Floor. precision@k ≈ base rate; makes imbalance legible. |
| `cvss` | Rank by base score | The brief's baseline. The thing to beat. |
| `epss` | Rank by EPSS as-of split date | Deployed state of the art. Credibility check. |

On EPSS, state two caveats plainly rather than burying them: it is trained partly
on KEV, so a KEV-derived target favours it; and its score must be taken *as of* a
date at or before the split boundary, never today's score.

### Step 3 — Models

**Deferred.** Model selection is in progress. The `Ranker` interface and registry
are in place (`src/vulnprio/models/`), so adding a family is one new file plus one
import line — nothing in the pipeline, CLI, or evaluation changes.

Whatever is chosen, the report must justify it. The draft's own argument is the
right one to make concrete: interpretable models with a track record on tabular,
imbalanced classification problems in security research.

Constraints that apply regardless of family:
- Implement `score`, never `predict`. There is no threshold — see [ADR 0001](decisions/0001-project-framing.md).
- Handle imbalance via class weighting in preference to resampling. If resampling
  is used, it must be fitted inside the CV fold, never before the split.
- All tuning happens on `valid`. `test` is touched once, at the end.

### Step 4 — Temporal split

Train on older CVEs, evaluate on newer ones, because that is the real setting.

```
publication date ──────────────────────────────────────────────────►
   [   train   ][embargo][ valid ][ test ]      (excluded)
                 180 days
                                        │
                                        └── labels still collected past here,
                                            up to observation_cutoff
```

The **embargo** is an addition to the draft. Exploitation is observed with a lag, so
CVEs published just before the training cutoff look unexploited mainly because
nobody has got to them yet. Without the embargo the model learns "recent implies
safe" — an artefact of the observation window. Reasoning in [ADR 0003](decisions/0003-temporal-split.md).

Note that the **label horizon extends past the test window**. A CVE published in
December 2023 and KEV-listed in March 2025 is a true positive for the 2023 test set.

### Step 5 — Evaluation

**Primary:** precision@k and recall@k at k ∈ {0.5, 1, 2, 5, 10, 20}%.
**Headline:** precision@10%, with a 95% bootstrap CI, against the CVSS baseline.
**Secondary:** lift@k, PR-AUC. ROC-AUC reported for comparability only.

**Tie-breaking is a first-class decision.** CVSS scores clump — tens of thousands
of CVEs sit at exactly 9.8, which is above the cut for any realistic budget. So the
"top 10% by CVSS" is genuinely ambiguous, and reported baseline precision can vary
by roughly 2x on tie policy alone. Since model scores barely tie, a careless default
systematically disadvantages the baseline and flatters the project's own conclusion.
Three policies are implemented and the headline states which was used. See
[ADR 0004](decisions/0004-tie-breaking.md).

### Step 6 — Ablations

Re-run with one feature block disabled at a time (a config edit, not a code edit).
This answers Q8 and produces the most interesting table in the report.

The `references` ablation is mandatory, not optional — see §5 item 4.

---

## 5. Data leakage

The draft's final heading, expanded. This section is the project's main
methodological risk, and `src/vulnprio/leakage.py` makes each guard executable.

**1. Temporal leakage** — training on CVEs published after the test window.
*Prevented structurally* by the temporal split; asserted by
`leakage.assert_temporal_order`.

**2. Label-date leakage** — using `kev_date_added`, or anything derived from it,
as a feature. It exists only because the CVE is a positive.
*Prevented* by `leakage.assert_no_forbidden_columns`, run before every fit.

**3. Description leakage** — NVD descriptions are edited over time. A CVE that gets
exploited often has its description rewritten afterwards to say so. Training TF-IDF
on text containing "exploited in the wild" or "actively exploited" is training on
the answer. *Mitigated* by a phrase blocklist applied before vectorisation, and
quantified by `leakage.leakage_audit`.

**4. Post-hoc reference leakage** — reference URLs and tags accumulate after
publication. An `Exploit` tag or a Metasploit link may be *why* the CVE reached
KEV. *Not fully preventable.* Handled by reporting the reference ablation both ways
and treating the no-reference result as the conservative headline.

**Residual risk, stated honestly.** NVD serves current-state records and does not
expose historical field revisions, so we cannot reconstruct what a CVE's
description or reference list looked like on its publication date. Leaks 3 and 4
are therefore mitigated, not eliminated. Saying so is part of the deliverable.

---

## 6. Architecture

Module tree below. For the *data flow* - what each stage reads and writes, and the
invariant enforced at each boundary - see **[architecture.md](architecture.md)**.

```
src/vulnprio/
  config.py           Typed, validated config; experiments extend a base file
  paths.py            Canonical locations
  leakage.py          Executable leakage guards
  cli.py              One command per pipeline stage
  data/
    sources/          One class per external source; snapshot manifests
    schema.py         The column contract everything else depends on
    parse.py          Raw payloads -> tidy parquet
    labels.py         Target construction
    splits.py         Temporal split with embargo
  features/
    base.py           FeatureBlock ABC + registry
    cvss.py           Reference implementation - copy this shape
    temporal.py cwe.py cpe.py references.py text.py
    build.py          Assembly
  models/
    base.py           Ranker ABC + registry  (no models yet, by design)
  baselines/
    cvss.py epss.py random_ranker.py
  evaluation/
    metrics.py        precision@k, recall@k, lift, PR-AUC, bootstrap CI
    report.py plots.py
```

Three properties are worth the ceremony:

**Feature blocks are independently toggleable.** Ablation — the analysis that
answers Q8 — is a config edit. It also lets group members write `cwe.py` and
`text.py` simultaneously without conflicts, since blocks share only the schema.

**Blocks are sklearn transformers.** Anything *fitted* (TF-IDF vocabulary, category
levels, vendor frequencies) is fitted on training rows only. Fitting a vectoriser
on the full corpus before splitting is a leak, and this shape makes it hard to do
by accident.

**Baselines implement the same interface as models.** The evaluation code cannot
treat them differently, so the comparison is structurally fair.

---

## 7. Reproducibility

- Every run is `(raw snapshot, resolved config, seed)`. All three are recorded:
  snapshots carry a `snapshot.json` manifest, and each run writes
  `config.resolved.yaml` beside its results.
- `data/raw/`, `data/interim/`, `data/processed/` are gitignored. Raw data is
  reproducibly downloadable; derived data is a deterministic function of it.
- `data/external/` **is** committed — small, hand-checked lookup tables.
- Label collection is frozen at `observation_cutoff`, so a rerun in three months
  reproduces today's numbers.
- `pytest` covers config validation, labelling, splitting, leakage guards, and
  metrics.

---

## 8. Deliverables

| Artefact | Location |
|---|---|
| EDA notebook + figures | `notebooks/01_eda.ipynb`, `reports/figures/` |
| Baseline comparison table | `reports/results/<experiment>/comparison.csv` |
| Ablation table | `reports/results/ablation_*/` |
| Decision records | `docs/decisions/` |
| Report | `reports/` |

---

## 9. Scope boundaries

**In scope:** NVD + KEV + EPSS; tabular, CWE, CPE, reference and TF-IDF text
features; ranking evaluation with bootstrap CIs; feature ablation.

**Out of scope for the first pass**, and each with a reason:

- *Sentence-embedding or transformer text models.* Establish that TF-IDF beats CVSS
  before reaching for anything heavier.
- *Survival / time-to-exploit modelling.* KEV gives `dateAdded` — when exploitation
  became *known*, not when it began. A survival model on it would partly model
  CISA's publication process.
- *Rolling-origin evaluation.* Better than a single split; deferred for report
  length. A stretch goal, not a rejected idea.
- *Exploit-availability sources (ExploitDB, Metasploit).* These record
  proof-of-concept availability, not real-world exploitation. Mixing them would
  blur the research question.

---

## 10. Open questions

1. **Which model families?** In progress. Registry is ready.
2. **Embargo length.** 180 days is a judgement call. Sensitivity check at
   0 / 90 / 180 / 365 days before fixing it.
3. **Headline tie policy.** `pessimistic` is the conservative default; confirm the
   bracket is reported either way.
4. **CVSS v2 fallback.** Currently out of window. If corpus size becomes a problem,
   revisit — but it needs a separate feature path.
