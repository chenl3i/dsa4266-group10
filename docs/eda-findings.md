# UGR'16 v2: data checks and EDA findings

Every number here comes from `scripts/eda_checks.py`, run against
`data/external/ugr16v2/UGR16v2.{X,Y}{train,test}.csv`. All splits are chronological.
The proposal ([DSA4266.md](DSA4266.md)) is the source of truth for the method; this file
records the evidence behind it and the risks the code must guard against.

Test-label rows (episode lengths, per-category counts, schedule) are **dataset
characterisation only**. Per the proposal, they must not set window lengths, thresholds or
any other model setting.

## Findings

| Metric | Train | Test |
|---|---|---|
| Rows / features / label columns | 98,262 / 134 / 8 | 43,200 / 134 / 8 |
| Date range | 2016-03-19 00:00 → 2016-05-31 23:59 | 2016-07-28 00:00 → 2016-08-26 23:59 |
| X/Y key alignment, sorted | yes, yes | yes, yes |
| Duplicate keys / duplicate rows | 0 / 0 | 0 / 0 |
| Missing minutes | 8,298 (2 gaps) | 0 |
| Gap 1 | 02:00–02:59 on 2016-03-27 (DST, so timestamps are local time, likely Europe/Madrid [VERIFY]) | none |
| Gap 2 | 2016-04-01 00:02 → 2016-04-06 17:19 | none |
| NaN / negative / inf | 0 / 0 / 0 | 0 / 0 / 0 |
| Strictly constant features | 0 | n/a |
| Features ≥ 99% zeros | 1 (`protocoligmp`, 99.997%) | n/a |
| **Positive minutes (7-label target)** | **0 (0.00%)** | **1,364 (3.16%)** |
| Positives in model-fitting / validation rows (78,609 / 19,653; cut at 2016-05-18 08:27) | **0 / 0** | n/a |
| Per category, minutes > 0: dos, scan11, scan44, nerisbotnet, idpscan, sshscan, spam | all 0 | 450, 98, 97, 732, 9, 0, 0 |
| Test attack window | n/a | 2016-07-28 00:00 → 2016-08-09 11:49; Aug 10–26 has no attacks |
| Attack minutes per day (Jul 28–Aug 8) | n/a | 88–124 per day, a regular injection schedule |
| Attack minutes by hour | n/a | alternates ~75 (odd hours) and ~40 (even hours) |
| Episodes (any attack) | n/a | 144; median 4, p90 11, max 61; 4 single-minute |
| Episodes per category | n/a | dos 90 (median 4), scan11 24 (4), scan44 24 (4), botnet 12 (61 each), idpscan 5 (1) |
| Minutes with ≥ 2 attack labels | 0 | 22 of 1,364 (dos+scan11: 15, dos+scan44: 7) |
| Median attacking-flow share of a positive minute | n/a | dos 15%, scan11 12%, scan44 40%, botnet 1.2%, idpscan 67% |
| Blacklist > 0 | 100% of minutes (median 253 flows) | 100% (median 272); P(blacklist \| attack) = 1.0 |
| Group sums = total flows (srcip, dstip, protocol, srctos, npackets) | 100% | ≥ 99.998% |
| nbytes bins sum < total | 9.7% of minutes | 12.1% |
| sport / dport sum ÷ total (median) | 1.43 / 1.51 | 1.40 / 1.51 |
| Max tcpflag count ÷ protocoltcp (median) | 1.52 | 1.18 |
| Label count > total flows | 0 minutes | 0 minutes |
| Flows per minute (median, min–max) | 94,894 (2,360–769,303) | 80,519 (52,928–455,305) |
| Benign hourly median, max ÷ min | 1.40 (peak 17:00, trough 05:00) | not run |
| Benign weekday vs weekend median | ~101k vs ~87k | not run |
| Skew: median / features > 10 | 6.0 / 55 of 134 | n/a |
| Skew after log1p: median / > 10 | 0.8 / 2 | n/a |
| Features > 50% zeros | 48 | n/a |
| Rank of raw matrix | 128 of 134 | n/a |
| Pairs with \|r\| > 0.95 / > 0.99 (log1p) | 24 / 1 | n/a |
| Per-feature train-vs-test AUC > 0.8 | 16 features (top: `dporttelnet` 0.99, `sportmds` 0.98) | n/a |
| Adversarial LR AUC (fit on first halves, evaluated on second halves) | 0.996 | n/a |

## Risks the code must guard against

1. **No positives anywhere in train.** This is why the proposal is one-class: every model
   fits background minutes only, and validation can measure reconstruction, not detection.
   Detection is only ever measured on test.
2. **Covariate shift.** Sixteen features separate train from test with AUC > 0.8, and an
   adversarial classifier reaches 0.996. The 99th-percentile threshold calibrated on late
   training minutes will not hold its 1% alert rate on test. Report the observed test FPR
   next to recall, as the proposal commits to.
3. **The attack schedule is learnable.** Attacks follow a daily, alternating-hour
   injection pattern. `Row` and anything derived from it (hour, weekday, day index) must
   never be a model input. `Row` is the first column of the X files, so the loader must
   drop it explicitly.
4. **Blacklist stays out of X and y.** It is non-zero in every minute and in every attack
   minute, and it lives in the label file. Using it as a feature puts label information
   into X.
5. **Gaps are hard boundaries.** Windows must not span the DST hour, the April outage, or
   the train/test boundary. Gaps are never imputed.
6. **Fit-on-fit-split only.** The log1p/scaler parameters and the constant-feature list
   come from the first 78,609 training rows. No random splits or shuffled folds anywhere.

## Open questions

1. Do port bins overlap (sport/dport sums are 1.4–1.5× the flow total), and are TCP-flag
   counts per packet rather than per flow? Byte bins fall short of the total in about 10%
   of minutes. Confirm against the UGR'16 v2 data description before treating any group
   as a partition. [VERIFY]
2. `labelanomalyidpscan` has only 9 positive minutes in test, and SSH scan and spam have
   none. The proposal reports per-category recall with sample counts; decide whether
   idpscan gets a row or a footnote.
