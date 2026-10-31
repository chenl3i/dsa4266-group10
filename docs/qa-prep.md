# DSA4266 Group 10: Q&A prep checklist

One self-contained list per member. Sections 1–16 are shared material every member
must be able to answer alone. The last section in each list is that member's own
deep dive. Numbers are from `main.tex` (30 Oct draft), the code on `working` and the
saved runs; re-check them if results change before the 3 Nov lock.

**Contents:** [Ang Chen Lei](#ang-chen-lei) · [Lu Phone Mon](#lu-phone-mon) · [Soon Huici Kesha](#soon-huici-kesha) · [Vanessa Lim Dian Qin](#vanessa-lim-dian-qin) · [Team flags](#team-flags-to-settle-before-the-presentation)

---

## Ang Chen Lei

**Your areas:** Training loop, PCA baseline, single-minute MLP, reproducibility.

#### 1. Problem statement (report 2 marks)

- [ ] Say the problem in one sentence: flag the minutes of ISP traffic that contain an attack, while keeping false alarms near 1% so a SOC analyst is not flooded.
- [ ] State the three research questions: (i) does a nonlinear model beat PCA? (ii) do the preceding minutes help? (iii) does recurrence (LSTM/GRU) or convolution (1D-CNN) use that context better?
- [ ] Explain the cost trade-off: a missed attack leaves systems exposed, and every false alarm costs analyst time. That is why we report precision, FPR and alert rate as well as recall.
- [ ] Explain the framing. It is one-class anomaly detection scored by reconstruction error, because the training file has **zero** positive minutes in all seven attack labels, so a supervised classifier cannot be trained.
- [ ] Answer "why not oversample (SMOTE) or reweight?": there are no training positives to oversample or reweight.
- [ ] Know our stance, "we do not assume more complexity means better detection", and the answer the results gave (see section 11).

#### 2. Review of applicable techniques (report 3 marks)

- [ ] Know the three cited papers: Maciá-Fernández et al. 2018 (UGR'16, built for cyclostationarity-based NIDS), Camacho et al. 2023 (data quality changes model comparisons on UGR'16) and Camacho et al. 2025 (quality tools, giving the UGR'16v2 release we use).
- [ ] Map the technique families to this data:
  - supervised classifiers: ruled out, no training positives;
  - linear subspace: PCA residual;
  - density or boundary methods: Isolation Forest, One-Class SVM, LOF;
  - reconstruction: autoencoders;
  - sequence models: LSTM autoencoders, dilated causal (TCN-style) 1D-CNNs.
- [ ] Explain why everything is scored by reconstruction error: one scoring rule for linear and deep models makes the comparison like-for-like, and the per-feature error shows which counters caused an alert.
- [ ] Explain why PCA is the baseline. It is the classic linear method for network-traffic anomalies (residual subspace). A linear autoencoder trained with MSE spans the same subspace as PCA, so PCA is the linear special case of our MLP autoencoder. *(Background only; not cited in the report.)*
- [ ] Answer "why not Isolation Forest or One-Class SVM?" honestly. They are out of scope: we fixed one scoring family so model differences are not scoring differences. A kernel OCSVM also scales poorly with ~78k rows. Name them as future work; do not claim we tested them.

#### 3. Data study ("is the data balanced?")

- [ ] Describe UGR'16v2: one row is one minute of traffic at a Spanish ISP, with 134 numeric flow counters (protocol, source/destination IP categories, port categories, TCP flags, type of service, packet- and byte-count bins).
- [ ] Know the sizes: train 98,262 minutes (19 Mar–31 May 2016); test 43,200 minutes (28 Jul–26 Aug 2016). `X*` files hold features and `Y*` files hold 8 label counts. They are joined on `Row` (`YYYYMMDDHHMM`, local time).
- [ ] Name the 8 labels: DoS, scan11, scan44, Neris botnet, UDP scan (idpscan), SSH scan, spam and **blacklist**.
- [ ] Explain why blacklist is excluded: it is non-zero in *every* minute of both files, including every attack minute, so it carries no signal. It marks IP reputation, not an attack.
- [ ] Answer "is it balanced?": no.
  - Train has 0% positives.
  - Test has 1,364 attack minutes (3.16%), all between 28 Jul and 9 Aug; the other 17 days have none.
  - After windowing, test has 1,357 attacks among 43,181 scored minutes (3.14%).
- [ ] Know the test attack minutes per category: DoS 450 (90 episodes); scan11 98; scan44 97; Neris botnet 732 (12 episodes, median 61 min long); idpscan 9; SSH scan 0; spam 0.
- [ ] Know the quality checks:
  - no missing, negative or infinite values, and no duplicate timestamps or rows;
  - groups that should partition the flows sum to the total in ≥ 99.998% of minutes;
  - port and TCP-flag groups sum to *more* than the total (overlapping bins), so they are kept unchanged;
  - spikes are kept, because they may be the anomalies.
- [ ] Know the gaps. The 02:00 hour on 27 Mar is missing (DST change), and an outage runs from 1 Apr 00:02 to 6 Apr 17:19. Together they remove 8,298 minutes. They are **not** imputed.
- [ ] Know the seasonality: a daily cycle (hourly median flows vary ×1.4, peaking at 17:00) and higher weekday than weekend volume.
- [ ] Know the drift. 16 features separate train from test with univariate AUC > 0.8, and a logistic regression separates the two periods with AUC 0.996. So we expected the 1% calibration alert rate *not* to hold on test.
- [ ] Know the injection schedule: test attacks run at 88–124 minutes per day, alternating odd and even hours. This is why `Row` and anything derived from it (hour, weekday, day index) is **never** a model input; a model would learn the schedule.
- [ ] Know the rule: test labels describe the data only. They never set the window length, threshold or any other setting.

#### 4. Preprocessing and normalisation

- [ ] Know the order: `log1p` → drop columns constant in the fit rows → z-score with the fit-row mean and **population** std (ddof=0, as in sklearn `StandardScaler`).
- [ ] Explain why log1p: the counts are heavy-tailed, and log1p lowers the median skewness from 6.0 to 0.8. The +1 keeps zeros defined.
- [ ] Explain the leakage rule: every step is learned on the **78,609 fit rows only**, then applied unchanged to tune, calib and test.
- [ ] Know that constant removal drops only `protocoligmp`, leaving **133** features.
- [ ] Explain why the 48 mostly-zero features (> 50% zeros) are kept: a rarely used port can be exactly what an attack lights up.
- [ ] Explain why we standardise: the score is the MSE averaged over features, so unscaled high-count features would dominate it. It is also why a synthetic shift of "m standard deviations" means something.
- [ ] Explain the consequence of drift: test minutes can lie outside the fit range, which raises scores on benign minutes and so raises the test FPR.

#### 5. Training, validation and test splits

- [ ] Know the chronological splits (no shuffling):
  - **fit** = rows 0–78,608 (19 Mar → 18 May 08:26);
  - **tune** = 9,826 minutes (→ 25 May 04:12);
  - **calib** = 9,827 minutes (→ 31 May);
  - **test** = the whole test file.
- [ ] Know each part's job:
  - fit: model weights and preprocessing;
  - tune: early stopping and model selection;
  - calib: the threshold and the synthetic sanity check only;
  - test: final evaluation, run once.
- [ ] Explain why validation is split in two: a threshold set on the same data used to pick the model would be optimistic. Each half covers about 6.8 days, close to a weekly cycle, so both see weekday and weekend traffic (`TUNE_FRACTION = 0.5`).
- [ ] Explain why not random or k-fold splits: minutes are autocorrelated, so a random split leaks neighbouring minutes. A deployed detector always trains on the past and scores the future.
- [ ] Know that mini-batches are shuffled **within the fit rows only**. Whole windows are shuffled, so the minute order inside each window is kept.
- [ ] Know that validation has no labelled attacks, so it measures *reconstruction*, not *detection*. Detection is measured on test only.

#### 6. Temporal windows

- [ ] Know that the window length is **20 minutes**, fixed before any model was evaluated. Have the team's agreed one-line reason ready (see the flags at the end). **Never** justify 20 with test episode lengths; that would use test labels to set a model setting.
- [ ] Describe the window: a window ending at minute *t* holds minutes *t−19 … t*. It is causal and never crosses a gap or partition boundary, so the first 19 minutes of each contiguous segment get no window.
- [ ] Know that all six models, PCA and the single-minute MLP included, are scored on the **same eligible minutes**: 43,181 test minutes with 1,357 attacks. The test file is one contiguous segment, so only its first 19 minutes are lost.
- [ ] Know the score: the training loss is MSE over the whole window, but the detection score is the MSE of the **final minute only**. Every alert therefore refers to one minute, *t*, as it does for the single-minute models.

#### 7. The six models end to end

- [ ] **PCA**: full SVD, no whitening, **59 components (≈ 90.3% of fit variance)**. The elbow rule chose it from {42, 55, 59, 63, 73}, which are the ranks for 80/90/95% variance plus the 90% rank ± 4. The score is the mean squared error of `inverse_transform(transform(x))`. No epochs and no seeds.
- [ ] **MLP autoencoder** (single minute):
  - 133 → 64 → 32 → **16** → 32 → 64 → 133, with ReLU on hidden layers;
  - linear bottleneck and output, because standardised values can be negative;
  - **22,485** parameters;
  - latent size picked by the elbow rule over {4, 8, 16, 32}.
- [ ] **LSTM autoencoder** (20-minute windows):
  - an LSTM encoder (64 units) whose last hidden state *h_n* is mapped by a linear layer to a 16-d latent;
  - the latent is repeated 20 times and fed to an LSTM decoder (64 units);
  - a linear layer maps each step back to 133 features;
  - **81,621** parameters; defaults fixed a priori, not tuned.
- [ ] **GRU**: the same architecture with a GRU cell (3 gates instead of 4), giving **63,637** parameters.
- [ ] **1D-CNN autoencoder** (20-minute windows):
  - features are channels;
  - 4 **causal dilated** conv layers (kernel 3, dilations 1/2/4/8, 32 channels, GELU, left padding only) give a receptive field of 1 + 2·(1+2+4+8) = **31 ≥ 20**;
  - the last time position goes through a linear layer to a 16-d latent;
  - a linear layer expands it to 32 channels × 20 steps, then a conv (k = 3) and a 1×1 conv give 133 features;
  - **41,013** parameters; not tuned.
- [ ] **Windowed MLP** (control):
  - flattens 20 × 133 = 2,660 values → 64 (ReLU, dropout 0.2) → **16** → 64 → 2,660;
  - Gaussian input noise of 0.1 sd makes it a denoising autoencoder;
  - learning rate 1e-4;
  - latent size from the rule "smallest of {8, 16, 32} within 5% of the best tune loss";
  - **345,332** parameters, the largest model.
- [ ] Explain why a bottleneck: it forces compression, so the network cannot learn the identity map. Normal traffic reconstructs well, and unusual traffic, ideally, does not.
- [ ] Explain why the windowed MLP exists: it sees the same 20 minutes as the LSTM and CNN but has no notion of order. It separates "more minutes of input" from "using the order of minutes".

#### 8. Training protocol

- [ ] Know the shared loop for every neural model: MSE loss, Adam, batch 256, up to 50 epochs, early stopping with patience 5 on tune loss (a strict improvement is needed), then the best-epoch weights are restored. The learning rate is 1e-3, except 1e-4 for the windowed MLP.
- [ ] Know the seeds: 0, 1 and 2, reported as mean ± sd. PCA is deterministic, so it has one run.
- [ ] Know the reproducibility steps:
  - seeded `random`, NumPy and PyTorch;
  - deterministic algorithms and 1 thread for PCA and MLP;
  - an audit retrained the MLP and got bit-identical weights and scores;
  - training-data SHA-256 checked before evaluation;
  - unit tests and CI on every change.
- [ ] Know that everything ran on CPU, and that training and inference times are recorded for the cost table.

#### 9. Threshold and sanity check

- [ ] Know the threshold: the **99th percentile of the model's own calibration scores** (≈ 1.01% calib alert rate), then frozen. A minute alerts when score > threshold.
- [ ] Explain why it is not tuned on test: that leaks labels into a setting. Test-derived operating points are reported only as retrospective comparisons.
- [ ] Explain why 99%: it encodes an assumed SOC budget of about 1 alert per 100 minutes, and it was fixed in the proposal before any result.
- [ ] Describe the synthetic sanity check:
  - inject into 1% of calib minutes;
  - add m ∈ {2, 4} fit-period sd to k ∈ {1, 5} random features per minute;
  - fixed seed 0, so every model sees the same altered minutes;
  - metric: ROC-AUC.
  It shows the model *reacts*; it is not detection and never touches test.
- [ ] Know the deviation from the proposal: the proposal said the sanity check would use the *tuning* block, but the code uses the *calibration* block. Know the team's reason.

#### 10. Evaluation metrics ("are they fair? which should we focus on?")

- [ ] Explain why not accuracy: predicting "normal" for every minute already scores 96.9% on test.
- [ ] Know the metrics at the frozen threshold: recall = TP/(TP+FN); precision = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN); FPR = FP/(FP+TN); alert rate = alerts / all minutes (the analyst's workload).
- [ ] Know the threshold-free metrics (test-derived, retrospective only):
  - **AP**: area under the precision–recall curve; a random scorer gets the prevalence, ≈ 0.031;
  - **recall at FPR ≤ 1%**: the best recall over thresholds whose FPR is at most 1%;
  - **standardised partial ROC-AUC** over FPR 0–1% (McClish): 0.5 is random, 1 is perfect.
- [ ] Explain why AP and partial AUC instead of full ROC-AUC: with 3% positives and a ~1% FPR budget, only the low-FPR region matters. Full ROC-AUC flatters models by counting FPRs no SOC would accept.
- [ ] Explain the fairness issue: at the frozen threshold, models land at **different test FPRs (0.85%–4.3%)**, so their recall mixes sensitivity with alert budget. Recall@FPR ≤ 1% and AP compare them on equal terms.
- [ ] Know how per-category recall is counted: in **minutes**, not episodes; SSH scan and spam are NaN (absent); idpscan has n = 9, too few to conclude anything.
- [ ] Be ready for "which metric should we focus on?": **recall at a controlled FPR or alert rate**, plus precision, because the SOC constraint is analyst time. F1 summarises one threshold only.
- [ ] Know the table checks: mean ± sd over 3 seeds, and a check that every run scored the same 43,181 minutes.

#### 11. Results (memorise the headline numbers)

| Model | Recall | Precision | F1 | Test FPR | AP | Recall@FPR≤1% | pAUC |
|---|---|---|---|---|---|---|---|
| PCA | 0.231 | 0.470 | 0.310 | 0.85% | 0.241 | 0.245 | 0.585 |
| MLP | 0.662 | 0.334 | 0.444 | 4.29% | **0.461** | **0.433** | **0.632** |
| Windowed MLP | 0.567 | 0.392 | **0.464** | 2.85% | 0.433 | 0.335 | 0.604 |
| LSTM | 0.600 | 0.365 | 0.454 | 3.39% | 0.426 | 0.327 | 0.603 |
| GRU | 0.542 | 0.399 | 0.459 | 2.65% | 0.421 | 0.324 | 0.600 |
| 1D-CNN | 0.611 | 0.375 | 0.462 | 3.36% | 0.446 | 0.346 | 0.605 |

- [ ] **RQ (i), nonlinear vs PCA: yes, clearly.** MLP vs PCA: AP 0.461 vs 0.241, recall@1% 0.433 vs 0.245, F1 0.444 vs 0.310.
- [ ] **RQ (ii), does temporal context help: no evidence.** Every windowed model (AP 0.42–0.45, recall@1% 0.32–0.35) is *below* the single-minute MLP (0.461, 0.433).
- [ ] **RQ (iii), recurrent vs convolutional:**
  - 1D-CNN ≥ LSTM ≥ GRU on AP (0.446 / 0.426 / 0.421), all within about one sd of each other;
  - LSTM and CNN are no better than the windowed-MLP control, so there is no evidence they use the *order* of minutes;
  - the CNN has the largest seed variance (recall sd 0.068).
- [ ] Know the trade-offs at the frozen threshold:
  - MLP has the best recall (0.662) but the highest FPR (4.3%, alert rate 6.2%);
  - PCA is the most precise (0.470) but misses most attacks;
  - the F1s of the windowed models are within 0.01 of each other, so do not over-read them.
- [ ] Know the **calibration-to-test shift**: a 1.0% calib alert rate became a 0.85–4.3% test FPR, consistent with the EDA drift (AUC 0.996). PCA is the only model whose FPR did not rise.
- [ ] Know the per-category recall:
  - scans are easy (neural models 0.92–1.0);
  - DoS is about 0.45–0.55;
  - **botnet separates the models: PCA 0.022 vs MLP 0.663** (temporal models 0.50–0.57);
  - idpscan is 5/9–6/9.
- [ ] Know the **sanity-check paradox**: PCA has the *highest* synthetic AUC (e.g. 0.881 at m = 2, k = 5) but the *worst* real detection. Sparse synthetic spikes are not what real attacks look like, so the check proves sensitivity, not detection.
- [ ] Know the cost (CPU training time):
  - MLP ≈ 3 s;
  - GRU ≈ 85 s; LSTM ≈ 136 s; windowed MLP ≈ 210 s;
  - CNN ≈ 679 s, but its timings were taken while other jobs ran, so they are inflated;
  - CNN inference is the slowest at 12.7 s.
- [ ] Have the one-line conclusion ready: **the simplest nonlinear model (single-minute MLP) is best on threshold-free metrics and cheapest to train; extra temporal machinery did not pay off on this benchmark.**

#### 12. Results discussion: loss curves and "did it learn anything?"

- [ ] Describe the loss curves: train loss keeps falling while tune loss bottoms out early. The LSTM's best epochs were 4, 2 and 12; the GRU's were 2, 3 and 2. For LSTM seed 0 at the stop, train loss is ≈ 0.49 and tune loss ≈ 0.70. The gap reflects fit→tune drift plus some overfitting, and early stopping caught it.
- [ ] Know the windowed MLP's first version: hidden 128, no regularisation, lr 1e-3. Its tune loss was lowest at **epoch 1**, so it was memorising. Dropout, input noise and a lower learning rate fixed that, and were judged on tune loss only.
- [ ] Know the MLP latent sweep tune MSE: 0.76 (4) → 0.69 (8) → 0.58 (16) → 0.44 (32). The elbow rule picked 16. **Lower reconstruction loss ≠ better detection**: a wider bottleneck can reconstruct attacks too.
- [ ] Know the PCA tune MSE: 0.31 at 42 components down to 0.11 at 73. The variance is spread out (10 components ≈ 46%, 40 ≈ 79%), so the traffic is genuinely high-dimensional.
- [ ] Give the evidence that the models learned something real:
  - AP 0.46 vs a random 0.031;
  - sanity AUC ≥ 0.97 for large shifts;
  - botnet is caught by the nonlinear models but not by PCA;
  - per-feature errors point at specific counters.
- [ ] Know the interpretability tools: per-feature squared error for each alert (`feature_errors.npy`) says *which counters* were unusual; PCA components; saved latents (`latents.npy`) for embedding plots. Be ready to explain any embedding or loss figure in the deck.

#### 13. Typical data-science problems and what we did about them

- [ ] **Leakage**: fit-only preprocessing; no `Row` or calendar features; chronological splits; a frozen threshold; test labels never set anything.
- [ ] **Class imbalance**: one-class framing, and metrics that handle imbalance (AP, recall@FPR, partial AUC).
- [ ] **Distribution drift**: measured in EDA, and the observed test FPR is reported instead of assuming the 1% carries over.
- [ ] **Overfitting**: early stopping on tune loss, bottlenecks, small fixed grids, and dropout plus denoising for the windowed MLP.
- [ ] **Reproducibility and correctness**: 3 seeds, the audit, data hashes, one shared pipeline, unit tests and CI.
- [ ] **Fairness of comparison**:
  - same preprocessing, eligible minutes, scoring minute and threshold rule for every model;
  - the asymmetry to admit: PCA, MLP and windowed MLP had small selection grids, while LSTM, GRU and CNN ran at fixed defaults (zero tuning budget).

#### 14. Limitations (have three ready without notes)

- [ ] Minute aggregation dilutes short or low-volume attacks and loses flow order within a minute.
- [ ] Background "normal" training traffic may contain unlabelled attacks.
- [ ] Excluding blacklist narrows the scope.
- [ ] Drift inflates test FPR.
- [ ] SSH scan and spam are absent from test, and idpscan has only 9 minutes.
- [ ] Minutes right after a gap are never scored (19-minute blind spot).
- [ ] The data is one ISP in 2016, so these are benchmark results, not deployment readiness.
- [ ] Autoencoders can reconstruct attacks well, as PCA did on botnet.
- [ ] Only 3 seeds and no significance test.
- [ ] LSTM and CNN were not tuned.

#### 15. Likely professor questions: answer each aloud in under 30 s

- [ ] "Validation has no attacks, so how do you know model selection was right?" We don't, for detection. Selection is on reconstruction plus the sanity check, and that is a stated limitation.
- [ ] "Your test FPR is 4%, not 1%. Is that a failure?" It is drift, shown in EDA (AUC 0.996), and we report it honestly. Fix: recalibrate the threshold periodically on recent traffic (future work).
- [ ] "Why score only the final minute?" So every alert maps to one minute, the same as the single-minute models, and the comparison is like-for-like.
- [ ] "Are the differences significant?" MLP vs LSTM AP is 0.461 ± 0.010 vs 0.426 ± 0.018, about 2 sd apart. The temporal models are within 1 sd of each other. With 3 seeds there was no formal test, so claim only the large gaps.
- [ ] "Why is MLP recall high but precision low?" Its frozen threshold landed at a 4.3% test FPR; drift pushed more benign minutes over it. Compare at a fixed FPR instead (recall@1% 0.433, still the best).
- [ ] "What would you do next?" Recalibrate on recent traffic; use episode-level metrics; tune the LSTM and CNN with the same budget; run more seeds; add Isolation Forest; move to flow-level data.

#### 16. Presentation and Q&A

- [ ] Know the main takeaway of every slide you present in one sentence, and how it leads to the conclusion.
- [ ] Be able to explain any figure in the deck: what each axis is, what the reader should notice, and why it matters.
- [ ] Prepare **two technical questions** for other groups (5 marks for participation in their presentations).

#### 17. Your deep dive: questions the professor will direct at you

- [ ] Explain the `train_autoencoder` contract: the model must return the same shape as its input, `(batch, features)` or `(batch, length, features)`. That is why the CNN transposes to channels-first internally and back.
- [ ] Explain the seeding: `set_seed` runs *before* the model is built, so weight initialisation is seeded too. A separate `torch.Generator` drives the mini-batch permutation.
- [ ] Explain how losses are computed:
  - train loss is the batch-size-weighted mean over the epoch;
  - tune loss is computed in `eval()` mode, so dropout and noise are off and it is comparable across settings;
  - a non-finite loss raises an error instead of training silently on NaNs.
- [ ] Explain why the best weights use `deepcopy(state_dict())`: a plain `state_dict()` shares storage with the live weights and would keep changing. Patience 5 needs a *strict* improvement.
- [ ] Explain the timing: `time.perf_counter()`, with a device sync first so GPU work is finished before the clock stops.
- [ ] Describe `score_autoencoder`: it batches 1024 rows, takes the final minute (`x[:, -1, :]` for windows) and returns per-feature squared errors and their mean. `encode_autoencoder` gives the row-aligned latents for embedding plots.
- [ ] Describe PCA in detail:
  - `PCAScorer` wraps `sklearn` PCA (`svd_solver="full"`, `whiten=False`);
  - whitening only rescales the component coordinates; the reconstruction, and so the score, is the same without it, and unwhitened coordinates keep their variance scale for latent plots;
  - the score is the per-row mean of the squared `inverse_transform(transform(X))` error.
- [ ] Know how the PCA rank grid was built: the ranks reaching 80%, 90% and 95% cumulative variance (42, 59, 73) plus the 90% rank ± 4 (55, 63).
- [ ] Explain the elbow rule: pick the point with the largest normalised drop below the chord joining the endpoints, falling back to minimum loss. It was fixed *before* the grid ran and judged on tune MSE only. It chose **PCA 59 (≈ 90.3% variance)** and **MLP latent 16**.
- [ ] Know the tune MSE numbers:
  - PCA: 0.314 (42), 0.221 (55), 0.194 (59), 0.170 (63), 0.112 (73);
  - MLP: ≈ 0.76 (4), 0.69 (8), 0.58 (16), 0.44 (32).
- [ ] Derive the MLP's 22,485 parameters (Linear layers 133→64→32→16→32→64→133, weights plus biases). Explain why ReLU is on hidden layers only and why the output is linear: z-scores can be negative.
- [ ] Explain the PCA–linear-autoencoder link: with linear activations and MSE, an autoencoder learns the PCA subspace, so the MLP differs from PCA *only* through its nonlinearity. That is how we isolate RQ (i).
- [ ] Know the reproducibility audit (`reproducibility.json`): retraining seed 0 gave an identical history, weights and scores, and PCA refit identically. The scope is the same CPU, versions and one thread; it is **not** a cross-platform guarantee. `manifest.json` records versions, the git commit, and source and data SHA-256 hashes.
- [ ] Know the evaluation flow (`experiments.py evaluate --window 20`): it loads the frozen checkpoints, checks the data hash, then scores PCA and MLP on the *same* window-20 eligible minutes so they compare with the windowed models.
- [ ] Interpret the results:
  - PCA has the lowest test FPR (0.85%) and best precision, but recall 0.23 and botnet 0.022;
  - MLP is best on AP and recall@1% but has the largest FPR inflation (4.3%);
  - be ready with a hedged reason why the MLP's calibration threshold transferred worse.
- [ ] Know the cost: MLP trains in ≈ 3 s per seed and PCA fits in 0.2 s. Use these to argue the simplest model is also the cheapest.
- [ ] Answer "why only one PCA run?": PCA with a full SVD is deterministic, so seeds do not apply.

---

## Lu Phone Mon

**Your areas:** Data loading, targets, chronological splits, synthetic sanity check, LSTM/GRU.

#### 1. Problem statement (report 2 marks)

- [ ] Say the problem in one sentence: flag the minutes of ISP traffic that contain an attack, while keeping false alarms near 1% so a SOC analyst is not flooded.
- [ ] State the three research questions: (i) does a nonlinear model beat PCA? (ii) do the preceding minutes help? (iii) does recurrence (LSTM/GRU) or convolution (1D-CNN) use that context better?
- [ ] Explain the cost trade-off: a missed attack leaves systems exposed, and every false alarm costs analyst time. That is why we report precision, FPR and alert rate as well as recall.
- [ ] Explain the framing. It is one-class anomaly detection scored by reconstruction error, because the training file has **zero** positive minutes in all seven attack labels, so a supervised classifier cannot be trained.
- [ ] Answer "why not oversample (SMOTE) or reweight?": there are no training positives to oversample or reweight.
- [ ] Know our stance, "we do not assume more complexity means better detection", and the answer the results gave (see section 11).

#### 2. Review of applicable techniques (report 3 marks)

- [ ] Know the three cited papers: Maciá-Fernández et al. 2018 (UGR'16, built for cyclostationarity-based NIDS), Camacho et al. 2023 (data quality changes model comparisons on UGR'16) and Camacho et al. 2025 (quality tools, giving the UGR'16v2 release we use).
- [ ] Map the technique families to this data:
  - supervised classifiers: ruled out, no training positives;
  - linear subspace: PCA residual;
  - density or boundary methods: Isolation Forest, One-Class SVM, LOF;
  - reconstruction: autoencoders;
  - sequence models: LSTM autoencoders, dilated causal (TCN-style) 1D-CNNs.
- [ ] Explain why everything is scored by reconstruction error: one scoring rule for linear and deep models makes the comparison like-for-like, and the per-feature error shows which counters caused an alert.
- [ ] Explain why PCA is the baseline. It is the classic linear method for network-traffic anomalies (residual subspace). A linear autoencoder trained with MSE spans the same subspace as PCA, so PCA is the linear special case of our MLP autoencoder. *(Background only; not cited in the report.)*
- [ ] Answer "why not Isolation Forest or One-Class SVM?" honestly. They are out of scope: we fixed one scoring family so model differences are not scoring differences. A kernel OCSVM also scales poorly with ~78k rows. Name them as future work; do not claim we tested them.

#### 3. Data study ("is the data balanced?")

- [ ] Describe UGR'16v2: one row is one minute of traffic at a Spanish ISP, with 134 numeric flow counters (protocol, source/destination IP categories, port categories, TCP flags, type of service, packet- and byte-count bins).
- [ ] Know the sizes: train 98,262 minutes (19 Mar–31 May 2016); test 43,200 minutes (28 Jul–26 Aug 2016). `X*` files hold features and `Y*` files hold 8 label counts. They are joined on `Row` (`YYYYMMDDHHMM`, local time).
- [ ] Name the 8 labels: DoS, scan11, scan44, Neris botnet, UDP scan (idpscan), SSH scan, spam and **blacklist**.
- [ ] Explain why blacklist is excluded: it is non-zero in *every* minute of both files, including every attack minute, so it carries no signal. It marks IP reputation, not an attack.
- [ ] Answer "is it balanced?": no.
  - Train has 0% positives.
  - Test has 1,364 attack minutes (3.16%), all between 28 Jul and 9 Aug; the other 17 days have none.
  - After windowing, test has 1,357 attacks among 43,181 scored minutes (3.14%).
- [ ] Know the test attack minutes per category: DoS 450 (90 episodes); scan11 98; scan44 97; Neris botnet 732 (12 episodes, median 61 min long); idpscan 9; SSH scan 0; spam 0.
- [ ] Know the quality checks:
  - no missing, negative or infinite values, and no duplicate timestamps or rows;
  - groups that should partition the flows sum to the total in ≥ 99.998% of minutes;
  - port and TCP-flag groups sum to *more* than the total (overlapping bins), so they are kept unchanged;
  - spikes are kept, because they may be the anomalies.
- [ ] Know the gaps. The 02:00 hour on 27 Mar is missing (DST change), and an outage runs from 1 Apr 00:02 to 6 Apr 17:19. Together they remove 8,298 minutes. They are **not** imputed.
- [ ] Know the seasonality: a daily cycle (hourly median flows vary ×1.4, peaking at 17:00) and higher weekday than weekend volume.
- [ ] Know the drift. 16 features separate train from test with univariate AUC > 0.8, and a logistic regression separates the two periods with AUC 0.996. So we expected the 1% calibration alert rate *not* to hold on test.
- [ ] Know the injection schedule: test attacks run at 88–124 minutes per day, alternating odd and even hours. This is why `Row` and anything derived from it (hour, weekday, day index) is **never** a model input; a model would learn the schedule.
- [ ] Know the rule: test labels describe the data only. They never set the window length, threshold or any other setting.

#### 4. Preprocessing and normalisation

- [ ] Know the order: `log1p` → drop columns constant in the fit rows → z-score with the fit-row mean and **population** std (ddof=0, as in sklearn `StandardScaler`).
- [ ] Explain why log1p: the counts are heavy-tailed, and log1p lowers the median skewness from 6.0 to 0.8. The +1 keeps zeros defined.
- [ ] Explain the leakage rule: every step is learned on the **78,609 fit rows only**, then applied unchanged to tune, calib and test.
- [ ] Know that constant removal drops only `protocoligmp`, leaving **133** features.
- [ ] Explain why the 48 mostly-zero features (> 50% zeros) are kept: a rarely used port can be exactly what an attack lights up.
- [ ] Explain why we standardise: the score is the MSE averaged over features, so unscaled high-count features would dominate it. It is also why a synthetic shift of "m standard deviations" means something.
- [ ] Explain the consequence of drift: test minutes can lie outside the fit range, which raises scores on benign minutes and so raises the test FPR.

#### 5. Training, validation and test splits

- [ ] Know the chronological splits (no shuffling):
  - **fit** = rows 0–78,608 (19 Mar → 18 May 08:26);
  - **tune** = 9,826 minutes (→ 25 May 04:12);
  - **calib** = 9,827 minutes (→ 31 May);
  - **test** = the whole test file.
- [ ] Know each part's job:
  - fit: model weights and preprocessing;
  - tune: early stopping and model selection;
  - calib: the threshold and the synthetic sanity check only;
  - test: final evaluation, run once.
- [ ] Explain why validation is split in two: a threshold set on the same data used to pick the model would be optimistic. Each half covers about 6.8 days, close to a weekly cycle, so both see weekday and weekend traffic (`TUNE_FRACTION = 0.5`).
- [ ] Explain why not random or k-fold splits: minutes are autocorrelated, so a random split leaks neighbouring minutes. A deployed detector always trains on the past and scores the future.
- [ ] Know that mini-batches are shuffled **within the fit rows only**. Whole windows are shuffled, so the minute order inside each window is kept.
- [ ] Know that validation has no labelled attacks, so it measures *reconstruction*, not *detection*. Detection is measured on test only.

#### 6. Temporal windows

- [ ] Know that the window length is **20 minutes**, fixed before any model was evaluated. Have the team's agreed one-line reason ready (see the flags at the end). **Never** justify 20 with test episode lengths; that would use test labels to set a model setting.
- [ ] Describe the window: a window ending at minute *t* holds minutes *t−19 … t*. It is causal and never crosses a gap or partition boundary, so the first 19 minutes of each contiguous segment get no window.
- [ ] Know that all six models, PCA and the single-minute MLP included, are scored on the **same eligible minutes**: 43,181 test minutes with 1,357 attacks. The test file is one contiguous segment, so only its first 19 minutes are lost.
- [ ] Know the score: the training loss is MSE over the whole window, but the detection score is the MSE of the **final minute only**. Every alert therefore refers to one minute, *t*, as it does for the single-minute models.

#### 7. The six models end to end

- [ ] **PCA**: full SVD, no whitening, **59 components (≈ 90.3% of fit variance)**. The elbow rule chose it from {42, 55, 59, 63, 73}, which are the ranks for 80/90/95% variance plus the 90% rank ± 4. The score is the mean squared error of `inverse_transform(transform(x))`. No epochs and no seeds.
- [ ] **MLP autoencoder** (single minute):
  - 133 → 64 → 32 → **16** → 32 → 64 → 133, with ReLU on hidden layers;
  - linear bottleneck and output, because standardised values can be negative;
  - **22,485** parameters;
  - latent size picked by the elbow rule over {4, 8, 16, 32}.
- [ ] **LSTM autoencoder** (20-minute windows):
  - an LSTM encoder (64 units) whose last hidden state *h_n* is mapped by a linear layer to a 16-d latent;
  - the latent is repeated 20 times and fed to an LSTM decoder (64 units);
  - a linear layer maps each step back to 133 features;
  - **81,621** parameters; defaults fixed a priori, not tuned.
- [ ] **GRU**: the same architecture with a GRU cell (3 gates instead of 4), giving **63,637** parameters.
- [ ] **1D-CNN autoencoder** (20-minute windows):
  - features are channels;
  - 4 **causal dilated** conv layers (kernel 3, dilations 1/2/4/8, 32 channels, GELU, left padding only) give a receptive field of 1 + 2·(1+2+4+8) = **31 ≥ 20**;
  - the last time position goes through a linear layer to a 16-d latent;
  - a linear layer expands it to 32 channels × 20 steps, then a conv (k = 3) and a 1×1 conv give 133 features;
  - **41,013** parameters; not tuned.
- [ ] **Windowed MLP** (control):
  - flattens 20 × 133 = 2,660 values → 64 (ReLU, dropout 0.2) → **16** → 64 → 2,660;
  - Gaussian input noise of 0.1 sd makes it a denoising autoencoder;
  - learning rate 1e-4;
  - latent size from the rule "smallest of {8, 16, 32} within 5% of the best tune loss";
  - **345,332** parameters, the largest model.
- [ ] Explain why a bottleneck: it forces compression, so the network cannot learn the identity map. Normal traffic reconstructs well, and unusual traffic, ideally, does not.
- [ ] Explain why the windowed MLP exists: it sees the same 20 minutes as the LSTM and CNN but has no notion of order. It separates "more minutes of input" from "using the order of minutes".

#### 8. Training protocol

- [ ] Know the shared loop for every neural model: MSE loss, Adam, batch 256, up to 50 epochs, early stopping with patience 5 on tune loss (a strict improvement is needed), then the best-epoch weights are restored. The learning rate is 1e-3, except 1e-4 for the windowed MLP.
- [ ] Know the seeds: 0, 1 and 2, reported as mean ± sd. PCA is deterministic, so it has one run.
- [ ] Know the reproducibility steps:
  - seeded `random`, NumPy and PyTorch;
  - deterministic algorithms and 1 thread for PCA and MLP;
  - an audit retrained the MLP and got bit-identical weights and scores;
  - training-data SHA-256 checked before evaluation;
  - unit tests and CI on every change.
- [ ] Know that everything ran on CPU, and that training and inference times are recorded for the cost table.

#### 9. Threshold and sanity check

- [ ] Know the threshold: the **99th percentile of the model's own calibration scores** (≈ 1.01% calib alert rate), then frozen. A minute alerts when score > threshold.
- [ ] Explain why it is not tuned on test: that leaks labels into a setting. Test-derived operating points are reported only as retrospective comparisons.
- [ ] Explain why 99%: it encodes an assumed SOC budget of about 1 alert per 100 minutes, and it was fixed in the proposal before any result.
- [ ] Describe the synthetic sanity check:
  - inject into 1% of calib minutes;
  - add m ∈ {2, 4} fit-period sd to k ∈ {1, 5} random features per minute;
  - fixed seed 0, so every model sees the same altered minutes;
  - metric: ROC-AUC.
  It shows the model *reacts*; it is not detection and never touches test.
- [ ] Know the deviation from the proposal: the proposal said the sanity check would use the *tuning* block, but the code uses the *calibration* block. Know the team's reason.

#### 10. Evaluation metrics ("are they fair? which should we focus on?")

- [ ] Explain why not accuracy: predicting "normal" for every minute already scores 96.9% on test.
- [ ] Know the metrics at the frozen threshold: recall = TP/(TP+FN); precision = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN); FPR = FP/(FP+TN); alert rate = alerts / all minutes (the analyst's workload).
- [ ] Know the threshold-free metrics (test-derived, retrospective only):
  - **AP**: area under the precision–recall curve; a random scorer gets the prevalence, ≈ 0.031;
  - **recall at FPR ≤ 1%**: the best recall over thresholds whose FPR is at most 1%;
  - **standardised partial ROC-AUC** over FPR 0–1% (McClish): 0.5 is random, 1 is perfect.
- [ ] Explain why AP and partial AUC instead of full ROC-AUC: with 3% positives and a ~1% FPR budget, only the low-FPR region matters. Full ROC-AUC flatters models by counting FPRs no SOC would accept.
- [ ] Explain the fairness issue: at the frozen threshold, models land at **different test FPRs (0.85%–4.3%)**, so their recall mixes sensitivity with alert budget. Recall@FPR ≤ 1% and AP compare them on equal terms.
- [ ] Know how per-category recall is counted: in **minutes**, not episodes; SSH scan and spam are NaN (absent); idpscan has n = 9, too few to conclude anything.
- [ ] Be ready for "which metric should we focus on?": **recall at a controlled FPR or alert rate**, plus precision, because the SOC constraint is analyst time. F1 summarises one threshold only.
- [ ] Know the table checks: mean ± sd over 3 seeds, and a check that every run scored the same 43,181 minutes.

#### 11. Results (memorise the headline numbers)

| Model | Recall | Precision | F1 | Test FPR | AP | Recall@FPR≤1% | pAUC |
|---|---|---|---|---|---|---|---|
| PCA | 0.231 | 0.470 | 0.310 | 0.85% | 0.241 | 0.245 | 0.585 |
| MLP | 0.662 | 0.334 | 0.444 | 4.29% | **0.461** | **0.433** | **0.632** |
| Windowed MLP | 0.567 | 0.392 | **0.464** | 2.85% | 0.433 | 0.335 | 0.604 |
| LSTM | 0.600 | 0.365 | 0.454 | 3.39% | 0.426 | 0.327 | 0.603 |
| GRU | 0.542 | 0.399 | 0.459 | 2.65% | 0.421 | 0.324 | 0.600 |
| 1D-CNN | 0.611 | 0.375 | 0.462 | 3.36% | 0.446 | 0.346 | 0.605 |

- [ ] **RQ (i), nonlinear vs PCA: yes, clearly.** MLP vs PCA: AP 0.461 vs 0.241, recall@1% 0.433 vs 0.245, F1 0.444 vs 0.310.
- [ ] **RQ (ii), does temporal context help: no evidence.** Every windowed model (AP 0.42–0.45, recall@1% 0.32–0.35) is *below* the single-minute MLP (0.461, 0.433).
- [ ] **RQ (iii), recurrent vs convolutional:**
  - 1D-CNN ≥ LSTM ≥ GRU on AP (0.446 / 0.426 / 0.421), all within about one sd of each other;
  - LSTM and CNN are no better than the windowed-MLP control, so there is no evidence they use the *order* of minutes;
  - the CNN has the largest seed variance (recall sd 0.068).
- [ ] Know the trade-offs at the frozen threshold:
  - MLP has the best recall (0.662) but the highest FPR (4.3%, alert rate 6.2%);
  - PCA is the most precise (0.470) but misses most attacks;
  - the F1s of the windowed models are within 0.01 of each other, so do not over-read them.
- [ ] Know the **calibration-to-test shift**: a 1.0% calib alert rate became a 0.85–4.3% test FPR, consistent with the EDA drift (AUC 0.996). PCA is the only model whose FPR did not rise.
- [ ] Know the per-category recall:
  - scans are easy (neural models 0.92–1.0);
  - DoS is about 0.45–0.55;
  - **botnet separates the models: PCA 0.022 vs MLP 0.663** (temporal models 0.50–0.57);
  - idpscan is 5/9–6/9.
- [ ] Know the **sanity-check paradox**: PCA has the *highest* synthetic AUC (e.g. 0.881 at m = 2, k = 5) but the *worst* real detection. Sparse synthetic spikes are not what real attacks look like, so the check proves sensitivity, not detection.
- [ ] Know the cost (CPU training time):
  - MLP ≈ 3 s;
  - GRU ≈ 85 s; LSTM ≈ 136 s; windowed MLP ≈ 210 s;
  - CNN ≈ 679 s, but its timings were taken while other jobs ran, so they are inflated;
  - CNN inference is the slowest at 12.7 s.
- [ ] Have the one-line conclusion ready: **the simplest nonlinear model (single-minute MLP) is best on threshold-free metrics and cheapest to train; extra temporal machinery did not pay off on this benchmark.**

#### 12. Results discussion: loss curves and "did it learn anything?"

- [ ] Describe the loss curves: train loss keeps falling while tune loss bottoms out early. The LSTM's best epochs were 4, 2 and 12; the GRU's were 2, 3 and 2. For LSTM seed 0 at the stop, train loss is ≈ 0.49 and tune loss ≈ 0.70. The gap reflects fit→tune drift plus some overfitting, and early stopping caught it.
- [ ] Know the windowed MLP's first version: hidden 128, no regularisation, lr 1e-3. Its tune loss was lowest at **epoch 1**, so it was memorising. Dropout, input noise and a lower learning rate fixed that, and were judged on tune loss only.
- [ ] Know the MLP latent sweep tune MSE: 0.76 (4) → 0.69 (8) → 0.58 (16) → 0.44 (32). The elbow rule picked 16. **Lower reconstruction loss ≠ better detection**: a wider bottleneck can reconstruct attacks too.
- [ ] Know the PCA tune MSE: 0.31 at 42 components down to 0.11 at 73. The variance is spread out (10 components ≈ 46%, 40 ≈ 79%), so the traffic is genuinely high-dimensional.
- [ ] Give the evidence that the models learned something real:
  - AP 0.46 vs a random 0.031;
  - sanity AUC ≥ 0.97 for large shifts;
  - botnet is caught by the nonlinear models but not by PCA;
  - per-feature errors point at specific counters.
- [ ] Know the interpretability tools: per-feature squared error for each alert (`feature_errors.npy`) says *which counters* were unusual; PCA components; saved latents (`latents.npy`) for embedding plots. Be ready to explain any embedding or loss figure in the deck.

#### 13. Typical data-science problems and what we did about them

- [ ] **Leakage**: fit-only preprocessing; no `Row` or calendar features; chronological splits; a frozen threshold; test labels never set anything.
- [ ] **Class imbalance**: one-class framing, and metrics that handle imbalance (AP, recall@FPR, partial AUC).
- [ ] **Distribution drift**: measured in EDA, and the observed test FPR is reported instead of assuming the 1% carries over.
- [ ] **Overfitting**: early stopping on tune loss, bottlenecks, small fixed grids, and dropout plus denoising for the windowed MLP.
- [ ] **Reproducibility and correctness**: 3 seeds, the audit, data hashes, one shared pipeline, unit tests and CI.
- [ ] **Fairness of comparison**:
  - same preprocessing, eligible minutes, scoring minute and threshold rule for every model;
  - the asymmetry to admit: PCA, MLP and windowed MLP had small selection grids, while LSTM, GRU and CNN ran at fixed defaults (zero tuning budget).

#### 14. Limitations (have three ready without notes)

- [ ] Minute aggregation dilutes short or low-volume attacks and loses flow order within a minute.
- [ ] Background "normal" training traffic may contain unlabelled attacks.
- [ ] Excluding blacklist narrows the scope.
- [ ] Drift inflates test FPR.
- [ ] SSH scan and spam are absent from test, and idpscan has only 9 minutes.
- [ ] Minutes right after a gap are never scored (19-minute blind spot).
- [ ] The data is one ISP in 2016, so these are benchmark results, not deployment readiness.
- [ ] Autoencoders can reconstruct attacks well, as PCA did on botnet.
- [ ] Only 3 seeds and no significance test.
- [ ] LSTM and CNN were not tuned.

#### 15. Likely professor questions: answer each aloud in under 30 s

- [ ] "Validation has no attacks, so how do you know model selection was right?" We don't, for detection. Selection is on reconstruction plus the sanity check, and that is a stated limitation.
- [ ] "Your test FPR is 4%, not 1%. Is that a failure?" It is drift, shown in EDA (AUC 0.996), and we report it honestly. Fix: recalibrate the threshold periodically on recent traffic (future work).
- [ ] "Why score only the final minute?" So every alert maps to one minute, the same as the single-minute models, and the comparison is like-for-like.
- [ ] "Are the differences significant?" MLP vs LSTM AP is 0.461 ± 0.010 vs 0.426 ± 0.018, about 2 sd apart. The temporal models are within 1 sd of each other. With 3 seeds there was no formal test, so claim only the large gaps.
- [ ] "Why is MLP recall high but precision low?" Its frozen threshold landed at a 4.3% test FPR; drift pushed more benign minutes over it. Compare at a fixed FPR instead (recall@1% 0.433, still the best).
- [ ] "What would you do next?" Recalibrate on recent traffic; use episode-level metrics; tune the LSTM and CNN with the same budget; run more seeds; add Isolation Forest; move to flow-level data.

#### 16. Presentation and Q&A

- [ ] Know the main takeaway of every slide you present in one sentence, and how it leads to the conclusion.
- [ ] Be able to explain any figure in the deck: what each axis is, what the reader should notice, and why it matters.
- [ ] Prepare **two technical questions** for other groups (5 marks for participation in their presentations).

#### 17. Your deep dive: questions the professor will direct at you

- [ ] Explain the `read_split` guards: the X and Y `Row` keys must be identical, unique and strictly increasing, or it raises an error. `Row` is parsed with `%Y%m%d%H%M` into a DatetimeIndex and then **dropped** from X. Explain why: a re-exported file could otherwise silently pair the wrong label with a minute.
- [ ] Describe the `Part` dataclass (`X`, `y`, `labels`, `segment` on one index). `y` = any of the 7 attack labels > 0. `labels` keeps the 7 counts for per-category recall, and blacklist is dropped.
- [ ] Explain `segment_ids`: a new segment starts wherever the next timestamp is not exactly 1 minute later, and ids restart at 0 in each part. So the fit part has 3 segments (start, after DST, after the outage), and tune, calib and test have 1 each.
- [ ] Derive the split sizes: `FIT_ROWS = 78,609`; the rest is 19,653; `int(19,653 × 0.5)` = 9,826 tune, leaving 9,827 calib. Know the end timestamps of each part.
- [ ] Know the real-data tests: they pin the split sizes, gap dates and label alignment, so a changed CSV fails CI.
- [ ] Explain the synthetic-injection mechanics:
  - `round(0.01 × n)` rows are chosen without replacement;
  - each row gets its *own* k distinct columns (an argsort of random keys);
  - the shift is added in standardised space;
  - the seed is fixed at 0 rather than the run seed, so every model sees the same altered rows.
- [ ] Explain the sanity grid: (m, k) ∈ {2, 4} × {1, 5}, chosen so the check shows *where* models start reacting instead of every model scoring AUC = 1.
- [ ] Explain why m = 2, k = 1 is near chance (≈ 0.55): one 2-sd bump on one of 133 features adds only about 2²/133 ≈ 0.03 to the MSE, while normal scores are ≈ 0.5–0.7.
- [ ] Explain the windowed subtlety: an injected minute also raises the scores of later windows that contain it, but those windows stay labelled "clean" because only the final minute is scored.
- [ ] Know the sanity results: PCA tops it (0.881 at m = 2, k = 5) while the neural models score 0.74–0.78, yet PCA is worst on real attacks. That is why the check only shows that a model reacts, not that it detects.
- [ ] Describe the LSTM architecture:
  - the encoder's `h_n` (not `c_n`) feeds a linear layer to the 16-d latent;
  - the latent is repeated per time step (RepeatVector-style) and fed to the decoder;
  - a linear layer gives 133 features per minute;
  - it works for any window length.
- [ ] Explain why read `h_n` and not `c_n`: `h_n` is the cell's exposed output, while `c_n` is internal memory. The single-vector bottleneck stops the model copying the window through.
- [ ] Derive the parameter counts. An LSTM layer has 4·(in·h + h·h + 2h); PyTorch keeps two bias vectors. That gives 50,944 (encoder) + 1,040 + 20,992 (decoder) + 8,645 = **81,621**. A GRU has 3 gates, so **63,637**.
- [ ] Explain LSTM vs GRU: gating fixes the vanishing gradients of a plain RNN; a GRU merges the gates (no separate cell state), so it is smaller and faster. Results: GRU recall 0.542 vs 0.600 and FPR 2.65% vs 3.39%, with AP almost equal (0.421 vs 0.426).
- [ ] Know the LSTM training: best epochs 4, 2 and 12 (runs of 9, 7 and 17 epochs), 28–81 s per seed. Tune loss plateaus at about 0.70 while train loss keeps falling.
- [ ] Explain why the LSTM did not beat the MLP (hedged): the decoder must rebuild 20 minutes from 16 numbers, so the final-minute reconstruction is "averaged" across the window. Attacks show up at the minute level, and the MLP spends its whole bottleneck on one minute.
- [ ] Know the EDA evidence (`scripts/eda_checks.py`): gap dates and lengths, drift AUC 0.996, the injection schedule, skewness before and after log1p. You own these numbers in Q&A.

---

## Soon Huici Kesha

**Your areas:** Preprocessing, gap-safe windows, 1D-CNN autoencoder.

#### 1. Problem statement (report 2 marks)

- [ ] Say the problem in one sentence: flag the minutes of ISP traffic that contain an attack, while keeping false alarms near 1% so a SOC analyst is not flooded.
- [ ] State the three research questions: (i) does a nonlinear model beat PCA? (ii) do the preceding minutes help? (iii) does recurrence (LSTM/GRU) or convolution (1D-CNN) use that context better?
- [ ] Explain the cost trade-off: a missed attack leaves systems exposed, and every false alarm costs analyst time. That is why we report precision, FPR and alert rate as well as recall.
- [ ] Explain the framing. It is one-class anomaly detection scored by reconstruction error, because the training file has **zero** positive minutes in all seven attack labels, so a supervised classifier cannot be trained.
- [ ] Answer "why not oversample (SMOTE) or reweight?": there are no training positives to oversample or reweight.
- [ ] Know our stance, "we do not assume more complexity means better detection", and the answer the results gave (see section 11).

#### 2. Review of applicable techniques (report 3 marks)

- [ ] Know the three cited papers: Maciá-Fernández et al. 2018 (UGR'16, built for cyclostationarity-based NIDS), Camacho et al. 2023 (data quality changes model comparisons on UGR'16) and Camacho et al. 2025 (quality tools, giving the UGR'16v2 release we use).
- [ ] Map the technique families to this data:
  - supervised classifiers: ruled out, no training positives;
  - linear subspace: PCA residual;
  - density or boundary methods: Isolation Forest, One-Class SVM, LOF;
  - reconstruction: autoencoders;
  - sequence models: LSTM autoencoders, dilated causal (TCN-style) 1D-CNNs.
- [ ] Explain why everything is scored by reconstruction error: one scoring rule for linear and deep models makes the comparison like-for-like, and the per-feature error shows which counters caused an alert.
- [ ] Explain why PCA is the baseline. It is the classic linear method for network-traffic anomalies (residual subspace). A linear autoencoder trained with MSE spans the same subspace as PCA, so PCA is the linear special case of our MLP autoencoder. *(Background only; not cited in the report.)*
- [ ] Answer "why not Isolation Forest or One-Class SVM?" honestly. They are out of scope: we fixed one scoring family so model differences are not scoring differences. A kernel OCSVM also scales poorly with ~78k rows. Name them as future work; do not claim we tested them.

#### 3. Data study ("is the data balanced?")

- [ ] Describe UGR'16v2: one row is one minute of traffic at a Spanish ISP, with 134 numeric flow counters (protocol, source/destination IP categories, port categories, TCP flags, type of service, packet- and byte-count bins).
- [ ] Know the sizes: train 98,262 minutes (19 Mar–31 May 2016); test 43,200 minutes (28 Jul–26 Aug 2016). `X*` files hold features and `Y*` files hold 8 label counts. They are joined on `Row` (`YYYYMMDDHHMM`, local time).
- [ ] Name the 8 labels: DoS, scan11, scan44, Neris botnet, UDP scan (idpscan), SSH scan, spam and **blacklist**.
- [ ] Explain why blacklist is excluded: it is non-zero in *every* minute of both files, including every attack minute, so it carries no signal. It marks IP reputation, not an attack.
- [ ] Answer "is it balanced?": no.
  - Train has 0% positives.
  - Test has 1,364 attack minutes (3.16%), all between 28 Jul and 9 Aug; the other 17 days have none.
  - After windowing, test has 1,357 attacks among 43,181 scored minutes (3.14%).
- [ ] Know the test attack minutes per category: DoS 450 (90 episodes); scan11 98; scan44 97; Neris botnet 732 (12 episodes, median 61 min long); idpscan 9; SSH scan 0; spam 0.
- [ ] Know the quality checks:
  - no missing, negative or infinite values, and no duplicate timestamps or rows;
  - groups that should partition the flows sum to the total in ≥ 99.998% of minutes;
  - port and TCP-flag groups sum to *more* than the total (overlapping bins), so they are kept unchanged;
  - spikes are kept, because they may be the anomalies.
- [ ] Know the gaps. The 02:00 hour on 27 Mar is missing (DST change), and an outage runs from 1 Apr 00:02 to 6 Apr 17:19. Together they remove 8,298 minutes. They are **not** imputed.
- [ ] Know the seasonality: a daily cycle (hourly median flows vary ×1.4, peaking at 17:00) and higher weekday than weekend volume.
- [ ] Know the drift. 16 features separate train from test with univariate AUC > 0.8, and a logistic regression separates the two periods with AUC 0.996. So we expected the 1% calibration alert rate *not* to hold on test.
- [ ] Know the injection schedule: test attacks run at 88–124 minutes per day, alternating odd and even hours. This is why `Row` and anything derived from it (hour, weekday, day index) is **never** a model input; a model would learn the schedule.
- [ ] Know the rule: test labels describe the data only. They never set the window length, threshold or any other setting.

#### 4. Preprocessing and normalisation

- [ ] Know the order: `log1p` → drop columns constant in the fit rows → z-score with the fit-row mean and **population** std (ddof=0, as in sklearn `StandardScaler`).
- [ ] Explain why log1p: the counts are heavy-tailed, and log1p lowers the median skewness from 6.0 to 0.8. The +1 keeps zeros defined.
- [ ] Explain the leakage rule: every step is learned on the **78,609 fit rows only**, then applied unchanged to tune, calib and test.
- [ ] Know that constant removal drops only `protocoligmp`, leaving **133** features.
- [ ] Explain why the 48 mostly-zero features (> 50% zeros) are kept: a rarely used port can be exactly what an attack lights up.
- [ ] Explain why we standardise: the score is the MSE averaged over features, so unscaled high-count features would dominate it. It is also why a synthetic shift of "m standard deviations" means something.
- [ ] Explain the consequence of drift: test minutes can lie outside the fit range, which raises scores on benign minutes and so raises the test FPR.

#### 5. Training, validation and test splits

- [ ] Know the chronological splits (no shuffling):
  - **fit** = rows 0–78,608 (19 Mar → 18 May 08:26);
  - **tune** = 9,826 minutes (→ 25 May 04:12);
  - **calib** = 9,827 minutes (→ 31 May);
  - **test** = the whole test file.
- [ ] Know each part's job:
  - fit: model weights and preprocessing;
  - tune: early stopping and model selection;
  - calib: the threshold and the synthetic sanity check only;
  - test: final evaluation, run once.
- [ ] Explain why validation is split in two: a threshold set on the same data used to pick the model would be optimistic. Each half covers about 6.8 days, close to a weekly cycle, so both see weekday and weekend traffic (`TUNE_FRACTION = 0.5`).
- [ ] Explain why not random or k-fold splits: minutes are autocorrelated, so a random split leaks neighbouring minutes. A deployed detector always trains on the past and scores the future.
- [ ] Know that mini-batches are shuffled **within the fit rows only**. Whole windows are shuffled, so the minute order inside each window is kept.
- [ ] Know that validation has no labelled attacks, so it measures *reconstruction*, not *detection*. Detection is measured on test only.

#### 6. Temporal windows

- [ ] Know that the window length is **20 minutes**, fixed before any model was evaluated. Have the team's agreed one-line reason ready (see the flags at the end). **Never** justify 20 with test episode lengths; that would use test labels to set a model setting.
- [ ] Describe the window: a window ending at minute *t* holds minutes *t−19 … t*. It is causal and never crosses a gap or partition boundary, so the first 19 minutes of each contiguous segment get no window.
- [ ] Know that all six models, PCA and the single-minute MLP included, are scored on the **same eligible minutes**: 43,181 test minutes with 1,357 attacks. The test file is one contiguous segment, so only its first 19 minutes are lost.
- [ ] Know the score: the training loss is MSE over the whole window, but the detection score is the MSE of the **final minute only**. Every alert therefore refers to one minute, *t*, as it does for the single-minute models.

#### 7. The six models end to end

- [ ] **PCA**: full SVD, no whitening, **59 components (≈ 90.3% of fit variance)**. The elbow rule chose it from {42, 55, 59, 63, 73}, which are the ranks for 80/90/95% variance plus the 90% rank ± 4. The score is the mean squared error of `inverse_transform(transform(x))`. No epochs and no seeds.
- [ ] **MLP autoencoder** (single minute):
  - 133 → 64 → 32 → **16** → 32 → 64 → 133, with ReLU on hidden layers;
  - linear bottleneck and output, because standardised values can be negative;
  - **22,485** parameters;
  - latent size picked by the elbow rule over {4, 8, 16, 32}.
- [ ] **LSTM autoencoder** (20-minute windows):
  - an LSTM encoder (64 units) whose last hidden state *h_n* is mapped by a linear layer to a 16-d latent;
  - the latent is repeated 20 times and fed to an LSTM decoder (64 units);
  - a linear layer maps each step back to 133 features;
  - **81,621** parameters; defaults fixed a priori, not tuned.
- [ ] **GRU**: the same architecture with a GRU cell (3 gates instead of 4), giving **63,637** parameters.
- [ ] **1D-CNN autoencoder** (20-minute windows):
  - features are channels;
  - 4 **causal dilated** conv layers (kernel 3, dilations 1/2/4/8, 32 channels, GELU, left padding only) give a receptive field of 1 + 2·(1+2+4+8) = **31 ≥ 20**;
  - the last time position goes through a linear layer to a 16-d latent;
  - a linear layer expands it to 32 channels × 20 steps, then a conv (k = 3) and a 1×1 conv give 133 features;
  - **41,013** parameters; not tuned.
- [ ] **Windowed MLP** (control):
  - flattens 20 × 133 = 2,660 values → 64 (ReLU, dropout 0.2) → **16** → 64 → 2,660;
  - Gaussian input noise of 0.1 sd makes it a denoising autoencoder;
  - learning rate 1e-4;
  - latent size from the rule "smallest of {8, 16, 32} within 5% of the best tune loss";
  - **345,332** parameters, the largest model.
- [ ] Explain why a bottleneck: it forces compression, so the network cannot learn the identity map. Normal traffic reconstructs well, and unusual traffic, ideally, does not.
- [ ] Explain why the windowed MLP exists: it sees the same 20 minutes as the LSTM and CNN but has no notion of order. It separates "more minutes of input" from "using the order of minutes".

#### 8. Training protocol

- [ ] Know the shared loop for every neural model: MSE loss, Adam, batch 256, up to 50 epochs, early stopping with patience 5 on tune loss (a strict improvement is needed), then the best-epoch weights are restored. The learning rate is 1e-3, except 1e-4 for the windowed MLP.
- [ ] Know the seeds: 0, 1 and 2, reported as mean ± sd. PCA is deterministic, so it has one run.
- [ ] Know the reproducibility steps:
  - seeded `random`, NumPy and PyTorch;
  - deterministic algorithms and 1 thread for PCA and MLP;
  - an audit retrained the MLP and got bit-identical weights and scores;
  - training-data SHA-256 checked before evaluation;
  - unit tests and CI on every change.
- [ ] Know that everything ran on CPU, and that training and inference times are recorded for the cost table.

#### 9. Threshold and sanity check

- [ ] Know the threshold: the **99th percentile of the model's own calibration scores** (≈ 1.01% calib alert rate), then frozen. A minute alerts when score > threshold.
- [ ] Explain why it is not tuned on test: that leaks labels into a setting. Test-derived operating points are reported only as retrospective comparisons.
- [ ] Explain why 99%: it encodes an assumed SOC budget of about 1 alert per 100 minutes, and it was fixed in the proposal before any result.
- [ ] Describe the synthetic sanity check:
  - inject into 1% of calib minutes;
  - add m ∈ {2, 4} fit-period sd to k ∈ {1, 5} random features per minute;
  - fixed seed 0, so every model sees the same altered minutes;
  - metric: ROC-AUC.
  It shows the model *reacts*; it is not detection and never touches test.
- [ ] Know the deviation from the proposal: the proposal said the sanity check would use the *tuning* block, but the code uses the *calibration* block. Know the team's reason.

#### 10. Evaluation metrics ("are they fair? which should we focus on?")

- [ ] Explain why not accuracy: predicting "normal" for every minute already scores 96.9% on test.
- [ ] Know the metrics at the frozen threshold: recall = TP/(TP+FN); precision = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN); FPR = FP/(FP+TN); alert rate = alerts / all minutes (the analyst's workload).
- [ ] Know the threshold-free metrics (test-derived, retrospective only):
  - **AP**: area under the precision–recall curve; a random scorer gets the prevalence, ≈ 0.031;
  - **recall at FPR ≤ 1%**: the best recall over thresholds whose FPR is at most 1%;
  - **standardised partial ROC-AUC** over FPR 0–1% (McClish): 0.5 is random, 1 is perfect.
- [ ] Explain why AP and partial AUC instead of full ROC-AUC: with 3% positives and a ~1% FPR budget, only the low-FPR region matters. Full ROC-AUC flatters models by counting FPRs no SOC would accept.
- [ ] Explain the fairness issue: at the frozen threshold, models land at **different test FPRs (0.85%–4.3%)**, so their recall mixes sensitivity with alert budget. Recall@FPR ≤ 1% and AP compare them on equal terms.
- [ ] Know how per-category recall is counted: in **minutes**, not episodes; SSH scan and spam are NaN (absent); idpscan has n = 9, too few to conclude anything.
- [ ] Be ready for "which metric should we focus on?": **recall at a controlled FPR or alert rate**, plus precision, because the SOC constraint is analyst time. F1 summarises one threshold only.
- [ ] Know the table checks: mean ± sd over 3 seeds, and a check that every run scored the same 43,181 minutes.

#### 11. Results (memorise the headline numbers)

| Model | Recall | Precision | F1 | Test FPR | AP | Recall@FPR≤1% | pAUC |
|---|---|---|---|---|---|---|---|
| PCA | 0.231 | 0.470 | 0.310 | 0.85% | 0.241 | 0.245 | 0.585 |
| MLP | 0.662 | 0.334 | 0.444 | 4.29% | **0.461** | **0.433** | **0.632** |
| Windowed MLP | 0.567 | 0.392 | **0.464** | 2.85% | 0.433 | 0.335 | 0.604 |
| LSTM | 0.600 | 0.365 | 0.454 | 3.39% | 0.426 | 0.327 | 0.603 |
| GRU | 0.542 | 0.399 | 0.459 | 2.65% | 0.421 | 0.324 | 0.600 |
| 1D-CNN | 0.611 | 0.375 | 0.462 | 3.36% | 0.446 | 0.346 | 0.605 |

- [ ] **RQ (i), nonlinear vs PCA: yes, clearly.** MLP vs PCA: AP 0.461 vs 0.241, recall@1% 0.433 vs 0.245, F1 0.444 vs 0.310.
- [ ] **RQ (ii), does temporal context help: no evidence.** Every windowed model (AP 0.42–0.45, recall@1% 0.32–0.35) is *below* the single-minute MLP (0.461, 0.433).
- [ ] **RQ (iii), recurrent vs convolutional:**
  - 1D-CNN ≥ LSTM ≥ GRU on AP (0.446 / 0.426 / 0.421), all within about one sd of each other;
  - LSTM and CNN are no better than the windowed-MLP control, so there is no evidence they use the *order* of minutes;
  - the CNN has the largest seed variance (recall sd 0.068).
- [ ] Know the trade-offs at the frozen threshold:
  - MLP has the best recall (0.662) but the highest FPR (4.3%, alert rate 6.2%);
  - PCA is the most precise (0.470) but misses most attacks;
  - the F1s of the windowed models are within 0.01 of each other, so do not over-read them.
- [ ] Know the **calibration-to-test shift**: a 1.0% calib alert rate became a 0.85–4.3% test FPR, consistent with the EDA drift (AUC 0.996). PCA is the only model whose FPR did not rise.
- [ ] Know the per-category recall:
  - scans are easy (neural models 0.92–1.0);
  - DoS is about 0.45–0.55;
  - **botnet separates the models: PCA 0.022 vs MLP 0.663** (temporal models 0.50–0.57);
  - idpscan is 5/9–6/9.
- [ ] Know the **sanity-check paradox**: PCA has the *highest* synthetic AUC (e.g. 0.881 at m = 2, k = 5) but the *worst* real detection. Sparse synthetic spikes are not what real attacks look like, so the check proves sensitivity, not detection.
- [ ] Know the cost (CPU training time):
  - MLP ≈ 3 s;
  - GRU ≈ 85 s; LSTM ≈ 136 s; windowed MLP ≈ 210 s;
  - CNN ≈ 679 s, but its timings were taken while other jobs ran, so they are inflated;
  - CNN inference is the slowest at 12.7 s.
- [ ] Have the one-line conclusion ready: **the simplest nonlinear model (single-minute MLP) is best on threshold-free metrics and cheapest to train; extra temporal machinery did not pay off on this benchmark.**

#### 12. Results discussion: loss curves and "did it learn anything?"

- [ ] Describe the loss curves: train loss keeps falling while tune loss bottoms out early. The LSTM's best epochs were 4, 2 and 12; the GRU's were 2, 3 and 2. For LSTM seed 0 at the stop, train loss is ≈ 0.49 and tune loss ≈ 0.70. The gap reflects fit→tune drift plus some overfitting, and early stopping caught it.
- [ ] Know the windowed MLP's first version: hidden 128, no regularisation, lr 1e-3. Its tune loss was lowest at **epoch 1**, so it was memorising. Dropout, input noise and a lower learning rate fixed that, and were judged on tune loss only.
- [ ] Know the MLP latent sweep tune MSE: 0.76 (4) → 0.69 (8) → 0.58 (16) → 0.44 (32). The elbow rule picked 16. **Lower reconstruction loss ≠ better detection**: a wider bottleneck can reconstruct attacks too.
- [ ] Know the PCA tune MSE: 0.31 at 42 components down to 0.11 at 73. The variance is spread out (10 components ≈ 46%, 40 ≈ 79%), so the traffic is genuinely high-dimensional.
- [ ] Give the evidence that the models learned something real:
  - AP 0.46 vs a random 0.031;
  - sanity AUC ≥ 0.97 for large shifts;
  - botnet is caught by the nonlinear models but not by PCA;
  - per-feature errors point at specific counters.
- [ ] Know the interpretability tools: per-feature squared error for each alert (`feature_errors.npy`) says *which counters* were unusual; PCA components; saved latents (`latents.npy`) for embedding plots. Be ready to explain any embedding or loss figure in the deck.

#### 13. Typical data-science problems and what we did about them

- [ ] **Leakage**: fit-only preprocessing; no `Row` or calendar features; chronological splits; a frozen threshold; test labels never set anything.
- [ ] **Class imbalance**: one-class framing, and metrics that handle imbalance (AP, recall@FPR, partial AUC).
- [ ] **Distribution drift**: measured in EDA, and the observed test FPR is reported instead of assuming the 1% carries over.
- [ ] **Overfitting**: early stopping on tune loss, bottlenecks, small fixed grids, and dropout plus denoising for the windowed MLP.
- [ ] **Reproducibility and correctness**: 3 seeds, the audit, data hashes, one shared pipeline, unit tests and CI.
- [ ] **Fairness of comparison**:
  - same preprocessing, eligible minutes, scoring minute and threshold rule for every model;
  - the asymmetry to admit: PCA, MLP and windowed MLP had small selection grids, while LSTM, GRU and CNN ran at fixed defaults (zero tuning budget).

#### 14. Limitations (have three ready without notes)

- [ ] Minute aggregation dilutes short or low-volume attacks and loses flow order within a minute.
- [ ] Background "normal" training traffic may contain unlabelled attacks.
- [ ] Excluding blacklist narrows the scope.
- [ ] Drift inflates test FPR.
- [ ] SSH scan and spam are absent from test, and idpscan has only 9 minutes.
- [ ] Minutes right after a gap are never scored (19-minute blind spot).
- [ ] The data is one ISP in 2016, so these are benchmark results, not deployment readiness.
- [ ] Autoencoders can reconstruct attacks well, as PCA did on botnet.
- [ ] Only 3 seeds and no significance test.
- [ ] LSTM and CNN were not tuned.

#### 15. Likely professor questions: answer each aloud in under 30 s

- [ ] "Validation has no attacks, so how do you know model selection was right?" We don't, for detection. Selection is on reconstruction plus the sanity check, and that is a stated limitation.
- [ ] "Your test FPR is 4%, not 1%. Is that a failure?" It is drift, shown in EDA (AUC 0.996), and we report it honestly. Fix: recalibrate the threshold periodically on recent traffic (future work).
- [ ] "Why score only the final minute?" So every alert maps to one minute, the same as the single-minute models, and the comparison is like-for-like.
- [ ] "Are the differences significant?" MLP vs LSTM AP is 0.461 ± 0.010 vs 0.426 ± 0.018, about 2 sd apart. The temporal models are within 1 sd of each other. With 3 seeds there was no formal test, so claim only the large gaps.
- [ ] "Why is MLP recall high but precision low?" Its frozen threshold landed at a 4.3% test FPR; drift pushed more benign minutes over it. Compare at a fixed FPR instead (recall@1% 0.433, still the best).
- [ ] "What would you do next?" Recalibrate on recent traffic; use episode-level metrics; tune the LSTM and CNN with the same budget; run more seeds; add Isolation Forest; move to flow-level data.

#### 16. Presentation and Q&A

- [ ] Know the main takeaway of every slide you present in one sentence, and how it leads to the conclusion.
- [ ] Be able to explain any figure in the deck: what each axis is, what the reader should notice, and why it matters.
- [ ] Prepare **two technical questions** for other groups (5 marks for participation in their presentations).

#### 17. Your deep dive: questions the professor will direct at you

- [ ] Explain `Preprocessor.fit`: it computes log1p, then keeps columns whose fit-part std > 0, and stores `feature_names_`, `mean_` and `scale_`.
- [ ] Explain `transform`: it raises `RuntimeError` if not fitted and `ValueError` if a kept column is missing (a dropped constant column may be absent), and returns float32.
- [ ] Explain ddof = 0 (population std): it matches sklearn's `StandardScaler`. With 78k rows the difference from ddof = 1 is negligible, but it keeps us consistent with the library.
- [ ] Know the numbers: median skewness 6.0 → 0.8 after log1p; `protocoligmp` dropped; 134 → 133 features; 48 mostly-zero features kept.
- [ ] Explain why the scaler is fitted on fit rows and not all training rows: fitting on tune or calib would let preprocessing see the traffic used to pick models and set the threshold. That is leakage, and unlike deployment.
- [ ] Explain the `eligible_positions` algorithm: find the segment boundaries; within each segment of length ≥ L, positions run from `start + L − 1` to `end − 1`; shorter segments give no windows. With L = 1 every row is eligible.
- [ ] Explain `make_windows`: it returns `(m, 20, 133)` float32 windows plus their end positions. The window at position *t* is `X[t−19 : t+1]`.
- [ ] Know the memory cost: test windows alone are 43,181 × 20 × 133 × 4 B ≈ **460 MB**, and fit windows are ≈ 840 MB. That is why window tensors are expensive on CPU.
- [ ] Know the real-data boundary checks (your week-2 branch): no window crosses the DST gap, the outage or a partition boundary.
- [ ] Describe the CNN architecture in detail:
  - 4 causal dilated conv layers: kernel 3, dilations 1, 2, 4, 8, 32 channels, GELU, **left-only padding**, so position *t* never sees later minutes;
  - receptive field 1 + (k−1)·Σd = 31 minutes, so the **last position sees the whole 20-minute window**;
  - the encoder reads that last position and a linear layer maps it to the 16-d latent;
  - decoder: Linear(16 → 32 × 20) → GELU → Conv1d(32, 32, k = 3, pad 1) → GELU → 1×1 Conv1d(32 → 133), with a linear output.
- [ ] Explain the design history:
  - an earlier plain or strided stack covered only the **last 7 minutes**, which made the CNN a shorter-context model than the LSTM;
  - a transposed-conv decoder gave the scored final minute a single edge tap;
  - the linear expansion gives every output minute its own weights, which is why the model needs `window_length = 20` exactly.
- [ ] Derive the **41,013** parameters: 12,800 + 3 × 3,104 + 528 + 10,880 + 3,104 + 4,389. The CNN is about half the LSTM's size, with the same 16-d latent. It was not tuned, matching the LSTM's zero tuning budget.
- [ ] Explain why GELU: it is smooth and does not create "dead" units the way ReLU can.
- [ ] Know the CNN results:
  - best temporal AP (0.446) and recall@1% (0.346);
  - recall 0.611 ± **0.068**, the largest seed variance, so be ready to explain it;
  - best idpscan (6/9) and scan44 (0.997);
  - slowest model: training 679 ± 363 s (contended machine) and inference 12.7 s.
- [ ] **Fix the report before presenting.** The `main.tex` CNN paragraph still describes the *old* model: stride 2, a 16 × 10 bottleneck map, a ConvTranspose decoder and 28,821 parameters. Rewrite it to match `models/cnn.py`, and confirm PR #6 is merged into `working`.

---

## Vanessa Lim Dian Qin

**Your areas:** Threshold, evaluation metrics, results format, comparison table, windowed MLP control.

#### 1. Problem statement (report 2 marks)

- [ ] Say the problem in one sentence: flag the minutes of ISP traffic that contain an attack, while keeping false alarms near 1% so a SOC analyst is not flooded.
- [ ] State the three research questions: (i) does a nonlinear model beat PCA? (ii) do the preceding minutes help? (iii) does recurrence (LSTM/GRU) or convolution (1D-CNN) use that context better?
- [ ] Explain the cost trade-off: a missed attack leaves systems exposed, and every false alarm costs analyst time. That is why we report precision, FPR and alert rate as well as recall.
- [ ] Explain the framing. It is one-class anomaly detection scored by reconstruction error, because the training file has **zero** positive minutes in all seven attack labels, so a supervised classifier cannot be trained.
- [ ] Answer "why not oversample (SMOTE) or reweight?": there are no training positives to oversample or reweight.
- [ ] Know our stance, "we do not assume more complexity means better detection", and the answer the results gave (see section 11).

#### 2. Review of applicable techniques (report 3 marks)

- [ ] Know the three cited papers: Maciá-Fernández et al. 2018 (UGR'16, built for cyclostationarity-based NIDS), Camacho et al. 2023 (data quality changes model comparisons on UGR'16) and Camacho et al. 2025 (quality tools, giving the UGR'16v2 release we use).
- [ ] Map the technique families to this data:
  - supervised classifiers: ruled out, no training positives;
  - linear subspace: PCA residual;
  - density or boundary methods: Isolation Forest, One-Class SVM, LOF;
  - reconstruction: autoencoders;
  - sequence models: LSTM autoencoders, dilated causal (TCN-style) 1D-CNNs.
- [ ] Explain why everything is scored by reconstruction error: one scoring rule for linear and deep models makes the comparison like-for-like, and the per-feature error shows which counters caused an alert.
- [ ] Explain why PCA is the baseline. It is the classic linear method for network-traffic anomalies (residual subspace). A linear autoencoder trained with MSE spans the same subspace as PCA, so PCA is the linear special case of our MLP autoencoder. *(Background only; not cited in the report.)*
- [ ] Answer "why not Isolation Forest or One-Class SVM?" honestly. They are out of scope: we fixed one scoring family so model differences are not scoring differences. A kernel OCSVM also scales poorly with ~78k rows. Name them as future work; do not claim we tested them.

#### 3. Data study ("is the data balanced?")

- [ ] Describe UGR'16v2: one row is one minute of traffic at a Spanish ISP, with 134 numeric flow counters (protocol, source/destination IP categories, port categories, TCP flags, type of service, packet- and byte-count bins).
- [ ] Know the sizes: train 98,262 minutes (19 Mar–31 May 2016); test 43,200 minutes (28 Jul–26 Aug 2016). `X*` files hold features and `Y*` files hold 8 label counts. They are joined on `Row` (`YYYYMMDDHHMM`, local time).
- [ ] Name the 8 labels: DoS, scan11, scan44, Neris botnet, UDP scan (idpscan), SSH scan, spam and **blacklist**.
- [ ] Explain why blacklist is excluded: it is non-zero in *every* minute of both files, including every attack minute, so it carries no signal. It marks IP reputation, not an attack.
- [ ] Answer "is it balanced?": no.
  - Train has 0% positives.
  - Test has 1,364 attack minutes (3.16%), all between 28 Jul and 9 Aug; the other 17 days have none.
  - After windowing, test has 1,357 attacks among 43,181 scored minutes (3.14%).
- [ ] Know the test attack minutes per category: DoS 450 (90 episodes); scan11 98; scan44 97; Neris botnet 732 (12 episodes, median 61 min long); idpscan 9; SSH scan 0; spam 0.
- [ ] Know the quality checks:
  - no missing, negative or infinite values, and no duplicate timestamps or rows;
  - groups that should partition the flows sum to the total in ≥ 99.998% of minutes;
  - port and TCP-flag groups sum to *more* than the total (overlapping bins), so they are kept unchanged;
  - spikes are kept, because they may be the anomalies.
- [ ] Know the gaps. The 02:00 hour on 27 Mar is missing (DST change), and an outage runs from 1 Apr 00:02 to 6 Apr 17:19. Together they remove 8,298 minutes. They are **not** imputed.
- [ ] Know the seasonality: a daily cycle (hourly median flows vary ×1.4, peaking at 17:00) and higher weekday than weekend volume.
- [ ] Know the drift. 16 features separate train from test with univariate AUC > 0.8, and a logistic regression separates the two periods with AUC 0.996. So we expected the 1% calibration alert rate *not* to hold on test.
- [ ] Know the injection schedule: test attacks run at 88–124 minutes per day, alternating odd and even hours. This is why `Row` and anything derived from it (hour, weekday, day index) is **never** a model input; a model would learn the schedule.
- [ ] Know the rule: test labels describe the data only. They never set the window length, threshold or any other setting.

#### 4. Preprocessing and normalisation

- [ ] Know the order: `log1p` → drop columns constant in the fit rows → z-score with the fit-row mean and **population** std (ddof=0, as in sklearn `StandardScaler`).
- [ ] Explain why log1p: the counts are heavy-tailed, and log1p lowers the median skewness from 6.0 to 0.8. The +1 keeps zeros defined.
- [ ] Explain the leakage rule: every step is learned on the **78,609 fit rows only**, then applied unchanged to tune, calib and test.
- [ ] Know that constant removal drops only `protocoligmp`, leaving **133** features.
- [ ] Explain why the 48 mostly-zero features (> 50% zeros) are kept: a rarely used port can be exactly what an attack lights up.
- [ ] Explain why we standardise: the score is the MSE averaged over features, so unscaled high-count features would dominate it. It is also why a synthetic shift of "m standard deviations" means something.
- [ ] Explain the consequence of drift: test minutes can lie outside the fit range, which raises scores on benign minutes and so raises the test FPR.

#### 5. Training, validation and test splits

- [ ] Know the chronological splits (no shuffling):
  - **fit** = rows 0–78,608 (19 Mar → 18 May 08:26);
  - **tune** = 9,826 minutes (→ 25 May 04:12);
  - **calib** = 9,827 minutes (→ 31 May);
  - **test** = the whole test file.
- [ ] Know each part's job:
  - fit: model weights and preprocessing;
  - tune: early stopping and model selection;
  - calib: the threshold and the synthetic sanity check only;
  - test: final evaluation, run once.
- [ ] Explain why validation is split in two: a threshold set on the same data used to pick the model would be optimistic. Each half covers about 6.8 days, close to a weekly cycle, so both see weekday and weekend traffic (`TUNE_FRACTION = 0.5`).
- [ ] Explain why not random or k-fold splits: minutes are autocorrelated, so a random split leaks neighbouring minutes. A deployed detector always trains on the past and scores the future.
- [ ] Know that mini-batches are shuffled **within the fit rows only**. Whole windows are shuffled, so the minute order inside each window is kept.
- [ ] Know that validation has no labelled attacks, so it measures *reconstruction*, not *detection*. Detection is measured on test only.

#### 6. Temporal windows

- [ ] Know that the window length is **20 minutes**, fixed before any model was evaluated. Have the team's agreed one-line reason ready (see the flags at the end). **Never** justify 20 with test episode lengths; that would use test labels to set a model setting.
- [ ] Describe the window: a window ending at minute *t* holds minutes *t−19 … t*. It is causal and never crosses a gap or partition boundary, so the first 19 minutes of each contiguous segment get no window.
- [ ] Know that all six models, PCA and the single-minute MLP included, are scored on the **same eligible minutes**: 43,181 test minutes with 1,357 attacks. The test file is one contiguous segment, so only its first 19 minutes are lost.
- [ ] Know the score: the training loss is MSE over the whole window, but the detection score is the MSE of the **final minute only**. Every alert therefore refers to one minute, *t*, as it does for the single-minute models.

#### 7. The six models end to end

- [ ] **PCA**: full SVD, no whitening, **59 components (≈ 90.3% of fit variance)**. The elbow rule chose it from {42, 55, 59, 63, 73}, which are the ranks for 80/90/95% variance plus the 90% rank ± 4. The score is the mean squared error of `inverse_transform(transform(x))`. No epochs and no seeds.
- [ ] **MLP autoencoder** (single minute):
  - 133 → 64 → 32 → **16** → 32 → 64 → 133, with ReLU on hidden layers;
  - linear bottleneck and output, because standardised values can be negative;
  - **22,485** parameters;
  - latent size picked by the elbow rule over {4, 8, 16, 32}.
- [ ] **LSTM autoencoder** (20-minute windows):
  - an LSTM encoder (64 units) whose last hidden state *h_n* is mapped by a linear layer to a 16-d latent;
  - the latent is repeated 20 times and fed to an LSTM decoder (64 units);
  - a linear layer maps each step back to 133 features;
  - **81,621** parameters; defaults fixed a priori, not tuned.
- [ ] **GRU**: the same architecture with a GRU cell (3 gates instead of 4), giving **63,637** parameters.
- [ ] **1D-CNN autoencoder** (20-minute windows):
  - features are channels;
  - 4 **causal dilated** conv layers (kernel 3, dilations 1/2/4/8, 32 channels, GELU, left padding only) give a receptive field of 1 + 2·(1+2+4+8) = **31 ≥ 20**;
  - the last time position goes through a linear layer to a 16-d latent;
  - a linear layer expands it to 32 channels × 20 steps, then a conv (k = 3) and a 1×1 conv give 133 features;
  - **41,013** parameters; not tuned.
- [ ] **Windowed MLP** (control):
  - flattens 20 × 133 = 2,660 values → 64 (ReLU, dropout 0.2) → **16** → 64 → 2,660;
  - Gaussian input noise of 0.1 sd makes it a denoising autoencoder;
  - learning rate 1e-4;
  - latent size from the rule "smallest of {8, 16, 32} within 5% of the best tune loss";
  - **345,332** parameters, the largest model.
- [ ] Explain why a bottleneck: it forces compression, so the network cannot learn the identity map. Normal traffic reconstructs well, and unusual traffic, ideally, does not.
- [ ] Explain why the windowed MLP exists: it sees the same 20 minutes as the LSTM and CNN but has no notion of order. It separates "more minutes of input" from "using the order of minutes".

#### 8. Training protocol

- [ ] Know the shared loop for every neural model: MSE loss, Adam, batch 256, up to 50 epochs, early stopping with patience 5 on tune loss (a strict improvement is needed), then the best-epoch weights are restored. The learning rate is 1e-3, except 1e-4 for the windowed MLP.
- [ ] Know the seeds: 0, 1 and 2, reported as mean ± sd. PCA is deterministic, so it has one run.
- [ ] Know the reproducibility steps:
  - seeded `random`, NumPy and PyTorch;
  - deterministic algorithms and 1 thread for PCA and MLP;
  - an audit retrained the MLP and got bit-identical weights and scores;
  - training-data SHA-256 checked before evaluation;
  - unit tests and CI on every change.
- [ ] Know that everything ran on CPU, and that training and inference times are recorded for the cost table.

#### 9. Threshold and sanity check

- [ ] Know the threshold: the **99th percentile of the model's own calibration scores** (≈ 1.01% calib alert rate), then frozen. A minute alerts when score > threshold.
- [ ] Explain why it is not tuned on test: that leaks labels into a setting. Test-derived operating points are reported only as retrospective comparisons.
- [ ] Explain why 99%: it encodes an assumed SOC budget of about 1 alert per 100 minutes, and it was fixed in the proposal before any result.
- [ ] Describe the synthetic sanity check:
  - inject into 1% of calib minutes;
  - add m ∈ {2, 4} fit-period sd to k ∈ {1, 5} random features per minute;
  - fixed seed 0, so every model sees the same altered minutes;
  - metric: ROC-AUC.
  It shows the model *reacts*; it is not detection and never touches test.
- [ ] Know the deviation from the proposal: the proposal said the sanity check would use the *tuning* block, but the code uses the *calibration* block. Know the team's reason.

#### 10. Evaluation metrics ("are they fair? which should we focus on?")

- [ ] Explain why not accuracy: predicting "normal" for every minute already scores 96.9% on test.
- [ ] Know the metrics at the frozen threshold: recall = TP/(TP+FN); precision = TP/(TP+FP); F1 = 2TP/(2TP+FP+FN); FPR = FP/(FP+TN); alert rate = alerts / all minutes (the analyst's workload).
- [ ] Know the threshold-free metrics (test-derived, retrospective only):
  - **AP**: area under the precision–recall curve; a random scorer gets the prevalence, ≈ 0.031;
  - **recall at FPR ≤ 1%**: the best recall over thresholds whose FPR is at most 1%;
  - **standardised partial ROC-AUC** over FPR 0–1% (McClish): 0.5 is random, 1 is perfect.
- [ ] Explain why AP and partial AUC instead of full ROC-AUC: with 3% positives and a ~1% FPR budget, only the low-FPR region matters. Full ROC-AUC flatters models by counting FPRs no SOC would accept.
- [ ] Explain the fairness issue: at the frozen threshold, models land at **different test FPRs (0.85%–4.3%)**, so their recall mixes sensitivity with alert budget. Recall@FPR ≤ 1% and AP compare them on equal terms.
- [ ] Know how per-category recall is counted: in **minutes**, not episodes; SSH scan and spam are NaN (absent); idpscan has n = 9, too few to conclude anything.
- [ ] Be ready for "which metric should we focus on?": **recall at a controlled FPR or alert rate**, plus precision, because the SOC constraint is analyst time. F1 summarises one threshold only.
- [ ] Know the table checks: mean ± sd over 3 seeds, and a check that every run scored the same 43,181 minutes.

#### 11. Results (memorise the headline numbers)

| Model | Recall | Precision | F1 | Test FPR | AP | Recall@FPR≤1% | pAUC |
|---|---|---|---|---|---|---|---|
| PCA | 0.231 | 0.470 | 0.310 | 0.85% | 0.241 | 0.245 | 0.585 |
| MLP | 0.662 | 0.334 | 0.444 | 4.29% | **0.461** | **0.433** | **0.632** |
| Windowed MLP | 0.567 | 0.392 | **0.464** | 2.85% | 0.433 | 0.335 | 0.604 |
| LSTM | 0.600 | 0.365 | 0.454 | 3.39% | 0.426 | 0.327 | 0.603 |
| GRU | 0.542 | 0.399 | 0.459 | 2.65% | 0.421 | 0.324 | 0.600 |
| 1D-CNN | 0.611 | 0.375 | 0.462 | 3.36% | 0.446 | 0.346 | 0.605 |

- [ ] **RQ (i), nonlinear vs PCA: yes, clearly.** MLP vs PCA: AP 0.461 vs 0.241, recall@1% 0.433 vs 0.245, F1 0.444 vs 0.310.
- [ ] **RQ (ii), does temporal context help: no evidence.** Every windowed model (AP 0.42–0.45, recall@1% 0.32–0.35) is *below* the single-minute MLP (0.461, 0.433).
- [ ] **RQ (iii), recurrent vs convolutional:**
  - 1D-CNN ≥ LSTM ≥ GRU on AP (0.446 / 0.426 / 0.421), all within about one sd of each other;
  - LSTM and CNN are no better than the windowed-MLP control, so there is no evidence they use the *order* of minutes;
  - the CNN has the largest seed variance (recall sd 0.068).
- [ ] Know the trade-offs at the frozen threshold:
  - MLP has the best recall (0.662) but the highest FPR (4.3%, alert rate 6.2%);
  - PCA is the most precise (0.470) but misses most attacks;
  - the F1s of the windowed models are within 0.01 of each other, so do not over-read them.
- [ ] Know the **calibration-to-test shift**: a 1.0% calib alert rate became a 0.85–4.3% test FPR, consistent with the EDA drift (AUC 0.996). PCA is the only model whose FPR did not rise.
- [ ] Know the per-category recall:
  - scans are easy (neural models 0.92–1.0);
  - DoS is about 0.45–0.55;
  - **botnet separates the models: PCA 0.022 vs MLP 0.663** (temporal models 0.50–0.57);
  - idpscan is 5/9–6/9.
- [ ] Know the **sanity-check paradox**: PCA has the *highest* synthetic AUC (e.g. 0.881 at m = 2, k = 5) but the *worst* real detection. Sparse synthetic spikes are not what real attacks look like, so the check proves sensitivity, not detection.
- [ ] Know the cost (CPU training time):
  - MLP ≈ 3 s;
  - GRU ≈ 85 s; LSTM ≈ 136 s; windowed MLP ≈ 210 s;
  - CNN ≈ 679 s, but its timings were taken while other jobs ran, so they are inflated;
  - CNN inference is the slowest at 12.7 s.
- [ ] Have the one-line conclusion ready: **the simplest nonlinear model (single-minute MLP) is best on threshold-free metrics and cheapest to train; extra temporal machinery did not pay off on this benchmark.**

#### 12. Results discussion: loss curves and "did it learn anything?"

- [ ] Describe the loss curves: train loss keeps falling while tune loss bottoms out early. The LSTM's best epochs were 4, 2 and 12; the GRU's were 2, 3 and 2. For LSTM seed 0 at the stop, train loss is ≈ 0.49 and tune loss ≈ 0.70. The gap reflects fit→tune drift plus some overfitting, and early stopping caught it.
- [ ] Know the windowed MLP's first version: hidden 128, no regularisation, lr 1e-3. Its tune loss was lowest at **epoch 1**, so it was memorising. Dropout, input noise and a lower learning rate fixed that, and were judged on tune loss only.
- [ ] Know the MLP latent sweep tune MSE: 0.76 (4) → 0.69 (8) → 0.58 (16) → 0.44 (32). The elbow rule picked 16. **Lower reconstruction loss ≠ better detection**: a wider bottleneck can reconstruct attacks too.
- [ ] Know the PCA tune MSE: 0.31 at 42 components down to 0.11 at 73. The variance is spread out (10 components ≈ 46%, 40 ≈ 79%), so the traffic is genuinely high-dimensional.
- [ ] Give the evidence that the models learned something real:
  - AP 0.46 vs a random 0.031;
  - sanity AUC ≥ 0.97 for large shifts;
  - botnet is caught by the nonlinear models but not by PCA;
  - per-feature errors point at specific counters.
- [ ] Know the interpretability tools: per-feature squared error for each alert (`feature_errors.npy`) says *which counters* were unusual; PCA components; saved latents (`latents.npy`) for embedding plots. Be ready to explain any embedding or loss figure in the deck.

#### 13. Typical data-science problems and what we did about them

- [ ] **Leakage**: fit-only preprocessing; no `Row` or calendar features; chronological splits; a frozen threshold; test labels never set anything.
- [ ] **Class imbalance**: one-class framing, and metrics that handle imbalance (AP, recall@FPR, partial AUC).
- [ ] **Distribution drift**: measured in EDA, and the observed test FPR is reported instead of assuming the 1% carries over.
- [ ] **Overfitting**: early stopping on tune loss, bottlenecks, small fixed grids, and dropout plus denoising for the windowed MLP.
- [ ] **Reproducibility and correctness**: 3 seeds, the audit, data hashes, one shared pipeline, unit tests and CI.
- [ ] **Fairness of comparison**:
  - same preprocessing, eligible minutes, scoring minute and threshold rule for every model;
  - the asymmetry to admit: PCA, MLP and windowed MLP had small selection grids, while LSTM, GRU and CNN ran at fixed defaults (zero tuning budget).

#### 14. Limitations (have three ready without notes)

- [ ] Minute aggregation dilutes short or low-volume attacks and loses flow order within a minute.
- [ ] Background "normal" training traffic may contain unlabelled attacks.
- [ ] Excluding blacklist narrows the scope.
- [ ] Drift inflates test FPR.
- [ ] SSH scan and spam are absent from test, and idpscan has only 9 minutes.
- [ ] Minutes right after a gap are never scored (19-minute blind spot).
- [ ] The data is one ISP in 2016, so these are benchmark results, not deployment readiness.
- [ ] Autoencoders can reconstruct attacks well, as PCA did on botnet.
- [ ] Only 3 seeds and no significance test.
- [ ] LSTM and CNN were not tuned.

#### 15. Likely professor questions: answer each aloud in under 30 s

- [ ] "Validation has no attacks, so how do you know model selection was right?" We don't, for detection. Selection is on reconstruction plus the sanity check, and that is a stated limitation.
- [ ] "Your test FPR is 4%, not 1%. Is that a failure?" It is drift, shown in EDA (AUC 0.996), and we report it honestly. Fix: recalibrate the threshold periodically on recent traffic (future work).
- [ ] "Why score only the final minute?" So every alert maps to one minute, the same as the single-minute models, and the comparison is like-for-like.
- [ ] "Are the differences significant?" MLP vs LSTM AP is 0.461 ± 0.010 vs 0.426 ± 0.018, about 2 sd apart. The temporal models are within 1 sd of each other. With 3 seeds there was no formal test, so claim only the large gaps.
- [ ] "Why is MLP recall high but precision low?" Its frozen threshold landed at a 4.3% test FPR; drift pushed more benign minutes over it. Compare at a fixed FPR instead (recall@1% 0.433, still the best).
- [ ] "What would you do next?" Recalibrate on recent traffic; use episode-level metrics; tune the LSTM and CNN with the same budget; run more seeds; add Isolation Forest; move to flow-level data.

#### 16. Presentation and Q&A

- [ ] Know the main takeaway of every slide you present in one sentence, and how it leads to the conclusion.
- [ ] Be able to explain any figure in the deck: what each axis is, what the reader should notice, and why it matters.
- [ ] Prepare **two technical questions** for other groups (5 marks for participation in their presentations).

#### 17. Your deep dive: questions the professor will direct at you

- [ ] Explain `fit_threshold`: `np.quantile(calib, 0.99)` with linear interpolation, returned as a plain `float` (JSON-safe). Non-finite scores raise an error, because a NaN would never alert or would make the threshold NaN.
- [ ] Explain the `threshold_metrics` conventions:
  - alerts use a strict `>`;
  - precision and F1 are 0.0 when there are no alerts, so a silent model scores as useless instead of crashing;
  - recall is NaN without positives and FPR is NaN without negatives;
  - F1 is computed as 2TP/(2TP+FP+FN), which equals 2PR/(P+R).
- [ ] Explain `ranking_metrics`:
  - AP uses `average_precision_score` (a step-wise sum, not trapezoidal, so it is not over-optimistic);
  - recall@FPR ≤ 1% is the max TPR over `roc_curve(..., drop_intermediate=False)` points with FPR ≤ 0.01; keeping every threshold means no point inside the budget is skipped;
  - partial AUC is `roc_auc_score(max_fpr=0.01)`, McClish-standardised so 0.5 = random.
- [ ] Explain `per_category_recall`: it counts minutes, not episodes. Detecting one minute of a 61-minute botnet episode is *not* counted as detecting the episode. It is NaN for categories with no positives, because "0% detected" would be false for SSH scan and spam. Labels must be aligned to the scored minutes first.
- [ ] Know the results format (`save_run` → `reports/results/<model>/seed<seed>/`):
  - `scores.csv` (timestamp, part, score);
  - `meta.json` (settings, threshold, metrics, timings, sanity results, calib alert rate);
  - `history.csv`, `feature_errors.npy`, `latents.npy`.
  Explain why: every table and figure is built from saved runs, never by re-running models.
- [ ] Explain `make_comparison_table.py`: it loads every run, **checks that all runs scored the same minutes**, recomputes the metrics and reports mean ± sd per model.
- [ ] Explain the windowed MLP architecture: flatten 2,660 → Linear 64 → ReLU → Dropout 0.2 → Linear 16 → Linear 64 → ReLU → Linear 2,660 → reshape. It has 345,332 parameters, almost all in the two 2,660 × 64 layers.
- [ ] Tell the regularisation story:
  - v1 (hidden 128, no dropout or noise, lr 1e-3) had its best tune loss at **epoch 1** for latents 8 and 16, so it was memorising;
  - the fix is dropout 0.2, Gaussian input noise of 0.1 sd, and lr 1e-4;
  - all three were fixed together and judged on tune loss only, never on test.
- [ ] Explain why denoising works: the loss compares the output with the *clean* input, so the model must learn structure rather than copy. Noise and dropout act only in `train()` mode, so tune loss and scoring see clean inputs.
- [ ] Explain the selection rule: the smallest latent in {8, 16, 32} whose best tune loss is within **5%** of the lowest. It picked **16**. Know your sweep's actual tune losses per latent.
- [ ] Explain why it is the control: it gets the same 20 minutes but no notion of order. LSTM, GRU and CNN did **not** beat it on AP, F1 or recall@1%, so there is no evidence they exploit the temporal order.
- [ ] Know its results:
  - best F1 (0.464) by a hair;
  - recall 0.567, test FPR 2.85%;
  - the most stable across seeds (AP sd 0.0008).
- [ ] Be ready for "are your metrics fair?":
  - use AP and recall@FPR instead of ROC-AUC under 3% prevalence;
  - frozen-threshold numbers mix sensitivity with drift-inflated FPR;
  - minute-level recall over-weights long episodes (botnet has 732 of the 1,357 attack minutes).

---

## Team flags to settle before the presentation

These came up while checking `main.tex` against the code on `working`:

- [ ] **CNN paragraph is outdated** in `main.tex`: it describes stride 2, a 16 × 10 map and 28,821 parameters; the code uses causal dilated convs, a linear decoder and 41,013 parameters (Kesha).
- [ ] **"Why 20 minutes?"** Agree one sentence everyone gives, based on operational reasoning or literature and *never* on test episode lengths (Phone + Kesha).
- [ ] **Model-selection wording:** PCA and MLP use the elbow rule; the windowed MLP uses the 5% rule; LSTM, GRU and CNN use fixed defaults. Agree how to state this asymmetry.
- [ ] **Sanity check block:** the proposal said tuning block; the code uses calibration. Agree the reason.
- [ ] **Report gaps:** PCA variance explained (≈ 90.3%) is not in the text; MLP widths still read "starting point 64-32-8"; the abstract finding and the Conclusion are TODO.
- [ ] **Timing caveat:** CNN train times were measured on a busy machine. Rerun them or say so.
