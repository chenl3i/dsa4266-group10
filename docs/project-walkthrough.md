# Project walkthrough: nonlinear and temporal autoencoders on UGR'16v2

A plain-language tour of the whole project, in report order. The numbers come from
`main.tex` (30 Oct draft), the code on `working` and the saved runs. Results may still
change before the 3 Nov lock.

---

## 1. Problem and research questions

- **Goal:** in per-minute ISP traffic data, flag the specific minutes that contain
  attacks while keeping false alarms low (target ≈ 1% of minutes).
- **Who cares:** a Security Operations Centre (SOC). A missed attack leaves systems
  exposed; every false alarm costs an analyst's time.
- **Main question:** do neural autoencoders beat a simple PCA baseline at detecting
  attacks?
  1. **Does nonlinearity help?** MLP vs PCA, both on single minutes.
  2. **Does preceding context help?** Sliding-window models vs single-minute models.
  3. **Which uses time context better?** LSTM vs 1D-CNN.
- **Stance:** we don't assume a more complex model detects better.

## 2. How detection works

- **An autoencoder** compresses each minute's 133 numbers down to 16, then tries to
  rebuild the original 133 from those 16. PCA does the same with 59 components.
- **It is trained on normal traffic only**, so it learns to rebuild *normal* minutes well.
- **An unusual minute** (e.g. an attack) gets rebuilt worse.
- **Reconstruction error** = mean of (input − rebuild)² over the features. A small
  error means the minute looks normal; a large error means it looks unusual and raises
  an alert.
- **Why this approach:** the training file has no attack minutes, so a normal
  classifier ("attack vs not") can't be trained. Learning what normal looks like and
  flagging departures from it is the one-class alternative.

## 3. Data

- **One row = one minute of traffic**, with 134 numeric count features: flows per
  protocol, source/destination IP category, port category, TCP flag, type of service,
  and packet-count and byte-count bins.
- **Files:** `Xtrain` / `Xtest` hold the traffic features; `Ytrain` / `Ytest` hold 8
  label counts per minute (how many flows belonged to each attack type). Each X row
  pairs with its Y row through the `Row` timestamp (`YYYYMMDDHHMM`, local time).
- **Size:** train 98,262 minutes (Mar–May 2016); test 43,200 minutes (Jul–Aug 2016).
- **Attacks:** training data has **zero** attack minutes. Test has **1,364 / 43,200
  attack minutes (3.16%)**, all between 28 Jul and 9 Aug.

| Test attack type | Minutes | Episodes | Median episode length |
|---|---|---|---|
| DoS | 450 | 90 | 4 min |
| Scan (scan11) | 98 | 24 | 4 min |
| Scan (scan44) | 97 | 24 | 4 min |
| Neris botnet | 732 | 12 | 61 min |
| UDP scan (idpscan) | 9 | 5 | 1 min |
| SSH scan, spam | 0 | 0 | – |

The rows add up to 1,386, not 1,364: some minutes contain two attack types.

- **Blacklist label excluded** from the target. It marks IP reputation (traffic
  involving known-bad IPs), not attack behaviour. It is non-zero in *every* minute of
  both files, so including it would label 100% of minutes as attacks.

## 4. Cleaning and EDA

1. Parsed the timestamp keys, sorted each partition chronologically and aligned the
   feature and label files. The loader refuses mismatched, repeated or unsorted keys.
2. Found no missing, negative or infinite values, no duplicate timestamps and no
   duplicate rows.
3. Checked that the feature groups add up. Groups that split each minute's flows into
   categories (IP, protocol, type of service, packet count) sum to the total flow count
   in ≥ 99.998% of minutes. Port and TCP-flag groups sum to more than the total
   (overlapping bins), so they are kept as they are.
4. Found two recording gaps: the DST hour on 27 Mar and an outage from 1 to 6 Apr,
   8,298 minutes in total. They are kept as gaps, not filled in.
5. Found a daily cycle (peak 17:00) and busier weekdays than weekends.
6. Found strong train→test drift: a classifier tells the two periods apart with
   AUC 0.996, so the 1% alert rate may not hold on test.
7. Found that test attacks follow an injection schedule (88–124 attack minutes per
   day, more in odd hours than even hours). So the timestamp, and anything derived from
   it like hour or weekday, is **never** a model input.
8. **Rule:** test labels are used only to describe the data, never to set any model
   setting.

## 5. Preprocessing

1. **log(1 + x)** reduces the effect of very large counts; median feature skewness
   drops from 6.0 to 0.8.
2. **Constant features are removed.** Only one feature is dropped (`protocoligmp`),
   leaving 133.
3. **Mostly-zero features are kept**, because they can still help identify unusual
   traffic.
4. **Standardise** with the mean and population SD of the **model-fitting rows only**,
   then apply those same values unchanged to the validation and test sets.

## 6. Target definition

- A minute counts as an **attack** if any of the 7 non-blacklist label counts is > 0.
- The 7 individual labels are kept too, to report recall per attack type.

## 7. Splits (chronological, no shuffling)

| Part | Minutes | Dates | Used for |
|---|---|---|---|
| Fit | 78,609 | 19 Mar – 18 May | training models and fitting preprocessing |
| Tune | 9,826 | 18 – 25 May | early stopping and model selection |
| Calib | 9,827 | 25 – 31 May | threshold and sanity check |
| Test | 43,200 | 28 Jul – 26 Aug | final evaluation only |

- **Why split by time, not randomly:** neighbouring minutes are alike, so a random
  split would leak information. A real detector also trains on the past and scores the
  future.
- **Why separate tune and calib:** the model is chosen to do well on tune, so its tune
  scores are optimistically low. A threshold set there would sit too low. Calib is
  untouched until the model is finished.
- **Why 50/50:** each half is ~6.8 days, close to a week, so both see weekdays and a
  weekend.
- **Neither has attacks:** validation measures how well a model rebuilds normal
  traffic, not detection. Detection is measured only on test.

## 8. Shared pipeline (every model goes through this)

1. Load and split the data.
2. Fit preprocessing on the model-fitting rows.
3. Build 20-minute windows ending at minute *t*. Windows never cross a gap or split
   boundary, so the first 19 minutes after each break get no window.
4. Train or fit the model.
5. Fix the threshold on the calibration block.
6. Write each run's scores and settings to a standard results format, from which all
   tables and figures are produced.
7. **Testing:** unit tests on small synthetic data, plus tests on the real files that
   pin the split sizes, gap dates and label alignment. CI runs them on every change.

All six models are scored on the **same 43,181 test minutes** (1,357 attacks). The 19
lost minutes are the start of the test file, which has no 20-minute history.

## 9. The six models

| Model | Sees | Idea in one line | Size |
|---|---|---|---|
| **PCA** | 1 minute | Linear: keep the top 59 directions of variation (≈ 90% of variance), rebuild from them | – |
| **MLP autoencoder** | 1 minute | Nonlinear: 133 → 64 → 32 → **16** → 32 → 64 → 133 | 22,485 params |
| **Windowed MLP** | 20 minutes, flattened | Control: same 20 minutes, but no sense of order (20 × 133 = 2,660 numbers → 64 → 16 → back) | 345,332 params |
| **LSTM autoencoder** | 20 minutes in order | Recurrent: reads the window minute by minute, squeezes it into 16 numbers, rebuilds it | 81,621 params |
| **GRU** | 20 minutes in order | Same as the LSTM with a simpler recurrent cell | 63,637 params |
| **1D-CNN autoencoder** | 20 minutes in order | Convolutional: causal dilated filters slide over time, squeeze into 16 numbers, rebuild | 41,013 params |

- **Why the bottleneck:** squeezing to 16 numbers stops the model from simply copying
  its input. It has to learn the main patterns of normal traffic.
- **Why the windowed MLP exists:** it gets the same 20 minutes as the LSTM and CNN but
  can't use their order. If the LSTM or CNN beat it, the gain comes from *using time
  order*, not just from *seeing more minutes*.
- **PCA vs MLP:** a linear autoencoder learns the same subspace as PCA, so the MLP
  differs from PCA mainly in its nonlinearity. That is what research question 1 tests.
- **CNN details:** "causal" means each position only sees its own minute and earlier
  ones. "Dilated" means the filters skip minutes (1, 2, 4, 8 apart), so four layers
  reach back 31 minutes and cover the whole 20-minute window.

## 10. Training

- **Same loop for every neural model:** MSE loss, Adam optimiser, batches of 256, up
  to 50 epochs. Learning rate 1e-3 (1e-4 for the windowed MLP).
- **Early stopping:** after each epoch, measure the loss on tune. Stop when it hasn't
  improved for 5 epochs, then restore the best epoch's weights.
- **Seeds:** each neural model is trained 3 times (seeds 0, 1, 2) and results are
  reported as mean ± sd. PCA is exact, so it needs one run.
- **Reproducibility:** seeds are fixed everywhere; a re-run of the MLP gave
  bit-identical weights and scores.

## 11. Model selection (judged on tune loss only)

| Model | What was chosen | Rule |
|---|---|---|
| PCA | 59 components from {42, 55, 59, 63, 73} | elbow of the tune-loss curve |
| MLP | bottleneck 16 from {4, 8, 16, 32} | elbow of the tune-loss curve |
| Windowed MLP | bottleneck 16 from {8, 16, 32} | smallest within 5% of the best tune loss |
| LSTM, GRU, CNN | fixed defaults (bottleneck 16) | not tuned |

- **Why not just pick the lowest loss:** a bigger bottleneck rebuilds everything
  better, attacks included. Lower reconstruction loss does not mean better detection,
  so the rules prefer the smaller model once gains flatten.
- **Windowed MLP fix:** its first version memorised the training data (best tune loss
  at epoch 1). Adding dropout, input noise and a lower learning rate fixed that.

## 12. Scoring, threshold and sanity check

- **Score:** LSTM, CNN and the windowed MLP rebuild all 20 minutes, but only the
  **final minute's** error is the score. Every alert then refers to one minute, the same
  as for the single-minute models.
- **Threshold:**
  - score every calibration minute;
  - set the threshold at the **99th percentile**, so 1% of calib minutes alert (about 1
    alarm per 100 normal minutes);
  - **freeze** it and apply it unchanged to test.
- **Sanity check:**
  - inject fake spikes into 1% of a copy of calib: add 2 or 4 standard deviations to 1
    or 5 random features;
  - check that the scores pick them out (ROC-AUC);
  - this shows the models *react* to unusual input; it is not attack detection.

## 13. Evaluation metrics

**At the frozen threshold** (what an analyst would actually see):

| Metric | Meaning |
|---|---|
| Recall | share of attack minutes that raised an alert |
| Precision | share of alerts that were real attacks |
| F1 | balance of precision and recall |
| FPR | share of normal minutes that raised an alert |
| Alert rate | share of all minutes that raised an alert (analyst workload) |

**Across all thresholds** (looking back at test, comparison only):

| Metric | Meaning |
|---|---|
| Average precision (AP) | area under the precision–recall curve; random guessing ≈ 0.031 |
| Recall @ FPR ≤ 1% | best recall if alerts on normal minutes are capped at 1% |
| Partial AUC (0–1% FPR) | ranking quality in the low-false-alarm region; 0.5 = random, 1 = perfect |

- **Why not accuracy:** saying "normal" every time is already 96.9% accurate.
- **Why both kinds:** the frozen threshold landed at different test FPRs for different
  models, so their recalls aren't on equal terms. Recall @ FPR ≤ 1% and AP compare them
  fairly.
- **Per attack type:** recall is counted in minutes, not episodes.

## 14. Results

| Model | Recall | Precision | F1 | Test FPR | AP | Recall@FPR≤1% |
|---|---|---|---|---|---|---|
| PCA | 0.231 | 0.470 | 0.310 | 0.85% | 0.241 | 0.245 |
| MLP | 0.662 | 0.334 | 0.444 | 4.29% | **0.461** | **0.433** |
| Windowed MLP | 0.567 | 0.392 | **0.464** | 2.85% | 0.433 | 0.335 |
| LSTM | 0.600 | 0.365 | 0.454 | 3.39% | 0.426 | 0.327 |
| GRU | 0.542 | 0.399 | 0.459 | 2.65% | 0.421 | 0.324 |
| 1D-CNN | 0.611 | 0.375 | 0.462 | 3.36% | 0.446 | 0.346 |

Neural models are the mean over 3 seeds.

**Answers to the research questions:**

1. **Nonlinearity helps, clearly.** The MLP roughly doubles PCA's AP (0.46 vs 0.24)
   and recall @ 1% FPR (0.43 vs 0.25).
2. **Preceding minutes did not help.** Every 20-minute model scores below the
   single-minute MLP on AP and recall @ 1% FPR.
3. **No clear winner between LSTM and CNN.** The CNN is slightly ahead (AP 0.446 vs
   0.426), but within about one standard deviation, and neither beats the windowed-MLP
   control. So there is no evidence they use the *order* of minutes.

**Per attack type (recall):**

| Model | DoS | scan11 | scan44 | Botnet | idpscan |
|---|---|---|---|---|---|
| PCA | 0.37 | 0.76 | 0.78 | **0.02** | 0.56 |
| MLP | 0.55 | 0.92 | 0.99 | **0.66** | 0.56 |
| LSTM / CNN | 0.51 / 0.54 | 0.94 | 0.99 | 0.57 | 0.56 / 0.67 |

- Scans are easy for every neural model.
- DoS is caught about half the time.
- **Botnet is the big separator:** PCA misses almost all of it, and the nonlinear models
  catch over half.
- idpscan has only 9 minutes, too few to conclude anything.

**False alarms on test:** calibrated for 1%, five of six models alerted on 2.6–4.3% of
normal test minutes. That is the train→test drift seen in the EDA.

**Sanity check:** every model reacts strongly to big spikes (AUC ≈ 0.97+). PCA scores
*best* here but *worst* on real attacks, because fake single-feature spikes don't look
like real attacks.

**Cost (CPU training time per run):** MLP ≈ 3 s, GRU ≈ 85 s, LSTM ≈ 136 s, windowed MLP
≈ 210 s, CNN ≈ 680 s (measured on a busy machine, so likely inflated).

## 15. Discussion: did the models learn something meaningful?

- **Loss curves:**
  - training loss keeps falling, but tune loss bottoms out within a few epochs (LSTM
    best epochs 4, 2, 12);
  - the gap reflects drift between the fit and tune periods plus some overfitting;
  - early stopping caught it.
- **Evidence of real learning:**
  - AP 0.46 vs 0.031 for random guessing;
  - botnet is caught by the nonlinear models;
  - the sanity check passes on large spikes.
- **Why temporal context didn't help (our best reading, not proven):**
  - attacks show up within single minutes, so earlier minutes add little;
  - a window model must rebuild 20 minutes from 16 numbers, so its rebuild of the
    scored final minute is less sharp than the MLP's, which spends all 16 numbers on
    one minute.
- **Interpretability:** each alert comes with a per-feature error, showing *which*
  traffic counters were unusual. The saved bottleneck values can be plotted to look at
  how models organise traffic.

## 16. Limitations

- Minute-level counts can hide short or low-volume attacks, and lose the order of
  flows within a minute.
- "Normal" training traffic may contain unlabelled attacks.
- Excluding blacklist activity narrows what we evaluate.
- Drift between training and test inflates false alarms.
- SSH scan and spam never occur in test, and idpscan has only 9 minutes.
- The first 19 minutes after any gap are never scored.
- The data comes from one ISP in 2016, so these are benchmark results, not proof of
  readiness for current networks.
- Autoencoders can rebuild some attacks well, as PCA did with botnet.
- Only 3 seeds, no significance test, and the LSTM, GRU and CNN were not tuned.

## 17. Conclusion (our reading; the report's conclusion is still TODO)

A simple nonlinear model, the single-minute MLP autoencoder, clearly beats the linear
PCA baseline, especially on botnet traffic. Giving models 20 minutes of context, whether
flattened, recurrent or convolutional, did not improve detection on this benchmark and
cost far more compute. The calibrated 1% false-alarm rate did not survive the shift to
the test period, which matters as much for deployment as the choice of model.

## 18. Glossary

| Term | Meaning |
|---|---|
| Flow | one network connection (e.g. one TCP session); features count flows per category |
| Autoencoder | a network trained to rebuild its own input through a narrow middle layer |
| Bottleneck / latent | the narrow middle layer (16 numbers here) |
| Reconstruction error | how far the rebuild is from the input (mean squared error) |
| Epoch | one full pass over the training data |
| Early stopping | stop training once the validation loss stops improving |
| Percentile threshold | a cut-off chosen so a set share (here 1%) of calibration scores lie above it |
| Drift | the data distribution changing over time |
| FPR | false-positive rate: share of normal minutes wrongly flagged |
| AP | average precision: summarises precision vs recall over all thresholds |
| Seed | the number that fixes randomness, so a run can be repeated exactly |

## 19. Where things live in the code (`src/ugr16ad/`)

| Step | File |
|---|---|
| Load, align, split, segments | `data.py` |
| log1p, constant removal, scaling | `preprocess.py` |
| 20-minute windows | `windows.py` |
| Training loop, scoring | `train.py` |
| Models | `models/pca.py`, `mlp.py`, `windowed_mlp.py`, `lstm.py`, `cnn.py` |
| Fake-anomaly sanity check | `synthetic.py` |
| Threshold and metrics | `evaluate.py` |
| Saving runs | `results.py` |
| Run scripts and comparison table | `scripts/run_*.py`, `scripts/make_comparison_table.py` |
