"""Read-only EDA checks on UGR'16 v2 for the proposal's Data Cleaning / EDA sections.

Reads the four CSVs in data/external/ugr16v2/; writes nothing. No random splits: every
split here is chronological (first 80% / last 20% of train, or contiguous halves).

Run: ./.venv/Scripts/python.exe scripts/eda_checks.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ugr16ad.paths import UGR16_DIR

LABELS = [
    "labeldos",
    "labelscan11",
    "labelscan44",
    "labelnerisbotnet",
    "labelanomalyidpscan",
    "labelanomalysshscan",
    "labelanomalyspam",
]
BLACKLIST = "labelblacklist"


def load(split):
    X = pd.read_csv(UGR16_DIR / f"UGR16v2.X{split}.csv")
    Y = pd.read_csv(UGR16_DIR / f"UGR16v2.Y{split}.csv")
    return X, Y


def section(title):
    print(f"\n=== {title} ===")


def run_lengths(mask):
    """Lengths of consecutive True runs in a boolean array (ignores time gaps)."""
    m = np.asarray(mask, dtype=int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


data = {s: load(s) for s in ("train", "test")}

section("Parsing, alignment, duplicates, gaps")
for s, (X, Y) in data.items():
    ts = pd.to_datetime(X["Row"].astype(str), format="%Y%m%d%H%M", errors="coerce")
    print(f"{s}: rows={len(X)} X_cols={X.shape[1] - 1} Y_cols={Y.shape[1] - 1}")
    print(f"  unparseable timestamps={ts.isna().sum()}  keys aligned={X.Row.equals(Y.Row)}")
    print(f"  range {ts.min()} -> {ts.max()}  sorted={ts.is_monotonic_increasing}")
    dup_key = X.Row.duplicated(keep=False)
    print(
        f"  duplicate keys={dup_key.sum()}  exact duplicate rows="
        f"{pd.concat([X, Y.drop(columns='Row')], axis=1).duplicated().sum()}"
    )
    expected = int((ts.max() - ts.min()) / pd.Timedelta(minutes=1)) + 1
    step = ts.diff().dropna()
    gaps = step[step > pd.Timedelta(minutes=1)]
    print(f"  expected minutes={expected} missing={expected - ts.nunique()} gap events={len(gaps)}")
    for i, g in gaps.items():
        print(f"    gap before {ts[i]}: {g}")
    Xn = X.drop(columns="Row")
    print(
        f"  NaN={Xn.isna().sum().sum()} negative X={(Xn < 0).sum().sum()} "
        f"negative Y={(Y.drop(columns='Row') < 0).sum().sum()} "
        f"inf={np.isinf(Xn.to_numpy(dtype=float)).sum()}"
    )
    print(f"  all-zero feature rows={(Xn.sum(axis=1) == 0).sum()}")

section("Grouped count consistency (group sums should all equal total flows)")
groups = {
    "srcip": r"^srcip",
    "dstip": r"^dstip",
    "sport": r"^sport",
    "dport": r"^dport",
    "protocol": r"^protocol",
    "srctos": r"^srctos",
    "npackets": r"^npackets",
    "nbytes": r"^nbytes",
}
for s, (X, _) in data.items():
    sums = pd.DataFrame({g: X.filter(regex=p).sum(axis=1) for g, p in groups.items()})
    ref = sums["srcip"]
    print(f"{s}: columns per group={ {g: X.filter(regex=p).shape[1] for g, p in groups.items()} }")
    for g in sums:
        print(f"  {g:9s} == srcip total in {(sums[g] == ref).mean():.4%} of rows")
    flags = X.filter(regex=r"^tcpflags")
    print(
        f"  tcpflags_* > protocoltcp in {(flags.gt(X.protocoltcp, axis=0)).any(axis=1).sum()} rows"
    )
    Y = data[s][1]
    print(
        f"  any label count > total flows: "
        f"{(Y[LABELS + [BLACKLIST]].gt(ref, axis=0)).any(axis=1).sum()} rows"
    )

section("Constant / near-constant features (train only)")
Xtr = data["train"][0].drop(columns="Row")
const = Xtr.columns[Xtr.nunique() <= 1].tolist()
zero_frac = (Xtr == 0).mean()
print(f"constant in train: {len(const)} -> {const}")
print(f"features >=99% zeros in train: {(zero_frac >= 0.99).sum()}")
Xte = data["test"][0].drop(columns="Row")
print(
    f"of the train-constant ones, non-constant in test: "
    f"{[c for c in const if Xte[c].nunique() > 1]}"
)

section("Target prevalence (target = any of the 7 non-blacklist labels > 0)")
rows = []
for s, (_, Y) in data.items():
    y = (Y[LABELS] > 0).any(axis=1)
    rows.append(
        {
            "split": s,
            "n": len(Y),
            "pos": int(y.sum()),
            "prev": y.mean(),
            **{c: int((Y[c] > 0).sum()) for c in LABELS + [BLACKLIST]},
        }
    )
print(pd.DataFrame(rows).T.to_string())

section("Chronological 80/20 inside train")
Ytr = data["train"][1]
ytr = (Ytr[LABELS] > 0).any(axis=1).to_numpy()
cut = int(0.8 * len(ytr))
print(
    f"cut at {Ytr.Row.iloc[cut]}; first80 pos={ytr[:cut].sum()} ({ytr[:cut].mean():.4%}), "
    f"last20 pos={ytr[cut:].sum()} ({ytr[cut:].mean():.4%})"
)
for c in LABELS:
    print(
        f"  {c:22s} first80={(Ytr[c].iloc[:cut] > 0).sum():6d} "
        f"last20={(Ytr[c].iloc[cut:] > 0).sum():6d}"
    )
days = pd.to_datetime(Ytr.Row.astype(str), format="%Y%m%d%H%M").dt.date
print("positive minutes per day in train (non-zero days):")
print(pd.Series(ytr, index=days).groupby(level=0).sum().loc[lambda s: s > 0].to_string())
Yte = data["test"][1]
dte = pd.to_datetime(Yte.Row.astype(str), format="%Y%m%d%H%M").dt.date
print("positive minutes per day in test:")
print(
    pd.Series((Yte[LABELS] > 0).any(axis=1).to_numpy(), index=dte)
    .groupby(level=0)
    .sum()
    .to_string()
)

section("Categories present only in one split")
for c in LABELS:
    a, b = (data["train"][1][c] > 0).sum(), (data["test"][1][c] > 0).sum()
    tag = (
        "train only"
        if a and not b
        else "test only"
        if b and not a
        else "neither"
        if not (a or b)
        else "both"
    )
    print(f"  {c:22s} {tag}")

section("Attack episode run lengths (consecutive positive minutes)")
for s, (_, Y) in data.items():
    y = (Y[LABELS] > 0).any(axis=1)
    rl = run_lengths(y)
    if len(rl):
        print(
            f"{s} any-attack: episodes={len(rl)} median={np.median(rl):.0f} "
            f"p90={np.percentile(rl, 90):.0f} max={rl.max()} single-minute={(rl == 1).sum()}"
        )
    for c in LABELS:
        rl = run_lengths(Y[c] > 0)
        if len(rl):
            print(f"  {c:22s} episodes={len(rl):5d} median={np.median(rl):6.0f} max={rl.max():6d}")

section("Label co-occurrence within a minute")
for s, (_, Y) in data.items():
    B = Y[LABELS] > 0
    k = B.sum(axis=1)
    print(
        f"{s}: minutes with >=2 attack labels={int((k >= 2).sum())} "
        f"(of {int((k >= 1).sum())} positive)"
    )
    co = B.T.astype(int) @ B.astype(int)
    print(co.loc[co.sum(axis=1) > 0, co.sum(axis=0) > 0].to_string())

section("Blacklist prevalence and overlap")
for s, (_, Y) in data.items():
    bl = Y[BLACKLIST] > 0
    y = (Y[LABELS] > 0).any(axis=1)
    print(
        f"{s}: blacklist>0 in {bl.mean():.4%} of minutes; "
        f"median count={Y[BLACKLIST].median():.0f}; "
        f"P(blacklist|attack)={bl[y].mean() if y.any() else float('nan'):.4f}; "
        f"P(attack|blacklist)={y[bl].mean():.4%}"
    )

section("Benign daily pattern (train, negatives only, total flows)")
X, Y = data["train"]
ts = pd.to_datetime(X.Row.astype(str), format="%Y%m%d%H%M")
tot = X.filter(regex=r"^srcip").sum(axis=1)
neg = ~(Y[LABELS] > 0).any(axis=1)
by_hour = tot[neg].groupby(ts[neg].dt.hour).median()
by_dow = tot[neg].groupby(ts[neg].dt.dayofweek).median()
print(
    f"hourly median flows min/max = {by_hour.min():.0f}/{by_hour.max():.0f} "
    f"(ratio {by_hour.max() / by_hour.min():.2f}; "
    f"peak hour {by_hour.idxmax()}, trough {by_hour.idxmin()})"
)
print(f"day-of-week median flows (Mon=0): {by_dow.round(0).to_dict()}")

section("Skew and zero-inflation (train)")
nz = Xtr.loc[:, Xtr.nunique() > 1]
sk = nz.skew()
print(f"features: median skew={sk.median():.1f}, skew>10: {(sk > 10).sum()}/{len(sk)}")
print(
    f"median zero fraction={zero_frac.median():.2%}; features >50% zeros: {(zero_frac > 0.5).sum()}"
)
sk_log = np.log1p(nz).skew()
print(f"after log1p: median skew={sk_log.median():.1f}, skew>10: {(sk_log > 10).sum()}")

section("Multicollinearity (train, log1p, non-constant)")
L = np.log1p(nz)
r = L.corr().abs().to_numpy()
iu = np.triu_indices_from(r, 1)
print(f"pairs |r|>0.95: {(r[iu] > 0.95).sum()}  |r|>0.99: {(r[iu] > 0.99).sum()}  of {len(iu[0])}")
print(
    f"rank of raw train matrix={np.linalg.matrix_rank(nz.to_numpy(dtype=float))} "
    f"of {nz.shape[1]} columns"
)

section("Covariate shift: per-feature train-vs-test AUC and adversarial LR (contiguous halves)")
Xte_nz = Xte[nz.columns]
lab = np.r_[np.zeros(len(nz)), np.ones(len(Xte_nz))]
comb = pd.concat([nz, Xte_nz], ignore_index=True)
shift = {}
for c in nz.columns:
    a = roc_auc_score(lab, comb[c])
    shift[c] = max(a, 1 - a)
shift = pd.Series(shift).sort_values(ascending=False)
print(shift.head(10).round(4).to_string())
print(f"features with train-vs-test AUC>0.8: {(shift > 0.8).sum()}")
# Fit on first half of each period, score on second half: chronological, never random.
h_tr, h_te = len(nz) // 2, len(Xte_nz) // 2
fit = pd.concat([nz.iloc[:h_tr], Xte_nz.iloc[:h_te]])
ev = pd.concat([nz.iloc[h_tr:], Xte_nz.iloc[h_te:]])
yf = np.r_[np.zeros(h_tr), np.ones(h_te)]
ye = np.r_[np.zeros(len(nz) - h_tr), np.ones(len(Xte_nz) - h_te)]
sc = StandardScaler().fit(np.log1p(fit))
clf = LogisticRegression(max_iter=2000).fit(sc.transform(np.log1p(fit)), yf)
print(
    f"adversarial LR AUC (fit on first halves, eval on second halves) = "
    f"{roc_auc_score(ye, clf.predict_proba(sc.transform(np.log1p(ev)))[:, 1]):.4f}"
)
