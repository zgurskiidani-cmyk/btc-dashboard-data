"""BTC regime model v2: on-chain/price features plus optional macro features.

Data (both from github.com/zgurskiidani-cmyk/btc-dashboard-data, data/):
  btc_daily.csv    date, price, mvrv, realized_price   (Coin Metrics community API)
  macro_daily.csv  date, us10y, real10y, usd_broad     (FRED DGS10, DFII10, DTWEXBGS)

Base features: ln_mvrv, ln_p_200d, ln_p_200w, ln_p_ath, cyc_sin, cyc_cos (halving-cycle phase).
Macro features (lagged one day so only already-published values are used):
  d_real10y_90  = change of the 10y TIPS real yield over 90 days, percentage points
  d_usd_90      = log change of the broad dollar index over 90 days
Targets: price higher after 180 / 365 days; drop of 20% or more within 90 days.
Method: L2 logistic regression on weekly samples, walk-forward by calendar year with a purge gap
equal to the horizon, year-block bootstrap. Usage: model2.py BTC_CSV MACRO_CSV OUT_DIR [--final]
"""
import sys, json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BTC, MACRO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FINAL = "--final" in sys.argv
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"])
CYCLE = 1458.0
BASE = ["ln_mvrv", "ln_p_200d", "ln_p_200w", "ln_p_ath", "cyc_sin", "cyc_cos"]
MAC = ["d_real10y_90", "d_usd_90"]
VARIANTS = {"base": BASE, "macro": BASE + MAC, "macro_no_cycle": BASE[:4] + MAC, "mvrv_only": ["ln_mvrv"]}
TARGETS = {"p180": 180, "p365": 365, "pdd90": 90}
CS = [0.01, 0.03, 0.1, 0.3, 1.0]
rng = np.random.default_rng(42)

b = pd.read_csv(BTC, parse_dates=["date"]).dropna(subset=["price", "mvrv"]).set_index("date").asfreq("D").ffill()
m = pd.read_csv(MACRO, parse_dates=["date"]).set_index("date").asfreq("D").ffill().shift(1)  # publish lag
df = b.join(m, how="left")
p = df["price"]
df["ln_mvrv"] = np.log(df["mvrv"])
df["ln_p_200d"] = np.log(p / p.rolling(200).mean())
df["ln_p_200w"] = np.log(p / p.rolling(1400).mean())
df["ln_p_ath"] = np.log(p / p.cummax())
dsh = pd.Series([(t - HALVINGS[HALVINGS <= t].max()).days if (HALVINGS <= t).any() else np.nan for t in df.index], index=df.index)
df["dsh"] = dsh
df["cyc_sin"], df["cyc_cos"] = np.sin(2 * np.pi * dsh / CYCLE), np.cos(2 * np.pi * dsh / CYCLE)
df["d_real10y_90"] = df["real10y"] - df["real10y"].shift(90)
df["d_usd_90"] = np.log(df["usd_broad"] / df["usd_broad"].shift(90))

last = df.index[-1]
df["y_p180"] = (p.shift(-180) > p).astype(float)
df["y_p365"] = (p.shift(-365) > p).astype(float)
fut_min = p[::-1].rolling(90, min_periods=90).min()[::-1].shift(-1)
df["y_pdd90"] = (fut_min < 0.8 * p).astype(float)
for t, h in TARGETS.items():
    df.loc[df.index > last - pd.Timedelta(days=h), f"y_{t}"] = np.nan

w = df.dropna(subset=BASE + MAC).iloc[::7].copy()
w["year"] = w.index.year


def fit(X, y, c):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    return LogisticRegression(C=c, max_iter=3000).fit((X - mu) / sd, y), mu, sd


def pred(model, X):
    m_, mu, sd = model
    return m_.predict_proba((X - mu) / sd)[:, 1]


def walk_forward(t, h, feats, c):
    ycol = f"y_{t}"
    preds, base = pd.Series(np.nan, index=w.index), pd.Series(np.nan, index=w.index)
    for Y in range(2018, last.year + 1):
        cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=h)
        tr, te = w[(w.index < cut) & w[ycol].notna()], w[w.year == Y]
        if len(te) == 0 or tr[ycol].nunique() < 2:
            continue
        preds.loc[te.index] = pred(fit(tr[feats].values, tr[ycol].values, c), te[feats].values)
        base.loc[te.index] = tr[ycol].mean()
    return preds, base


metrics, oos, calib = [], {}, []
for t, h in TARGETS.items():
    ycol = f"y_{t}"
    for vname, feats in VARIANTS.items():
        for c in (CS if vname in ("base", "macro") else [0.1]):
            preds, base = walk_forward(t, h, feats, c)
            ev = w[preds.notna() & w[ycol].notna()]
            yv, pv, bv = ev[ycol].values, preds[ev.index].values, base[ev.index].values
            br, bb = np.mean((pv - yv) ** 2), np.mean((bv - yv) ** 2)
            # skill by test year, to see whether a gain is broad or comes from one year
            by_year = {int(Y): round(1 - np.mean((pv[ev.index.year == Y] - yv[ev.index.year == Y]) ** 2) /
                                     max(np.mean((bv[ev.index.year == Y] - yv[ev.index.year == Y]) ** 2), 1e-9), 2)
                       for Y in sorted(set(ev.index.year))}
            metrics.append({"target": t, "variant": vname, "C": c, "n_weeks": int(len(ev)),
                            "test_years": f"{ev.index.year.min()}-{ev.index.year.max()}",
                            "brier": round(br, 4), "brier_baseline": round(bb, 4),
                            "skill_vs_baseline": round(1 - br / bb, 3), "auc": round(roc_auc_score(yv, pv), 3),
                            "base_rate": round(yv.mean(), 3), "skill_by_year": by_year})
            oos[(t, vname, c)] = preds

if not FINAL:
    for r in metrics:
        print(r["target"], f"{r['variant']:15s}", f"C={r['C']:<5}", "skill", r["skill_vs_baseline"], "auc", r["auc"], "n", r["n_weeks"], r["skill_by_year"])
    sys.exit(0)

# ---- final outputs for the chosen variant ----
CHOSEN, PRIMARY_C = sys.argv[sys.argv.index("--final") + 1], float(sys.argv[sys.argv.index("--final") + 2])
FEATS = VARIANTS[CHOSEN]
for t in TARGETS:
    pv_all = oos[(t, CHOSEN, PRIMARY_C)]
    ev = w[pv_all.notna() & w[f"y_{t}"].notna()]
    pv, yv = pv_all[ev.index].values, ev[f"y_{t}"].values
    bins = np.linspace(0, 1, 6)
    idx = np.clip(np.digitize(pv, bins) - 1, 0, 4)
    for k in range(5):
        sel = idx == k
        if sel.sum():
            calib.append({"target": t, "bin": f"{int(bins[k]*100)}-{int(bins[k+1]*100)}%", "predicted": round(pv[sel].mean(), 3),
                          "observed": round(yv[sel].mean(), 3), "n_weeks": int(sel.sum())})

coef_rows, boot_rows = [], []
for t in TARGETS:
    ycol = f"y_{t}"
    tr = w[w[ycol].notna()]
    for c in CS:
        mdl, mu, sd = fit(tr[FEATS].values, tr[ycol].values, c)
        coef_rows.append({"target": t, "C": c, "term": "intercept", "mean": 0.0, "sd": 1.0, "coef": round(float(mdl.intercept_[0]), 5)})
        for f, a, s, k in zip(FEATS, mu, sd, mdl.coef_[0]):
            coef_rows.append({"target": t, "C": c, "term": f, "mean": round(float(a), 5), "sd": round(float(s), 5), "coef": round(float(k), 5)})
    yrs, r = sorted(tr.year.unique()), 0
    while r < 200:
        bs = pd.concat([tr[tr.year == yy] for yy in rng.choice(yrs, size=len(yrs), replace=True)])
        if bs[ycol].nunique() < 2:
            continue
        mdl, mu, sd = fit(bs[FEATS].values, bs[ycol].values, PRIMARY_C)
        row = {"target": t, "rep": r, "intercept": round(float(mdl.intercept_[0]), 4)}
        for f, a, s, k in zip(FEATS, mu, sd, mdl.coef_[0]):
            row[f"{f}__mean"], row[f"{f}__sd"], row[f"{f}__coef"] = round(float(a), 4), round(float(s), 4), round(float(k), 4)
        boot_rows.append(row)
        r += 1

hist = pd.DataFrame({"date": w.index.strftime("%Y-%m-%d"), "price": w["price"].round(2).values, "mvrv": w["mvrv"].round(3).values})
for t in TARGETS:
    hist[t] = oos[(t, CHOSEN, PRIMARY_C)].round(3).values
hist = hist[hist["p180"].notna()]

d = df.dropna(subset=["ln_mvrv"]).copy()
d = d[d.index >= "2011-01-01"]
edges = [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 99]
labels = ["< 0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", "1.5-2.0", "2.0-3.0", "> 3.0"]
d["bucket"] = pd.cut(d["mvrv"], edges, labels=labels)
d["r180"], d["r365"] = d["price"].shift(-180) / d["price"] - 1, d["price"].shift(-365) / d["price"] - 1
base_rows = []
def add_rates(name, g, order):
    s180, s365, sdd = g["r180"].dropna(), g["r365"].dropna(), g["y_pdd90"].dropna()
    base_rows.append({"order": order, "state": name, "days": int(len(g)),
                      "episodes": int((g.index.to_series().diff().dt.days.fillna(999) > 60).sum()),
                      "p_up_180": round((s180 > 0).mean(), 3), "median_180": round(s180.median(), 3),
                      "p_up_365": round((s365 > 0).mean(), 3), "median_365": round(s365.median(), 3), "p_dd20_90": round(sdd.mean(), 3)})
for i, lab in enumerate(labels):
    add_rates(f"MVRV {lab}", d[d["bucket"] == lab], i)
add_rates("MVRV 1.0-1.3 & below 200D", d[d["mvrv"].between(1.0, 1.3) & (d["ln_p_200d"] < 0)], 7)
add_rates("MVRV 1.0-1.3 & below 200D & 200W", d[d["mvrv"].between(1.0, 1.3) & (d["ln_p_200d"] < 0) & (d["ln_p_200w"] < 0)], 8)

meta = {"data_last_date": str(last.date()), "train_start": str(w.index.min().date()), "features": FEATS, "variant": CHOSEN,
        "primary_C": PRIMARY_C, "C_sensitivity": CS}
for name, obj in [("coef", {"rows": coef_rows, "meta": meta}), ("boot", boot_rows), ("metrics", [{k: v for k, v in r.items() if k != "skill_by_year"} for r in metrics]),
                  ("calibration", calib), ("history", hist.to_dict("records")), ("base_rates", base_rows)]:
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), allow_nan=False)
print("written", OUT, meta)
