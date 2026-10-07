"""이중차분(DiD) 추정: 서울 기후동행카드(2024-01-27) 도입이 서울 지하철 평일 승차인원에 미친 영향.

plan.yaml 과 README의 '계획 보완 1, 2'를 그대로 구현한다. 결과를 본 뒤 규칙을 바꾸지 않는다.

입력 : data/processed/station_month.csv  (src/prepare.py 의 출력)
출력 : results/*.csv, results/*.png, results/summary.json

사용: python src/estimate.py --panel data/processed/station_month.csv --out results
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf

PRE_START = pd.Period("2022-07", "M")
PRE_END = pd.Period("2023-12", "M")
TRANSITION = pd.Period("2024-01", "M")          # 도입 달: DiD 추정에서 제외
POST_START = pd.Period("2024-02", "M")
POST_END = pd.Period("2024-12", "M")
REF_MONTH = pd.Period("2023-12", "M")           # 이벤트 스터디 기준월(-1)
OUTLIER_LOG_JUMP = float(np.log(2))             # 월 대비 월 로그 변화가 이 값을 넘으면 급변


def load_panel(path):
    df = pd.read_csv(path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df["unit"] = df["city"] + "_" + df["line"].astype(str) + "_" + df["station_code"].astype(str)
    df["treat"] = (df["city"] == "seoul").astype(int)
    df["month_s"] = df["month"].astype(str)
    return df


def balanced_sample(df, outcome="mean_weekday_boardings"):
    """계획 보완 1: 30개월 모두 자료가 있고 월평균/중앙값 평일 승차인원이 모두 0보다 큰 역-호선."""
    n_months = df["month"].nunique()
    g = df.groupby("unit").agg(n=("month", "nunique"),
                               mn_mean=("mean_weekday_boardings", "min"),
                               mn_med=("median_weekday_boardings", "min"))
    keep = g[(g["n"] == n_months) & (g["mn_mean"] > 0) & (g["mn_med"] > 0)].index
    return df[df["unit"].isin(keep)].copy()


def remove_jumpers(df):
    """계획 보완 2: 월 대비 월 로그 변화가 ln(2)를 한 번이라도 넘는 역-호선 제외."""
    d = df.sort_values(["unit", "month"]).copy()
    d["ly"] = np.log(d["mean_weekday_boardings"])
    d["dly"] = d.groupby("unit")["ly"].diff().abs()
    bad = d.loc[d["dly"] > OUTLIER_LOG_JUMP, "unit"].unique()
    return df[~df["unit"].isin(bad)].copy(), sorted(bad.tolist())


def fit_did(d, outcome, post_start, drop_months=(), end=None, start=None):
    d = d.copy()
    if start is not None:
        d = d[d["month"] >= start]
    if end is not None:
        d = d[d["month"] <= end]
    d = d[~d["month"].isin(list(drop_months))]
    d["post"] = (d["month"] >= post_start).astype(int)
    d["tp"] = d["treat"] * d["post"]
    groups = pd.factorize(d["unit"])[0]
    m = smf.ols(f"{outcome} ~ tp + C(unit) + C(month_s)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    b, se = float(m.params["tp"]), float(m.bse["tp"])
    lo, hi = (float(x) for x in m.conf_int().loc["tp"])
    return {"coef": b, "se": se, "ci_low": lo, "ci_high": hi, "p": float(m.pvalues["tp"]),
            "n_obs": int(m.nobs), "n_units": int(d["unit"].nunique()),
            "n_treated_units": int(d.loc[d["treat"] == 1, "unit"].nunique()),
            "n_control_units": int(d.loc[d["treat"] == 0, "unit"].nunique())}


def event_study(d, outcome):
    d = d.copy()
    months = sorted(d["month"].unique())
    evcols = []
    for m_ in months:
        if m_ == REF_MONTH:
            continue
        c = "ev_" + str(m_).replace("-", "")
        d[c] = ((d["month"] == m_) & (d["treat"] == 1)).astype(int)
        evcols.append((m_, c))
    groups = pd.factorize(d["unit"])[0]
    f = f"{outcome} ~ " + " + ".join(c for _, c in evcols) + " + C(unit) + C(month_s)"
    res = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = res.conf_int()
    rows = [{"month": str(m_), "coef": float(res.params[c]), "se": float(res.bse[c]),
             "ci_low": float(ci.loc[c, 0]), "ci_high": float(ci.loc[c, 1])} for m_, c in evcols]
    rows.append({"month": str(REF_MONTH), "coef": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    es = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    pre_cols = [c for m_, c in evcols if m_ < REF_MONTH]
    wt = res.wald_test(", ".join(f"{c} = 0" for c in pre_cols), use_f=True, scalar=True)
    pre = {"n_pre_coefs": len(pre_cols), "wald_F": float(wt.statistic), "p_value": float(wt.pvalue)}
    return es, pre


def placebo_in_time(d, outcome, actual):
    """사전 기간 안에서만 가짜 도입월 p를 바꿔가며 같은 모형을 추정한다."""
    pre = d[(d["month"] >= PRE_START) & (d["month"] <= PRE_END)]
    rows = []
    for p in pd.period_range("2022-12", "2023-08", freq="M"):
        r = fit_did(pre, outcome, post_start=p + 1, drop_months=[p])
        rows.append({"placebo_month": str(p), **r})
    pl = pd.DataFrame(rows)
    rank = float((pl["coef"].abs() >= abs(actual)).mean())
    return pl, rank


def make_plots(out, d, es, pre, pl, actual, outcome):
    # 1) 도시별 평균 로그 승차인원(2023-12 = 0)
    t = d.groupby(["city", "month"])[outcome].mean().unstack("city")
    t = t - t.loc[REF_MONTH]
    labels = [str(m_) for m_ in t.index]
    fig, ax = plt.subplots(figsize=(8, 4))
    for city in t.columns:
        ax.plot(labels, t[city].values, marker="o", label=city)
    ax.axvline(labels.index(str(TRANSITION)), color="gray", ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Mean log weekday boardings (relative to 2023-12)"); ax.legend()
    ax.set_xticks(range(0, len(labels), 3)); ax.set_xticklabels(labels[::3], rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(out, "trends.png"), dpi=130); plt.close(fig)

    # 2) 이벤트 스터디
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(es["month"], es["coef"], yerr=[es["coef"] - es["ci_low"], es["ci_high"] - es["coef"]], fmt="o", capsize=2)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(list(es["month"]).index(str(TRANSITION)), color="gray", ls="--")
    ax.set_title(f"Event study (ref 2023-12). Pre-trend Wald p={pre['p_value']:.3f}")
    ax.set_xticks(range(0, len(es), 3)); ax.set_xticklabels(es["month"][::3], rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(out, "event_study.png"), dpi=130); plt.close(fig)

    # 3) 가짜 도입일 분포
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(pl["placebo_month"], pl["coef"], color="lightgray", label="placebo (pre-period only)")
    ax.axhline(actual, color="C3", label=f"actual = {actual:.3f}")
    ax.axhline(0, color="k", lw=0.5); ax.legend(); ax.set_title("Placebo-in-time vs actual estimate")
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(out, "placebo.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/station_month.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw = load_panel(a.panel)
    bal = balanced_sample(raw)
    bal["ly"] = np.log(bal["mean_weekday_boardings"])
    bal["lmed"] = np.log(bal["median_weekday_boardings"])
    sizes = {"all_units": int(raw["unit"].nunique()), "balanced_units": int(bal["unit"].nunique()),
             "balanced_seoul": int(bal.loc[bal["treat"] == 1, "unit"].nunique()),
             "balanced_busan": int(bal.loc[bal["treat"] == 0, "unit"].nunique())}

    # (1) 주 추정
    main_r = fit_did(bal, "ly", POST_START, drop_months=[TRANSITION])
    pd.DataFrame([main_r]).to_csv(os.path.join(a.out, "did_main.csv"), index=False)

    # (2) 이벤트 스터디와 사전 추세 검정
    es, pre = event_study(bal, "ly")
    es.to_csv(os.path.join(a.out, "event_study.csv"), index=False)

    # (3) 가짜 도입일
    pl, rank = placebo_in_time(bal, "ly", main_r["coef"])
    pl.to_csv(os.path.join(a.out, "placebo.csv"), index=False)

    # (4) 호선별 (서울 각 호선 대 부산 전체)
    rows = []
    for ln in sorted(bal.loc[bal["treat"] == 1, "line"].unique()):
        sub = bal[(bal["treat"] == 0) | ((bal["treat"] == 1) & (bal["line"] == ln))]
        rows.append({"seoul_line": int(ln), **fit_did(sub, "ly", POST_START, drop_months=[TRANSITION])})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "by_line.csv"), index=False)

    # (5) 재추정
    rb = []
    rb.append({"spec": "pre 12 months only (2023-01~2023-12)", **fit_did(bal, "ly", POST_START, [TRANSITION], start=pd.Period("2023-01", "M"))})
    rb.append({"spec": "post 6 months only (2024-02~2024-07)", **fit_did(bal, "ly", POST_START, [TRANSITION], end=pd.Period("2024-07", "M"))})
    rb.append({"spec": "level outcome (persons/day)", **fit_did(bal, "mean_weekday_boardings", POST_START, [TRANSITION])})
    rb.append({"spec": "median-based outcome", **fit_did(bal, "lmed", POST_START, [TRANSITION])})
    nj, jumpers = remove_jumpers(bal)
    rb.append({"spec": f"drop month-to-month jumps > 2x ({len(jumpers)} units dropped)", **fit_did(nj, "ly", POST_START, [TRANSITION])})
    unb = raw[raw["mean_weekday_boardings"] > 0].copy()
    unb["ly"] = np.log(unb["mean_weekday_boardings"])
    rb.append({"spec": "unbalanced panel (months with boardings > 0)", **fit_did(unb, "ly", POST_START, [TRANSITION])})
    pd.DataFrame(rb).to_csv(os.path.join(a.out, "robustness.csv"), index=False)

    make_plots(a.out, bal, es, pre, pl, main_r["coef"], "ly")

    summary = {
        "versions": {"pandas": pd.__version__, "numpy": np.__version__, "statsmodels": statsmodels.__version__},
        "sample": sizes,
        "main_did": main_r,
        "main_effect_percent": float(np.expm1(main_r["coef"]) * 100),
        "main_effect_percent_ci": [float(np.expm1(main_r["ci_low"]) * 100), float(np.expm1(main_r["ci_high"]) * 100)],
        "pretrend_test": pre,
        "placebo": {"n_placebos": int(len(pl)), "share_abs_ge_actual": rank,
                    "placebo_coef_min": float(pl["coef"].min()), "placebo_coef_max": float(pl["coef"].max())},
        "jumper_units_dropped": jumpers,
    }
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
