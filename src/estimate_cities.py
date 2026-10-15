"""계획 2: 도시 x 월 패널로 서울과 비교 도시 4곳을 비교한다 (plan_v2.yaml 규칙 그대로).

입력 : data/processed/city_month.csv, data/processed/station_month_5cities.csv
출력 : results/cities_*.csv, results/cities_summary.json, results/cities_*.png

사용: python src/estimate_cities.py --plan plan_v2.yaml --panel data/processed/city_month.csv --out results
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANNED = ["입증", "때문에", "덕분에", "확실히", "명백히"]


def gap_series(cm, y, treated, controls):
    w = cm.pivot(index="month", columns="city", values=y).sort_index()
    return w[treated] - w[controls].mean(axis=1)


def did_from_gap(gap, pre_months, post_months):
    """도시·월 고정효과 모형에서 대조 도시를 같은 무게로 두면 tau = 사후 격차 평균 - 사전 격차 평균."""
    return float(gap[post_months].mean() - gap[pre_months].mean())


def rmspe_ratio(gap, pre_months, post_months):
    base = gap[pre_months].mean()
    pre = np.sqrt(((gap[pre_months] - base) ** 2).mean())
    post = np.sqrt(((gap[post_months] - base) ** 2).mean())
    return float(post / pre)


def twfe(cm, y, treated, controls, post0):
    d = cm[cm["city"].isin([treated, *controls])].copy()
    d["D"] = ((d["city"] == treated) & (d["month"] >= post0)).astype(int)
    fit = smf.ols(f"{y} ~ D + C(city) + C(month)", data=d).fit()
    return float(fit.params["D"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="plan_v2.yaml")
    ap.add_argument("--panel", default="data/processed/city_month.csv")
    ap.add_argument("--stations", default="data/processed/station_month_5cities.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    plan = yaml.safe_load(open(a.plan, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    cm = pd.read_csv(a.panel)

    treated = plan["policy"]["treated_city"]
    controls = sorted(c for c in cm["city"].unique() if c != treated)
    months = sorted(cm["month"].unique())
    pre0, pre1 = plan["window"]["pre"]
    post0, post1 = plan["window"]["post"]
    pre_m = [m for m in months if pre0 <= m <= pre1]
    post_m = [m for m in months if post0 <= m <= post1]
    out = {"treated": treated, "controls": controls, "pre_months": len(pre_m), "post_months": len(post_m)}

    # 1) 주 추정
    gap = gap_series(cm, "log_weekday", treated, controls)
    tau = did_from_gap(gap, pre_m, post_m)
    tau_ols = twfe(cm, "log_weekday", treated, controls, post0)
    out["main"] = {"tau_log": tau, "tau_pct": 100 * (np.exp(tau) - 1), "tau_ols_check": tau_ols}

    # 2) 가짜 처치 도시 비교 (placebo-in-space): 서울을 빼고 대조 도시끼리 돌려 본다
    rows = [{"city": treated, "tau": tau, "rmspe_ratio": rmspe_ratio(gap, pre_m, post_m)}]
    for c in controls:
        others = [x for x in controls if x != c]
        g = gap_series(cm, "log_weekday", c, others)
        rows.append({"city": c, "tau": did_from_gap(g, pre_m, post_m), "rmspe_ratio": rmspe_ratio(g, pre_m, post_m)})
    space = pd.DataFrame(rows)
    space["abs_tau_rank"] = space["tau"].abs().rank(ascending=False, method="min").astype(int)
    space["ratio_rank"] = space["rmspe_ratio"].rank(ascending=False, method="min").astype(int)
    space.to_csv(os.path.join(a.out, "cities_placebo_space.csv"), index=False)
    srow = space[space["city"] == treated].iloc[0]
    out["placebo_space"] = {"abs_tau_rank": int(srow["abs_tau_rank"]), "ratio_rank": int(srow["ratio_rank"]),
                            "n": len(space), "min_possible_p": 1 / len(space)}

    # 3) 사전 격차 추세: 사전 18개월 격차에 직선, Newey-West(3) 표준오차
    pre_gap = gap[pre_m].reset_index(drop=True)
    X = sm.add_constant(np.arange(len(pre_gap)))
    fit = sm.OLS(pre_gap.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    lo, hi = fit.conf_int()[1]
    out["pre_trend"] = {"slope_per_month": float(fit.params[1]), "ci95": [float(lo), float(hi)],
                        "includes_zero": bool(lo <= 0 <= hi)}
    # 사전 추세를 그대로 늘렸을 때 사후 구간에서 예상되는 격차 변화 (참고)
    t_post = np.array([months.index(m) for m in post_m])
    t_pre = np.array([months.index(m) for m in pre_m])
    out["pre_trend"]["extrapolated_change"] = float(fit.params[1] * (t_post.mean() - t_pre.mean()))

    # 4) 가짜 시작월 (사전 구간만)
    fake = "2023-08"
    f_pre = [m for m in pre_m if m < fake]
    f_post = [m for m in pre_m if m >= fake]
    tau_fake = did_from_gap(gap, f_pre, f_post)
    out["placebo_time"] = {"fake_start": fake, "tau": tau_fake, "abs_ratio_to_main": abs(tau_fake) / abs(tau) if tau else None}

    # 5) 민감도
    sens = []
    for name, ctrls, y, pre_set in [
        ("주 추정 (대조 4곳, 평일)", controls, "log_weekday", pre_m),
        ("대조 부산·광주만", ["busan", "gwangju"], "log_weekday", pre_m),
        ("사전 기준 2023-10~12만", controls, "log_weekday", [m for m in pre_m if m >= "2023-10"]),
        ("사전 기준 2023-10~12, 대조 부산·광주", ["busan", "gwangju"], "log_weekday", [m for m in pre_m if m >= "2023-10"]),
        ("주말 승차 (대조 4곳)", controls, "log_weekend", pre_m),
    ]:
        g = gap_series(cm, y, treated, ctrls)
        t = did_from_gap(g, pre_set, post_m)
        sens.append({"spec": name, "tau_log": t, "tau_pct": 100 * (np.exp(t) - 1)})
    # 역 단위 참고 분석: 역 고정효과 + 도시별이 아닌 공통 월 고정효과, 역 군집 표준오차 (과소추정 가능)
    st = pd.read_csv(a.stations)
    st = st[st["balanced"] & st["month"].isin(months)].copy()
    st["y"] = np.log(st["mean_weekday"])
    st["sid"] = st["city"] + "_" + st["line"].astype(str) + "_" + st["station_code"].astype(str)
    st["D"] = ((st["city"] == treated) & (st["month"] >= post0)).astype(int)
    f2 = smf.ols("y ~ D + C(sid) + C(month)", data=st).fit(cov_type="cluster", cov_kwds={"groups": st["sid"]})
    ci = f2.conf_int().loc["D"]
    sens.append({"spec": "역 단위 (참고, 역 군집 SE)", "tau_log": float(f2.params["D"]),
                 "tau_pct": 100 * (np.exp(f2.params["D"]) - 1), "ci95_log": [float(ci[0]), float(ci[1])]})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(a.out, "cities_sensitivity.csv"), index=False)
    out["sensitivity"] = sens.to_dict(orient="records")

    # 6) 판정 (plan_v2.yaml verdict 규칙)
    bp = sens.set_index("spec").loc["대조 부산·광주만", "tau_log"]
    cond = {
        "pre_trend_ci_includes_zero": out["pre_trend"]["includes_zero"],
        "placebo_time_less_than_half": abs(tau_fake) < 0.5 * abs(tau),
        "seoul_rank1_both": out["placebo_space"]["abs_tau_rank"] == 1 and out["placebo_space"]["ratio_rank"] == 1,
        "same_sign_busan_gwangju": bool(np.sign(bp) == np.sign(tau)),
    }
    if all(cond.values()):
        verdict = "효과 근거 있음"
    elif (not cond["pre_trend_ci_includes_zero"]) or (abs(tau_fake) >= abs(tau)):
        verdict = "판단 불가"
    else:
        verdict = "조건부"
    out["verdict"] = {"label": verdict, "conditions": cond}

    # 7) 월별 격차 표와 그림 (기준월 2023-12 = 0)
    ref = gap["2023-12"]
    ev = pd.DataFrame({"month": gap.index, "gap_minus_ref": (gap - ref).values})
    ev.to_csv(os.path.join(a.out, "cities_gap_by_month.csv"), index=False)
    x = np.arange(len(gap))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, (gap - ref).values, marker="o", color="tab:red", label="Seoul - mean(4 controls)")
    for c in controls:
        others = [o for o in controls if o != c]
        g = gap_series(cm, "log_weekday", c, others)
        ax.plot(x, (g - g["2023-12"]).values, color="grey", alpha=0.5, lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(months.index(post0) - 0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log weekday boardings gap (ref 2023-12)")
    ax.set_title("Seoul vs controls (red) and placebo cities (grey); 2024-01 excluded")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "cities_gap.png"), dpi=130)
    plt.close(fig)

    w = cm.pivot(index="month", columns="city", values="log_weekday").sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    for c in w.columns:
        ax.plot(x, (w[c] - w[c][pre_m].mean()).values, marker="." if c == treated else None,
                lw=2.2 if c == treated else 1, label=c)
    ax.axvline(months.index(post0) - 0.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(months[::2], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log weekday boardings (minus own pre mean)")
    ax.set_title("Monthly weekday boardings by city")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "cities_trends.png"), dpi=130)
    plt.close(fig)

    # 8) 요약 문장 + 금지 표현 점검
    s = (f"판정: {verdict}. 서울과 비교 도시 4곳 평균의 평일 승차 격차는 사후 3개월에 사전 평균보다 "
         f"{out['main']['tau_pct']:+.1f}% 달랐다. 이 값은 같은 계산을 대조 도시에 해 본 5개 값 중 "
         f"크기 기준 {out['placebo_space']['abs_tau_rank']}번째였다.")
    out["summary_sentence"] = s
    out["wording_check"] = {"banned_found": [b for b in BANNED if b in s]}
    with open(os.path.join(a.out, "cities_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(space.to_string())


if __name__ == "__main__":
    main()
