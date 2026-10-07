"""탐색적 분석 (사전등록 밖). estimate.py 결과를 해석하기 위한 보조 확인이며, 주 결론의 근거로 쓰지 않는다.

확인 질문
1) 로그 기준 효과(약 0)와 수준(명/일) 기준 효과(양수)가 왜 다르게 보이나? -> 서울 기준 수준 효과를 %로 환산, 역 규모별 분해
2) 처치가 도시 단위이므로 역 단위 군집 표준오차가 과소 추정될 수 있다. -> 도시 2개의 월별 합계 시계열로 줄여 보면?

사용: python src/exploratory.py --panel data/processed/station_month.csv --out results
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate import (POST_START, PRE_END, PRE_START, REF_MONTH, TRANSITION,  # noqa: E402
                      balanced_sample, load_panel)


def city_series(bal, col):
    return bal.groupby(["city", "month"])[col].sum().unstack("city")


def simple_did(s, post_start, drop=(), pre_end=None, start=None):
    """서울-부산 로그 차이의 사후 평균 - 사전 평균. s: index=month, columns=[busan, seoul] (로그)"""
    gap = s["seoul"] - s["busan"]
    gap = gap[~gap.index.isin(list(drop))]
    if start is not None:
        gap = gap[gap.index >= start]
    if pre_end is not None:
        gap = gap[gap.index <= pre_end]
    pre = gap[gap.index < post_start]
    post = gap[gap.index >= post_start]
    return float(post.mean() - pre.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/station_month.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    bal = balanced_sample(load_panel(a.panel))
    out = {}

    # 1) 수준 효과를 %로: 서울 사전 기간 평균 역별 승차인원 대비
    pre = bal[(bal["month"] <= PRE_END)]
    base = pre.groupby("city")["mean_weekday_boardings"].mean()
    out["pre_period_mean_boardings_per_station_month"] = {k: float(v) for k, v in base.items()}
    out["level_effect_persons_per_day_from_estimate_csv"] = None
    rb = pd.read_csv(os.path.join(a.out, "robustness.csv"))
    lvl = rb.loc[rb["spec"].str.startswith("level"), "coef"].iloc[0]
    out["level_effect_persons_per_day_from_estimate_csv"] = float(lvl)
    out["level_effect_pct_of_seoul_pre_mean"] = float(lvl / base["seoul"] * 100)

    # 규모별: 서울 사전 평균 기준 규모 5분위로 나누어 (사후-사전) 변화를 부산 평균 변화와 비교
    d = bal[(bal["month"] != TRANSITION)].copy()
    d["post"] = (d["month"] >= POST_START).astype(int)
    unit_chg = d.groupby(["unit", "city", "post"])["mean_weekday_boardings"].mean().unstack("post")
    unit_chg["chg"] = unit_chg[1] - unit_chg[0]
    unit_chg["pre"] = unit_chg[0]
    unit_chg = unit_chg.reset_index()
    uc = unit_chg.copy()
    uc["size_q"] = pd.qcut(uc["pre"], 5, labels=False, duplicates="drop") + 1
    tab = uc.groupby(["city", "size_q"])["chg"].agg(["mean", "count"]).unstack("city")
    tab.columns = [f"{a_}_{b_}" for a_, b_ in tab.columns]
    tab.to_csv(os.path.join(a.out, "exploratory_by_size_quintile.csv"))
    out["pct_change_by_city"] = {
        c: float(g["chg"].sum() / g["pre"].sum() * 100) for c, g in unit_chg.groupby("city")}

    # 2) 도시 합계 시계열 DiD + 가짜 도입일 (사전 기간 안)
    ls = np.log(city_series(bal, "mean_weekday_boardings"))
    actual = simple_did(ls, POST_START, drop=[TRANSITION])
    pl = []
    for p in pd.period_range("2022-12", "2023-08", freq="M"):
        sub = ls[(ls.index >= PRE_START) & (ls.index <= PRE_END)]
        pl.append(simple_did(sub, p + 1, drop=[p]))
    out["city_total_log_did"] = actual
    out["city_total_placebo"] = {"values": [float(x) for x in pl],
                                 "share_abs_ge_actual": float(np.mean(np.abs(pl) >= abs(actual)))}
    gap = (ls["seoul"] - ls["busan"]).rename("seoul_minus_busan_log")
    gap.reset_index().assign(month=lambda x: x["month"].astype(str)).to_csv(
        os.path.join(a.out, "exploratory_city_gap.csv"), index=False)

    with open(os.path.join(a.out, "exploratory_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(tab.round(1).to_string())


if __name__ == "__main__":
    sys.exit(main())
