"""계획 2: 다섯 도시(서울, 부산, 대구, 광주, 대전) 원본 CSV를 도시 x 월 패널로 정리한다.

입력 : data/raw/ 아래 원본 CSV (저장소에는 올리지 않음. 받는 곳은 README 참고)
        seoul_<날짜>.csv, busan_<연도>.csv            (계획 1과 같음)
        daegu_<YYYYMMDD>.csv, gwangju_<YYYYMMDD>.csv, daejeon_<YYYYMMDD>.csv
        (세 도시 파일은 연초부터 파일 날짜까지 누적. 같은 날짜가 두 파일에 있으면 뒤 파일을 쓴다)
출력 : data/processed/city_month.csv      도시 x 월 (분석용)
       data/processed/station_month_5cities.csv  역 x 월 (참고 분석용)
       data/processed/qc_report_cities.txt

사용: python src/prepare_cities.py --plan plan_v2.yaml --raw data/raw --out data/processed
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_busan, load_seoul  # noqa: E402  (계획 1의 정리 규칙을 그대로 쓴다)


def read_any(path):
    """파일마다 인코딩이 달라서(UTF-8 BOM 또는 CP949) 차례로 시도한다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 알 수 없음: {path}")


def _board_only(df, flow_col, qc, label):
    vals = df[flow_col].astype(str).str.strip()
    qc.append(f"       구분 값(공백 제거 후): {sorted(vals.unique().tolist())}")
    return df[vals == "승차"].copy()


def load_daegu(path, qc):
    """대구: 월, 일, 역번호, 역명, 승하차, 05~24시 시간대, 일계. 연도 열이 없어 파일 이름의 연도를 붙인다."""
    df, enc = read_any(path)
    cols = list(df.columns)
    year = int(re.search(r"(\d{4})\d{4}", os.path.basename(path)).group(1))
    hours, total = cols[5:-1], cols[-1]
    qc.append(f"[대구] {os.path.basename(path)} {enc} 행수={len(df):,} 시간대열={len(hours)} 연도={year}")
    hsum = df[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    diff = (pd.to_numeric(df[total], errors="coerce") - hsum).abs()
    qc.append(f"       일계 vs 시간대 합: 불일치 행 {(diff > 0).sum():,} (최대 차이 {diff.max():,.0f})")
    b = _board_only(df, cols[4], qc, "대구")
    b["date"] = pd.to_datetime(dict(year=year, month=b[cols[0]].astype(int), day=b[cols[1]].astype(int)))
    b["boardings"] = pd.to_numeric(b[total], errors="coerce")
    b["station_code"] = pd.to_numeric(b[cols[2]], errors="coerce")
    b["station_name"] = b[cols[3]].astype(str).str.strip()
    return b[["date", "station_code", "station_name", "boardings"]]


def _load_date_hours(path, qc, label):
    """광주·대전: 날짜, 역번호, 역명, 구분, 시간대별 인원(합계 열 없음)."""
    df, enc = read_any(path)
    cols = list(df.columns)
    hours = cols[4:]
    qc.append(f"[{label}] {os.path.basename(path)} {enc} 행수={len(df):,} 시간대열={len(hours)}")
    b = _board_only(df, cols[3], qc, label)
    b["date"] = pd.to_datetime(b[cols[0]].astype(str).str.strip())
    b["boardings"] = b[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    b["station_code"] = pd.to_numeric(b[cols[1]], errors="coerce")
    b["station_name"] = b[cols[2]].astype(str).str.strip()
    return b[["date", "station_code", "station_name", "boardings"]]


def load_cumulative(city, files, loader, qc):
    """누적 파일 여러 개를 합친다. 같은 (역, 날짜)는 날짜가 늦은 파일 값을 쓴다."""
    parts = []
    for order, f in enumerate(sorted(files)):
        p = loader(f, qc) if loader is load_daegu else loader(f, qc, city)
        # 한 파일 안에서 같은 역-날짜가 두 번 나오는 경우(예: 대구 2023 파일의 12월이 두 번 들어 있음,
        # 두 묶음 합계 차이 약 0.007%)에는 파일에서 뒤에 나오는 행을 쓴다.
        inner = p.duplicated(["station_code", "date"], keep=False)
        if inner.any():
            days = sorted(p.loc[inner, "date"].dt.strftime("%Y-%m").unique())
            qc.append(f"       파일 안 중복 역-날짜 {int(inner.sum()) // 2:,}건 (월: {days}) → 뒤에 나오는 행 사용")
            p = p.drop_duplicates(["station_code", "date"], keep="last")
        p["_order"] = order
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    before = len(d)
    d = (d.sort_values("_order", kind="stable").drop_duplicates(["station_code", "date"], keep="last")
         .drop(columns="_order"))
    qc.append(f"[{city}] 파일 사이 겹침 제거: {before - len(d):,}행 (같은 역-날짜가 두 파일에 있던 행)")
    # 같은 역-날짜에 승차 행이 여러 개면 더한다 (환승역 게이트 분리 등)
    d = d.groupby(["date", "station_code"], as_index=False).agg(
        station_name=("station_name", "first"), boardings=("boardings", "sum"))
    d["city"] = city
    d["line"] = 0
    return d


def month_index(months, start):
    s = pd.Period(start, "M")
    return [(pd.Period(m, "M") - s).n for m in months]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="plan_v2.yaml")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    plan = yaml.safe_load(open(a.plan, encoding="utf-8"))
    os.makedirs(a.out, exist_ok=True)
    qc = []

    pre0, pre1 = plan["window"]["pre"]
    post0, post1 = plan["window"]["post"]
    excluded = set(plan["window"].get("excluded", []))
    months = [str(p) for p in pd.period_range(pre0, post1, freq="M") if str(p) not in excluded]

    # 1) 역-일 승차 (도시별)
    parts = []
    s = [load_seoul(f, qc) for f in sorted(glob.glob(os.path.join(a.raw, "seoul_*.csv")))]
    if s:
        seoul = pd.concat(s, ignore_index=True)
        parts.append(seoul.dropna(subset=["line"]))
    b = [load_busan(f, qc) for f in sorted(glob.glob(os.path.join(a.raw, "busan_*.csv")))]
    if b:
        parts.append(pd.concat(b, ignore_index=True).dropna(subset=["line"]))
    loaders = {"daegu": load_daegu, "gwangju": _load_date_hours, "daejeon": _load_date_hours}
    for city, fn in loaders.items():
        files = glob.glob(os.path.join(a.raw, f"{city}_*.csv"))
        if files:
            parts.append(load_cumulative(city, files, fn, qc))
    daily = pd.concat(parts, ignore_index=True)
    daily["line"] = daily["line"].astype(int)
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    daily = daily[daily["month"].isin(months)].copy()

    dup = daily.duplicated(["city", "line", "station_code", "date"]).sum()
    qc.append(f"\n중복 (도시, 호선, 역번호, 일자): {dup:,}")
    for city, g in daily.groupby("city"):
        days = set(pd.date_range(pre0 + "-01", pd.Period(post1, "M").end_time.normalize()))
        days = {d for d in days if str(d.to_period("M")) in months}
        missing = sorted(days - set(g["date"].unique()))
        qc.append(f"[{city}] 대상 {len(days)}일 중 자료 없는 날 {len(missing)}일"
                  + (f" 예: {[d.strftime('%Y-%m-%d') for d in missing[:5]]}" if missing else ""))

    # 2) 역-월: 평일(월~금) / 주말(토·일) 하루 평균 승차
    daily["dow"] = daily["date"].dt.dayofweek
    daily["daytype"] = np.where(daily["dow"] < 5, "weekday", "weekend")
    sm = (daily.groupby(["city", "line", "station_code", "month", "daytype"], as_index=False)["boardings"].mean()
          .pivot_table(index=["city", "line", "station_code", "month"], columns="daytype", values="boardings")
          .reset_index().rename(columns={"weekday": "mean_weekday", "weekend": "mean_weekend"}))
    names = daily.groupby(["city", "line", "station_code"])["station_name"].first().reset_index()
    sm = sm.merge(names, on=["city", "line", "station_code"], how="left")

    # 3) 도시별 균형 표본: 모든 대상 월에 평일·주말 승차가 0보다 큰 역만
    ok = (sm[(sm["mean_weekday"] > 0) & (sm["mean_weekend"] > 0)]
          .groupby(["city", "line", "station_code"])["month"].nunique())
    keep = ok[ok == len(months)].index
    sm["balanced"] = sm.set_index(["city", "line", "station_code"]).index.isin(keep)
    qc.append(f"\n대상 월 {len(months)}개: {months[0]} ~ {months[-1]} (제외: {sorted(excluded)})")
    for city, g in sm.groupby("city"):
        allst = g[["line", "station_code"]].drop_duplicates().shape[0]
        bal = g[g["balanced"]][["line", "station_code"]].drop_duplicates().shape[0]
        dropped = g[~g["balanced"]].groupby(["line", "station_code"])["station_name"].first()
        qc.append(f"[{city}] 역(-호선) {allst}개 중 균형 표본 {bal}개"
                  + (f", 제외: {', '.join(f'{n}({c})' for (l, c), n in dropped.items())}" if len(dropped) else ""))
    sm.to_csv(os.path.join(a.out, "station_month_5cities.csv"), index=False, encoding="utf-8")

    # 4) 도시 x 월
    cm = (sm[sm["balanced"]].groupby(["city", "month"], as_index=False)
          .agg(n_stations=("station_code", "size"), weekday_total=("mean_weekday", "sum"),
               weekend_total=("mean_weekend", "sum")))
    cm["log_weekday"] = np.log(cm["weekday_total"])
    cm["log_weekend"] = np.log(cm["weekend_total"])
    cm["month_idx"] = month_index(cm["month"], pre0)
    cm["treated"] = (cm["city"] == plan["policy"]["treated_city"]).astype(int)
    cm["post"] = (cm["month"] >= post0).astype(int)
    cm = cm.sort_values(["city", "month"])
    cm.to_csv(os.path.join(a.out, "city_month.csv"), index=False, encoding="utf-8")
    n_per_city = cm.groupby("city")["month"].nunique().to_dict()
    qc.append(f"\n도시 x 월 패널: {len(cm)}행, 도시별 월 수 {n_per_city}")
    qc.append(f"도시별 역 수가 월마다 같은지: {cm.groupby('city')['n_stations'].nunique().eq(1).all()}")

    with open(os.path.join(a.out, "qc_report_cities.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(qc) + "\n")
    print("\n".join(qc))


if __name__ == "__main__":
    sys.exit(main())
