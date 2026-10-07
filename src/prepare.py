"""원본 CSV(서울교통공사, 부산교통공사)를 역-월 단위 패널로 정리한다.

입력 : data/raw/*.csv (공공데이터포털에서 받은 원본, 저장소에는 올리지 않음. 출처는 README 참고)
출력 : data/processed/station_month.csv  (역-월 단위, 평일 평균 일 승차인원)
       data/processed/qc_report.txt      (품질 점검 결과)

사용: python src/prepare.py --raw data/raw --out data/processed
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

HOUR_COLS_SEOUL = slice(7, 27)   # 06시 이전 ~ 24시 이후 (20개)
HOUR_COLS_BUSAN = slice(6, 30)   # 01-02시 ~ 24-01시 (24개)


def read_cp949(path, **kw):
    return pd.read_csv(path, encoding="cp949", **kw)


def load_seoul(path, qc):
    """서울: 일자, 호선, 역번호, 역명, 승하차구분, 승객유형별 시간대 인원 -> 역-일 승차인원"""
    df = read_cp949(path)
    cols = list(df.columns)
    date, line, code, name, flow = cols[1], cols[2], cols[3], cols[4], cols[5]
    hours = cols[HOUR_COLS_SEOUL]
    qc.append(f"[서울] {os.path.basename(path)} 행수={len(df):,} 시간대열={len(hours)}")
    flows = df[flow].astype(str).str.strip().unique().tolist()
    qc.append(f"       승하차구분 값: {flows}")
    board = df[df[flow].astype(str).str.strip() == "승차"].copy()
    board["boardings"] = board[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    # 호선 표기가 파일마다 다르다('1', '1.0', '1호선' 세 가지). 정수 부분만 남긴다. 빈 값은 NA로 두고 main()에서 역번호로 채운다.
    board["line"] = pd.to_numeric(board[line].astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce").astype("Int64")
    n_na = int(board["line"].isna().sum())
    if n_na:
        qc.append(f"       호선 값이 비어 있는 승차 행 {n_na}개: " + ", ".join(sorted(set(f"{r[date]} {r[name]}({r[code]})" for _, r in board[board['line'].isna()].iterrows()))))
    board["date"] = pd.to_datetime(board[date])
    # dropna=False: 호선이 비어 있는 행이 groupby에서 조용히 버려지는 것을 막는다.
    out = (board.groupby(["date", "line", code, name], as_index=False, dropna=False)["boardings"].sum()
           .rename(columns={code: "station_code", name: "station_name"}))
    out["city"] = "seoul"
    return out


def load_busan(path, qc):
    """부산: 역번호, 역명, 년월일, 요일, 구분(승차/하차), 합계, 시간대별 인원 -> 역-일 승차인원"""
    df = read_cp949(path)
    cols = list(df.columns)
    code, name, date, dow, flow, total = cols[0], cols[1], cols[2], cols[3], cols[4], cols[5]
    hours = cols[HOUR_COLS_BUSAN]
    qc.append(f"[부산] {os.path.basename(path)} 행수={len(df):,} 시간대열={len(hours)}")
    flows = df[flow].astype(str).str.strip().unique().tolist()
    qc.append(f"       구분 값: {flows}")
    hsum = df[hours].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    diff = (pd.to_numeric(df[total], errors="coerce") - hsum).abs()
    qc.append(f"       합계 vs 시간대 합: 불일치 행 {(diff > 0).sum():,} / {len(df):,} (최대 차이 {diff.max():,.0f})")
    board = df[df[flow].astype(str).str.strip() == "승차"].copy()
    board["boardings"] = pd.to_numeric(board[total], errors="coerce")
    board["date"] = pd.to_datetime(board[date])
    c = pd.to_numeric(board[code], errors="coerce")
    board["station_code"] = c
    # 부산은 호선 열이 없어 역번호 대역으로 호선을 정한다: 1xx=1호선, 2xx=2호선, 3xx=3호선, 4xx=4호선
    # (1호선 역번호는 95~134 처럼 100 미만도 있어 200 미만은 1호선으로 본다)
    board["line"] = pd.cut(c, bins=[0, 199, 299, 399, 499], labels=[1, 2, 3, 4]).astype("Int64")
    out = (board.groupby(["date", "line", "station_code", board[name]], as_index=False)["boardings"].sum()
           .rename(columns={name: "station_name"}))
    out["city"] = "busan"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    qc = []

    parts = []
    for f in sorted(glob.glob(os.path.join(a.raw, "seoul_*.csv"))):
        parts.append(load_seoul(f, qc))
    for f in sorted(glob.glob(os.path.join(a.raw, "busan_*.csv"))):
        parts.append(load_busan(f, qc))
    daily = pd.concat(parts, ignore_index=True)
    # 서울: 호선이 비어 있는 행은 같은 역번호의 다른 행에서 호선을 찾아 채운다(역번호당 호선이 하나로 정해진 경우만).
    s = daily[daily["city"] == "seoul"]
    lk = s.dropna(subset=["line"]).groupby("station_code")["line"].agg(lambda x: x.mode().iloc[0] if x.nunique() == 1 else pd.NA)
    miss = daily["line"].isna() & (daily["city"] == "seoul")
    if miss.any():
        daily.loc[miss, "line"] = daily.loc[miss, "station_code"].map(lk).astype("Int64")
        qc.append(f"서울 호선 빈 값 {int(miss.sum())}개를 역번호로 채움 -> 모두 채워짐: {bool(daily.loc[miss, 'line'].notna().all())}")
    left = daily[daily["line"].isna()]
    if len(left):
        qc.append(f"호선을 정할 수 없어 제외한 행 {len(left):,}개: " + str(left.groupby("city")["station_code"].unique().to_dict()))
    daily = daily.dropna(subset=["line"])
    daily["line"] = daily["line"].astype(int)
    qc.append(f"\n역-일 행수: {len(daily):,}  기간: {daily['date'].min().date()} ~ {daily['date'].max().date()}")

    # 분석 창: plan.yaml 의 window 와 동일하게 2022-07-01 ~ 2024-12-31
    daily = daily[(daily["date"] >= "2022-07-01") & (daily["date"] <= "2024-12-31")].copy()

    # 중복 확인 (같은 역-호선-일자가 두 번 이상 나오면 안 된다)
    dup = daily.duplicated(["city", "line", "station_code", "date"]).sum()
    qc.append(f"중복 (도시, 호선, 역번호, 일자): {dup:,}")

    # 일자 연속성: 도시별로 빠진 날짜
    for city, g in daily.groupby("city"):
        full = pd.date_range("2022-07-01", "2024-12-31")
        missing = sorted(set(full) - set(g["date"].unique()))
        qc.append(f"[{city}] 전체 {len(full)}일 중 자료 없는 날 {len(missing)}일"
                  + (f" 예: {[d.strftime('%Y-%m-%d') for d in missing[:5]]}" if missing else ""))

    # 평일(월~금)만
    daily["dow"] = daily["date"].dt.dayofweek
    wk = daily[daily["dow"] < 5].copy()
    wk["month"] = wk["date"].dt.to_period("M").astype(str)

    panel = (wk.groupby(["city", "line", "station_code", "station_name", "month"], as_index=False)
             .agg(weekday_days=("boardings", "size"),
                  mean_weekday_boardings=("boardings", "mean"),
                  median_weekday_boardings=("boardings", "median")))
    panel.to_csv(os.path.join(a.out, "station_month.csv"), index=False, encoding="utf-8")

    qc.append(f"\n역-월 패널 행수: {len(panel):,}")
    for city, g in panel.groupby("city"):
        qc.append(f"[{city}] 역-호선 수={g[['line','station_code']].drop_duplicates().shape[0]}, "
                  f"월 수={g['month'].nunique()}, 호선={sorted(g['line'].unique().tolist())}")
        full_n = g["month"].nunique()
        cnt = g.groupby(["line", "station_code"])["month"].nunique()
        qc.append(f"       모든 월에 자료가 있는 역 {(cnt == full_n).sum()} / {len(cnt)}"
                  f"  (일부 월만 있는 역 {(cnt < full_n).sum()}: 신설/폐쇄/명칭변경 가능성)")
        zero = (g["mean_weekday_boardings"] <= 0).sum()
        qc.append(f"       평균 승차인원 0 이하인 역-월 {zero}")
        part = cnt[cnt < full_n]
        for (ln, sc), n in part.items():
            sub = g[(g['line'] == ln) & (g['station_code'] == sc)]
            qc.append(f"         부분 기간 역: {ln}호선 {sub['station_name'].iloc[0]}({sc}) {n}개월 {sub['month'].min()}~{sub['month'].max()}")
        zr = g[g['mean_weekday_boardings'] <= 0]
        for (ln, sc), sub in zr.groupby(['line', 'station_code']):
            qc.append(f"         승차 0인 역: {ln}호선 {sub['station_name'].iloc[0]}({sc}) {len(sub)}개월 ({sub['month'].min()}~{sub['month'].max()})")

    with open(os.path.join(a.out, "qc_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(qc) + "\n")
    print("\n".join(qc))


if __name__ == "__main__":
    sys.exit(main())
