# 케이스: 서울 기후동행카드 시범사업 전후 지하철 승차 흐름 비교

- 작성: 문정미 (jm-git-163)
- 주제 분류: 대중교통 요금 지원 (`public-transit`)
- 판정: **조건부** (자세한 내용은 [discussion.md](discussion.md))

## 파일
| 파일 | 내용 |
|---|---|
| [plan.yaml](plan.yaml) | 분석 계획서 (멘토 스키마 형식, 내용은 계획 2 원본 `plan_v2.yaml`과 같음) |
| [discussion.md](discussion.md) | 해석, 한계, 판정 근거 |
| [../REPORT_v2.md](../REPORT_v2.md) | 결과 전체 (표, 그림, 민감도) |
| [../README.md](../README.md) | 저장소 전체 안내, 실행 방법, 계획 변경 기록 |

## 순서 (이 저장소 기준)
1. `plan_v2.yaml`을 자료를 열기 전에 올림 (커밋 87f5222)
2. `src/prepare_cities.py`와 `src/estimate_cities.py`를 결과를 보기 전에 올림 (커밋 e6e1043)
3. 결과 올림 (`results/cities_*`, `REPORT_v2.md`)
4. 멘토 스키마 형식으로 계획서를 옮김 (`cases/plan.yaml`, 내용 동일)

## 아직 안 한 것
- 멘토 공용 저장소(`CausalInferenceLab/policy-effect-analytics-agent`)의 `cases/` 폴더에 PR로 올리는 것
- 멘토 엔진(`core/`)으로 같은 계산을 다시 돌려 보는 것
- 대구·광주·대전 원본 파일을 `fetch.py`로 자동으로 받는 것 (지금은 수동으로 받아 `data/raw/`에 둠)
