# final: 최종 중심성 전략 v2

이 폴더는 중심성 전략의 최종 구현과 산출물입니다.

`final_strategy_v2.py`는 원천 데이터에서 매월 네트워크 중심성 점수를 직접 계산하고, Z-Score Composite 기준으로 Q1~Q5 포트폴리오와 Q5 수수료후 성과를 산출합니다.

## 핵심 파일

| 파일 | 설명 |
| --- | --- |
| `final_strategy_v2.py` | 최종 전략 v2 메인 실행 파일 |
| `final_v2_output.txt` | 실행 로그와 성과 요약 |
| `monthly_scores.csv` | 월별 composite score 저장 결과 |
| `final_strategy_v2.png` | 성과 시각화 |
| `final_strategy_v2_nav.xlsx` | Q5 NAV 결과 |
| `export_q5_top40_nav.py` | Q5/Top40 NAV 산출 |
| `export_double_sort_heatmap.py` | 시가총액 x 점수 double sort heatmap 산출 |
| `export_final_strategy_cost_sensitivity.py` | 거래비용 민감도 산출 |
| `export_ff4_ols.py` | Carhart 4-Factor OLS 산출 |
| `final_figure/` | 최종 그림과 엑셀 산출물 |

## 최종 실행 결과 요약

`final_v2_output.txt` 기준:

| 항목 | 값 |
| --- | ---: |
| 기간 | 2000-01 ~ 2025-12 |
| 종목 수 | 3,751 |
| Q5 수수료후 CAGR | 17.62% |
| Q5 수수료후 Sharpe | 0.60 |
| Q5 수수료후 MDD | -54.4% |
| Carhart 4-Factor Alpha | 16.59% |
| Alpha t-stat | 4.50 |

## 재실행 시 필요한 대용량 데이터

아래 원천 데이터는 GitHub repo에 포함하지 않았습니다.

- `수정주가_거래대금필터_code.csv`
- `현금배당포함.csv`
- `시가총액.csv`
- `거래대금.csv`
- `60_거래대금.csv`

원본 로컬 데이터가 같은 폴더에 있어야 `final_strategy_v2.py`를 처음부터 다시 실행할 수 있습니다.
