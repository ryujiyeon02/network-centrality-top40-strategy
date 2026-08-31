# 1st: 초기 중심성 전략 실험

이 폴더는 3주차 중심성 전략의 초기 최종본입니다.

핵심은 `daily_scores.csv`로 계산된 일별 네트워크 중심성 점수를 활용해 Q5 종목군을 만들고, Jaccard 유사도가 충분히 낮아질 때만 리밸런싱하는 것입니다.

## 핵심 파일

| 파일 | 설명 |
| --- | --- |
| `final_strategy.py` | Jaccard 0.3 + Entry Delay + ScorePow 가중 전략 |
| `strategy_description_final.txt` | 전략 원리와 성과를 정리한 최종 설명 문서 |
| `strategy_full_description.txt` | 전체 계산 프로시저 설명 |
| `daily_scores_trigger_backtest.py` | 일별 점수 trigger 백테스트 |
| `daily_scores_trigger_test.py` | trigger 테스트 |
| `best_from2010.py` | 2010년 이후 성과 기준 실험 |
| `winsorize_amihud_test.py` | 윈저라이징 및 Amihud 거래비용 실험 |

## 전략 특징

- 일별 composite score 기반 Q5 종목군 산출
- Q5 구성 변화가 충분히 클 때만 리밸런싱
- Jaccard threshold는 0.3
- 리드-래그 데이터에서 계산한 평균 lag를 Entry Delay로 사용
- ScorePow 방식으로 점수 높은 종목에 비중 집중
- Amihud 기준 유동성 하위 종목에는 높은 거래비용 적용

## 제외된 원천 데이터

아래 파일은 크기가 크거나 로컬 원천 데이터 성격이 강해 GitHub에는 포함하지 않았습니다.

- `daily_scores.csv`
- `lead_lag_pairs.csv`
- `total adj close.csv`
- `trading value.csv`
- `trading value 60.csv`
- `수정주가.csv`
- `시가총액.csv`
