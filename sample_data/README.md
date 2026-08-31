# 합성 샘플 데이터

공개 저장소에는 가격·시가총액·거래대금·팩터 원본이 포함되지 않습니다. 이 폴더는 `final/` 코드가 기대하는 열 구조를 설명하기 위한 합성 예시입니다.

```bash
python scripts/generate_sample_data.py
```

| 샘플 | 원본 입력의 논리 구조 |
| --- | --- |
| `prices_sample.csv` | `Date`와 종목코드별 수정주가 |
| `total_return_sample.csv` | `코드명` 날짜 인덱스와 종목명별 배당포함지수 |
| `market_cap_sample.csv` | `코드명` 날짜 인덱스와 종목명별 시가총액 |
| `trading_value_sample.csv` | `Date`와 종목코드별 거래대금 |
| `trading_value_60_sample.csv` | `Date`와 종목코드별 60일 거래대금 기준값 |
| `factors_sample.csv` | `Date`, `KOSPI`, `HML`, `SMB`, `MOM`, `CD91` |

샘플 값은 인위적으로 생성했으며 공개된 백테스트 결과에는 사용하지 않았습니다.
