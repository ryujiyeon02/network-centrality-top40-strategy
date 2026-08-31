import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SCORES_PATH = Path(__file__).resolve().parent / "daily_scores.csv"


def load_q5_sets(scores_path: Path) -> pd.Series:
    scores = pd.read_csv(scores_path, usecols=["date", "ticker", "quintile"], parse_dates=["date"])
    scores = scores[scores["quintile"] == 5].copy()
    by_date = scores.groupby("date")["ticker"].agg(lambda x: set(x))
    return by_date.sort_index()


def compute_daily_changes(q5_sets: pd.Series) -> pd.DataFrame:
    rows = []
    prev_set = None

    for date, current_set in q5_sets.items():
        if prev_set is not None:
            entered = len(current_set - prev_set)
            exited = len(prev_set - current_set)
            overlap = len(current_set & prev_set)
            rows.append(
                {
                    "date": date,
                    "prev_count": len(prev_set),
                    "curr_count": len(current_set),
                    "entered": entered,
                    "exited": exited,
                    "overlap": overlap,
                    "change_ratio": entered / max(len(prev_set), 1),
                    "overlap_ratio": overlap / max(len(prev_set), 1),
                }
            )
        prev_set = current_set

    return pd.DataFrame(rows)


def get_month_end_dates(dates: pd.Index) -> list[pd.Timestamp]:
    month_ends = pd.Series(dates, index=dates).groupby(dates.to_period("M")).tail(1).index.tolist()
    if month_ends and month_ends[0] != dates[0]:
        month_ends = [dates[0]] + month_ends
    return month_ends


def get_trigger_dates(q5_sets: pd.Series, threshold: float) -> pd.DataFrame:
    dates = list(q5_sets.index)
    anchor_date = dates[0]
    anchor_set = q5_sets.iloc[0]
    rows = [
        {
            "date": anchor_date,
            "change_from_anchor": 0.0,
            "anchor_date": anchor_date,
            "days_since_anchor": 0,
        }
    ]

    for date in dates[1:]:
        current_set = q5_sets.loc[date]
        change_ratio = len(current_set - anchor_set) / max(len(anchor_set), 1)
        if change_ratio >= threshold:
            rows.append(
                {
                    "date": date,
                    "change_from_anchor": change_ratio,
                    "anchor_date": anchor_date,
                    "days_since_anchor": (date - anchor_date).days,
                }
            )
            anchor_date = date
            anchor_set = current_set

    return pd.DataFrame(rows)


def summarize_frequency(dates: list[pd.Timestamp]) -> dict[str, float]:
    if len(dates) <= 1:
        return {"count": len(dates), "avg_gap_days": 0.0, "median_gap_days": 0.0}

    gaps = pd.Series(dates).diff().dropna().dt.days
    return {
        "count": len(dates),
        "avg_gap_days": float(gaps.mean()),
        "median_gap_days": float(gaps.median()),
    }


def build_threshold_sensitivity(q5_sets: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        trigger_df = get_trigger_dates(q5_sets, threshold)
        summary = summarize_frequency(trigger_df["date"].tolist())
        rows.append(
            {
                "threshold": threshold,
                "rebalances": summary["count"],
                "avg_gap_days": summary["avg_gap_days"],
                "median_gap_days": summary["median_gap_days"],
                "avg_trigger_change": trigger_df["change_from_anchor"].iloc[1:].mean() if len(trigger_df) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_month_end_drift(q5_sets: pd.Series, trigger_dates: list[pd.Timestamp]) -> pd.DataFrame:
    month_ends = get_month_end_dates(q5_sets.index)
    trigger_iter = iter(trigger_dates)
    active_date = next(trigger_iter)
    active_set = q5_sets.loc[active_date]
    next_trigger = next(trigger_iter, None)
    rows = []

    for date in q5_sets.index:
        if next_trigger is not None and date == next_trigger:
            active_date = date
            active_set = q5_sets.loc[date]
            next_trigger = next(trigger_iter, None)

        if date in month_ends:
            target_set = q5_sets.loc[date]
            drift_ratio = len(target_set - active_set) / max(len(active_set), 1)
            rows.append(
                {
                    "date": date,
                    "active_date": active_date,
                    "drift_ratio": drift_ratio,
                    "days_since_rebalance": (date - active_date).days,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="daily_scores.csv 기반 Q5 구성 변화/트리거 분석")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--threshold-grid", type=float, nargs="*", default=[0.30, 0.35, 0.40, 0.45])
    args = parser.parse_args()

    q5_sets = load_q5_sets(args.scores)
    daily_changes = compute_daily_changes(q5_sets)
    month_end_dates = get_month_end_dates(q5_sets.index)
    trigger_df = get_trigger_dates(q5_sets, args.threshold)
    trigger_dates = trigger_df["date"].tolist()
    month_end_drift = compute_month_end_drift(q5_sets, trigger_dates)
    sensitivity_df = build_threshold_sensitivity(q5_sets, args.threshold_grid)

    monthly_summary = summarize_frequency(month_end_dates)
    trigger_summary = summarize_frequency(trigger_dates)

    print(f"scores: {args.scores}")
    print(f"threshold: {args.threshold:.0%}")

    print("\n[Universe]")
    print(f"period: {q5_sets.index[0].date()} ~ {q5_sets.index[-1].date()}")
    print(f"trading_days: {len(q5_sets)}")
    print(f"avg_q5_names: {q5_sets.map(len).mean():.2f}")
    print(f"median_q5_names: {q5_sets.map(len).median():.2f}")

    print("\n[Daily Change Stats]")
    print(f"avg_change_ratio: {daily_changes['change_ratio'].mean():.2%}")
    print(f"median_change_ratio: {daily_changes['change_ratio'].median():.2%}")
    print(f"max_change_ratio: {daily_changes['change_ratio'].max():.2%}")
    print(f"hit_rate_over_threshold: {(daily_changes['change_ratio'] >= args.threshold).mean():.2%}")

    print("\n[Frequency Comparison]")
    print(f"monthly_rebalances: {monthly_summary['count']}")
    print(f"monthly_avg_gap_days: {monthly_summary['avg_gap_days']:.2f}")
    print(f"monthly_median_gap_days: {monthly_summary['median_gap_days']:.2f}")
    print(f"trigger_rebalances: {trigger_summary['count']}")
    print(f"trigger_avg_gap_days: {trigger_summary['avg_gap_days']:.2f}")
    print(f"trigger_median_gap_days: {trigger_summary['median_gap_days']:.2f}")

    print("\n[Threshold Sensitivity]")
    sensitivity_view = sensitivity_df.copy()
    sensitivity_view["threshold"] = sensitivity_view["threshold"].map(lambda x: f"{x:.0%}")
    sensitivity_view["avg_gap_days"] = sensitivity_view["avg_gap_days"].map(lambda x: f"{x:.2f}")
    sensitivity_view["median_gap_days"] = sensitivity_view["median_gap_days"].map(lambda x: f"{x:.2f}")
    sensitivity_view["avg_trigger_change"] = sensitivity_view["avg_trigger_change"].map(lambda x: f"{x:.2%}")
    print(sensitivity_view.to_string(index=False))

    print("\n[Trigger Events]")
    print(f"avg_change_from_anchor: {trigger_df['change_from_anchor'].iloc[1:].mean():.2%}")
    print(f"median_change_from_anchor: {trigger_df['change_from_anchor'].iloc[1:].median():.2%}")
    print(f"max_change_from_anchor: {trigger_df['change_from_anchor'].max():.2%}")

    print("\n[Month-End Drift Under Trigger Rule]")
    print(f"avg_drift_ratio: {month_end_drift['drift_ratio'].mean():.2%}")
    print(f"median_drift_ratio: {month_end_drift['drift_ratio'].median():.2%}")
    print(f"max_drift_ratio: {month_end_drift['drift_ratio'].max():.2%}")
    print(f"avg_days_since_rebalance: {month_end_drift['days_since_rebalance'].mean():.2f}")

    print("\n[Recent Trigger Events]")
    recent = trigger_df.tail(10).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["anchor_date"] = recent["anchor_date"].dt.strftime("%Y-%m-%d")
    print(recent.to_string(index=False))


if __name__ == "__main__":
    main()
