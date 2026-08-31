import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SCORES_PATH = BASE_DIR / "daily_scores.csv"
DEFAULT_PRICES_PATH = BASE_DIR / "수정주가.csv"
DEFAULT_TV_PATH = BASE_DIR / "trading value.csv"
DEFAULT_TV60_PATH = BASE_DIR / "trading value 60.csv"

LOW_COST = 0.003
HIGH_COST = 0.008
ILLIQ_THRESHOLD = 0.80


def normalize_ticker(name: str) -> str:
    value = str(name)
    replacements = [
        ("(주)", ""),
        ("㈜", ""),
        ("주식회사", ""),
        (" ", ""),
        (".", ""),
        (",", ""),
        ("-", ""),
        ("&", "앤"),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    return value.strip()


def load_scores(scores_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(
        scores_path,
        usecols=["date", "ticker", "quintile", "composite_score"],
        parse_dates=["date"],
    )
    scores = scores[scores["quintile"] == 5].copy()
    scores["norm_ticker"] = scores["ticker"].map(normalize_ticker)
    return scores.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_numeric_panel(path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    panel = pd.read_csv(path, index_col="Date", parse_dates=True, usecols=usecols, low_memory=False)
    return panel.apply(pd.to_numeric, errors="coerce").sort_index()


def prepare_data(
    scores_path: Path,
    prices_path: Path,
    tv_path: Path,
    tv60_path: Path,
) -> tuple[dict[pd.Timestamp, pd.Series], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = load_scores(scores_path)

    price_columns = pd.read_csv(prices_path, nrows=0).columns.tolist()[1:]
    price_name_map = {normalize_ticker(col): col for col in price_columns}

    scores["price_ticker"] = scores["norm_ticker"].map(price_name_map)
    scores = scores.dropna(subset=["price_ticker"]).copy()
    needed = sorted(scores["price_ticker"].drop_duplicates())

    usecols = ["Date"] + needed
    prices = load_numeric_panel(prices_path, usecols=usecols)
    tv = load_numeric_panel(tv_path, usecols=usecols)
    tv60 = load_numeric_panel(tv60_path, usecols=usecols)

    returns_daily = prices.pct_change(fill_method=None)
    daily_illiq = returns_daily.abs() / tv
    amihud_monthly = daily_illiq.resample("ME").mean()
    tv60_monthly = tv60.resample("ME").last()

    available_dates = pd.Index(sorted(scores["date"].unique())).intersection(returns_daily.index)
    scores = scores[scores["date"].isin(available_dates)].copy()

    targets = {}
    for trade_date, group in scores.groupby("date", sort=True):
        deduped = group[["price_ticker", "composite_score"]].drop_duplicates("price_ticker", keep="last").copy()
        raw_scores = deduped["composite_score"].to_numpy(dtype=float)
        shifted = raw_scores - np.nanmin(raw_scores) + 1e-8
        score_pow = shifted ** 1.5

        if not np.isfinite(score_pow).all() or score_pow.sum() <= 0:
            weights = np.ones(len(deduped)) / len(deduped)
        else:
            weights = score_pow / score_pow.sum()

        targets[trade_date] = pd.Series(weights, index=deduped["price_ticker"].tolist(), dtype=float)

    return targets, returns_daily, amihud_monthly, tv60_monthly


def get_monthly_dates(all_dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    dates = pd.Index(all_dates)
    month_ends = list(pd.Series(dates, index=dates).groupby(dates.to_period("M")).tail(1).index)
    if month_ends and month_ends[0] != all_dates[0]:
        month_ends = [all_dates[0]] + month_ends
    return month_ends


def get_trigger_dates(targets: dict[pd.Timestamp, pd.Series], threshold: float) -> list[pd.Timestamp]:
    all_dates = sorted(targets)
    trigger_dates = [all_dates[0]]
    anchor = set(targets[all_dates[0]].index)

    for trade_date in all_dates[1:]:
        current = set(targets[trade_date].index)
        change_ratio = len(current - anchor) / max(len(anchor), 1)
        if change_ratio >= threshold:
            trigger_dates.append(trade_date)
            anchor = current

    return trigger_dates


def latest_month_end(index: pd.Index, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = index[index <= date]
    if len(eligible) == 0:
        return None
    return eligible[-1]


def apply_universe_filter(
    target_weights: pd.Series,
    trade_date: pd.Timestamp,
    tv60_monthly: pd.DataFrame,
) -> pd.Series:
    ref_date = latest_month_end(tv60_monthly.index, trade_date)
    if ref_date is None:
        return target_weights

    tv60_now = tv60_monthly.loc[ref_date].dropna()
    if len(tv60_now) == 0:
        return target_weights

    investable = tv60_now[tv60_now > tv60_now.quantile(0.10)].index
    filtered = target_weights[target_weights.index.isin(investable)]
    if len(filtered) == 0:
        return pd.Series(dtype=float)
    return filtered / filtered.sum()


def get_illiquid_set(
    trade_date: pd.Timestamp,
    amihud_monthly: pd.DataFrame,
) -> set[str]:
    ref_date = latest_month_end(amihud_monthly.index, trade_date)
    if ref_date is None:
        return set()

    amihud_now = amihud_monthly.loc[ref_date].dropna()
    if len(amihud_now) == 0:
        return set()

    threshold_value = amihud_now.quantile(ILLIQ_THRESHOLD)
    return set(amihud_now[amihud_now >= threshold_value].index)


def run_backtest(
    targets: dict[pd.Timestamp, pd.Series],
    returns_daily: pd.DataFrame,
    amihud_monthly: pd.DataFrame,
    tv60_monthly: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    use_amihud_cost: bool,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    nav = 1.0
    holdings = pd.Series(dtype=float)
    nav_history = []
    ret_history = []
    turnover_rows = []
    rebalance_set = set(rebalance_dates)

    for trade_date in sorted(targets):
        if trade_date in rebalance_set:
            target_weights = apply_universe_filter(targets[trade_date], trade_date, tv60_monthly)
            prev_weights = holdings / holdings.sum() if holdings.sum() > 0 else pd.Series(dtype=float)

            all_names = prev_weights.index.union(target_weights.index)
            prev_weights = prev_weights.reindex(all_names, fill_value=0.0)
            target_weights = target_weights.reindex(all_names, fill_value=0.0)
            delta_weights = target_weights - prev_weights

            if use_amihud_cost:
                illiquid_set = get_illiquid_set(trade_date, amihud_monthly)
                cost_rate = np.where(delta_weights.index.isin(illiquid_set), HIGH_COST, LOW_COST)
            else:
                cost_rate = LOW_COST

            portfolio_nav = nav
            trade_amounts = np.abs(delta_weights) * portfolio_nav
            total_trade = float(trade_amounts.sum())
            trade_cost = float((trade_amounts * cost_rate).sum())
            turnover = total_trade / portfolio_nav if portfolio_nav > 0 else 0.0

            nav = max(nav - trade_cost, 0.0)
            holdings = target_weights * nav

            turnover_rows.append(
                {
                    "date": trade_date,
                    "turnover": turnover,
                    "turnover_one_way": turnover / 2.0,
                    "cost_rate": trade_cost / (nav + trade_cost) if nav + trade_cost > 0 else 0.0,
                }
            )

        realized_ret = returns_daily.loc[trade_date, holdings.index] if len(holdings.index) else pd.Series(dtype=float)
        realized_ret = realized_ret.fillna(0.0)
        holdings = holdings * (1 + realized_ret)

        next_nav = holdings.sum() if len(holdings) else nav
        port_ret = next_nav / nav - 1 if nav > 0 else 0.0
        nav = next_nav

        nav_history.append((trade_date, nav))
        ret_history.append((trade_date, port_ret))

    nav_series = pd.Series(dict(nav_history)).sort_index()
    ret_series = pd.Series(dict(ret_history)).sort_index()
    turnover_df = pd.DataFrame(turnover_rows)
    return nav_series, ret_series, turnover_df


def collapse_to_rebalance_path(nav_series: pd.Series, rebalance_dates: list[pd.Timestamp]) -> tuple[pd.Series, pd.Series]:
    sampled_dates = list(rebalance_dates)
    if nav_series.index[-1] != sampled_dates[-1]:
        sampled_dates = sampled_dates + [nav_series.index[-1]]

    sampled_nav = nav_series.reindex(sampled_dates).dropna()
    sampled_ret = sampled_nav.pct_change().fillna(0.0)
    return sampled_nav, sampled_ret


def metrics(nav_series: pd.Series, ret_series: pd.Series) -> dict[str, float]:
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = nav_series.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    periods_per_year = len(ret_series[ret_series.index > ret_series.index[0]]) / years if years > 0 else np.nan
    vol = ret_series.std() * np.sqrt(periods_per_year) if periods_per_year and periods_per_year > 0 else np.nan
    sharpe = (
        ret_series.mean() / ret_series.std() * np.sqrt(periods_per_year)
        if periods_per_year and periods_per_year > 0 and ret_series.std() > 0
        else np.nan
    )
    mdd = (nav_series / nav_series.cummax() - 1).min()
    return {
        "final_nav": float(nav_series.iloc[-1]),
        "cagr": float(cagr),
        "vol": float(vol),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "periods_per_year": float(periods_per_year) if periods_per_year == periods_per_year else np.nan,
    }


def print_result_block(
    title: str,
    rebalance_dates: list[pd.Timestamp],
    nav: pd.Series,
    turnover: pd.DataFrame,
) -> None:
    sampled_nav, sampled_ret = collapse_to_rebalance_path(nav, rebalance_dates)
    result = metrics(sampled_nav, sampled_ret)
    print(f"\n[{title}]")
    print(f"rebalances: {len(rebalance_dates)}")
    print(f"recorded_points: {len(sampled_nav)}")
    print(f"avg_holding_period_days: {pd.Series(sampled_nav.index).diff().dropna().dt.days.mean():.2f}")
    print(f"avg_turnover: {turnover['turnover'].mean():.2%}")
    print(f"avg_turnover_one_way: {turnover['turnover_one_way'].mean():.2%}")
    print(f"avg_cost_rate: {turnover['cost_rate'].mean():.3%}")
    print(f"final_nav: {result['final_nav']:.3f}")
    print(f"cagr: {result['cagr']:.2%}")
    print(f"vol: {result['vol']:.2%}")
    print(f"sharpe: {result['sharpe']:.2f}")
    print(f"mdd: {result['mdd']:.2%}")
    print(f"periods_per_year: {result['periods_per_year']:.2f}")


def collect_result_row(
    label: str,
    rebalance_dates: list[pd.Timestamp],
    nav: pd.Series,
    turnover: pd.DataFrame,
) -> dict[str, float]:
    sampled_nav, sampled_ret = collapse_to_rebalance_path(nav, rebalance_dates)
    result = metrics(sampled_nav, sampled_ret)
    avg_gap_days = pd.Series(sampled_nav.index).diff().dropna().dt.days.mean() if len(sampled_nav) > 1 else 0.0
    return {
        "label": label,
        "rebalances": len(rebalance_dates),
        "avg_gap_days": float(avg_gap_days),
        "avg_turnover": float(turnover["turnover"].mean()),
        "avg_turnover_one_way": float(turnover["turnover_one_way"].mean()),
        "avg_cost_rate": float(turnover["cost_rate"].mean()),
        "final_nav": result["final_nav"],
        "cagr": result["cagr"],
        "vol": result["vol"],
        "sharpe": result["sharpe"],
        "mdd": result["mdd"],
    }


def print_rebalance_schedule(name: str, rebalance_dates: list[pd.Timestamp], max_rows: int) -> None:
    schedule = pd.Series(rebalance_dates, name="rebalance_date")
    print(f"\n[{name} Rebalance Dates]")
    print(f"count: {len(schedule)}")

    yearly_counts = schedule.dt.year.value_counts().sort_index()
    print("by_year:")
    print(yearly_counts.to_string())

    head_n = min(max_rows, len(schedule))
    tail_n = min(max_rows, len(schedule))

    print(f"\nfirst_{head_n}:")
    for value in schedule.head(head_n):
        print(value.strftime("%Y-%m-%d"))

    print(f"\nlast_{tail_n}:")
    for value in schedule.tail(tail_n):
        print(value.strftime("%Y-%m-%d"))


def main() -> None:
    parser = argparse.ArgumentParser(description="daily_scores.csv 기반 월별 vs 트리거 Q5 백테스트")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--tv", type=Path, default=DEFAULT_TV_PATH)
    parser.add_argument("--tv60", type=Path, default=DEFAULT_TV60_PATH)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--threshold-grid", type=float, nargs="*", default=[0.30, 0.35, 0.40, 0.45])
    parser.add_argument("--cost-model", choices=["flat", "amihud"], default="amihud")
    parser.add_argument("--show-dates", action="store_true")
    parser.add_argument("--date-rows", type=int, default=20)
    args = parser.parse_args()

    print(f"scores: {args.scores}")
    print(f"prices: {args.prices}")
    print(f"tv: {args.tv}")
    print(f"tv60: {args.tv60}")
    print(f"threshold: {args.threshold:.0%}")
    print(f"cost_model: {args.cost_model}")

    targets, returns_daily, amihud_monthly, tv60_monthly = prepare_data(
        args.scores,
        args.prices,
        args.tv,
        args.tv60,
    )
    all_dates = sorted(targets)
    monthly_dates = get_monthly_dates(all_dates)
    trigger_dates = get_trigger_dates(targets, args.threshold)
    use_amihud_cost = args.cost_model == "amihud"

    monthly_nav, monthly_ret, monthly_turnover = run_backtest(
        targets, returns_daily, amihud_monthly, tv60_monthly, monthly_dates, use_amihud_cost
    )
    trigger_nav, trigger_ret, trigger_turnover = run_backtest(
        targets, returns_daily, amihud_monthly, tv60_monthly, trigger_dates, use_amihud_cost
    )

    print(f"\nperiod: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)} trading days)")
    print_result_block("Monthly Rebalance", monthly_dates, monthly_nav, monthly_turnover)
    print_result_block(f"Trigger Rebalance {args.threshold:.0%}", trigger_dates, trigger_nav, trigger_turnover)

    monthly_gap = pd.Series(monthly_dates).diff().dropna().dt.days
    trigger_gap = pd.Series(trigger_dates).diff().dropna().dt.days
    print("\n[Frequency]")
    print(f"monthly_avg_gap_days: {monthly_gap.mean():.2f}")
    print(f"trigger_avg_gap_days: {trigger_gap.mean():.2f}")
    print("\n[Notes]")
    print("universe filter: trading value 60 하위 10% 제외")
    print("cost model: Amihud 월평균 기준 상위 20% 종목 80bp, 나머지 30bp" if use_amihud_cost else "cost model: 일괄 30bp")
    print("price path: 일별 수정주가 사용")
    print("winsorized total return / 월별 거래정지 제외는 일별 트리거 경로와 완전히 동일하게 맞출 수 없음")

    summary_rows = [collect_result_row("monthly", monthly_dates, monthly_nav, monthly_turnover)]
    for threshold in args.threshold_grid:
        threshold_dates = get_trigger_dates(targets, threshold)
        threshold_nav, _, threshold_turnover = run_backtest(
            targets, returns_daily, amihud_monthly, tv60_monthly, threshold_dates, use_amihud_cost
        )
        summary_rows.append(
            collect_result_row(f"trigger_{threshold:.0%}", threshold_dates, threshold_nav, threshold_turnover)
        )

    summary_df = pd.DataFrame(summary_rows)
    display_df = summary_df.copy()
    for col in ["avg_gap_days"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")
    for col in ["avg_turnover", "avg_turnover_one_way", "avg_cost_rate", "cagr", "vol", "mdd"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")
    for col in ["final_nav", "sharpe"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if col == "final_nav" else f"{x:.2f}")

    print("\n[Threshold Comparison]")
    print(display_df.to_string(index=False))

    if args.show_dates:
        print_rebalance_schedule("Monthly", monthly_dates, args.date_rows)
        print_rebalance_schedule(f"Trigger {args.threshold:.0%}", trigger_dates, args.date_rows)


if __name__ == "__main__":
    main()
