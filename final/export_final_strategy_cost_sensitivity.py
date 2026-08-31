import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "final_strategy_cost_sensitivity.xlsx"

SCORES_PATH = BASE_DIR / "monthly_scores.csv"
PRICES_PATH = BASE_DIR / "수정주가_거래대금필터_code.csv"
TOTAL_RETURN_PATH = BASE_DIR / "현금배당포함.csv"
TV_PATH = BASE_DIR / "거래대금.csv"

ILLIQ_THRESHOLD = 0.80
COST_SCENARIOS = {
    "H0L0": (0.0, 0.0),
    "H80L30": (0.0080, 0.0030),
}


def load_numeric_csv(path, index_col):
    raw = pd.read_csv(path, index_col=index_col, parse_dates=True, low_memory=False)
    return raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", "").str.strip(), errors="coerce"))


def winsorize_monthly_returns(returns_monthly):
    wins = returns_monthly.copy()
    for col in wins.columns:
        s = wins[col].dropna()
        if len(s) > 10:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            wins[col] = wins[col].clip(lo, hi)
    return wins


def col_letter(n):
    result = []
    while n:
        n, rem = divmod(n - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result))


def make_sheet_xml(df):
    rows = []
    header = "".join(
        f'<c r="{col_letter(i + 1)}1" t="inlineStr"><is><t>{escape(str(col))}</t></is></c>'
        for i, col in enumerate(df.columns)
    )
    rows.append(f'<row r="1">{header}</row>')

    for r_idx, row in enumerate(df.itertuples(index=False), start=2):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            cell_ref = f"{col_letter(c_idx)}{r_idx}"
            if pd.isna(value):
                continue
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                text = pd.Timestamp(value).strftime("%Y-%m-%d")
                cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(text)}</t></is></c>')
            elif isinstance(value, str):
                cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                cells.append(f'<c r="{cell_ref}"><v>{float(value)}</v></c>')
        rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData>"
        "</worksheet>"
    )


def write_simple_xlsx(df, output_path, sheet_name="CostSensitivity"):
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/worksheets/sheet1.xml", make_sheet_xml(df))


def load_inputs():
    scores = pd.read_csv(SCORES_PATH, parse_dates=["date"])
    score_tickers = pd.Index(sorted(scores["ticker"].dropna().unique()))

    prices_full = load_numeric_csv(PRICES_PATH, "Date")
    total_adj_raw = pd.read_csv(TOTAL_RETURN_PATH, index_col="코드명", parse_dates=True, low_memory=False)
    name_to_code = dict(zip(total_adj_raw.columns, prices_full.columns))

    prices = prices_full.reindex(columns=prices_full.columns.intersection(score_tickers))
    total_adj = total_adj_raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", "").str.strip(), errors="coerce"))
    tv = load_numeric_csv(TV_PATH, "Date")
    tv = tv.reindex(columns=tv.columns.intersection(score_tickers))
    total_adj.columns = [name_to_code.get(c, c) for c in total_adj.columns]
    total_adj = total_adj.reindex(columns=total_adj.columns.intersection(score_tickers))

    prices_monthly = prices.resample("ME").last()
    total_adj_monthly = total_adj.resample("ME").last()
    returns_monthly = total_adj_monthly.pct_change(fill_method=None)
    returns_monthly = winsorize_monthly_returns(returns_monthly)
    returns_daily = prices.pct_change(fill_method=None)
    amihud_monthly = (returns_daily.abs() / tv).resample("ME").mean()
    spac_set = set(c for c in prices.columns if "스팩" in str(c) or "SPAC" in str(c))

    return scores, prices_monthly, total_adj_monthly, returns_monthly, amihud_monthly, spac_set


def build_top40_inputs(scores, prices_monthly, total_adj_monthly, returns_monthly, amihud_monthly, spac_set):
    score_dates = sorted(scores["date"].unique())
    rows = []

    for start in score_dates:
        next_dates = returns_monthly.index[returns_monthly.index > start]
        if len(next_dates) == 0:
            continue
        end = next_dates[0]

        nr = returns_monthly.loc[end].dropna() if end in returns_monthly.index else pd.Series(dtype=float)
        if len(nr) == 0:
            rows.append({"date": end, "weights": pd.Series(dtype=float), "ret_seg": pd.Series(dtype=float), "illiquid_set": set()})
            continue

        p_s = total_adj_monthly.loc[start] if start in total_adj_monthly.index else None
        p_e = total_adj_monthly.loc[end] if end in total_adj_monthly.index else None
        if p_s is not None and p_e is not None:
            common_idx = nr.index.intersection(p_s.dropna().index).intersection(p_e.dropna().index)
            suspended = p_s.reindex(common_idx)[p_s.reindex(common_idx) == p_e.reindex(common_idx)].index
            nr = nr.drop(suspended, errors="ignore")

        top40_codes = (
            scores.loc[scores["date"] == start]
            .sort_values("composite_score", ascending=False)
            .head(40)["ticker"]
            .tolist()
        )
        top40_filtered = [t for t in top40_codes if t not in spac_set]
        if start in prices_monthly.index:
            pr = prices_monthly.loc[start]
            top40_filtered = [t for t in top40_filtered if t in pr.index and pr[t] >= 1000]
        if len(top40_filtered) < 10:
            top40_filtered = top40_codes

        sel_tickers = pd.Index(top40_filtered).intersection(nr.index)
        weights = pd.Series(1.0 / len(sel_tickers), index=sel_tickers) if len(sel_tickers) > 0 else pd.Series(dtype=float)
        ret_seg = nr.reindex(sel_tickers, fill_value=0) if len(sel_tickers) > 0 else pd.Series(dtype=float)

        amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
        illiquid_set = set(amihud_now[amihud_now >= amihud_now.quantile(ILLIQ_THRESHOLD)].index) if len(amihud_now) > 0 else set()

        rows.append({"date": end, "weights": weights, "ret_seg": ret_seg, "illiquid_set": illiquid_set})

    return rows, score_dates[0] if score_dates else None


def simulate_cost_scenarios(portfolio_rows, start_date):
    results = {}

    for label, (high_cost, low_cost) in COST_SCENARIOS.items():
        nav = 1.0
        prev_pf = pd.Series(dtype=float)
        dates = []
        rets = []

        if start_date is not None:
            dates.append(start_date)
            rets.append(0.0)

        for row in portfolio_rows:
            end = row["date"]
            weights = row["weights"]
            ret_seg = row["ret_seg"]
            illiquid_set = row["illiquid_set"]

            if len(weights) == 0:
                dates.append(end)
                rets.append(0.0)
                continue

            prev_weights = prev_pf / prev_pf.sum() if prev_pf.sum() > 0 else pd.Series(dtype=float)
            all_index = weights.index.union(prev_weights.index)
            target_w = weights.reindex(all_index, fill_value=0)
            prev_w = prev_weights.reindex(all_index, fill_value=0)
            delta_w = target_w - prev_w
            trade_amounts = abs(delta_w) * nav
            cost_rate = np.where(delta_w.index.isin(illiquid_set), high_cost, low_cost)
            trade_cost = (trade_amounts * cost_rate).sum()

            nav_after = nav - trade_cost
            current_pv = weights * nav_after
            next_pv = current_pv * (1 + ret_seg)
            nav_new = next_pv.sum()
            portfolio_ret = nav_new / nav - 1 if nav > 0 else 0.0

            dates.append(end)
            rets.append(portfolio_ret)
            nav = nav_new
            prev_pf = next_pv

        results[label] = pd.Series(rets, index=pd.to_datetime(dates), name=label)

    return results


def main():
    scores, prices_monthly, total_adj_monthly, returns_monthly, amihud_monthly, spac_set = load_inputs()
    portfolio_rows, start_date = build_top40_inputs(
        scores=scores,
        prices_monthly=prices_monthly,
        total_adj_monthly=total_adj_monthly,
        returns_monthly=returns_monthly,
        amihud_monthly=amihud_monthly,
        spac_set=spac_set,
    )
    scenario_returns = simulate_cost_scenarios(portfolio_rows, start_date)

    all_dates = sorted(set().union(*[series.index for series in scenario_returns.values()]))
    out = pd.DataFrame(index=all_dates)
    out.index.name = "Date"
    for label, series in scenario_returns.items():
        out[f"{label}_Monthly_Return"] = series.reindex(out.index).fillna(0.0)
        out[f"{label}_Log_Cum_Return"] = np.log1p(out[f"{label}_Monthly_Return"]).cumsum()

    out = out.reset_index()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    write_simple_xlsx(out, OUTPUT_PATH)
    print(f"{OUTPUT_PATH.name} 저장 완료")


if __name__ == "__main__":
    main()
