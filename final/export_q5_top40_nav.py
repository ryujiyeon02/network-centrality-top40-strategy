import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SCORES_PATH = BASE_DIR / "monthly_scores.csv"
PRICES_PATH = BASE_DIR / "수정주가_거래대금필터_code.csv"
TOTAL_RETURN_PATH = BASE_DIR / "현금배당포함.csv"
TV_PATH = BASE_DIR / "거래대금.csv"
OUTPUT_PATH = BASE_DIR / "q5_top40_nav.xlsx"
EXPORT_BASE_DATE = pd.Timestamp("2009-12-31")
HIGH_COST = 0.008
LOW_COST = 0.003
ILLIQ_THRESHOLD = 0.80


def load_numeric_csv(path, index_col):
    raw = pd.read_csv(path, index_col=index_col, parse_dates=True, low_memory=False)
    return raw.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(",", "").str.strip(), errors="coerce"))


def build_code_mapping():
    prices_raw = pd.read_csv(PRICES_PATH, index_col="Date", parse_dates=True, low_memory=False)
    total_raw = pd.read_csv(TOTAL_RETURN_PATH, index_col="코드명", parse_dates=True, low_memory=False)

    code_cols = prices_raw.columns
    name_cols = total_raw.columns
    return dict(zip(name_cols, code_cols))


def winsorize_monthly_returns(returns_monthly):
    wins = returns_monthly.copy()
    for col in wins.columns:
        s = wins[col].dropna()
        if len(s) > 10:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            wins[col] = wins[col].clip(lo, hi)
    return wins


def compute_strategy_returns(scores, returns_monthly, total_adj_monthly, prices_monthly, amihud_monthly, spac_set, selector):
    rets = {}
    score_dates = sorted(scores["date"].unique())
    if not score_dates:
        return pd.Series(dtype=float)

    rets[EXPORT_BASE_DATE] = 0.0
    nav = 1.0
    prev_pf = pd.Series(dtype=float)

    monthly_dates = returns_monthly.index.sort_values()

    for start in score_dates:
        next_dates = monthly_dates[monthly_dates > start]
        if len(next_dates) == 0:
            continue
        end = next_dates[0]

        score_slice = scores[scores["date"] == start]
        selected = selector(score_slice)
        if len(selected) == 0:
            rets[end] = 0.0
            continue

        nr = returns_monthly.loc[end].dropna() if end in returns_monthly.index else pd.Series(dtype=float)
        if len(nr) == 0:
            rets[end] = 0.0
            continue

        p_s = total_adj_monthly.loc[start] if start in total_adj_monthly.index else None
        p_e = total_adj_monthly.loc[end] if end in total_adj_monthly.index else None
        if p_s is not None and p_e is not None:
            common_idx = nr.index.intersection(p_s.dropna().index).intersection(p_e.dropna().index)
            suspended = p_s.reindex(common_idx)[p_s.reindex(common_idx) == p_e.reindex(common_idx)].index
            nr = nr.drop(suspended, errors="ignore")

        filtered = [t for t in selected if t not in spac_set]
        if start in prices_monthly.index:
            pr = prices_monthly.loc[start]
            filtered = [t for t in filtered if t in pr.index and pr[t] >= 1000]
        if len(filtered) < 10:
            filtered = selected

        selected_idx = pd.Index(filtered).intersection(nr.index)
        if len(selected_idx) == 0:
            rets[end] = 0.0
            continue

        weights = pd.Series(1.0 / len(selected_idx), index=selected_idx)
        prev_weights = prev_pf / prev_pf.sum() if prev_pf.sum() > 0 else pd.Series(dtype=float)
        all_index = weights.index.union(prev_weights.index)
        target_w = weights.reindex(all_index, fill_value=0)
        prev_w = prev_weights.reindex(all_index, fill_value=0)
        delta_w = target_w - prev_w
        trade_amounts = abs(delta_w) * nav

        amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
        illiquid_set = set(amihud_now[amihud_now >= amihud_now.quantile(ILLIQ_THRESHOLD)].index) if len(amihud_now) > 0 else set()
        cost_rate = np.where(delta_w.index.isin(illiquid_set), HIGH_COST, LOW_COST)
        trade_cost = (trade_amounts * cost_rate).sum()

        nav_after = nav - trade_cost
        current_pv = weights * nav_after
        ret_seg = nr.reindex(selected_idx, fill_value=0)
        next_pv = current_pv * (1 + ret_seg)
        nav_new = next_pv.sum()
        rets[end] = nav_new / nav - 1 if nav > 0 else 0.0
        nav = nav_new
        prev_pf = next_pv

    return pd.Series(rets).sort_index()


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

    sheet_data = "".join(rows)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{sheet_data}</sheetData>"
        "</worksheet>"
    )


def col_letter(n):
    result = []
    while n:
        n, rem = divmod(n - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result))


def write_simple_xlsx(df, output_path):
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="NAV" sheetId="1" r:id="rId1"/></sheets>'
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


def main():
    code_mapping = build_code_mapping()

    prices = load_numeric_csv(PRICES_PATH, "Date")
    total_adj = load_numeric_csv(TOTAL_RETURN_PATH, "코드명")
    tv = load_numeric_csv(TV_PATH, "Date")
    total_adj.columns = [code_mapping.get(c, c) for c in total_adj.columns]
    prices_monthly = prices.resample("ME").last()
    total_adj_monthly = total_adj.resample("ME").last()
    returns_monthly = total_adj_monthly.pct_change(fill_method=None)
    returns_monthly = winsorize_monthly_returns(returns_monthly)
    returns_daily = prices.pct_change(fill_method=None)
    amihud_monthly = (returns_daily.abs() / tv).resample("ME").mean()
    spac_set = set(c for c in prices.columns if "스팩" in str(c) or "SPAC" in str(c))

    scores = pd.read_csv(SCORES_PATH, parse_dates=["date"])
    scores["date"] = pd.to_datetime(scores["date"])

    q5_rets = compute_strategy_returns(
        scores=scores,
        returns_monthly=returns_monthly,
        total_adj_monthly=total_adj_monthly,
        prices_monthly=prices_monthly,
        amihud_monthly=amihud_monthly,
        spac_set=spac_set,
        selector=lambda df: df.loc[df["quintile"] == 5, "ticker"].tolist(),
    )
    top40_rets = compute_strategy_returns(
        scores=scores,
        returns_monthly=returns_monthly,
        total_adj_monthly=total_adj_monthly,
        prices_monthly=prices_monthly,
        amihud_monthly=amihud_monthly,
        spac_set=spac_set,
        selector=lambda df: df.sort_values("composite_score", ascending=False).head(40)["ticker"].tolist(),
    )

    out = pd.DataFrame(index=q5_rets.index.union(top40_rets.index).sort_values())
    out.index.name = "Date"
    out["Q5_Monthly_Return"] = q5_rets.reindex(out.index).fillna(0.0)
    out["Q5_NAV"] = (1 + out["Q5_Monthly_Return"]).cumprod()
    out["Top40_Monthly_Return"] = top40_rets.reindex(out.index).fillna(0.0)
    out["Top40_NAV"] = (1 + out["Top40_Monthly_Return"]).cumprod()
    out["Q5_Log_NAV"] = np.log(out["Q5_NAV"].replace(0, np.nan))
    out["Top40_Log_NAV"] = np.log(out["Top40_NAV"].replace(0, np.nan))
    out = out.loc[out.index >= EXPORT_BASE_DATE].copy()
    out = out.reset_index()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    write_simple_xlsx(out, OUTPUT_PATH)
    print(f"{OUTPUT_PATH.name} 저장 완료")


if __name__ == "__main__":
    main()
