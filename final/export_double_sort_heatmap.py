import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(__file__).resolve().parent

SCORES_PATH = BASE_DIR / "monthly_scores.csv"
PRICES_PATH = BASE_DIR / "수정주가_거래대금필터_code.csv"
TOTAL_RETURN_PATH = BASE_DIR / "현금배당포함.csv"
MCAP_PATH = BASE_DIR / "시가총액.csv"
TV_PATH = BASE_DIR / "거래대금.csv"

HEATMAP_XLSX_PATH = BASE_DIR / "double_sort_cagr_heatmap.xlsx"
HEATMAP_HTML_ORIGINAL_PATH = BASE_DIR / "double_sort_cagr_heatmap_original.html"
HEATMAP_PNG_ORIGINAL_PATH = BASE_DIR / "double_sort_cagr_heatmap_original.png"
SHARPE_XLSX_PATH = BASE_DIR / "double_sort_sharpe_heatmap.xlsx"
SHARPE_HTML_PATH = BASE_DIR / "double_sort_sharpe_heatmap.html"
SHARPE_PNG_PATH = BASE_DIR / "double_sort_sharpe_heatmap.png"

ORIGINAL_HEATMAP_COLORS = [
    np.array([247, 251, 255]),
    np.array([107, 174, 214]),
    np.array([8, 81, 156]),
]
SHARPE_HEATMAP_COLORS = [
    np.array([232, 245, 233]),
    np.array([129, 199, 132]),
    np.array([27, 94, 32]),
]

CAP_CUT = 5
FACTOR_CUT = 5
HIGH_COST = 0.008
LOW_COST = 0.003
ILLIQ_THRESHOLD = 0.80


def load_numeric_csv(path, index_col):
    raw = pd.read_csv(path, index_col=index_col, parse_dates=True, low_memory=False)
    return raw.apply(
        lambda x: pd.to_numeric(
            x.astype(str).str.replace(",", "").str.replace('"', "").str.strip(),
            errors="coerce",
        )
    )


def build_code_mapping():
    prices_raw = pd.read_csv(PRICES_PATH, index_col="Date", parse_dates=True, low_memory=False)
    total_raw = pd.read_csv(TOTAL_RETURN_PATH, index_col="코드명", parse_dates=True, low_memory=False)
    return dict(zip(total_raw.columns, prices_raw.columns))


def winsorize_monthly_returns(returns_monthly):
    wins = returns_monthly.copy()
    for col in wins.columns:
        s = wins[col].dropna()
        if len(s) > 10:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            wins[col] = wins[col].clip(lo, hi)
    return wins


def cagr(series):
    s = series.dropna()
    if s.empty:
        return np.nan
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return (1 + s).prod() ** (1 / years) - 1


def sharpe_ratio(series):
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    rfm = (1 + 0.02) ** (1 / 12) - 1
    denom = (s - rfm).std()
    if denom == 0 or pd.isna(denom):
        return np.nan
    return (s - rfm).mean() / denom * np.sqrt(12)


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
            ref = f"{col_letter(c_idx)}{r_idx}"
            if pd.isna(value):
                continue
            if isinstance(value, str):
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{float(value)}</v></c>')
        rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(rows)}</sheetData>"
        "</worksheet>"
    )


def write_simple_xlsx(df, output_path, sheet_name="Heatmap"):
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


def color_for_value(value, vmin, vmax, colors):
    if pd.isna(value):
        return "#ffffff"
    ratio = 0.5 if vmax == vmin else (value - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.5:
        local = ratio / 0.5
        rgb = (colors[0] + local * (colors[1] - colors[0])).astype(int)
    else:
        local = (ratio - 0.5) / 0.5
        rgb = (colors[1] + local * (colors[2] - colors[1])).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def text_color_for_rgb(rgb):
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255.0
    return "#ffffff" if luminance < 0.52 else "#000000"


def rgb_for_value(value, vmin, vmax, colors):
    if pd.isna(value):
        return np.array([255, 255, 255])
    ratio = 0.5 if vmax == vmin else (value - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.5:
        local = ratio / 0.5
        return (colors[0] + local * (colors[1] - colors[0])).astype(int)
    local = (ratio - 0.5) / 0.5
    return (colors[1] + local * (colors[2] - colors[1])).astype(int)


def format_cell(value, value_format):
    if pd.isna(value):
        return ""
    if value_format == "percent":
        return f"{value:.1%}"
    return f"{value:.2f}"


def write_heatmap_html(matrix, output_path, colors, title, value_format):
    vmin = np.nanmin(matrix.values)
    vmax = np.nanmax(matrix.values)
    header_cells = "".join(f"<th>{escape(str(col))}</th>" for col in matrix.columns)
    body_rows = []
    for idx, row in matrix.iterrows():
        cells = [f"<th>{escape(str(idx))}</th>"]
        for value in row:
            rgb = rgb_for_value(value, vmin, vmax, colors)
            bg = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            fg = text_color_for_rgb(rgb)
            text = format_cell(value, value_format)
            cells.append(f'<td style="background:{bg}; color:{fg}; text-align:center; font-weight:600;">{text}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; }}
    h1 {{ font-size: 22px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; }}
    th, td {{ border: 1px solid #cfcfcf; padding: 10px 14px; min-width: 82px; }}
    th {{ background: #f7f7f7; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table>
    <thead>
      <tr><th>Market Cap Quintile (C)</th>{header_cells}</tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_heatmap_png(matrix, output_path, colors, title, value_format):
    cmap = LinearSegmentedColormap.from_list(
        "heatmap_palette",
        [tuple(color / 255.0) for color in colors],
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix.values, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.xaxis.tick_top()
    ax.set_title(title, pad=24)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            text = format_cell(value, value_format)
            rgb = rgb_for_value(value, np.nanmin(matrix.values), np.nanmax(matrix.values), colors)
            ax.text(j, i, text, ha="center", va="center", color=text_color_for_rgb(rgb), fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CAGR")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    code_mapping = build_code_mapping()

    prices = load_numeric_csv(PRICES_PATH, "Date")
    total_adj = load_numeric_csv(TOTAL_RETURN_PATH, "코드명")
    mcap = load_numeric_csv(MCAP_PATH, "코드명")
    tv = load_numeric_csv(TV_PATH, "Date")
    scores = pd.read_csv(SCORES_PATH, parse_dates=["date"])

    total_adj.columns = [code_mapping.get(c, c) for c in total_adj.columns]
    mcap.columns = [code_mapping.get(c, c) for c in mcap.columns]

    total_adj_monthly = total_adj.resample("ME").last()
    mcap_monthly = mcap.resample("ME").last()
    returns_monthly = winsorize_monthly_returns(total_adj_monthly.pct_change(fill_method=None))
    returns_daily = prices.pct_change(fill_method=None)
    amihud_monthly = (returns_daily.abs() / tv).resample("ME").mean()

    score_dates = sorted(scores["date"].unique())
    monthly_dates = returns_monthly.index.sort_values()
    rets = {(c, q): pd.Series(dtype=float) for c in range(1, CAP_CUT + 1) for q in range(1, FACTOR_CUT + 1)}
    navs = {(c, q): 1.0 for c in range(1, CAP_CUT + 1) for q in range(1, FACTOR_CUT + 1)}
    prev_pfs = {(c, q): pd.Series(dtype=float) for c in range(1, CAP_CUT + 1) for q in range(1, FACTOR_CUT + 1)}

    for start in score_dates:
        next_dates = monthly_dates[monthly_dates > start]
        if len(next_dates) == 0:
            continue
        end = next_dates[0]

        nr = returns_monthly.loc[end].dropna() if end in returns_monthly.index else pd.Series(dtype=float)
        if len(nr) == 0:
            for key in rets:
                rets[key].loc[end] = 0.0
            continue

        score_slice = scores[scores["date"] == start].copy()
        mc = mcap_monthly.loc[start].dropna() if start in mcap_monthly.index else pd.Series(dtype=float)
        score_slice["mcap"] = score_slice["ticker"].map(mc)
        score_slice = score_slice.dropna(subset=["mcap"])
        if score_slice.empty:
            for key in rets:
                rets[key].loc[end] = 0.0
            continue

        score_slice["cap_quintile"] = pd.qcut(score_slice["mcap"], q=CAP_CUT, labels=False, duplicates="drop") + 1

        p_s = total_adj_monthly.loc[start] if start in total_adj_monthly.index else None
        p_e = total_adj_monthly.loc[end] if end in total_adj_monthly.index else None
        if p_s is not None and p_e is not None:
            common_idx = nr.index.intersection(p_s.dropna().index).intersection(p_e.dropna().index)
            suspended = p_s.reindex(common_idx)[p_s.reindex(common_idx) == p_e.reindex(common_idx)].index
            nr = nr.drop(suspended, errors="ignore")

        amihud_now = amihud_monthly.loc[start].dropna() if start in amihud_monthly.index else pd.Series(dtype=float)
        illiquid_set = set(amihud_now[amihud_now >= amihud_now.quantile(ILLIQ_THRESHOLD)].index) if len(amihud_now) > 0 else set()

        for c in range(1, CAP_CUT + 1):
            for q in range(1, FACTOR_CUT + 1):
                key = (c, q)
                basket = score_slice.loc[
                    (score_slice["cap_quintile"] == c) & (score_slice["quintile"] == q),
                    "ticker",
                ]
                selected = pd.Index(basket).intersection(nr.index)
                if len(selected) == 0:
                    rets[key].loc[end] = 0.0
                    continue

                weights = pd.Series(1.0 / len(selected), index=selected)
                prev_pf = prev_pfs[key]
                prev_weights = prev_pf / prev_pf.sum() if prev_pf.sum() > 0 else pd.Series(dtype=float)
                all_index = weights.index.union(prev_weights.index)
                target_w = weights.reindex(all_index, fill_value=0)
                prev_w = prev_weights.reindex(all_index, fill_value=0)
                delta_w = target_w - prev_w

                nav = navs[key]
                trade_amounts = abs(delta_w) * nav
                cost_rate = np.where(delta_w.index.isin(illiquid_set), HIGH_COST, LOW_COST)
                trade_cost = (trade_amounts * cost_rate).sum()

                nav_after = nav - trade_cost
                current_pv = weights * nav_after
                ret_seg = nr.reindex(selected, fill_value=0)
                next_pv = current_pv * (1 + ret_seg)
                nav_new = next_pv.sum()
                rets[key].loc[end] = nav_new / nav - 1 if nav > 0 else 0.0
                navs[key] = nav_new
                prev_pfs[key] = next_pv

    cagr_matrix = pd.DataFrame(
        {
            f"Q({q}/5)": [cagr(rets[(c, q)]) for c in range(1, CAP_CUT + 1)]
            for q in range(1, FACTOR_CUT + 1)
        },
        index=[f"C({c}/5)" for c in range(1, CAP_CUT + 1)],
    )
    sharpe_matrix = pd.DataFrame(
        {
            f"Q({q}/5)": [sharpe_ratio(rets[(c, q)]) for c in range(1, CAP_CUT + 1)]
            for q in range(1, FACTOR_CUT + 1)
        },
        index=[f"C({c}/5)" for c in range(1, CAP_CUT + 1)],
    )

    heatmap_df = cagr_matrix.reset_index().rename(columns={"index": "Market Cap Quintile"})
    write_simple_xlsx(heatmap_df, HEATMAP_XLSX_PATH, sheet_name="CAGRHeatmap")
    sharpe_df = sharpe_matrix.reset_index().rename(columns={"index": "Market Cap Quintile"})
    write_simple_xlsx(sharpe_df, SHARPE_XLSX_PATH, sheet_name="SharpeHeatmap")
    write_heatmap_html(cagr_matrix, HEATMAP_HTML_ORIGINAL_PATH, ORIGINAL_HEATMAP_COLORS, "CAGR Heatmap by Cap vs Factor Quintile (Original Color)", "percent")
    write_heatmap_png(cagr_matrix, HEATMAP_PNG_ORIGINAL_PATH, ORIGINAL_HEATMAP_COLORS, "CAGR Heatmap by Cap vs Factor Quintile", "percent")
    write_heatmap_html(sharpe_matrix, SHARPE_HTML_PATH, SHARPE_HEATMAP_COLORS, "Sharpe Ratio Heatmap by Cap vs Factor Quintile", "number")
    write_heatmap_png(sharpe_matrix, SHARPE_PNG_PATH, SHARPE_HEATMAP_COLORS, "Sharpe Ratio Heatmap by Cap vs Factor Quintile", "number")

    print(f"{HEATMAP_XLSX_PATH.name} 저장 완료")
    print(f"{SHARPE_XLSX_PATH.name} 저장 완료")
    print(f"{HEATMAP_HTML_ORIGINAL_PATH.name} 저장 완료")
    print(f"{SHARPE_HTML_PATH.name} 저장 완료")
    print(f"{HEATMAP_PNG_ORIGINAL_PATH.name} 저장 완료")
    print(f"{SHARPE_PNG_PATH.name} 저장 완료")


if __name__ == "__main__":
    main()
