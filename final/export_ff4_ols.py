import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from IPython.display import display
except ImportError:
    display = None


BASE_DIR = Path(__file__).resolve().parent
NAV_PATH = BASE_DIR / "q5_top40_nav.xlsx"
FACTOR_PATH = BASE_DIR / "4팩터.csv"

RESULT_CSV_PATH = BASE_DIR / "ff4_ols_results.csv"
RESULT_XLSX_PATH = BASE_DIR / "ff4_ols_results.xlsx"
ALIGNED_CSV_PATH = BASE_DIR / "ff4_ols_aligned_monthly.csv"


def read_simple_xlsx(path):
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in root.find("a:sheetData", ns).findall("a:row", ns):
        vals = {}
        for cell in row.findall("a:c", ns):
            ref = cell.attrib.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                val = "".join(x.text or "" for x in cell.findall(".//a:t", ns))
            elif cell_type == "s":
                v = cell.find("a:v", ns)
                val = shared[int(v.text)] if v is not None else ""
            else:
                v = cell.find("a:v", ns)
                val = v.text if v is not None else ""
            vals[col] = val
        rows.append(vals)

    cols = sorted(rows[0].keys())
    headers = [rows[0][c] for c in cols]
    data = [[r.get(c, "") for c in cols] for r in rows[1:]]
    return pd.DataFrame(data, columns=headers)


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


def write_simple_xlsx(df, output_path, sheet_name="OLS"):
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


def load_factor_data():
    factors_raw = pd.read_csv(FACTOR_PATH, index_col="Date", parse_dates=True)
    for col in factors_raw.columns:
        factors_raw[col] = pd.to_numeric(
            factors_raw[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        )

    factors_monthly = factors_raw.resample("ME").last()

    kospi_factor_ret = factors_monthly["KOSPI"].pct_change() if "KOSPI" in factors_monthly.columns else factors_monthly.iloc[:, 0].pct_change()
    smb_ret = factors_monthly["SMB"].pct_change() if "SMB" in factors_monthly.columns else factors_monthly.iloc[:, 2].pct_change()
    hml_ret = factors_monthly["HML"].pct_change() if "HML" in factors_monthly.columns else factors_monthly.iloc[:, 1].pct_change()
    mom_ret = factors_monthly["MOM"].pct_change() if "MOM" in factors_monthly.columns else factors_monthly.iloc[:, 3].pct_change()
    cd91 = factors_monthly["CD91"] if "CD91" in factors_monthly.columns else factors_monthly.iloc[:, 4]
    rf_monthly = cd91 / 100 / 12
    mkt_factor = kospi_factor_ret - rf_monthly

    return pd.DataFrame(
        {
            "RF": rf_monthly,
            "MKT": mkt_factor,
            "SMB": smb_ret,
            "HML": hml_ret,
            "MOM": mom_ret,
        }
    )


def fit_ff4_model(return_series, factor_df, label):
    valid_dates = []
    for d in return_series.index:
        if d in factor_df.index and not np.isnan(factor_df.loc[d, "MKT"]):
            valid_dates.append(d)

    if len(valid_dates) <= 10:
        return None, None

    y = np.array([return_series.loc[d] for d in valid_dates], dtype=float)
    rf_a = np.array([factor_df.loc[d, "RF"] if d in factor_df.index else 0 for d in valid_dates], dtype=float)
    x_ols = factor_df.loc[valid_dates, ["MKT", "SMB", "HML", "MOM"]].copy()
    x_ols = sm.add_constant(x_ols)
    y_ols = y - rf_a
    model = sm.OLS(y_ols, x_ols).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

    summary_row = {
        "model": label,
        "nobs": int(model.nobs),
        "alpha_ann_pct": model.params["const"] * 12 * 100,
        "alpha_t": model.tvalues["const"],
        "beta_mkt": model.params["MKT"],
        "t_mkt": model.tvalues["MKT"],
        "beta_smb": model.params["SMB"],
        "t_smb": model.tvalues["SMB"],
        "beta_hml": model.params["HML"],
        "t_hml": model.tvalues["HML"],
        "beta_mom": model.params["MOM"],
        "t_mom": model.tvalues["MOM"],
        "r_squared": model.rsquared,
    }

    aligned = pd.DataFrame(
        {
            "Date": valid_dates,
            f"{label}_Return": y,
            "RF": rf_a,
            "MKT": factor_df.loc[valid_dates, "MKT"].values,
            "SMB": factor_df.loc[valid_dates, "SMB"].values,
            "HML": factor_df.loc[valid_dates, "HML"].values,
            "MOM": factor_df.loc[valid_dates, "MOM"].values,
            f"{label}_Excess_Return": y_ols,
        }
    )
    return summary_row, aligned, model


def build_regression_result_table(results_df):
    table = pd.DataFrame(
        {
            "Q5_coef": [np.nan] * 5,
            "Q5_t(HAC)": [np.nan] * 5,
            "Top40_coef": [np.nan] * 5,
            "Top40_t(HAC)": [np.nan] * 5,
        },
        index=["Alpha(ann. %)", "MKT", "SMB", "HML", "MOM"],
    )

    for _, row in results_df.iterrows():
        prefix = "Q5" if row["model"] == "Q5" else "Top40"
        table.loc["Alpha(ann. %)", f"{prefix}_coef"] = row["alpha_ann_pct"]
        table.loc["Alpha(ann. %)", f"{prefix}_t(HAC)"] = row["alpha_t"]
        table.loc["MKT", f"{prefix}_coef"] = row["beta_mkt"]
        table.loc["MKT", f"{prefix}_t(HAC)"] = row["t_mkt"]
        table.loc["SMB", f"{prefix}_coef"] = row["beta_smb"]
        table.loc["SMB", f"{prefix}_t(HAC)"] = row["t_smb"]
        table.loc["HML", f"{prefix}_coef"] = row["beta_hml"]
        table.loc["HML", f"{prefix}_t(HAC)"] = row["t_hml"]
        table.loc["MOM", f"{prefix}_coef"] = row["beta_mom"]
        table.loc["MOM", f"{prefix}_t(HAC)"] = row["t_mom"]

    return table.reset_index().rename(columns={"index": "Term"})


def main():
    nav_df = read_simple_xlsx(NAV_PATH)
    if "Date" not in nav_df.columns:
        raise KeyError(f"Expected 'Date' column, got {list(nav_df.columns)}")
    date_num = pd.to_numeric(nav_df["Date"], errors="coerce")
    if date_num.notna().all():
        nav_df["Date"] = pd.to_datetime(date_num, unit="D", origin="1899-12-30")
    else:
        nav_df["Date"] = pd.to_datetime(nav_df["Date"])
    nav_df["Q5_Monthly_Return"] = pd.to_numeric(nav_df["Q5_Monthly_Return"], errors="coerce")
    nav_df["Top40_Monthly_Return"] = pd.to_numeric(nav_df["Top40_Monthly_Return"], errors="coerce")

    q5_rets = nav_df.set_index("Date")["Q5_Monthly_Return"].dropna()
    top40_rets = nav_df.set_index("Date")["Top40_Monthly_Return"].dropna()

    if len(q5_rets) > 0 and abs(q5_rets.iloc[0]) < 1e-15:
        q5_rets = q5_rets.iloc[1:]
    if len(top40_rets) > 0 and abs(top40_rets.iloc[0]) < 1e-15:
        top40_rets = top40_rets.iloc[1:]

    factor_df = load_factor_data()

    q5_summary, q5_aligned, q5_model = fit_ff4_model(q5_rets, factor_df, "Q5")
    top40_summary, top40_aligned, top40_model = fit_ff4_model(top40_rets, factor_df, "Top40")

    results_df = pd.DataFrame([row for row in [q5_summary, top40_summary] if row is not None])
    regression_result_df = build_regression_result_table(results_df)
    aligned_df = q5_aligned.merge(top40_aligned, on=["Date", "RF", "MKT", "SMB", "HML", "MOM"], how="outer")
    aligned_df = aligned_df.sort_values("Date")
    aligned_df["Date"] = aligned_df["Date"].dt.strftime("%Y-%m-%d")

    results_df.to_csv(RESULT_CSV_PATH, index=False, encoding="utf-8-sig")
    aligned_df.to_csv(ALIGNED_CSV_PATH, index=False, encoding="utf-8-sig")
    write_simple_xlsx(results_df, RESULT_XLSX_PATH, sheet_name="FF4_OLS")

    print(q5_model.summary())
    print()
    print(top40_model.summary())
    print("\nOLS Regression Result")
    print(regression_result_df.to_string(index=False))
    if display is not None:
        display(regression_result_df)
    print(f"\n{RESULT_CSV_PATH.name} 저장 완료")
    print(f"{RESULT_XLSX_PATH.name} 저장 완료")
    print(f"{ALIGNED_CSV_PATH.name} 저장 완료")


if __name__ == "__main__":
    main()
