import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FACTOR_PATH = BASE_DIR / "4팩터.csv"
OUTPUT_PATH = BASE_DIR / "kospi_monthly_2010.xlsx"
EXPORT_BASE_DATE = pd.Timestamp("2009-12-31")


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


def col_letter(n):
    result = []
    while n:
        n, rem = divmod(n - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result))


def write_simple_xlsx(df, output_path, sheet_name="Sheet1"):
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


def main():
    factors = pd.read_csv(FACTOR_PATH, index_col="Date", parse_dates=True)
    for col in factors.columns:
        factors[col] = pd.to_numeric(factors[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")

    kospi_monthly = factors["KOSPI"].resample("ME").last().loc[EXPORT_BASE_DATE:].dropna()
    out = pd.DataFrame(index=kospi_monthly.index)
    out.index.name = "Date"
    out["KOSPI"] = kospi_monthly
    out["Monthly_Return"] = out["KOSPI"].pct_change()
    out = out.reset_index()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    write_simple_xlsx(out, OUTPUT_PATH, sheet_name="KOSPI")
    print(f"{OUTPUT_PATH.name} 저장 완료")


if __name__ == "__main__":
    main()
