"""Bundle every results_*.csv into a single Excel workbook (results_all.xlsx).

The CSVs remain the source of truth; this workbook is a derived, read-only
convenience. Re-run this script after any experiment driver regenerates its CSV.

Usage (from inside code/):
    python make_results_workbook.py
"""

import glob
import os
import sys

import pandas as pd

OUT = "results_all.xlsx"


def main():
    csvs = sorted(glob.glob("results_*.csv"))
    if not csvs:
        sys.exit("No results_*.csv found. Run this from inside code/.")

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        for path in csvs:
            sheet = os.path.splitext(path)[0].removeprefix("results_")[:31]
            df = pd.read_csv(path)
            df.to_excel(writer, sheet_name=sheet, index=False)

            ws = writer.sheets[sheet]
            ws.freeze_panes = "A2"
            for col_idx, col in enumerate(df.columns, start=1):
                width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200)))
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(width + 2, 60)

    print(f"Wrote {OUT}: {len(csvs)} sheets")
    for path in csvs:
        print(f"  {os.path.splitext(path)[0].removeprefix('results_')}  <-  {path}")


if __name__ == "__main__":
    main()
