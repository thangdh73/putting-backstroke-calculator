from __future__ import annotations

import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.predict import _predict_backstroke  # noqa: E402

FT_TO_M = 0.3048
HEAD_RX = re.compile(
    r"stimp\s*(?P<stimp>\d+(?:\.\d+)?)\s*m?.*?(?P<dir>(?:up|down)hill).*?\((?P<roll>\d+)\s*cm\s*past\)",
    re.I,
)


def _normalise(value: object) -> str:
    return str(value).replace("_", " ").replace("\u202f", " ").strip()


def _load_excel_rows() -> pd.DataFrame:
    workbook = pd.ExcelFile(ROOT / "data" / "Extracted_Backstroke_Table.xlsx")
    stacks = []
    for sheet_name in workbook.sheet_names:
        first_cell = _normalise(workbook.parse(sheet_name, nrows=1, header=None).iloc[0, 0])
        match = HEAD_RX.search(first_cell)
        if not match:
            raise ValueError(f"Cannot parse sheet header: {sheet_name}")
        raw = workbook.parse(sheet_name, header=None)
        header = None
        for index, value in raw.iloc[:, 0].items():
            if _normalise(value).lower() == "putt length (m)":
                header = index
                break
        if header is None:
            raise ValueError(f"Cannot find lookup table header: {sheet_name}")
        block = raw.iloc[header:].copy()
        block.columns = block.iloc[0]
        block = block.iloc[1:]
        long = block.melt("Putt Length (m)", var_name="Elevation (cm)", value_name="Backstroke (cm)")
        long["Stimp"] = float(match["stimp"])
        long["Direction"] = "Uphill" if match["dir"].lower().startswith("up") else "Downhill"
        long["Target Roll (cm)"] = float(match["roll"])
        for column in ["Putt Length (m)", "Elevation (cm)", "Backstroke (cm)"]:
            long[column] = pd.to_numeric(long[column], errors="coerce")
        stacks.append(long.dropna(subset=["Putt Length (m)", "Elevation (cm)", "Backstroke (cm)"]))
    return pd.concat(stacks, ignore_index=True)


def main() -> None:
    df = _load_excel_rows()
    max_error = 0.0
    failures = []

    for row in df.to_dict("records"):
        putt_m = float(row["Putt Length (m)"])
        elevation_cm = float(row["Elevation (cm)"])
        expected = float(row["Backstroke (cm)"])
        stimp_ft = float(row["Stimp"]) / FT_TO_M
        direction = str(row["Direction"])
        target_roll_cm = float(row["Target Roll (cm)"])
        actual, method = _predict_backstroke(putt_m, stimp_ft, direction, target_roll_cm, elevation_cm)
        error = abs(actual - expected)
        max_error = max(max_error, error)
        if error > 1e-9 or method != "exact SQL lookup":
            failures.append((putt_m, stimp_ft, direction, target_roll_cm, elevation_cm, expected, actual, method))

    if failures:
        for failure in failures[:10]:
            print("failure", failure)
        raise SystemExit(f"{len(failures)} lookup failures; max_error={max_error}")

    print(f"validated {len(df)} exact SQL lookups against Excel-derived source; max_error={max_error:.12f}")


if __name__ == "__main__":
    main()
