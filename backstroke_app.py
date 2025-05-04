#!/usr/bin/env python
# backstroke_app.py – stimp in feet + slope% → elevation(cm)

from __future__ import annotations
import pathlib, re
from typing import List, Tuple

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


##############################################################################
# 0.  Paths & constants
##############################################################################
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
EXCEL_PATH = DATA_DIR / "Extracted_Backstroke_Table.xlsx"

MODEL_PATH = BASE_DIR / "backstroke_model.joblib"
FRAME_PATH = BASE_DIR / "backstroke_data.parquet"

FT_TO_M = 0.3048       # 1 ft  → 0.3048 m
M_TO_FT = 3.280839895  # 1 m   → 3.2808 ft


##############################################################################
# 1.  Workbook → tidy DataFrame   (unchanged)
##############################################################################
def _normalise(s: str) -> str:
    return str(s).replace("_", " ").replace("\u202f", " ").strip()

HEAD_RX = re.compile(
    r"""stimp\s*(?P<stimp>\d+(?:\.\d+)?)      # stimp distance (m, in workbook)
        \s*m?
        [\s_-]*[–\-]?[\s_-]*
        (?P<dir>(?:up|down)hill)?             # direction
        .*?\(\s*(?P<slope>\d+)                # slope in header, ignored here
    """,
    re.I | re.X,
)

def _find_header_row(df: pd.DataFrame) -> int | None:
    for idx, val in df.iloc[:, 0].items():
        if _normalise(val).lower() == "putt length (m)":
            return idx
    return None

def _long_form(block: pd.DataFrame, meta: dict) -> pd.DataFrame:
    block = block.rename(columns=lambda c: str(c).strip())
    numeric_cols = [c for c in block.columns if c != "Putt Length (m)"]
    long = block.melt(
        id_vars="Putt Length (m)", value_vars=numeric_cols,
        var_name="Elevation (cm)", value_name="Backstroke (cm)"
    )
    long["Stimp"]      = float(meta["stimp"])            # meters in workbook
    long["Direction"]  = "Uphill" if meta.get("dir", "").lower().startswith("up") else "Downhill"
    long["Elevation (cm)"]  = pd.to_numeric(long["Elevation (cm)"],  errors="coerce")
    long["Backstroke (cm)"] = pd.to_numeric(long["Backstroke (cm)"], errors="coerce")
    return long.dropna(subset=["Backstroke (cm)"])

def load_workbook(path: pathlib.Path) -> pd.DataFrame:
    wb   = pd.ExcelFile(path)
    out  : List[pd.DataFrame] = []
    log  = st.sidebar.expander("🪵 Parser log", expanded=False)

    for sheet in wb.sheet_names:
        first_cell = _normalise(wb.parse(sheet, nrows=1, header=None).iloc[0, 0])
        m = HEAD_RX.match(first_cell)
        if not m:
            log.write(f"⏭️ Skipped '{sheet}' – header not recognised: {first_cell!r}")
            continue

        df_sheet  = wb.parse(sheet, header=None)
        hdr       = _find_header_row(df_sheet)
        if hdr is None:
            log.write(f"❓ '{sheet}' has no 'Putt Length' row; skipped")
            continue

        data_block         = df_sheet.iloc[hdr:]
        data_block.columns = data_block.iloc[0]
        data_block         = data_block.iloc[1:]

        try:
            out.append(_long_form(data_block, m.groupdict()))
            log.write(f"✔️ Parsed '{sheet}' ({len(data_block)} rows)")
        except Exception as exc:
            log.write(f"❗ Error '{sheet}' – {exc}")

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


##############################################################################
# 2.  Build model helpers  (PURE – no Streamlit calls)
##############################################################################
def build_model(df: pd.DataFrame):
    num = ["Putt Length (m)", "Stimp", "Elevation (cm)"]
    cat = ["Direction"]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([
                 ("imp", SimpleImputer(strategy="most_frequent")),
                 ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ]
    )
    gbr = GradientBoostingRegressor(random_state=42)
    return Pipeline([("pre", pre), ("reg", gbr)])

def _build_from_excel(excel_path: pathlib.Path):
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = load_workbook(excel_path)
    if df.empty:
        raise ValueError("No data parsed from workbook")

    model = build_model(df).fit(
        df[["Putt Length (m)", "Stimp", "Direction", "Elevation (cm)"]],
        df["Backstroke (cm)"],
    )
    FRAME_PATH.write_bytes(df.to_parquet())
    joblib.dump(model, MODEL_PATH)
    return df, model


##############################################################################
# 3.  Cached accessor (PURE)
##############################################################################
@st.cache_resource(show_spinner=False)
def _get_cached():
    if MODEL_PATH.exists() and FRAME_PATH.exists():
        return pd.read_parquet(FRAME_PATH), joblib.load(MODEL_PATH)
    return _build_from_excel(EXCEL_PATH)


##############################################################################
# 4.  Streamlit UI
##############################################################################
st.set_page_config(page_title="Backstroke Length Calculator", layout="centered")
st.title("⛳ Putter Backstroke Calculator")

with st.spinner("Loading model … (first time only)"):
    try:
        df, model = _get_cached()
    except Exception as err:
        st.error(str(err))
        st.stop()

# ───────────────────────── Maintenance sidebar ─────────────────────────────
with st.sidebar.expander("⚙️  Model maintenance", expanded=False):
    if st.button("Rebuild from bundled Excel"):
        MODEL_PATH.unlink(missing_ok=True)
        FRAME_PATH.unlink(missing_ok=True)
        _get_cached.clear()
        st.experimental_rerun()

    new_file = st.file_uploader("Upload new Excel & retrain", type=("xlsx", "xlsm"))
    if new_file and st.button("Train on uploaded file"):
        tmp = BASE_DIR / "uploaded_tmp.xlsx"
        tmp.write_bytes(new_file.read())
        try:
            df_up = load_workbook(tmp)
            if df_up.empty:
                st.error("Could not parse any sheet from the uploaded file.")
            else:
                model_up = build_model(df_up).fit(
                    df_up[["Putt Length (m)", "Stimp", "Direction", "Elevation (cm)"]],
                    df_up["Backstroke (cm)"],
                )
                FRAME_PATH.write_bytes(df_up.to_parquet())
                joblib.dump(model_up, MODEL_PATH)
                st.success("New model trained & cached. Reloading …")
                _get_cached.clear()
                st.experimental_rerun()
        finally:
            tmp.unlink(missing_ok=True)

# ───────────────────────── Prediction form ──────────────────────────────────
st.subheader("Make a prediction")

c1, c2 = st.columns(2)
with c1:
    putt_len = st.number_input("Putt length (m)", 0.5, 20.0, 3.0, 0.1)
    slope_pc = st.slider("Slope at ball position (%)", 0.0, 20.0, 5.0, 0.1)
with c2:
    stimp_ft  = st.number_input("Green speed – Stimp (ft)", 6.0, 15.0, 10.0, 0.1)
    direction = st.radio("Slope direction", ("Uphill", "Downhill"), horizontal=True)

if st.button("Predict"):
    stimp_m  = stimp_ft * FT_TO_M
    elev_cm  = putt_len * slope_pc        # given relation
    Xnew = pd.DataFrame(
        [{
            "Putt Length (m)" : putt_len,
            "Stimp"           : stimp_m,
            "Direction"       : direction,
            "Elevation (cm)"  : elev_cm,
        }]
    )
    y = model.predict(Xnew)[0]
    st.markdown(
        f"""
        **Inputs**

        • Putt length : {putt_len:.2f} m  
        • Stimp       : {stimp_ft:.2f} ft  → {stimp_m:.2f} m  
        • Slope       : {slope_pc:.2f}%  ({elev_cm:.1f} cm elevation)  
        • Direction   : {direction}

        **Required backstroke ≈ `{y:.1f} cm`**
        """,
        unsafe_allow_html=True,
    )
