#!/usr/bin/env python
# putt_backstroke_and_audio.py
# ------------------------------------------------------------------
# One Streamlit app that:
#   • loads / trains the Gradient-Boosting backstroke model
#   • predicts required backstroke length
#   • lets the user pick Core-Tempo (BPM) + Backswing-Ratio
#   • generates a stereo practice tone with those timing values
#
# ------------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------------
from __future__ import annotations
import io, pathlib, re
from math import sqrt, sin, pi, exp
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pydub import AudioSegment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------------
# 1. Paths & constants
# ------------------------------------------------------------------
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
EXCEL_PATH = DATA_DIR / "Extracted_Backstroke_Table.xlsx"

MODEL_PATH = BASE_DIR / "backstroke_model.joblib"
FRAME_PATH = BASE_DIR / "backstroke_data.parquet"

FT_TO_M = 0.3048
M_TO_FT = 3.280839895


# ------------------------------------------------------------------
# 2. ProfessionalPuttGenerator  (exactly as you supplied)
# ------------------------------------------------------------------
class ProfessionalPuttGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05

        self.max_backswing_in = 24.0
        self.max_velocity = 2.2
        self.min_velocity = 0.5
        self.club_length = 0.9

    # ── physics helpers ──────────────────────────────────────────
    @staticmethod
    def _to_m(ft: float) -> float:    # small helper
        return ft * FT_TO_M

    def calculate_pro_velocity(self, distance_ft, stimp, slope_percent):
        distance_m = self._to_m(distance_ft)
        base_speed = 0.36 * stimp * (1 - exp(-distance_m / 9.5))
        slope_effect = 1 + slope_percent * 0.003
        return np.clip(base_speed * slope_effect, self.min_velocity, self.max_velocity)

    def calculate_pro_swing(
        self,
        core_tempo_bpm: float,
        backswing_rhythm: float,
        distance_ft: float,
        stimp: float,
        slope_percent: float,
    ):
        dsi_time       = 30 / core_tempo_bpm
        backswing_time = dsi_time * backswing_rhythm
        velocity       = self.calculate_pro_velocity(distance_ft, stimp, slope_percent)

        base_length   = 6.5  # in  at 10-ft putt
        scaling       = min(0.8 * sqrt(distance_ft / 10), 3.5)
        backswing_in  = min(base_length * scaling, self.max_backswing_in)

        return {
            "dsi_time": dsi_time,
            "backswing_time": backswing_time,
            "backswing_length_in": backswing_in,
            "required_velocity": velocity,
            "is_capped": backswing_in >= self.max_backswing_in,
        }

    # ── simple DSP primitives ────────────────────────────────────
    def _exp_chirp(self):
        t = np.linspace(0, self.impact_chirp_duration,
                        int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2 * pi * (1200 + 3000 * t / self.impact_chirp_duration) * t)
        return chirp * np.linspace(1, 0, len(t))

    def _sweep(self, dur, f0, f1):
        N = int(self.sample_rate * dur)
        t = np.linspace(0, dur, N, False)
        freq = f0 + (f1 - f0) * t / dur
        tone = np.sin(2 * pi * freq * t)

        a, s = int(0.15 * N), int(0.7 * N)
        r = max(1, N - a - s)
        env = np.concatenate([np.linspace(0, 1, a),
                              np.ones(s),
                              np.linspace(1, 0, r)])[:N]
        return tone * env

    def generate_putt_audio(
        self,
        core_tempo_bpm,
        backswing_rhythm,
        distance_ft,
        stimp,
        slope_percent,
        handedness="right",
    ):
        p = self.calculate_pro_swing(
            core_tempo_bpm, backswing_rhythm,
            distance_ft, stimp, slope_percent
        )
        back = self._sweep(p["backswing_time"], 420, 580)
        down = self._sweep(p["dsi_time"],        580, 420)
        chirp = self._exp_chirp()
        beep  = self._sweep(self.backswing_beep_duration, 1500, 1500)

        if handedness == "right":
            pan_back = np.linspace(1, 0, len(back))
            pan_down = np.linspace(0, 1, len(down))
        else:
            pan_back = np.linspace(0, 1, len(back))
            pan_down = np.linspace(1, 0, len(down))

        L = np.concatenate([back * pan_back, down * pan_down])
        R = np.concatenate([back * (1 - pan_back), down * (1 - pan_down)])

        beep_pos = int(0.9 * len(back))
        L[beep_pos:beep_pos+len(beep)] += beep * 0.6
        R[beep_pos:beep_pos+len(beep)] += beep * 0.6

        impact_pos = len(back)
        L[impact_pos:impact_pos+len(chirp)] += chirp * 0.8
        R[impact_pos:impact_pos+len(chirp)] += chirp * 0.8

        peak = max(np.max(np.abs(L)), np.max(np.abs(R))) or 1.0
        return L / peak, R / peak, p

    def to_mp3_buffer(self, left, right):
        stereo = np.column_stack([(left * 32767).astype(np.int16),
                                  (right * 32767).astype(np.int16)])
        seg = AudioSegment(
            stereo.tobytes(),
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=2,
        )
        buf = io.BytesIO()
        seg.export(buf, format="mp3", bitrate="192k")
        return buf


# ------------------------------------------------------------------
# 3. Workbook → tidy DataFrame (unchanged helper functions)
# ------------------------------------------------------------------
def _normalise(s: str) -> str:
    return str(s).replace("_", " ").replace("\u202f", " ").strip()

HEAD_RX = re.compile(
    r"""stimp\s*(?P<stimp>\d+(?:\.\d+)?)\s*m?[\s_-]*[–\-]?[\s_-]*
        (?P<dir>(?:up|down)hill)?.*?\(\s*(?P<slope>\d+)""",
    re.I | re.X,
)

def _find_header_row(df: pd.DataFrame) -> int | None:
    for idx, val in df.iloc[:, 0].items():
        if _normalise(val).lower() == "putt length (m)":
            return idx
    return None

def _long_form(block: pd.DataFrame, meta: dict) -> pd.DataFrame:
    block = block.rename(columns=lambda c: str(c).strip())
    long = block.melt(
        id_vars="Putt Length (m)",
        var_name="Elevation (cm)",
        value_name="Backstroke (cm)",
    )
    long["Stimp"] = float(meta["stimp"])
    long["Direction"] = "Uphill" if meta.get("dir", "").lower().startswith("up") else "Downhill"
    long["Elevation (cm)"] = pd.to_numeric(long["Elevation (cm)"], errors="coerce")
    long["Backstroke (cm)"] = pd.to_numeric(long["Backstroke (cm)"], errors="coerce")
    return long.dropna(subset=["Backstroke (cm)"])

def load_workbook(path: pathlib.Path) -> pd.DataFrame:
    wb = pd.ExcelFile(path)
    out: List[pd.DataFrame] = []
    log = st.sidebar.expander("🪵 Parser log", expanded=False)

    for sh in wb.sheet_names:
        first = _normalise(wb.parse(sh, nrows=1, header=None).iloc[0, 0])
        m = HEAD_RX.match(first)
        if not m:
            log.write(f"⏭️ skipped '{sh}' – header not recognised: {first!r}")
            continue
        df = wb.parse(sh, header=None)
        hdr = _find_header_row(df)
        if hdr is None:
            log.write(f"❓ '{sh}' has no 'Putt Length' row")
            continue

        block = df.iloc[hdr:]
        block.columns = block.iloc[0]
        block = block.iloc[1:]
        try:
            out.append(_long_form(block, m.groupdict()))
            log.write(f"✔️ parsed '{sh}'")
        except Exception as e:
            log.write(f"❗ '{sh}' error {e}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------
# 4. Model helpers (unchanged)
# ------------------------------------------------------------------
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
    return Pipeline([("pre", pre), ("reg", GradientBoostingRegressor(random_state=42))])

def _build_from_excel(path: pathlib.Path):
    if not path.exists():
        raise FileNotFoundError(f"Excel not found: {path}")
    df = load_workbook(path)
    if df.empty:
        raise ValueError("Workbook parsed zero rows.")
    model = build_model(df).fit(
        df[["Putt Length (m)", "Stimp", "Direction", "Elevation (cm)"]],
        df["Backstroke (cm)"],
    )
    FRAME_PATH.write_bytes(df.to_parquet())
    joblib.dump(model, MODEL_PATH)
    return df, model

@st.cache_resource(show_spinner=False)
def get_model_and_data():
    if MODEL_PATH.exists() and FRAME_PATH.exists():
        return pd.read_parquet(FRAME_PATH), joblib.load(MODEL_PATH)
    return _build_from_excel(EXCEL_PATH)


# ------------------------------------------------------------------
# 5. Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Putter Backstroke + Audio Trainer", layout="centered")
st.title("⛳ Putter Backstroke Calculator & Audio Trainer")

with st.spinner("Loading resources…"):
    try:
        data_df, model = get_model_and_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

# ── maintenance sidebar (same logic) ────────────────────────────
with st.sidebar.expander("⚙️ Model maintenance", expanded=False):
    if st.button("Rebuild from bundled Excel"):
        MODEL_PATH.unlink(missing_ok=True); FRAME_PATH.unlink(missing_ok=True)
        get_model_and_data.clear(); st.experimental_rerun()

    up = st.file_uploader("Upload new workbook & retrain", type=("xlsx", "xlsm"))
    if up and st.button("Train on upload"):
        tmp = BASE_DIR / "uploaded_tmp.xlsx"; tmp.write_bytes(up.read())
        try:
            new_df = load_workbook(tmp)
            if new_df.empty:
                st.error("Parser found no rows.")
            else:
                new_mod = build_model(new_df).fit(
                    new_df[["Putt Length (m)", "Stimp", "Direction", "Elevation (cm)"]],
                    new_df["Backstroke (cm)"],
                )
                FRAME_PATH.write_bytes(new_df.to_parquet())
                joblib.dump(new_mod, MODEL_PATH)
                st.success("New model trained. Reloading…")
                get_model_and_data.clear(); st.experimental_rerun()
        finally:
            tmp.unlink(missing_ok=True)

# ── prediction + audio form ─────────────────────────────────────
st.subheader("Make a prediction & generate tone")

colL, colR = st.columns(2)
with colL:
    putt_len_m = st.number_input("Putt length (m)", 0.5, 20.0, 3.0, 0.1)
    slope_pc   = st.slider("Slope at ball position (%)", 0.0, 20.0, 5.0, 0.1)
    direction  = st.radio("Slope direction", ("Uphill", "Downhill"), horizontal=True)

with colR:
    stimp_ft   = st.number_input("Green speed – Stimp (ft)", 6.0, 15.0, 10.0, 0.1)
    tempo_bpm  = st.slider("Core Tempo (BPM)", 65, 120, 90,
                           help="Downswing timing (BPM ↔ 30/BPM s)")
    rhythm     = st.slider("Backswing Ratio", 1.8, 2.4, 2.1, 0.1)
    handedness = st.radio("Handedness", ["Right", "Left"], horizontal=True)

if st.button("Predict & Play Tone", type="primary"):
    # 5.1 model prediction
    stimp_m = stimp_ft * FT_TO_M
    elev_cm = putt_len_m * slope_pc
    X = pd.DataFrame([{
        "Putt Length (m)": putt_len_m,
        "Stimp": stimp_m,
        "Direction": direction,
        "Elevation (cm)": elev_cm,
    }])
    back_cm = float(model.predict(X)[0])

    # 5.2 audio generation
    gen = ProfessionalPuttGenerator()
    distance_ft = putt_len_m * M_TO_FT
    L, R, swing = gen.generate_putt_audio(
        tempo_bpm, rhythm, distance_ft, stimp_ft, slope_pc, handedness.lower()
    )
    mp3_buf = gen.to_mp3_buffer(L, R)

    # 5.3 display results
    st.subheader("Results")
    st.markdown(
        f"""
        **Inputs**

        • Putt length&nbsp;: {putt_len_m:.2f} m  
        • Stimp&nbsp;: {stimp_ft:.2f} ft → {stimp_m:.2f} m  
        • Slope&nbsp;: {slope_pc:.2f} % ({elev_cm:.1f} cm elevation)  
        • Direction&nbsp;: {direction}

        **Predicted backstroke ≈ `{back_cm:.1f} cm`**

        **Audio timing**

        • Backswing time&nbsp;: {swing['backswing_time']:.3f} s  
        • Downswing time&nbsp;: {swing['dsi_time']:.3f} s  
        • Back : Down ratio&nbsp;: {rhythm:.2f}  
        """,
        unsafe_allow_html=True,
    )

    st.audio(mp3_buf, format="audio/mp3")
    st.download_button(
        "Download MP3",
        mp3_buf,
        file_name=f"putt_{putt_len_m:.1f}m_{stimp_ft:.1f}st.mp3",
        mime="audio/mp3",
    )

st.caption("Data-driven backstroke model + SoundTempo-style training tone · Not affiliated with any commercial product")
