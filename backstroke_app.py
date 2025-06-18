#!/usr/bin/env python
# backstroke_app.py  –  Excel-matched with SoundTempo option
# ------------------------------------------------------------------
#  • Predict backstroke length directly from Excel lookup/interpolation
#  • Generate SoundTempo-style tone (repeatable)
# ------------------------------------------------------------------

import io, pathlib, re, shutil, wave
from math import sqrt, sin, pi, exp
import numpy as np
import pandas as pd
import streamlit as st
from pydub import AudioSegment
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

# ── constants ──────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
EXCEL_PATH = DATA_DIR / "Extracted_Backstroke_Table.xlsx"

FT_TO_M  = 0.3048
M_TO_FT  = 3.280839895
CM_TO_IN = 0.393700787

# ── SoundTempo‑style tone generator ─────────
class ProfessionalPuttGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        self.max_backswing_in = 24.0
        self.max_velocity = 2.2
        self.min_velocity = 0.5

    def _velocity(self, dist_ft, stimp, slope_pc):
        d_m = dist_ft * FT_TO_M
        v = 0.36 * stimp * (1 - exp(-d_m / 9.5))
        return float(np.clip(v * (1 + slope_pc * 0.003), self.min_velocity, self.max_velocity))

    def _swing(self, bpm, ratio, dist_ft, stimp, slope_pc):
        dsi = 30 / bpm
        back_t = dsi * ratio
        base = 6.5  # inches at 10‑ft putt
        back_len = min(base * min(0.8 * sqrt(dist_ft / 10), 3.5), self.max_backswing_in)
        return {"dsi_time": dsi, "backswing_time": back_t, "backswing_length_in": back_len,
                "required_velocity": self._velocity(dist_ft, stimp, slope_pc)}

    def _exp_chirp(self):
        t = np.linspace(0, self.impact_chirp_duration,
                        int(self.sample_rate * self.impact_chirp_duration), False)
        chirp = np.sin(2*pi*(1200 + 3000*t/self.impact_chirp_duration)*t)
        return chirp * np.linspace(1, 0, len(t))

    def _sweep(self, dur, f0, f1):
        N = int(self.sample_rate * dur)
        t = np.linspace(0, dur, N, False)
        tone = np.sin(2*pi*(f0+(f1-f0)*t/dur)*t)
        a, s = int(0.15*N), int(0.7*N); r = max(1, N-a-s)
        env = np.concatenate([np.linspace(0,1,a), np.ones(s), np.linspace(1,0,r)])[:N]
        return tone * env

    def generate(self, bpm, ratio, dist_ft, stimp, slope_pc, handed="right"):
        p = self._swing(bpm, ratio, dist_ft, stimp, slope_pc)
        back = self._sweep(p["backswing_time"], 420, 580)
        down = self._sweep(p["dsi_time"], 580, 420)
        chirp = self._exp_chirp()
        beep  = self._sweep(self.backswing_beep_duration, 1500, 1500)

        pan_back = np.linspace(1, 0, len(back)) if handed=="right" else np.linspace(0,1,len(back))
        pan_down = np.linspace(0, 1, len(down)) if handed=="right" else np.linspace(1,0,len(down))

        L = np.concatenate([back*pan_back, down*pan_down])
        R = np.concatenate([back*(1-pan_back), down*(1-pan_down)])

        bp = int(0.9*len(back))
        L[bp:bp+len(beep)] += beep*0.6; R[bp:bp+len(beep)] += beep*0.6
        ip = len(back)
        L[ip:ip+len(chirp)] += chirp*0.8; R[ip:ip+len(chirp)] += chirp*0.8

        pk = max(np.abs(L).max(), np.abs(R).max()) or 1
        return L/pk, R/pk, p

    def to_audio_buffer(self, left, right):
        pcm = np.column_stack([(left*32767).astype(np.int16),
                               (right*32767).astype(np.int16)])
        if shutil.which("ffmpeg"):
            seg = AudioSegment(pcm.tobytes(), frame_rate=self.sample_rate,
                               sample_width=2, channels=2)
            buf = io.BytesIO(); seg.export(buf, format="mp3", bitrate="192k")
            buf.seek(0); return buf, "audio/mp3", ".mp3"
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        buf.seek(0); return buf, "audio/wav", ".wav"

# ╭─────────────────────────────╮
# │ Excel Grid Loader + Helper │
# ╰─────────────────────────────╯
@st.cache_data(show_spinner=False)
def load_excel_grid(xlsx_file):
    df = pd.read_excel(xlsx_file, sheet_name=0, index_col=0)
    def _make_float(val):
        import re
        try: return float(re.sub(r"[^\d.]+", "", str(val)))
        except: return np.nan
    df.columns = [_make_float(c) for c in df.columns]
    df.index = [_make_float(i) for i in df.index]
    df = df.replace("N/A", np.nan)
    return df

def build_interpolator(df):
    xs = df.index.to_numpy()
    ys = df.columns.to_numpy()
    zs = df.astype(float).to_numpy()
    return RegularGridInterpolator((xs, ys), zs, bounds_error=False, fill_value=np.nan)

st.set_page_config(page_title="Backstroke Calculator + Sound", layout="centered")
st.title("⛳ Backstroke Calculator (Excel-matched) + SoundTempo Tone")

side = st.sidebar
side.header("Load Excel Table")
default_path = DATA_DIR / "Extracted_Backstroke_Table.xlsx"
path_str = side.text_input("Path to Excel table:", str(default_path))
uploaded = side.file_uploader("Or upload Excel file (.xlsx)", type=["xlsx"])

if uploaded:
    df = load_excel_grid(uploaded)
elif Path(path_str).exists():
    df = load_excel_grid(path_str)
else:
    df = None

if df is None or df.empty:
    st.warning("No Excel table found. Please upload or specify a valid path.")
    st.stop()

interp = build_interpolator(df)
side.success("Table loaded ✔")
st.caption("Backstroke is a direct lookup/interpolation from the Excel grid. No ML is used.")

# ╭─────────────────────────────╮
# │ User Inputs                │
# ╰─────────────────────────────╯
c1,c2=st.columns(2)
with c1:
    putt_m   = st.number_input("Putt length (m)", float(df.index.min()), float(df.index.max()), float(df.index.min()), 0.1)
    elev_cm  = st.number_input("Slope elevation (cm)", float(df.columns.min()), float(df.columns.max()), float(df.columns.min()), 1.0)
    unit     = st.selectbox("Display backstroke in",("cm","inches"))
with c2:
    stimp_ft = st.number_input("Stimp (ft)",6.0,15.0,10.0,0.1)
    tempo    = st.number_input("Core Tempo (BPM)",65,120,90)
    ratio    = st.number_input("Backswing Ratio",1.8,3.0,2.1,0.1)
    hand     = st.radio("Handedness",("Right","Left"),horizontal=True)
    repeat_n = st.number_input("Repeat count",1,20,1)

input_point = (putt_m, elev_cm)
back_cm = float(interp(input_point))
if np.isnan(back_cm):
    st.error("Input is outside the Excel grid—no value to interpolate.")
else:
    back_display = back_cm if unit=="cm" else back_cm*CM_TO_IN
    unit_label   = "cm" if unit=="cm" else "in"
    is_exact = (putt_m in df.index) and (elev_cm in df.columns)
    msg = "Exact Excel cell match" if is_exact else "Interpolated between cells"
    st.metric(f"Backstroke length ({unit_label}) [{msg}]", f"{back_display:.2f}")

    # ---------- audio ------------------------------------------
    gen=ProfessionalPuttGenerator()
    L,R,swing=gen.generate(tempo,ratio,putt_m*M_TO_FT,stimp_ft,0,hand.lower())
    if repeat_n>1: L,R=np.tile(L,repeat_n),np.tile(R,repeat_n)
    buf,mime,ext = gen.to_audio_buffer(L,R)

    # ---------- output -----------------------------------------
    st.markdown(f"""
    **Backstroke:** `{back_display:.2f} {unit_label}`  
    **Slope elevation used:** `{elev_cm:.2f} cm`  

    **Audio timings**  
    • Backswing = {swing['backswing_time']:.3f} s  
    • Downswing = {swing['dsi_time']:.3f} s  
    • Ratio     = {ratio:.2f}  
    • Repeat    = {repeat_n}×
    """)
    st.audio(buf,format=mime)
    st.download_button("Download audio",buf,
                       file_name=f"putt_{putt_m:.1f}m{ext}",mime=mime)

with st.expander("Show Lookup Table / Preview"):
    st.dataframe(df.style.format("{:.1f}"))

st.caption("If your inputs match Excel cell positions, this will return exactly the Excel value. For in-between values, bilinear interpolation is used.")
