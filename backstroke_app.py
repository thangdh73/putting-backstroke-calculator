#!/usr/bin/env python
# putt_backstroke_and_audio.py  –  v2 with “Repeat count” option
# ------------------------------------------------------------------
#  • Predict backstroke length from workbook-trained model
#  • Generate SoundTempo-style tone
#  • Repeat the tone N times before playing / downloading
# ------------------------------------------------------------------
#!/usr/bin/env python
# putt_backstroke_and_audio.py  –  backstroke model + repeatable audio tone
# ------------------------------------------------------------------------

from __future__ import annotations
import io, pathlib, re, shutil, wave, base64
from math import sqrt, sin, pi, exp
from typing import List

import joblib, numpy as np, pandas as pd, streamlit as st
from pydub import AudioSegment
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
EXCEL_PATH = DATA_DIR / "Extracted_Backstroke_Table.xlsx"
MODEL_PATH = BASE_DIR / "backstroke_model.joblib"
FRAME_PATH = BASE_DIR / "backstroke_data.parquet"

FT_TO_M = 0.3048
M_TO_FT = 3.280839895

# ── Sound-tone generator ────────────────────────────────────────────────
class ProfessionalPuttGenerator:
    def __init__(self):
        self.sample_rate = 44100
        self.impact_chirp_duration = 0.05
        self.backswing_beep_duration = 0.05
        self.max_backswing_in = 24.0
        self.max_velocity = 2.2
        self.min_velocity = 0.5

    # ---------- physics helpers -----------------------------------------
    def _velocity(self, dist_ft, stimp, slope_pc):
        d_m = dist_ft * FT_TO_M
        v = 0.36 * stimp * (1 - exp(-d_m / 9.5))
        return float(np.clip(v * (1 + slope_pc * 0.003),
                             self.min_velocity, self.max_velocity))

    def _swing(self, bpm, ratio, dist_ft, stimp, slope_pc):
        dsi = 30 / bpm
        back_t = dsi * ratio
        base = 6.5  # in at 10 ft
        back_len = min(base * min(0.8 * sqrt(dist_ft / 10), 3.5),
                       self.max_backswing_in)
        return {
            "dsi_time": dsi,
            "backswing_time": back_t,
            "backswing_length_in": back_len,
            "required_velocity": self._velocity(dist_ft, stimp, slope_pc),
        }

    # ---------- DSP primitives ------------------------------------------
    def _exp_chirp(self):
        t = np.linspace(0, self.impact_chirp_duration,
                        int(self.sample_rate * self.impact_chirp_duration),
                        False)
        chirp = np.sin(2 * pi * (1200 + 3000 * t /
                                 self.impact_chirp_duration) * t)
        return chirp * np.linspace(1, 0, len(t))

    def _sweep(self, dur, f0, f1):
        N = int(self.sample_rate * dur)
        t = np.linspace(0, dur, N, False)
        tone = np.sin(2 * pi * (f0 + (f1 - f0) * t / dur) * t)
        a, s = int(0.15 * N), int(0.7 * N)
        r = max(1, N - a - s)
        env = np.concatenate([np.linspace(0, 1, a),
                              np.ones(s),
                              np.linspace(1, 0, r)])[:N]
        return tone * env

    def generate(self, bpm, ratio, dist_ft, stimp, slope_pc, handed="right"):
        p = self._swing(bpm, ratio, dist_ft, stimp, slope_pc)
        back = self._sweep(p["backswing_time"], 420, 580)
        down = self._sweep(p["dsi_time"], 580, 420)
        chirp = self._exp_chirp()
        beep  = self._sweep(self.backswing_beep_duration, 1500, 1500)

        # -------- stereopan (fixed lengths) -----------------------------
        if handed == "right":
            pan_back = np.linspace(1, 0, len(back))
            pan_down = np.linspace(0, 1, len(down))
        else:
            pan_back = np.linspace(0, 1, len(back))
            pan_down = np.linspace(1, 0, len(down))

        L = np.concatenate([back * pan_back,  down * pan_down])
        R = np.concatenate([back * (1 - pan_back),
                            down * (1 - pan_down)])

        # cues
        bp = int(0.9 * len(back))
        L[bp:bp+len(beep)] += beep * 0.6
        R[bp:bp+len(beep)] += beep * 0.6

        ip = len(back)
        L[ip:ip+len(chirp)] += chirp * 0.8
        R[ip:ip+len(chirp)] += chirp * 0.8

        pk = max(np.abs(L).max(), np.abs(R).max()) or 1
        return L / pk, R / pk, p

    # ---------- MP3/WAV export (FFmpeg optional) ------------------------
    def to_audio_buffer(self, left, right, prefer_mp3=True):
        pcm = np.column_stack([(left * 32767).astype(np.int16),
                               (right * 32767).astype(np.int16)])
        if prefer_mp3 and shutil.which("ffmpeg"):
            seg = AudioSegment(pcm.tobytes(), frame_rate=self.sample_rate,
                               sample_width=2, channels=2)
            buf = io.BytesIO(); seg.export(buf, format="mp3", bitrate="192k")
            buf.seek(0); return buf, "audio/mp3", ".mp3"

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        buf.seek(0); return buf, "audio/wav", ".wav"


# ── data‐model helpers (unchanged logic) ────────────────────────────────────
def _normalise(s: str) -> str: return str(s).replace("_", " ").replace("\u202f", " ").strip()
HEAD_RX = re.compile(r"""stimp\s*(?P<stimp>\d+(?:\.\d+)?)\s*m?[\s_-]*[–\-]?[\s_-]*
                         (?P<dir>(?:up|down)hill)?.*?\(\s*(?P<slope>\d+)""", re.I|re.X)

def _find_header(df):  # find row with "Putt Length (m)"
    for i, v in df.iloc[:, 0].items():
        if _normalise(v).lower() == "putt length (m)": return i
    return None

def _long(block, meta):
    block.columns = block.iloc[0]; block = block.iloc[1:]
    long = block.melt("Putt Length (m)", var_name="Elevation (cm)", value_name="Backstroke (cm)")
    long["Stimp"] = float(meta["stimp"])
    long["Direction"] = "Uphill" if meta.get("dir", "").lower().startswith("up") else "Downhill"
    long["Elevation (cm)"]  = pd.to_numeric(long["Elevation (cm)"],  errors="coerce")
    long["Backstroke (cm)"] = pd.to_numeric(long["Backstroke (cm)"], errors="coerce")
    return long.dropna(subset=["Backstroke (cm)"])

def load_workbook(xlsx):
    wb = pd.ExcelFile(xlsx); out: List[pd.DataFrame] = []; log = st.sidebar.expander("🪵 Parser log", False)
    for sh in wb.sheet_names:
        m = HEAD_RX.match(_normalise(wb.parse(sh, nrows=1, header=None).iloc[0, 0]))
        if not m: log.write(f"⏭️ {sh} – header not recognised"); continue
        df = wb.parse(sh, header=None); hdr = _find_header(df)
        if hdr is None: log.write(f"❓ {sh} – no header row"); continue
        out.append(_long(df.iloc[hdr:], m.groupdict())); log.write(f"✔️ {sh}")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def build_model(df):
    num = ["Putt Length (m)", "Stimp", "Elevation (cm)"]; cat = ["Direction"]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                             ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                               ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    return Pipeline([("pre", pre), ("reg", GradientBoostingRegressor(random_state=42))])

def _build_from_excel(x):
    df = load_workbook(x)
    model = build_model(df).fit(df[["Putt Length (m)", "Stimp", "Direction", "Elevation (cm)"]],
                                df["Backstroke (cm)"])
    FRAME_PATH.write_bytes(df.to_parquet()); joblib.dump(model, MODEL_PATH)
    return df, model

@st.cache_resource(show_spinner=False)
def get_model_and_data():
    if MODEL_PATH.exists() and FRAME_PATH.exists():
        return pd.read_parquet(FRAME_PATH), joblib.load(MODEL_PATH)
    return _build_from_excel(EXCEL_PATH)

# ── UI ─────────────────────────────────────────────────────────────────────
st.set_page_config("Backstroke + Audio Trainer", layout="centered")
st.title("⛳ Backstroke Calculator + SoundTempo Tone")

with st.spinner("Loading …"):
    try:
        df_data, model = get_model_and_data()
    except Exception as e:
        st.error(str(e)); st.stop()

c1, c2 = st.columns(2)
with c1:
    putt_m  = st.number_input("Putt length (m)", 0.5, 20.0, 3.0, 0.1)
    slope   = st.number_input("Slope at ball (%)", 0.0, 5.0, 2.5, 0.1)
    dirn    = st.radio("Slope direction", ("Uphill", "Downhill"), horizontal=True)
with c2:
    stimp_ft = st.number_input("Stimp (ft)", 6.0, 15.0, 10.0, 0.1)
    tempo    = st.number_input("Core Tempo (BPM)", 65, 120, 90)
    ratio    = st.number_input("Backswing Ratio", 1.8, 3.0, 2.1, 0.1)
    hand     = st.radio("Handedness", ("Right", "Left"), horizontal=True)
    repeat_n = st.number_input("Repeat count", 1, 20, 1)

if st.button("Predict & Play"):
    # model prediction
    X = pd.DataFrame([{
        "Putt Length (m)": putt_m,
        "Stimp": stimp_ft * FT_TO_M,
        "Direction": dirn,
        "Elevation (cm)": putt_m * slope,
    }])
    back_cm = float(model.predict(X)[0])

    # audio
    gen = ProfessionalPuttGenerator()
    L, R, swing = gen.generate(tempo, ratio, putt_m * M_TO_FT,
                               stimp_ft, slope, hand.lower())
    if repeat_n > 1:
        L, R = np.tile(L, repeat_n), np.tile(R, repeat_n)

    buf, mime, ext = gen.to_audio_buffer(L, R)
    st.markdown(f"""
    **Predicted backstroke:** `{back_cm:.1f} cm`  

    **Audio timings**  
    • Backswing = {swing['backswing_time']:.3f} s  
    • Downswing = {swing['dsi_time']:.3f} s  
    • Ratio     = {ratio:.2f}  
    • Repeat    = {repeat_n}×
    """)
    st.audio(buf, format=mime)
    st.download_button("Download audio", buf,
                       file_name=f"putt_{putt_m:.1f}m{ext}", mime=mime)

st.caption("Workbook-driven backstroke model • Repeatable SoundTempo training tone")
