from __future__ import annotations

import base64
import io
import json
import math
import pathlib
import sqlite3
import wave
from http.server import BaseHTTPRequestHandler


BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "backstroke_observations.sqlite"
TABLE_NAME = "backstroke_observations"

FT_TO_M = 0.3048
M_TO_FT = 3.280839895
CM_TO_IN = 0.393700787


def _float(payload: dict, key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(payload: dict, key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _predict_backstroke(putt_m: float, stimp_ft: float, direction: str, elevation_cm: float) -> float:
    stimp_m = stimp_ft * FT_TO_M
    direction = "Uphill" if str(direction).lower().startswith("up") else "Downhill"

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f'''
            SELECT
                "Putt Length (m)",
                "Stimp",
                "Direction",
                "Elevation (cm)",
                "Backstroke (cm)"
            FROM "{TABLE_NAME}"
            WHERE "Direction" = ?
            ''',
            (direction,),
        ).fetchall()

    if not rows:
        raise RuntimeError("No matching SQL rows found for prediction.")

    scored = []
    for row_putt_m, row_stimp_m, _, row_elev_cm, row_back_cm in rows:
        distance = (
            ((row_putt_m - putt_m) / 3.0) ** 2
            + ((row_stimp_m - stimp_m) / 0.6) ** 2
            + ((row_elev_cm - elevation_cm) / 15.0) ** 2
        )
        scored.append((distance, row_back_cm))
    scored.sort(key=lambda item: item[0])

    nearest = scored[:8]
    if nearest[0][0] == 0:
        return float(nearest[0][1])

    weighted_total = 0.0
    weight_sum = 0.0
    for distance, back_cm in nearest:
        weight = 1.0 / (distance + 1e-6)
        weighted_total += weight * back_cm
        weight_sum += weight
    return float(weighted_total / weight_sum)


def _swing(bpm: float, ratio: float, dist_ft: float, stimp: float, slope_pc: float) -> dict:
    dsi = 30.0 / bpm
    back_t = dsi * ratio
    d_m = dist_ft * FT_TO_M
    velocity = 0.36 * stimp * (1 - math.exp(-d_m / 9.5))
    return {
        "dsi_time": dsi,
        "backswing_time": back_t,
        "backswing_length_in": min(6.5 * min(0.8 * math.sqrt(dist_ft / 10), 3.5), 24.0),
        "required_velocity": _clamp(velocity * (1 + slope_pc * 0.003), 0.5, 2.2),
    }


def _sweep(sample_rate: int, duration: float, f0: float, f1: float) -> list[float]:
    count = max(1, int(sample_rate * duration))
    samples = []
    for index in range(count):
        t = index / sample_rate
        position = index / max(1, count - 1)
        freq = f0 + (f1 - f0) * position
        tone = math.sin(2 * math.pi * freq * t)
        if position < 0.15:
            env = position / 0.15
        elif position > 0.85:
            env = (1 - position) / 0.15
        else:
            env = 1.0
        samples.append(tone * max(0.0, env))
    return samples


def _chirp(sample_rate: int, duration: float) -> list[float]:
    count = max(1, int(sample_rate * duration))
    samples = []
    for index in range(count):
        t = index / sample_rate
        position = index / max(1, count - 1)
        freq = 1200 + 3000 * position
        samples.append(math.sin(2 * math.pi * freq * t) * (1 - position))
    return samples


def _generate_wav(payload: dict) -> tuple[bytes, dict]:
    sample_rate = 22050
    bpm = _clamp(_float(payload, "tempo", 90), 65, 120)
    ratio = _clamp(_float(payload, "ratio", 2.1), 1.8, 3.0)
    putt_m = _clamp(_float(payload, "putt_m", 3.0), 0.5, 20.0)
    stimp_ft = _clamp(_float(payload, "stimp_ft", 10.0), 6.0, 15.0)
    slope_pc = _clamp(_float(payload, "slope_pc", 2.5), 0.0, 5.0)
    handed = str(payload.get("handedness", payload.get("handed", "right"))).lower()
    repeat_n = _clamp(_int(payload, "repeat_n", 1), 1, 20)

    swing = _swing(bpm, ratio, putt_m * M_TO_FT, stimp_ft, slope_pc)
    back = _sweep(sample_rate, swing["backswing_time"], 420, 580)
    down = _sweep(sample_rate, swing["dsi_time"], 580, 420)
    beep = _sweep(sample_rate, 0.05, 1500, 1500)
    chirp = _chirp(sample_rate, 0.05)

    left = []
    right = []
    for index, sample in enumerate(back):
        pan = index / max(1, len(back) - 1)
        left_pan = 1 - pan if handed == "right" else pan
        left.append(sample * left_pan)
        right.append(sample * (1 - left_pan))
    for index, sample in enumerate(down):
        pan = index / max(1, len(down) - 1)
        left_pan = pan if handed == "right" else 1 - pan
        left.append(sample * left_pan)
        right.append(sample * (1 - left_pan))

    beep_pos = int(0.9 * len(back))
    impact_pos = len(back)
    for offset, sample in enumerate(beep):
        pos = beep_pos + offset
        if pos < len(left):
            left[pos] += sample * 0.6
            right[pos] += sample * 0.6
    for offset, sample in enumerate(chirp):
        pos = impact_pos + offset
        if pos < len(left):
            left[pos] += sample * 0.8
            right[pos] += sample * 0.8

    left *= int(repeat_n)
    right *= int(repeat_n)
    peak = max(max(abs(value) for value in left), max(abs(value) for value in right), 1.0)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for l_sample, r_sample in zip(left, right):
            l_pcm = int(_clamp(l_sample / peak, -1, 1) * 32767)
            r_pcm = int(_clamp(r_sample / peak, -1, 1) * 32767)
            wav.writeframesraw(l_pcm.to_bytes(2, "little", signed=True))
            wav.writeframesraw(r_pcm.to_bytes(2, "little", signed=True))
    return buffer.getvalue(), swing


def predict(payload: dict) -> dict:
    putt_m = _clamp(_float(payload, "putt_m", 3.0), 0.5, 20.0)
    slope_pc = _clamp(_float(payload, "slope_pc", 2.5), 0.0, 5.0)
    elevation_cm = putt_m * slope_pc
    repeat_count = int(_clamp(_int(payload, "repeat_n", 1), 1, 20))
    back_cm = _predict_backstroke(
        putt_m,
        _clamp(_float(payload, "stimp_ft", 10.0), 6.0, 15.0),
        payload.get("direction", "Uphill"),
        elevation_cm,
    )
    wav_bytes, swing = _generate_wav(payload)
    unit = str(payload.get("unit", "cm"))
    back_display = back_cm * CM_TO_IN if unit == "inches" else back_cm

    return {
        "predicted_backstroke": round(back_display, 2),
        "backstroke_cm": round(back_cm, 2),
        "unit_label": "in" if unit == "inches" else "cm",
        "elevation_cm": round(elevation_cm, 2),
        "repeat_count": repeat_count,
        "swing": {
            "backswing_time": round(swing["backswing_time"], 3),
            "dsi_time": round(swing["dsi_time"], 3),
            "ratio": round(_clamp(_float(payload, "ratio", 2.1), 1.8, 3.0), 2),
        },
        "audio": {
            "mime": "audio/wav",
            "filename": f"putt_{putt_m:.1f}m.wav",
            "base64": base64.b64encode(wav_bytes).decode("ascii"),
        },
    }


def _response(payload: dict) -> bytes:
    return json.dumps(predict(payload)).encode("utf-8")


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, b"")

    def do_GET(self) -> None:
        self._send(200, _response({}))

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, _response(payload))
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"))
