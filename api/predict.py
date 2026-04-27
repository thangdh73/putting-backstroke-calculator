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


def _bounds(values: list[float], target: float) -> tuple[float, float]:
    if target <= values[0]:
        return values[0], values[0]
    if target >= values[-1]:
        return values[-1], values[-1]
    for lower, upper in zip(values, values[1:]):
        if lower <= target <= upper:
            return lower, upper
    return values[-1], values[-1]


def _lerp(lower: float, upper: float, ratio: float) -> float:
    return lower + (upper - lower) * ratio


def _ratio(lower: float, upper: float, target: float) -> float:
    return 0.0 if lower == upper else (target - lower) / (upper - lower)


def _predict_backstroke(
    putt_m: float,
    stimp_ft: float,
    direction: str,
    target_roll_cm: float,
    elevation_cm: float,
) -> tuple[float, str]:
    stimp_m = stimp_ft * FT_TO_M
    direction = "Uphill" if str(direction).lower().startswith("up") else "Downhill"

    with sqlite3.connect(DB_PATH) as conn:
        exact = conn.execute(
            f'''
            SELECT "Backstroke (cm)"
            FROM "{TABLE_NAME}"
            WHERE "Direction" = ?
              AND ABS("Target Roll (cm)" - ?) < 0.000001
              AND ABS("Putt Length (m)" - ?) < 0.000001
              AND ABS("Stimp" - ?) < 0.000001
              AND ABS("Elevation (cm)" - ?) < 0.000001
            LIMIT 1
            ''',
            (direction, target_roll_cm, putt_m, stimp_m, elevation_cm),
        ).fetchone()
        if exact:
            return float(exact[0]), "exact SQL lookup"

        axis_values = {}
        for column in ("Putt Length (m)", "Stimp", "Elevation (cm)"):
            axis_values[column] = [
                row[0]
                for row in conn.execute(
                    f'SELECT DISTINCT "{column}" FROM "{TABLE_NAME}" '
                    f'WHERE "Direction" = ? AND ABS("Target Roll (cm)" - ?) < 0.000001 '
                    f'ORDER BY "{column}"',
                    (direction, target_roll_cm),
                )
            ]
        if any(not values for values in axis_values.values()):
            raise RuntimeError("No matching SQL rows found for prediction.")

        p0, p1 = _bounds(axis_values["Putt Length (m)"], putt_m)
        s0, s1 = _bounds(axis_values["Stimp"], stimp_m)
        e0, e1 = _bounds(axis_values["Elevation (cm)"], elevation_cm)
        corner_rows = conn.execute(
            f'''
            SELECT
                "Putt Length (m)",
                "Stimp",
                "Elevation (cm)",
                "Backstroke (cm)"
            FROM "{TABLE_NAME}"
            WHERE "Direction" = ?
              AND ABS("Target Roll (cm)" - ?) < 0.000001
              AND "Putt Length (m)" IN (?, ?)
              AND "Stimp" IN (?, ?)
              AND "Elevation (cm)" IN (?, ?)
            ''',
            (direction, target_roll_cm, p0, p1, s0, s1, e0, e1),
        ).fetchall()

        lookup = {(p, s, e): back for p, s, e, back in corner_rows}
        needed = {
            (p, s, e)
            for p in (p0, p1)
            for s in (s0, s1)
            for e in (e0, e1)
        }
        if needed.issubset(lookup):
            pr = _ratio(p0, p1, putt_m)
            sr = _ratio(s0, s1, stimp_m)
            er = _ratio(e0, e1, elevation_cm)
            interpolated_by_stimp = []
            for p in (p0, p1):
                interpolated_by_elevation = []
                for s in (s0, s1):
                    interpolated_by_elevation.append(
                        _lerp(lookup[(p, s, e0)], lookup[(p, s, e1)], er)
                    )
                interpolated_by_stimp.append(
                    _lerp(interpolated_by_elevation[0], interpolated_by_elevation[1], sr)
                )
            return _lerp(interpolated_by_stimp[0], interpolated_by_stimp[1], pr), "SQL interpolation"

        raise RuntimeError("SQL lookup grid is incomplete for these inputs.")


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
    target_roll_cm = _clamp(_float(payload, "target_roll_cm", 30.0), 15.0, 30.0)
    elevation_cm = putt_m * slope_pc
    repeat_count = int(_clamp(_int(payload, "repeat_n", 1), 1, 20))
    back_cm, method = _predict_backstroke(
        putt_m,
        _clamp(_float(payload, "stimp_ft", 10.0), 6.0, 16.0),
        payload.get("direction", "Uphill"),
        target_roll_cm,
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
        "target_roll_cm": round(target_roll_cm, 2),
        "method": method,
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
