"""Guided calibration: the phone tells you where to go, rx data is recorded per step,
and at the end each situation is summarised with suggested detection thresholds.

Run tx.py first, then: python3 calibrate.py  and open the printed address on your phone.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import adi

import phone_view

URI = "usb:0.1.5"            # RX board (left port), serial ...1f105137
FS  = 2000000                # sample rate, 2 MSPS
N   = 1024                   # tone length in samples
BIN = 102                    # cycles per N samples -> 102*FS/N = 199.2 kHz
WIN3 = 3.0                   # seconds per window for the breathing spread

# (key, what to do, seconds to get into position after pressing the button, seconds recorded)
# Key prefixes are the categories: empty_*, still_* / walking_* (same room), other_* (another room, through a wall)
STEP_SETS = {
    # Everything:  python3 calibrate.py
    "full": [
        ("empty",         "Leave. Go behind the metal door and close it.",                                        20, 30),
        ("still_2m",      "Stand completely still about 2 m in front of the receivers. Arms down, breathe normally.", 10, 30),
        ("still_5m",      "Stand completely still about 5 m in front of the receivers. Arms down, breathe normally.", 10, 30),
        ("walking",       "Walk around the room at a normal pace.",                                                5, 20),
        ("walking_slow",  "Walk slowly, small steps.",                                                             5, 20),
        ("other_still",   "Other room: go into the bathroom, close the door. Stand completely still, breathe normally.", 20, 30),
        ("other_moving",  "Other room: stay in the bathroom, move around normally.",                               5, 20),
        ("empty2",        "Leave again. Behind the door, close it.",                                              20, 30),
    ],
    # Only breathing in the other room, standing (sitting anchors the body and moves less):  python3 calibrate.py breath
    "breath": [
        ("other_still",   "Other room: go into the bathroom, close the door. STAND completely still (don't sit), breathe normally.", 20, 30),
    ],
    # Just an empty-apartment reference, to pair with a run that had none:  python3 calibrate.py empty
    "empty": [
        ("empty",         "Leave. Go behind the metal door and close it.",                                        20, 30),
    ],
    # Just the other room:  python3 calibrate.py other
    "other": [
        ("empty",         "Leave. Go behind the metal door and close it.",                                        20, 30),
        ("other_still",   "Other room: go into the bathroom, close the door. Stand completely still, breathe normally.", 20, 30),
        ("other_moving",  "Other room: stay in the bathroom, move around normally.",                               5, 20),
        ("empty2",        "Leave again. Behind the door, close it.",                                              20, 30),
    ],
}
STEPS = STEP_SETS[sys.argv[1] if len(sys.argv) > 1 else "full"]

# --- Radio: same setup as rx.py ---
sdr = adi.ad9361(uri=URI)
sdr.sample_rate = FS
sdr.rx_lo = sdr.tx_lo = 2480000000
sdr.rx_rf_bandwidth = sdr.tx_rf_bandwidth = 600000
sdr.rx_enabled_channels = [0, 1]
sdr.tx_enabled_channels = [0]   # TX1 on this same board, like rx.py
sdr.gain_control_mode_chan0 = sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = sdr.rx_hardwaregain_chan1 = 40
sdr.tx_hardwaregain_chan0 = -10
sdr.rx_buffer_size = 1048576
n = np.arange(N)
sdr.tx_cyclic_buffer = True
sdr.tx(np.exp(2j * np.pi * BIN * n / N) * 2**14)
mixer = np.exp(-2j * np.pi * BIN * n / N)
for _ in range(4): sdr.rx()


def measure():
    """One buffer -> presence (0.5 s spread), mean ratio r, both amplitudes, and the old per-channel
    phase spreads. Same maths as rx.py."""
    b0, b1 = sdr.rx()
    h0 = (b0.reshape(1024, 1024) * mixer).mean(axis=1)
    h1 = (b1.reshape(1024, 1024) * mixer).mean(axis=1)
    r = h1 / h0
    presence = np.abs(r - r.mean()).std() / np.abs(r.mean())
    spread0 = np.unwrap(np.angle(h0)).std()
    spread1 = np.unwrap(np.angle(h1)).std()
    # h0/h1 per buffer: with TX on this board each channel's phase holds steady across buffers,
    # so it can be followed over 3 s directly, without dividing by the other channel
    return presence, r.mean(), np.abs(h0).mean(), np.abs(h1).mean(), spread0, spread1, h0.mean(), h1.mean()


def window_spreads(t, phase):
    """Spread inside 3 s windows, straight line removed so slow drift doesn't count."""
    spreads = []
    for t0 in np.arange(t[0], t[-1] - WIN3, 1.0):
        m = (t >= t0) & (t < t0 + WIN3)
        if m.sum() >= 4:
            y = phase[m] - np.polyval(np.polyfit(t[m], phase[m], 1), t[m])
            spreads.append(y.std())
    return np.array(spreads)


def analyse(rows):
    """Summarise one step's recorded buffers."""
    t = np.array([row["t"] for row in rows])
    p = np.array([row["presence"] for row in rows])
    old = np.array([row["old_spread"] for row in rows])
    phase = np.unwrap(np.angle(np.array([row["r"] for row in rows])))

    spreads = window_spreads(t, phase)
    # "direct 3 s": the same 3 s spread on each channel's own phase, averaged over the two channels
    direct = (window_spreads(t, np.unwrap(np.angle(np.array([row["h0"] for row in rows]))))
              + window_spreads(t, np.unwrap(np.angle(np.array([row["h1"] for row in rows]))))) / 2

    # Breathing: resample to an even grid, compare 0.1-0.5 Hz (breathing) with the band just above (noise)
    fs = (len(t) - 1) / (t[-1] - t[0])
    tu = np.arange(t[0], t[-1], 1 / fs)
    y = np.interp(tu, t, phase)
    y -= np.polyval(np.polyfit(tu, y, 1), tu)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / fs)
    breath = (f >= 0.1) & (f <= 0.5)
    noise = (f >= 0.6) & (f <= 0.9 * fs / 2)
    ratio = spec[breath].mean() / spec[noise].mean() if noise.any() else float("nan")

    return {
        "presence_med": float(np.median(p)),
        "presence_p90": float(np.percentile(p, 90)),
        "old_phase_med": float(np.median(old)),       # original single-board measure, mean of both channels
        "old_phase_p90": float(np.percentile(old, 90)),
        "old_phase_all": old.tolist(),
        "spread3_med": float(np.median(spreads)) if len(spreads) else float("nan"),
        "spread3_p10": float(np.percentile(spreads, 10)) if len(spreads) else float("nan"),
        "spread3_p90": float(np.percentile(spreads, 90)) if len(spreads) else float("nan"),
        "direct3_med": float(np.median(direct)) if len(direct) else float("nan"),
        "direct3_p10": float(np.percentile(direct, 10)) if len(direct) else float("nan"),
        "direct3_p90": float(np.percentile(direct, 90)) if len(direct) else float("nan"),
        "breath_ratio": float(ratio),
        "breath_bpm": float(f[breath][np.argmax(spec[breath])] * 60),
    }


def suggest(summary):
    """Thresholds halfway (geometrically) between empty and the people steps, also as multiples of empty."""
    empty = [s for s in summary if s["key"].startswith("empty")]
    still = [s for s in summary if s["key"].startswith("still")]
    walk = [s for s in summary if s["key"].startswith("walking")]
    out = {}
    if still and walk:
        # Both on the 3 s spread: between the stillest walking and the most restless standing
        st = max(s["spread3_med"] for s in still)
        w = min(s["spread3_med"] for s in walk)
        out["movement_threshold"] = float(np.sqrt(st * w))
        out["movement_x_empty"] = (float(np.sqrt(st * w) / np.median([s["spread3_med"] for s in empty]))
                                   if empty else float("nan"))
        out["movement_separable"] = bool(min(s["spread3_p10"] for s in walk) > max(s["spread3_p90"] for s in still))
    if empty and still:
        e = max(s["spread3_med"] for s in empty)
        b = min(s["spread3_med"] for s in still)
        out["breath_spread_threshold"] = float(np.sqrt(e * b))
        out["breath_spread_x_empty"] = float(np.sqrt(e * b) / e)
        out["breath_separable"] = bool(b > 1.5 * e)
        out["breath_ratio_empty_max"] = max(s["breath_ratio"] for s in empty)
        out["breath_ratio_still_min"] = min(s["breath_ratio"] for s in still)
    # Other room, judged on its own: can each measure tell it apart from empty at all?
    other_still = [s for s in summary if s["key"] == "other_still"]
    if empty and other_still:
        o = other_still[0]
        out["other_room_spread3_separable"] = bool(o["spread3_p10"] > max(s["spread3_p90"] for s in empty))
        out["other_room_old_phase_separable"] = bool(
            np.percentile(o["old_phase_all"], 10) > max(np.percentile(s["old_phase_all"], 90) for s in empty))
        out["other_room_direct3_separable"] = bool(o["direct3_p10"] > max(s["direct3_p90"] for s in empty))
    return out


def add_x_empty(summary):
    """Each step's medians as multiples of the empty room's: the house-independent way to compare measures."""
    empty = [s for s in summary if s["key"].startswith("empty")]
    if not empty:
        return
    e_spread = np.median([s["spread3_med"] for s in empty])
    e_old = np.median([s["old_phase_med"] for s in empty])
    e_direct = np.median([s["direct3_med"] for s in empty])
    for s in summary:
        s["spread3_x_empty"] = float(s["spread3_med"] / e_spread)
        s["old_phase_x_empty"] = float(s["old_phase_med"] / e_old)
        s["direct3_x_empty"] = float(s["direct3_med"] / e_direct)


# --- Shared state between the measuring loop and the web page ---
# stage: "ready" (waiting for the button) -> "settle" (get into position) -> "record" -> next step ... -> "done"
state = {"stage": "ready", "step": 1, "n_steps": len(STEPS), "text": STEPS[0][1], "remaining": 0,
         "settle": STEPS[0][2], "presence": 0.0, "amp0": 0.0, "amp1": 0.0, "summary": [], "suggest": {}, "file": ""}
go = False


def press_go():
    global go
    if state["stage"] == "ready":   # ignore stray taps mid-step, or they would skip the next wait
        go = True


skip = False


def press_skip():
    global skip
    if state["stage"] == "settle":  # "I'm already in position": start recording now
        skip = True


phone_view.start(lambda: state, page=open(os.path.join(os.path.dirname(__file__), "calibrate.html")).read(),
                 actions={"/go": press_go, "/skip": press_skip})
print("Open the address above on your phone. Each step starts when you press its button.")

os.makedirs("calibration", exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join("calibration", f"calib_{stamp}.csv")

try:
    recorded = {key: [] for key, *_ in STEPS}
    t_start = time.time()
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["t", "step", "stage", "presence", "r_re", "r_im", "amp0", "amp1", "old0", "old1",
                         "h0_re", "h0_im", "h1_re", "h1_im"])

        for i, (key, text, settle, record) in enumerate(STEPS):
            # Wait for the button; keep measuring so the signal numbers on the page stay live
            go = False
            state = {**state, "stage": "ready", "step": i + 1, "text": text, "settle": settle, "record": record,
                     "remaining": 0}
            print(f"[{i + 1}/{len(STEPS)} {key}] waiting for the button: {text}")
            while not go:
                presence, r, a0, a1, *_ = measure()
                state = {**state, "presence": float(presence), "amp0": float(a0), "amp1": float(a1)}

            step_start = time.time()
            skip = False
            while (elapsed := time.time() - step_start) < settle + record:
                if skip and elapsed < settle:
                    step_start = time.time() - settle   # jump straight to recording
                    elapsed = settle
                skip = False
                presence, r, a0, a1, s0, s1, m0, m1 = measure()
                now = time.time() - t_start
                stage = "settle" if elapsed < settle else "record"
                remaining = (settle - elapsed) if stage == "settle" else (settle + record - elapsed)
                writer.writerow([f"{now:.3f}", key, stage, f"{presence:.6f}", r.real, r.imag, a0, a1,
                                 f"{s0:.6f}", f"{s1:.6f}", m0.real, m0.imag, m1.real, m1.imag])
                if stage == "record":
                    recorded[key].append({"t": now, "presence": presence, "r": r, "old_spread": (s0 + s1) / 2,
                                          "h0": m0, "h1": m1})
                state = {**state, "stage": stage, "step": i + 1, "text": text, "remaining": max(0, round(remaining)),
                         "presence": float(presence), "amp0": float(a0), "amp1": float(a1)}
                print(f"[{i + 1}/{len(STEPS)} {key} {stage}] {remaining:4.0f} s | presence {presence:.4f}")

    summary = [{"key": key, **analyse(recorded[key])} for key, *_ in STEPS]
    add_x_empty(summary)
    sug = suggest(summary)
    for s in summary:
        del s["old_phase_all"]   # only needed by suggest(); keeps the json and the page small
    with open(csv_path.replace(".csv", "_summary.json"), "w") as fj:
        json.dump({"steps": summary, "suggest": sug}, fj, indent=2)
    state = {**state, "stage": "done", "summary": summary, "suggest": sug, "file": csv_path}

    print(f"\nSaved {csv_path}")
    print(f"{'step':13} {'old phase':>9} {'x empty':>8} {'3s spread':>10} {'x empty':>8} "
          f"{'direct 3s':>10} {'x empty':>8} {'breath ratio':>13} {'bpm':>5}")
    for s in summary:
        print(f"{s['key']:13} {s['old_phase_med']:9.4f} {s.get('old_phase_x_empty', float('nan')):8.1f} "
              f"{s['spread3_med']:10.4f} {s.get('spread3_x_empty', float('nan')):8.1f} "
              f"{s['direct3_med']:10.4f} {s.get('direct3_x_empty', float('nan')):8.1f} "
              f"{s['breath_ratio']:13.1f} {s['breath_bpm']:5.0f}")
    print(json.dumps(sug, indent=2))
    print("\nResults are on the phone page too. Ctrl+C to quit.")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    sdr.tx_destroy_buffer()   # stop transmitting
    sdr.rx_destroy_buffer()
