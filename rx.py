import time
from collections import deque

import numpy as np
import adi
import matplotlib.pyplot as plt

import phone_view

URI = "usb:0.1.5"            # RX board (left port), serial ...1f105137
FS  = 2000000              # sample rate, 2 MSPS
N   = 1024                   # tone length in samples
BIN = 102                    # cycles per N samples -> 102*FS/N = 199.2 kHz

# Status from the 3 s spread of the ch1-ch0 phase, plus the breathing rhythm for the grey zone.
# Thresholds from the two calibrations on 2026-10-18 (empty ~0.0008, still 0.002-0.017, walking 0.008-0.06);
# on those recordings this gets empty 97%, breathing 80%, moving 98% right.
SPREAD_WINDOW = 3.0          # seconds: about one breath
BREATH_THR = 0.002           # above this: someone is there (at least breathing)
MOVE_THR = 0.008             # above this: grey zone, decided by the breathing rhythm
MOVE_SURE_THR = 0.04         # above this: moving, no matter what
RATIO_WINDOW = 20.0          # seconds of history for the breathing rhythm
BREATH_RATIO_THR = 30        # 0.1-0.5 Hz energy vs the band above it; standing still gave 40-190, walking 3-17

# 1. connect, and read something back to prove communication works
sdr = adi.ad9361(uri=URI)
print("connected:", sdr.sample_rate, sdr.rx_lo)

# 2. configure
sdr.sample_rate = FS
sdr.rx_lo = sdr.tx_lo = 2480000000
sdr.rx_rf_bandwidth = sdr.tx_rf_bandwidth = 600000

#Enable both channels
sdr.rx_enabled_channels = [0, 1]
sdr.tx_enabled_channels = [0]   # TX1 on this same board: shared clock, so the phase per channel is usable again
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 40
sdr.tx_hardwaregain_chan0 = -10

sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan1 = 40
sdr.tx_hardwaregain_chan1 = -10


 # At sample rate FS = 2000000 it will take about half a second to fill
 # the buffer for the receiver. Providing enough data to find a human.
sdr.rx_buffer_size = 1048576

# Build the tone and transmit it on repeat
n = np.arange(N)
sdr.tx_cyclic_buffer = True

# Transmit a 199.2 kHz tone (102 turns per 1024 samples); the board loops it forever
sdr.tx(np.exp(2j * np.pi * BIN * n / N) * 2**14)

mixer = np.exp(-2j * np.pi * BIN * n / N)       

#Warm up
for _ in range(4): sdr.rx()


def spread_last(times, phase):
    """Spread of the phase over the last SPREAD_WINDOW seconds, straight line removed so slow drift doesn't count."""
    t = np.array(times)
    m = t >= t[-1] - SPREAD_WINDOW
    if m.sum() < 4:
        return float("nan")
    y = phase[m] - np.polyval(np.polyfit(t[m], phase[m], 1), t[m])
    return float(y.std())


def breathing_rhythm(times, phase):
    """Over the last RATIO_WINDOW seconds: (energy at 0.1-0.5 Hz / energy just above it, breaths per minute)."""
    t = np.array(times)
    m = t >= t[-1] - RATIO_WINDOW
    t, y = t[m], phase[m]
    if t[-1] - t[0] < 0.9 * RATIO_WINDOW:
        return float("nan"), float("nan")
    fs = (len(t) - 1) / (t[-1] - t[0])
    tu = np.arange(t[0], t[-1], 1 / fs)          # buffers don't arrive evenly; resample before the FFT
    yu = np.interp(tu, t, y)
    yu -= np.polyval(np.polyfit(tu, yu, 1), tu)
    spec = np.abs(np.fft.rfft(yu * np.hanning(len(yu)))) ** 2
    f = np.fft.rfftfreq(len(yu), 1 / fs)
    breath = (f >= 0.1) & (f <= 0.5)
    noise = (f >= 0.6) & (f <= 0.9 * fs / 2)
    if not noise.any():
        return float("nan"), float("nan")
    return float(spec[breath].mean() / spec[noise].mean()), float(f[breath][np.argmax(spec[breath])] * 60)


def status_of(spread, ratio):
    if not np.isfinite(spread):
        return "WAIT"
    if spread <= BREATH_THR:
        return "EMPTY"
    if spread > MOVE_SURE_THR:
        return "MOVING"
    if spread > MOVE_THR:
        # Grey zone: close-up breathing and slow walking overlap here; a steady rhythm means breathing
        return "BREATHING" if np.isfinite(ratio) and ratio > BREATH_RATIO_THR else "MOVING"
    return "BREATHING"


COLORS = {"MOVING": "tab:red", "BREATHING": "tab:orange", "EMPTY": "tab:green", "WAIT": "tab:gray"}

# Live plot: interactive mode so the window updates without blocking the loop
# Top: 3 s spread with the two thresholds. Bottom: slow phase wave across buffers (breathing).
plt.ion()
fig, (ax, ax_breath) = plt.subplots(2, 1, sharex=True)
line, = ax.plot([], [])
ax.axhline(BREATH_THR, color=COLORS["BREATHING"], ls="--", lw=1)
ax.axhline(MOVE_THR, color=COLORS["MOVING"], ls=":", lw=1)       # grey zone starts
ax.axhline(MOVE_SURE_THR, color=COLORS["MOVING"], ls="--", lw=1)  # surely moving
ax.set_yscale("log")          # empty ~0.001, walking ~0.02: log keeps both visible
ax.set_ylabel("3 s spread (rad)")
line_breath, = ax_breath.plot([], [])
ax_breath.set_xlabel("time (s)")
ax_breath.set_ylabel("ch1-ch0 phase (rad)")
WINDOW = 30                   # seconds of history shown
times = deque(maxlen=200)     # more than fits in WINDOW; old values drop off automatically
values = deque(maxlen=200)    # 3 s spread per buffer
r_means = deque(maxlen=200)   # one complex ratio per buffer, ~2 per second: enough to see breathing
start = time.time()

# Same data as a web page, so it can be watched from a phone on the WiFi
MARKS = [{"v": BREATH_THR, "color": "#ea8c0c"}, {"v": MOVE_THR, "color": "#f2a0a0"}, {"v": MOVE_SURE_THR, "color": "#dc2626"}]
latest = {"t": [], "spread": [], "phase": [], "status": "WAIT", "bpm": None, "marks": MARKS}
phone_view.start(lambda: latest)

try:
    while True:
        #Receive
        b0, b1 = sdr.rx()
        chunks0 = b0.reshape(1024, 1024)
        chunks1 = b1.reshape(1024, 1024)

        # Same tone spinning backwards, size 1: multiplying by it removes the spin.
        # Then get the average of the complex numbers
        h0 = (chunks0 * mixer).mean(axis=1)
        h1 = (chunks1 * mixer).mean(axis=1)
                
        # amplitude0 = np.abs(h0).mean() # The size of the tone. It should be fairly steady.
        # amplitude1 = np.abs(h1).mean() # The size of the tone. It should be fairly steady.

        # # unwrap: np.angle jumps between +π and -π, which would fake a huge phase change
        # phase0 = np.unwrap(np.angle(h0)) # Our beloved phase that is used to detect humans
        # phase1 = np.unwrap(np.angle(h1)) # Our beloved phase that is used to detect humans

        # phase_spread0 = phase0.std() # The spread of the phase
        # phase_spread1 = phase1.std() # The spread of the phase

        r = h1 / h0                                    # amplitude AND phase of ch1 relative to ch0
        presence = np.abs(r - r.mean()).std() / np.abs(r.mean())

        # The original single-board measure: each channel's phase against the board's own clock
        spread0 = np.unwrap(np.angle(h0)).std()
        spread1 = np.unwrap(np.angle(h1)).std()




        now = time.time() - start
        times.append(now)
        r_means.append(r.mean())

        # unwrap across buffers so a slow drift past ±π doesn't show as a jump
        breath = np.unwrap(np.angle(np.array(r_means)))
        breath -= breath.mean()

        spread = spread_last(times, breath)
        values.append(spread)
        ratio, bpm = breathing_rhythm(times, breath)
        # Median of the last 3 spreads so one noisy buffer doesn't flip the status
        status = status_of(np.median(list(values)[-3:]), ratio)
        show_bpm = status == "BREATHING" and np.isfinite(ratio) and ratio > BREATH_RATIO_THR
        label = f"{status} {bpm:.0f}/min" if show_bpm else status

        print(f"{label:15} | 3 s spread: {spread:.4f} | breath ratio: {ratio:6.1f} | "
              f"old phase ch0: {spread0:.4f} ch1: {spread1:.4f} | "
              f"Amp ch0: {np.abs(h0).mean():.1f} | Amp ch1: {np.abs(h1).mean():.1f}")

        # Update the live plot: new data on the same line, rescale, redraw
        line.set_data(times, values)
        line_breath.set_data(times, breath)
        ax.set_title(label, color=COLORS[status], fontweight="bold")

        # Swap in a fresh dict so the web thread never sees a half-updated one
        latest = {"t": list(times), "spread": [v if np.isfinite(v) else None for v in values],
                  "phase": breath.tolist(), "status": status, "bpm": round(bpm) if show_bpm else None,
                  "marks": MARKS}

        for a in (ax, ax_breath):
            a.relim()
            a.autoscale_view()
        ax.set_xlim(max(0, now - WINDOW), max(WINDOW, now))   # scroll: always show the last 30 s (shared x)
        plt.pause(0.01)

        # Closing the window stops the program
        if not plt.fignum_exists(fig.number):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    sdr.tx_destroy_buffer()   # stop transmitting
    sdr.rx_destroy_buffer()



# stop transmitting (do this before plotting, the plot window blocks)

# spectrum
# spec = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x))) + 1e-9)
# freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS)) / 1e3
# plt.plot(freq, spec)
# plt.xlabel("frequency (kHz)")
# plt.ylabel("dB")
# plt.show()
