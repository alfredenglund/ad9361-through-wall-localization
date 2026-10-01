import time
from collections import deque

import numpy as np
import adi
import matplotlib.pyplot as plt

URI = "usb:5.2.5"            # replace with what iio_info -s shows
FS  = 2000000              # sample rate, 2 MSPS
N   = 1024                   # tone length in samples
BIN = 102                    # cycles per N samples -> 102*FS/N = 199.2 kHz

# 1. connect, and read something back to prove communication works
sdr = adi.ad9361(uri=URI)
print("connected:", sdr.sample_rate, sdr.rx_lo)

# 2. configure
sdr.sample_rate = FS
sdr.rx_lo = sdr.tx_lo = 2480000000
sdr.rx_rf_bandwidth = sdr.tx_rf_bandwidth = 600000
sdr.rx_enabled_channels = [0]
sdr.tx_enabled_channels = [0]
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = 40
sdr.tx_hardwaregain_chan0 = -10

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

# Live plot: interactive mode so the window updates without blocking the loop
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_yscale("log")          # still ~0.002, moving ~0.2: log keeps both visible
ax.set_xlabel("time (s)")
ax.set_ylabel("phase spread (rad)")
WINDOW = 30                   # seconds of history shown
times = deque(maxlen=200)     # more than fits in WINDOW; old values drop off automatically
values = deque(maxlen=200)
start = time.time()

try:
    while True:
        #Receive
        buffer = sdr.rx()
        chunks = buffer.reshape(1024, 1024)

        # Same tone spinning backwards, size 1: multiplying by it removes the spin.
        # Then get the average of the complex numbers
        h = (chunks * mixer).mean(axis=1)
        
        amplitude = np.abs(h).mean() # The size of the tone. It should be fairly steady.
        
        # unwrap: np.angle jumps between +π and -π, which would fake a huge phase change
        phase = np.unwrap(np.angle(h)) # Our beloved phase that is used to detect humans

        phase_spread = phase.std() # The spread of the phase

        print(f"Phase: {phase_spread:.4f} | Amplitude: {(amplitude):.1f}")

        # Update the live plot: new data on the same line, rescale, redraw
        now = time.time() - start
        times.append(now)
        values.append(phase_spread)
        line.set_data(times, values)
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(max(0, now - WINDOW), max(WINDOW, now))   # scroll: always show the last 30 s
        plt.pause(0.01)

        # Closing the window stops the program
        if not plt.fignum_exists(fig.number):
            break

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    sdr.tx_destroy_buffer()



# stop transmitting (do this before plotting, the plot window blocks)

# spectrum
# spec = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x))) + 1e-9)
# freq = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS)) / 1e3
# plt.plot(freq, spec)
# plt.xlabel("frequency (kHz)")
# plt.ylabel("dB")
# plt.show()
