import numpy as np
import matplotlib.pyplot as plt


def turbine_fft(values):
    samplerate = 12800  # Hz
    fft = np.fft.fft(values)
    N = len(fft)
    n = np.arange(N)
    T = N/samplerate
    freq = n/T
    fft_abs = np.abs(fft)

    return freq, fft_abs


def visualize_freq(freq, fft):
    plt.figure(figsize=(12, 6))
    plt.stem(freq, fft, 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 6400)
    plt.show()


def visualize_order(freq, fft):
    plt.figure(figsize=(12, 6))
    plt.stem(freq, fft, 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Order')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, 800)
    plt.show()
