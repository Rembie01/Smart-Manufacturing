import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_turbine_csv(number, columns="all"):
    # Read the CSV file into a DataFrame
    files = [None, 'Turbine1.csv', 'Turbine2.csv', 'Turbine3.csv', 'Turbine4.csv', 'Turbine5.csv', 'Turbine6.csv']
    if number == 0 or number > 6:
        return None
    if columns == "all":
        columns = 16384
    df = pd.read_csv(files[number], header=None, nrows=columns)

    # Assign column names
    df.columns = ['year', 'speed'] + [f'Sensor_{i-2}' for i in range(2, len(df.columns))]

    return df


def return_sensor_values(df, row):
    values = []
    for col in df.columns[2:]:
        values.append(df[col][row])

    return values


dataframe = read_turbine_csv(4, 50)

values = return_sensor_values(dataframe, 3)


def fft_and_visualize(values):
    samplerate = 12800  # Hz
    sampletime = 1.0 / samplerate
    time = np.arange(0, 1.28, sampletime)
    fft = np.fft.fft(values)
    N = len(fft)
    n = np.arange(N)
    T = N/samplerate
    freq = n/T
    fft_abs = np.abs(fft)

    plt.figure(figsize=(12, 6))
    plt.title()
    plt.subplot(121)
    plt.stem(freq, fft_abs, 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0, samplerate/2)

    plt.subplot(122)
    plt.plot(time, np.fft.ifft(fft), 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    return freq, fft


fft_and_visualize(values)
