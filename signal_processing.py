from numpy.core.fromnumeric import shape
from scipy.signal import butter, lfilter
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, rfft
import numpy as np

"""
You want to take the signals
Remove DC
Filter the signal with band pass filter
FFT filtered signal
Take magnitudes from the FFT signal by taking absolute value


"""

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_filter_all(data, lowcut, highcut, fs, channels, order=5 ):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = pd.DataFrame()
    for channel in channels:
        y[channel] = lfilter(b, a, data[channel])
    return y

def remove_dc(signal_df, channels):
    dc_free_df = pd.DataFrame()

    for channel in channels:
        dc_free_df[channel + '_dc'] = (signal_df[channel] - signal_df[channel].mean())

    return dc_free_df


def fft_signal(signal_df, channels):
    fft_df = pd.DataFrame()

    for channel in channels:
        fft_df[channel + '_fft'] = fft(signal_df[channel].to_numpy())

    return fft_df

def magnitude_spectrum(signal_df, channels):
    magnitude_df = pd.DataFrame()

    for channel in channels:
        magnitude_df[channel + '_mag'] = abs(fft(signal_df[channel].to_numpy()))

    return magnitude_df

def magnitude_spectrum_normalized(signal_df, samples, channels):
    magnitude_df = pd.DataFrame()

    for channel in channels:
        magnitude_df[channel + '_mag'] = (2/samples)*abs(fft(signal_df[channel].to_numpy()))

    return magnitude_df

# Band pass filter cut off frequencies
lowcut = 3.0
highcut = 12.0

# Read raw accelerometer data
df = pd.read_csv('accelerometer_data.csv', header=None)
df.columns = ['time', 'x_axis', 'y_axis', 'z_axis']

# Change timestamps to seconds
df['time'] -= df['time'][0] 

fs = 1 / (df['time'][2] - df['time'][1])
#T = (df['time'].iloc[-1] - df['time'][0])
samples = df['time'].size
T = 1 / fs
#samples = int(T * fs * 100)

# Filter a noisy signal.
t = np.linspace(0, df['time'][samples-1], samples)
freq_axis = np.linspace(0, fs, samples)
#freq_axis = np.arange(-np.pi+(np.pi/samples), np.pi-(np.pi/samples), 2*np.pi/(samples+1) )
#freq_axis = (freq_axis / (2*np.pi))*fs

plt.figure(1)
plt.clf()

# Remove DC offset from signal
dc_df = remove_dc(df, channels=['x_axis', 'y_axis', 'z_axis'])



# Plot Bandpass filtered DC-free signal
y_dc = butter_bandpass_filter(dc_df['x_axis_dc'], lowcut, highcut, fs, order=9)
#y_dc_y = butter_bandpass_filter(dc_df['y_axis_dc'], lowcut, highcut, fs, order=9)
#y_dc_z = butter_bandpass_filter(dc_df['z_axis_dc'], lowcut, highcut, fs, order=9)


plt.plot(t, y_dc, label='Filtered signal ( Hz) DC')
plt.legend()

freq_bin = int(samples / 2 )

filtered_df = butter_bandpass_filter_all(dc_df, lowcut, highcut, fs, channels=['x_axis_dc','y_axis_dc','z_axis_dc'], order=9)
magnitude_df = magnitude_spectrum(filtered_df, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])
#magnitude_df_normalized = magnitude_spectrum_normalized(filtered_df, samples, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])
fft_df = fft_signal(filtered_df, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])

#PSD = fft_df * np.conj(fft_df) / samples
#PSD = np.square(magnitude_df)
PSD = np.square(abs(fft_df))

print(PSD)

plt.figure(2)
plt.plot(freq_axis[:freq_bin], magnitude_df['x_axis_dc_mag'][:freq_bin], label='Fourier')
#plt.plot(freq_axis[:freq_bin], magnitude_df_normalized['x_axis_dc_mag'][:freq_bin], label='Fourier')

plt.legend()

plt.figure(3)
plt.plot(freq_axis[:freq_bin], PSD['x_axis_dc_fft'][:freq_bin], label='Power')
#plt.plot(freq_axis[:freq_bin], PSD['x_axis_dc_mag'][:freq_bin], label='Power')

plt.legend()

plt.show()