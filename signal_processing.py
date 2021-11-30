from numpy.core.fromnumeric import shape
from scipy.signal import butter, lfilter, welch, find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift, rfft
import numpy as np
from scipy.integrate import quad

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

def find_start_and_end(arr, low, high):
    start = 0
    stop = 0
    for i in range(len(arr)):
        if arr[i] > low and start == 0:
            start = i
        if arr[i] > high and stop == 0:
            stop = i
    return start, stop


def do_stuff(dataset, fs, channels):
    #75 samples per 3 seconds
    uus_df = pd.DataFrame()
    samples = int(3/(1/fs))
    start = 0
    stop = samples
    freq = int(samples/2)
    freq_a = np.linspace(0, fs, samples)
    for channel in channels:
        #freq_axis = np.linspace(0, fs, samples)
        for i in range((int(dataset[channel].size/samples)+1)):

            if stop - 2 > dataset[channel].size:
                df = dataset[channel][start:]
            else:
                df = dataset[channel][start:stop]
            start = stop
            stop += samples
            print(df)
            uus_df = fft(df[channel].to_numpy())
            uus_PSD = np.square(abs(uus_df))
            """ peaks, _ = find_peaks(uus_PSD[channel][:freq])
            plt.figure()
            plt.plot(freq_a[:freq],uus_PSD[channel][:freq])
            plt.plot(freq_a[peaks], uus_PSD[channel][:freq][peaks], "x")
            plt.plot(freq_a[:freq_bin], uus_PSD[channel][:freq], label='X Power') """
        start = 0
        stop = samples
        
        
    


    

# Band pass filter cut off frequencies
lowcut = 3.0
highcut = 12.0

# Read raw accelerometer data
df = pd.read_csv('data/accelerometer_data.csv', header=None)
df.columns = ['time', 'x_axis', 'y_axis', 'z_axis']

# Change timestamps to seconds
df['time'] -= df['time'][0] 

fs = 1 / (df['time'][2] - df['time'][1])
#T = (df['time'].iloc[-1] - df['time'][0])
samples = df['time'].size
#print(fs)
T = 1 / fs
#print(T)
#samples = int(T * fs * 150)
#print(samples)

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
#y_dc = butter_bandpass_filter(dc_df['x_axis_dc'].head(150), lowcut, highcut, fs, order=9)
y_dc = butter_bandpass_filter(dc_df['x_axis_dc'], lowcut, highcut, fs, order=9)
y_dc_y = butter_bandpass_filter(dc_df['y_axis_dc'], lowcut, highcut, fs, order=9)
y_dc_z = butter_bandpass_filter(dc_df['z_axis_dc'], lowcut, highcut, fs, order=9)


plt.plot(t, y_dc, label='Filtered signal X ( Hz) DC')
plt.plot(t, y_dc_y, label='Filtered signal Y ( Hz) DC')
plt.plot(t, y_dc_z, label='Filtered signal Z ( Hz) DC')

plt.legend()

freq_bin = int(samples / 2 )

filtered_df = butter_bandpass_filter_all(dc_df, lowcut, highcut, fs, channels=['x_axis_dc','y_axis_dc','z_axis_dc'], order=9)
magnitude_df = magnitude_spectrum(filtered_df, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])
#magnitude_df_normalized = magnitude_spectrum_normalized(filtered_df, samples, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])
fft_df = fft_signal(filtered_df, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])

#PSD = fft_df * np.conj(fft_df) / samples
#PSD = np.square(magnitude_df)
PSD = np.square(abs(fft_df))


low, high = find_start_and_end(freq_axis, 3 , 5)


plt.figure(2)
plt.plot(freq_axis[:freq_bin], magnitude_df['x_axis_dc_mag'][:freq_bin], label='X Fourier')
plt.plot(freq_axis[:freq_bin], magnitude_df['y_axis_dc_mag'][:freq_bin], label='Y Fourier')
plt.plot(freq_axis[:freq_bin], magnitude_df['z_axis_dc_mag'][:freq_bin], label='Z Fourier')
plt.ylim(bottom=0)
#plt.plot(freq_axis[:freq_bin], magnitude_df_normalized['x_axis_dc_mag'][:freq_bin], label='Fourier')

plt.legend()

plt.figure(3)
plt.plot(freq_axis[:freq_bin], PSD['x_axis_dc_fft'][:freq_bin], label='X Power')
plt.plot(freq_axis[:freq_bin], PSD['y_axis_dc_fft'][:freq_bin], label='Y Power')
plt.plot(freq_axis[:freq_bin], PSD['z_axis_dc_fft'][:freq_bin], label='Z Power')

plt.ylim(bottom=0)
#plt.plot(freq_axis[:freq_bin], PSD['x_axis_dc_mag'][:freq_bin], label='Power')
plt.legend()

# PSD limit 350
plt.figure(4)

peaks, _ = find_peaks(PSD['x_axis_dc_fft'][:freq_bin], height=350)
print(peaks)
if peaks.size > 0:
    print('asd')
plt.plot(freq_axis[:freq_bin],PSD['x_axis_dc_fft'][:freq_bin], label='X')
plt.plot(freq_axis[peaks], PSD['x_axis_dc_fft'][:freq_bin][peaks], "x")
print(freq_axis[peaks])

peaks, _ = find_peaks(PSD['y_axis_dc_fft'][:freq_bin], height=350)
print(peaks)
if peaks.size > 0:
    print('asd')
plt.plot(freq_axis[:freq_bin],PSD['y_axis_dc_fft'][:freq_bin], label='Y')
plt.plot(freq_axis[peaks], PSD['y_axis_dc_fft'][:freq_bin][peaks], "x")
print(freq_axis[peaks])

peaks, _ = find_peaks(PSD['z_axis_dc_fft'][:freq_bin], height=350)
print(peaks)
if peaks.size > 0:
    print('asd')
plt.plot(freq_axis[:freq_bin],PSD['z_axis_dc_fft'][:freq_bin], label='Z')
plt.plot(freq_axis[peaks], PSD['z_axis_dc_fft'][:freq_bin][peaks], "x")
print(freq_axis[peaks])
plt.legend()
#do_stuff(filtered_df, fs, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])

plt.show()