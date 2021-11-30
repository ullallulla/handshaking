import smbus
import time
import sys
import threading
from scipy.signal import butter, lfilter, find_peaks
import pandas as pd
import numpy as np
from scipy.fft import fft
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS





def start_system(bus):
    bus.write_byte_data(0x53, 0x2C, 0x0B)
    value = bus.read_byte_data(0x53, 0x31)
    value &= ~0x0F
    value |= 0x0B
    value |= 0x08
    bus.write_byte_data(0x53, 0x31, value)
    bus.write_byte_data(0x53, 0x2D, 0x08)


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

def getAxes(bus):
    bytes = bus.read_i2c_block_data(0x53, 0x32, 6)
        
    x = bytes[0] | (bytes[1] << 8)
    if(x & (1 << 16 - 1)):
        x = x - (1<<16)

    y = bytes[2] | (bytes[3] << 8)
    if(y & (1 << 16 - 1)):
        y = y - (1<<16)

    z = bytes[4] | (bytes[5] << 8)
    if(z & (1 << 16 - 1)):
        z = z - (1<<16)

    x = x * 0.004 
    y = y * 0.004
    z = z * 0.004
    
    timestamp = time.time()

    return x,y,z, timestamp


def processing(data):
    # Band pass filter cut off frequencies
    lowcut = 3.0
    highcut = 12.0
    tremor = []
    
    # Read raw accelerometer data
    df = pd.DataFrame(data, columns = ['time', 'x_axis', 'y_axis', 'z_axis'])
    
    # Change timestamps to seconds
    df['time'] -= df['time'][0] 

    fs = 1 / (df['time'][2] - df['time'][1])
    samples = df['time'].size
    print(f"{fs}=")
    
    T = 1 / fs
    t = np.linspace(0, df['time'][samples-1], samples)
    freq_axis = np.linspace(0, fs, samples)
    
    # Remove DC offset from signal
    dc_df = remove_dc(df, channels=['x_axis', 'y_axis', 'z_axis'])
    freq_bin = int(samples / 2 )
    filtered_df = butter_bandpass_filter_all(dc_df, lowcut, highcut, fs, channels=['x_axis_dc','y_axis_dc','z_axis_dc'], order=9)
    fft_df = fft_signal(filtered_df, channels=['x_axis_dc','y_axis_dc','z_axis_dc'])
    PSD = np.square(abs(fft_df))
    peaks, _ = find_peaks(PSD['x_axis_dc_fft'][:freq_bin], height=350)
    if peaks.size > 0:
        print(f"{peaks}")
        tremor.append(freq_axis[peaks])
        for peak in peaks:
            print(f"timestamp = {data[peak][0]}")
    peaks, _ = find_peaks(PSD['y_axis_dc_fft'][:freq_bin], height=350)
    if peaks.size > 0:
        print(f"{peaks}")
        tremor.append(freq_axis[peaks])
        for peak in peaks:
            print(f"timestamp = {data[peak][0]}")
    peaks, _ = find_peaks(PSD['z_axis_dc_fft'][:freq_bin], height=350)
    if peaks.size > 0:
        print(f"{peaks}")
        for peak in peaks:
            print(f"timestamp = {data[peak][0]}")
        tremor.append(freq_axis[peaks])
    print(tremor)
    
def send_data_to_cloud(dataset, token, org, bucket):
    with InfluxDBClient(url="https://westeurope-1.azure.cloud2.influxdata.com", token=token, org=org) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        payload = []
        for row in dataset:
            
            #change timestamp to unix nanoseconds
            timestamp = int(row[0]*1000000000)
                
            
            data = {
                "measurement": "sensor_data", 
                "tags": {
                    "host": "raspi"
                    }, 
                "time": timestamp,
                "fields": {
                    "x_axis": row[1],
                    "y_axis": row[2],
                    "z_axis": row[3]
                    }
                }
            payload.append(data)
        
        write_api.write(bucket, org, payload)
        client.close()

    
    
def main():
    # You can generate an API token from the "API Tokens Tab" in the UI
    # API TOKEN
    # DB ORG
    # DB BUCKET


    bus = smbus.SMBus(1)
    data = []
    start = 0
    t0 = time.time()
    t1 = time.time()
    try:
        start_system(bus)
        while True: 
            x, y, z, timestamp = getAxes(bus)
            
            
            if time.time()-t0 > 3:
                data_process = []

                data_process = data.copy()
                data_process = data_process[start:]
                start = len(data)

                t0 = time.time()
                data_process_thread = threading.Thread(target=processing, args=(data_process,))
                data_process_thread.start()
                
            if time.time() -t1 > 10:
                cloud_data = data.copy()
                data = []
                cloud_thread = threading.Thread(target=send_data_to_cloud, args=(cloud_data, token, org, bucket))
                cloud_thread.start()
                
                t1 = time.time()
            
            data.append([timestamp,x,y,z])
            time.sleep(0.04)
        
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()