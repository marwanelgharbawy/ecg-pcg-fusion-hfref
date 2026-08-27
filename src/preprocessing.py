import numpy as np
import scipy.signal as signal

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def preprocess_signals(ecg_raw, pcg_raw):
    # 1. Filtering
    # ECG: 0.5 to 40 Hz (removes baseline wander and high-freq noise)
    ecg_filtered = butter_bandpass_filter(ecg_raw, 0.5, 40.0, fs=500)
    
    # PCG: 25 to 400 Hz (focuses on main heart sound frequencies)
    pcg_filtered = butter_bandpass_filter(pcg_raw, 25.0, 400.0, fs=4000)
    
    # 2. Normalization (Z-score standardization)
    ecg_norm = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)
    pcg_norm = (pcg_filtered - np.mean(pcg_filtered)) / np.std(pcg_filtered)
    
    return ecg_norm, pcg_norm

def segment_signals(ecg, pcg, segment_length_sec=5):
    # 5 seconds = 2500 samples (ECG), 20000 samples (PCG)
    ecg_samples = segment_length_sec * 500
    pcg_samples = segment_length_sec * 4000
    
    ecg_segs = [ecg[i:i + ecg_samples] for i in range(0, len(ecg), ecg_samples) 
                if len(ecg[i:i + ecg_samples]) == ecg_samples]
    pcg_segs = [pcg[i:i + pcg_samples] for i in range(0, len(pcg), pcg_samples) 
                if len(pcg[i:i + pcg_samples]) == pcg_samples]
    
    return ecg_segs, pcg_segs