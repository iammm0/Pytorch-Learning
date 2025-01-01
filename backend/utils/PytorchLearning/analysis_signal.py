import numpy as np
from scipy.signal import find_peaks


def analyze_signal(signal, sampling_rate):
    # 振幅计算
    amplitude = np.max(np.abs(signal))

    # 频率和周期计算
    peaks, _ = find_peaks(signal)
    peak_intervals = np.diff(peaks) / sampling_rate  # 峰值间隔
    if len(peak_intervals) > 0:
        frequency = 1 / np.mean(peak_intervals)  # 平均频率
        period = 1 / frequency  # 周期
    else:
        frequency = 0
        period = 0

    # 相位计算
    fft_result = np.fft.fft(signal)
    phase = np.angle(fft_result[1])  # 第一频率分量的相位

    return amplitude, frequency, period, phase

