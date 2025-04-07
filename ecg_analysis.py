import numpy as np
from scipy.signal import welch

def calculate_hr(rri):
    """ RR間隔(ms) から心拍数(HR) を計算 """
    if  rri is None or rri == 0:
        return None
    return round(60000 / rri, 3)

def calculate_pnn50(rr_intervals):
    """ pNN50を計算 (50ms以上変化したRR間隔の割合) """
    if len(rr_intervals) < 2:
        return None
    diff_rr = np.diff(rr_intervals)  # RR間隔の差分
    nn50 = np.sum(np.abs(diff_rr) > 50)  # 50ms以上の変化
    return (nn50 / len(diff_rr)) * 100

def calculate_sdnn(rr_intervals):
    """ RR間隔の標準偏差 (SDNN) を計算 """
    if len(rr_intervals) == 0:
        return None
    return np.std(rr_intervals, ddof=1)

def calculate_rmssd(rr_intervals):
    """ RMSSD (連続するRR間隔の差の2乗平均平方根) を計算 """
    if len(rr_intervals) < 2:
        return None
    diff_rr = np.diff(rr_intervals)  # RR間隔の差分
    return np.sqrt(np.mean(diff_rr ** 2))

def calculate_lf_hf(rr_intervals, fs=4.0):
    """ LF/HF比 (交感神経・副交感神経バランス) を計算 """
    if len(rr_intervals) < 2:
        return None
    
    # RR間隔の時系列データを周波数解析
    rr_time = np.cumsum(rr_intervals) / 1000.0  # 秒単位
    rr_mean = np.mean(rr_intervals)
    interpolated_rr = np.interp(np.arange(0, rr_time[-1], 1/fs), rr_time, rr_intervals - rr_mean)

    # FFT (Welch法) でパワースペクトル密度 (PSD) を計算
    freqs, psd = welch(interpolated_rr, fs=fs, nperseg=len(interpolated_rr))

    # LF (0.04–0.15Hz) と HF (0.15–0.40Hz) のパワーを計算
    lf_power = np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)])
    hf_power = np.trapz(psd[(freqs >= 0.15) & (freqs < 0.40)])

    if hf_power == 0:
        return None  # ゼロ除算を防ぐ

    return lf_power / hf_power
