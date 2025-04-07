import numpy as np
from scipy.signal import welch

def calculate_hr(rri_buffer):

    if len(rri_buffer) < 10:  # 10個のRRIがない場合は計算しない
        return None
    
    # 最新10個のRRIを取り出す
    last_10_rri = list(rri_buffer)[-10:]
    # 最新10個のRRIの平均を計算
    avg_rri = np.mean(last_10_rri)
    # 平均RRIからHRを計算
    hr = 60 / (avg_rri / 1000)  

    return round(hr, 3)


def calculate_pnn50(rri_buffer):
    
    if len(rri_buffer) < 30 :
        return None
    
    # RRI間の差を計算
    rri_diffs = np.diff(rri_buffer)
    # 50ms以上の差を持つペアをカウント
    count_pnn50 = np.sum(rri_diffs >= 50)
    # pNN50を計算
    pnn50 = (count_pnn50 / len(rri_buffer) * 100)

    return pnn50

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
