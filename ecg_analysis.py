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
    if len(rri_buffer) < 30:
        return None

    # NumPy配列に変換して差分計算
    rri_array = np.array(rri_buffer)
    rri_diffs = np.abs(np.diff(rri_array))  # 差の絶対値を取ることも重要！

    # 50ms以上の変化がある箇所をカウント
    count_pnn50 = np.sum(rri_diffs > 50)

    # pNN50は「差分の数」に対してではなく、「全RRI数 - 1」に対しての割合
    pnn50 = (count_pnn50 / (len(rri_array) - 1)) * 100

    return round(pnn50, 3)


