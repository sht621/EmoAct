import cv2
import socket
import pickle
import struct
import time
from datetime import datetime
import neurokit2 as nk
from rrd import DataReceiver
from rrd_logger import CSVLogger
from collections import deque
import numpy as np
import threading


# labels.py からラベルを読み込む
from labels import KINETICS_ID_TO_LABEL

# UbuntuのIPアドレス
UBUNTU_IP = "192.168.65.120"
PORT = 9999

# mybeatセンサーデータ受信
receiver = DataReceiver()
logger = CSVLogger(receiver, csv_filename="rrd_log.csv", log_interval=1)

# ソケット作成（TCP）
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((UBUNTU_IP, PORT))
print("[INFO] Ubuntuに接続しました。")

# カメラ設定
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けませんでした")
    exit(1)

# --- ECGバッファと設定 ---
sample_rate = 500  # myBeatのサンプリング周波数に合わせる
ecg_buffer = deque(maxlen=sample_rate * 5)  # 5秒分保持
buffer_lock = threading.Lock()

# --- ECG取得スレッド ---
def ecg_acquisition():
    while True:
        ecg = logger.receiver.getEcgData()
        if ecg is not None:
            with buffer_lock:
                ecg_buffer.append(ecg)
        time.sleep(0.002)  # 約500Hzで取得

ecg_thread = threading.Thread(target=ecg_acquisition, daemon=True)
ecg_thread.start()

frame_counter = 0

receiver.start()  # データ受信開始

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの読み込みに失敗しました")
        continue

    cv2.imshow("Camera", frame)  # 追加：カメラの映像を表示

    # フレームをJPEG圧縮
    _, encoded_frame = cv2.imencode('.jpg', frame)

    # タイムスタンプ取得
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- 画像送信 ---
    data = {
        'image': encoded_frame.tobytes()
    }
    frame_counter += 1

        # --- 16フレームごとにRRIとセンサーデータを付加 ---
    if frame_counter >= 16:
        frame_counter = 0

        with buffer_lock:
            ecg_array = np.array(ecg_buffer)

        # RRI計算（2秒間以上のECGが必要）
        if len(ecg_array) >= sample_rate * 2:
            try:
                ecg_cleaned = nk.ecg_clean(ecg_array, sampling_rate=sample_rate)
                _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sample_rate)
                rri = np.diff(rpeaks['ECG_R_Peaks']) / sample_rate
                avg_rri = round(np.mean(rri) * 1000, 2) if len(rri) > 0 else None
            except Exception as e:
                print(f"[ERROR] RRI計算失敗: {e}")
                avg_rri = None
        else:
            avg_rri = None

        # センサーデータ取得
        temp = logger.receiver.getTempData()
        acc_x = logger.receiver.getAccXData()
        acc_y = logger.receiver.getAccYData()
        acc_z = logger.receiver.getAccZData()

        print(f"[DEBUG] サンプル数: {len(ecg_array)} RRI: {avg_rri}, Temp: {temp}, Acc: ({acc_x}, {acc_y}, {acc_z})")

        # データに追加
        data.update({
            'timestamp': timestamp,
            'rri': avg_rri,
            'temp': temp,
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z
        })

    # --- データ送信 ---
    message = pickle.dumps(data)
    message = struct.pack("Q", len(message)) + message
    client_socket.sendall(message)

    print(timestamp)

    # 'q' を押したらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] 'q'が押されたため、終了します。")
        break

    time.sleep(0.05)  # 適度な遅延

receiver.stop()  # データ受信を停止

cap.release()
cv2.destroyAllWindows()  # 追加：ウィンドウを閉じる
client_socket.close()
print("[INFO] 通信終了。")
