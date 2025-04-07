import asyncio
import cv2
import socket
import pickle
import struct
import time
from datetime import datetime
import neurokit2 as nk
from bleak import BleakClient
from collections import deque
import numpy as np
import threading
import ecg_analysis

# Polar H10のMACアドレス（実際のアドレスに置き換えてください）
address = "00:22:D0:3B:XX:XX"  # Polar H10 のMACアドレス
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"  # Polar H10の心拍数UUID

# 最新30個のRRIを保持するためのdeque
rri_buffer = deque(maxlen=30)

# Polar H10からRRIデータを受け取る際のコールバック関数
def handle_rri_data(sender, data):
    # データからRRI値（2バイト）を抽出
    rri = int.from_bytes(data[1:3], byteorder='little')  # RRI（単位: ms）
    
    # 最新RRI値をバッファに追加
    rri_buffer.append(rri)
    print(f"新しいRRI: {rri} ms")
    print(f"現在のRRIバッファ: {list(rri_buffer)}")

# Polar H10と接続してRRIデータを受け取る非同期関数
async def run_ble_client():
    async with BleakClient(address) as client:
        print(f"接続成功: {address}")
        
        # Polar H10 からの通知を開始（RRIデータ受信時にhandle_rri_dataが呼ばれる）
        await client.start_notify(HR_CHAR_UUID, handle_rri_data)
        
        # 通知を受け取る間は処理が続く
        while True:
            await asyncio.sleep(1)  # 1秒ごとに非同期処理を実行

# メイン処理の開始
async def main():
    await run_ble_client()

# 非同期処理の実行
asyncio.run(main())

# ソケット設定
UBUNTU_IP = "192.168.65.120"
PORT = 9999
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
        # Polar H10からのRRIデータはhandle_rri_data関数で受け取る
        time.sleep(0.002)  # 約500Hzで取得

ecg_thread = threading.Thread(target=ecg_acquisition, daemon=True)
ecg_thread.start()

frame_counter = 0

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

    # 16フレームごとにRRIとセンサーデータを付加
    if frame_counter >= 16:
        frame_counter = 0

        hr = ecg_analysis.calculate_hr(rri_buffer)
        pnn50 = ecg_analysis.calculate_pnn50(rri_buffer)

        # データに追加
        data.update({
            'timestamp': timestamp,
            'HR': hr,
            'pNN50': pnn50
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

client_socket.close()
cap.release()
cv2.destroyAllWindows()  # 追加：ウィンドウを閉じる
print("[INFO] 通信終了。")
