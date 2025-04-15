import asyncio
import cv2
import socket
import pickle
import struct
import time
from datetime import datetime
import neurokit2 as nk
from bleak import BleakClient, BleakScanner
from collections import deque
import numpy as np
import threading
import ecg_analysis
import sys


# --- Polar H10設定 ---
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

rri_buffer = deque(maxlen=30)
rri_lock = threading.Lock()
ble_connected = threading.Event()  # BLE接続成功を知らせるイベント

# --- RRIデータ受信コールバック ---
def handle_rri_data(sender, data):
    global latest_heart_rate  # ← これが必要！

    rr_intervals = []

    flags = data[0]
    index = 1

    hr_format_16bit = flags & 0x01 != 0
    sensor_contact_present = flags & 0x02 != 0
    sensor_contact_supported = flags & 0x04 != 0
    energy_expended_present = flags & 0x08 != 0
    rr_interval_present = flags & 0x10 != 0

    if hr_format_16bit:
        heart_rate = int.from_bytes(data[index:index + 2], byteorder='little')
        index += 2
    else:
        heart_rate = data[index]
        index += 1

    latest_heart_rate = heart_rate

    # print(f"Heart Rate (raw): {heart_rate} bpm")

    if sensor_contact_supported:
        contact_status = "ON" if sensor_contact_present else "OFF"
        print(f"Sensor contact: {contact_status}")

    if energy_expended_present:
        energy_expended = int.from_bytes(data[index:index + 2], byteorder='little')
        index += 2
        print(f"Energy Expended: {energy_expended} kJ")

    if rr_interval_present:
        while index + 1 < len(data):
            rri = int.from_bytes(data[index:index + 2], byteorder='little') / 1024.0
            rri_ms = int(rri * 1000)
            rr_intervals.append(rri_ms)
            index += 2

    with rri_lock:
        rri_buffer.extend(rr_intervals)

# --- BLEクライアント非同期処理 ---
async def run_ble_client():
    print("Polar H10 をスキャン中...")
    devices = await BleakScanner.discover(timeout=5.0)
    polar = None
    for d in devices:
        if d.name and "Polar H10" in d.name:
            polar = d
            break

    if polar is None:
        print("Polar H10 が見つかりませんでした。")
        sys.exit(1)

    try:
        async with BleakClient(polar.address) as client:
            print(f"接続成功: {polar.name} ({polar.address})")
            await client.start_notify(HR_CHAR_UUID, handle_rri_data)
            ble_connected.set()
            print("Polar H10 からRRIデータを受信中...")
            while True:
                await asyncio.sleep(1)
    except Exception as e:
        print(f"[FATAL] BLE接続エラー: {e}")
        sys.exit(1)

# --- BLEスレッド起動 ---
def start_ble_thread():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_ble_client())
    except Exception as e:
        print(f"[FATAL] BLEスレッドエラー: {e}")
        sys.exit(1)

# --- カメラとソケット送信処理 ---
def start_camera_and_socket():
    try:
        UBUNTU_IP = "192.168.65.120"
        PORT = 9999
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((UBUNTU_IP, PORT))
        print("[INFO] Ubuntuに接続しました。")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("カメラが開けませんでした")
            sys.exit(1)

        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームの読み込みに失敗しました")
                break

            cv2.imshow("Camera", frame)

            _, encoded_frame = cv2.imencode('.jpg', frame)

            data = {
                'image': encoded_frame.tobytes()
            }

            frame_counter += 1

            if frame_counter >= 16:
                frame_counter = 0
                with rri_lock:
                    rri_list = list(rri_buffer)

                hr = latest_heart_rate 
                pnn50 = ecg_analysis.calculate_pnn50(rri_list)

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

                data.update({
                    'timestamp': timestamp,
                    'HR': hr,
                    'pNN50': pnn50
                })

            message = pickle.dumps(data)
            message = struct.pack("Q", len(message)) + message
            client_socket.sendall(message)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q'が押されたため、終了します。")
                break

            time.sleep(0.05)

        client_socket.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 通信終了。")

    except Exception as e:
        print(f"[FATAL] カメラ・通信処理でエラーが発生しました: {e}")
        sys.exit(1)

# --- メイン処理 ---
try:
    ble_thread = threading.Thread(target=start_ble_thread, daemon=True)
    ble_thread.start()

    print("[INFO] Polar H10への接続を待機中...")
    ble_connected.wait()
    print("[INFO] Polar H10への接続に成功。カメラ送信を開始します。")

    start_camera_and_socket()

except Exception as e:
    print(f"[FATAL] メイン処理中にエラーが発生しました: {e}")
    sys.exit(1)
