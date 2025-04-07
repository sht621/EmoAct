import cv2
import socket
import pickle
import struct
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from datetime import datetime
import csv
from torchvision.models.video import R3D_18_Weights
import threading
import queue
import ecg_analysis
import time 
from collections import deque

from labels import KINETICS_ID_TO_LABEL

# モデル読み込み
yolo_model = YOLO('yolov8n.pt').to('cuda')
model = r3d_18(weights=R3D_18_Weights.DEFAULT).eval().to('cuda')

# 入力変換
transform = Compose([
    Resize((128, 171)),
    CenterCrop(112),
    ToTensor(),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

frame_buffer = []
data_queue = queue.Queue()

# CSVファイル初期化
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"log_{now}.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Action', 'HR', 'pNN50'])

# データ処理スレッド
def process_data():
    frame_buffer = []
    current_action_label = "Unknown"
    current_hr = "N/A"
    current_pnn50 = "N/A"
    latest_timestamp = "N/A"

    # FPS用変数
    last_frame_time = time.time()
    fps_list = deque(maxlen=16)
    avg_fps = 0.0

    # 前回記録した内容
    last_logged = {
        "timestamp": None,
        "action": None,
        "hr": None,
        "pnn50": None
    }

    while True:
        try:
            message = data_queue.get()
            if message is None:
                break

            # --- 画像復元 ---
            frame_data = np.frombuffer(message["image"], dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if frame is None:
                print("[ERROR] 画像のデコードに失敗しました。")
                continue

            # --- HR / pNN50 / timestamp を常に更新 ---
            latest_timestamp = message.get("timestamp", latest_timestamp)
            current_hr = message.get("HR", current_hr)
            current_pnn50 = message.get("pNN50", current_pnn50)

            # --- YOLO 検出 ---
            results = yolo_model(frame, conf=0.8)
            person_crop = None
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0].cpu().numpy()) == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        person_crop = frame[y1:y2, x1:x2]

            if person_crop is not None:
                frame_buffer.append(person_crop)
                if len(frame_buffer) > 16:
                    frame_buffer.pop(0)

            # フレームごとにFPS記録
            now = time.time()
            frame_fps = 1.0 / (now - last_frame_time)
            last_frame_time = now
            fps_list.append(frame_fps)

            if len(frame_buffer) == 16:

                avg_fps = sum(fps_list) / len(fps_list)

                processed_frames = [
                    transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    for f in frame_buffer
                ]
                video_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0).to('cuda')

                with torch.no_grad():
                    outputs = model(video_tensor)
                    pred = outputs.argmax(1).item()

                current_action_label = KINETICS_ID_TO_LABEL.get(pred, "Unknown")
                print(f"[INFO] 認識結果: {current_action_label}")

                # --- 重複回避：前回と全く同じならスキップ ---
                if (
                    current_action_label != last_logged["action"]
                    or current_hr != last_logged["hr"]
                    or current_pnn50 != last_logged["pnn50"]
                    or latest_timestamp != last_logged["timestamp"]
                ):
                    with open(filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            latest_timestamp, current_action_label,
                            current_hr, current_pnn50
                        ])

                    # ログ済みデータを更新
                    last_logged.update({
                        "timestamp": latest_timestamp,
                        "action": current_action_label,
                        "hr": current_hr,
                        "pnn50": current_pnn50
                    })

                frame_buffer.clear()

            # --- 表示 ---
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Action: {current_action_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, f"HR: {current_hr}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, f"pNN50: {current_pnn50}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {avg_fps:.2f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


            cv2.imshow("Real-time View", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"[ERROR] 処理スレッド例外: {e}")


# スレッド起動
processing_thread = threading.Thread(target=process_data, daemon=True)
processing_thread.start()

# ソケット通信開始
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
server_socket.bind(("0.0.0.0", 9999))
server_socket.listen(5)
print("[INFO] クライアント接続待機中...")
conn, addr = server_socket.accept()
print("[INFO] 接続:", addr)

# データ受信ループ
while True:
    try:
        packed_msg_size = conn.recv(struct.calcsize("Q"))
        if not packed_msg_size:
            print("[ERROR] クライアントが切断されました。")
            break

        msg_size = struct.unpack("Q", packed_msg_size)[0]
        data = b""
        while len(data) < msg_size:
            remaining_bytes = msg_size - len(data)
            chunk_size = 4096 if remaining_bytes > 4096 else remaining_bytes
            packet = conn.recv(chunk_size)
            if not packet:
                print("[ERROR] 受信データが途中で切断されました")
                break
            data += packet

        message = pickle.loads(data)
        data_queue.put(message)

    except Exception as e:
        print(f"[ERROR] 受信スレッド例外: {e}")
        break

# 終了処理
data_queue.put(None)
processing_thread.join()
cv2.destroyAllWindows()
conn.close()
server_socket.close()
print("[INFO] サーバー終了。")
