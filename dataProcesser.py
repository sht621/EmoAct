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

from labels import KINETICS_ID_TO_LABEL

yolo_model = YOLO('yolov8n.pt').to('cuda')
model = r3d_18(weights=R3D_18_Weights.DEFAULT).to('cuda')
model.eval()

transform = Compose([
    Resize((128, 171)),
    CenterCrop(112),
    ToTensor(),
    Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

frame_buffer = []
data_queue = queue.Queue()

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"log_{now}.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Action', 'RRI', 'Temp', 'Acc_x', 'Acc_y', 'Acc_z', 'HR'])

def process_data():
    frame_buffer = []
    while True:
        try:
            message = data_queue.get()
            if message is None:
                break

            frame_data = np.frombuffer(message["image"], dtype=np.uint8)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

            if frame is None:
                print("[ERROR] 画像のデコードに失敗しました。")
                continue

            results = yolo_model(frame, conf=0.5)
            person_crop = None
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0].cpu().numpy()) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        person_crop = frame[y1:y2, x1:x2]

            if person_crop is not None:
                frame_buffer.append(person_crop)
                if len(frame_buffer) > 16:
                    frame_buffer.pop(0)

            if len(frame_buffer) == 16:
                processed_frames = [
                    transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    for f in frame_buffer
                ]
                video_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3).unsqueeze(0).to('cuda')

                with torch.no_grad():
                    outputs = model(video_tensor)
                    pred = outputs.argmax(1).item()

                action_label = KINETICS_ID_TO_LABEL.get(pred, "Unknown")
                print(f"[INFO] 認識結果: {action_label}")

                timestamp, rri, temp, acc_x, acc_y, acc_z = (
                    message["timestamp"], message["rri"], message["temp"],
                    message["acc_x"], message["acc_y"], message["acc_z"]
                )
                
                hr = None
                if rri is not None:
                    hr = ecg_analysis.calculate_hr(rri)

                with open(filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, action_label, rri, temp, acc_x, acc_y, acc_z, hr])

                frame_buffer.clear()

        except Exception as e:
            print(f"[ERROR] 処理スレッド例外: {e}")

processing_thread = threading.Thread(target=process_data, daemon=True)
processing_thread.start()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
server_socket.bind(("0.0.0.0", 9999))
server_socket.listen(5)
print("[INFO] クライアント接続待機中...")
conn, addr = server_socket.accept()
print("[INFO] 接続:", addr)

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

data_queue.put(None)
processing_thread.join()
conn.close()
server_socket.close()
print("[INFO] サーバー終了。")
