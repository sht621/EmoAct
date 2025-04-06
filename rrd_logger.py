import csv
import time
from datetime import datetime
from rrd import DataReceiver  # rrd.py から DataReceiver をインポート

class CSVLogger:
    def __init__(self, receiver, csv_filename="rrd_data.csv", log_interval=1):
        """
        DataReceiver のデータを CSV に記録するクラス
        
        Args:
            receiver (DataReceiver): rrd.py の DataReceiver インスタンス
            csv_filename (str): 保存する CSV のファイル名
            log_interval (int): データを記録する間隔（秒）
        """
        self.receiver = receiver
        self.csv_filename = csv_filename
        self.log_interval = log_interval
        self.logging = False

        # CSV にヘッダーを書き込む（最初の1回のみ）
        with open(self.csv_filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "ecg", "temp", "acc_x", "acc_y", "acc_z"])

    def log_to_csv(self):
        """データを定期的に取得し、CSV に記録する"""
        self.logging = True
        print(f"データ記録を開始（{self.csv_filename} に {self.log_interval}秒ごとに保存）...")

        while self.logging:
            # 現在のタイムスタンプを取得
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # DataReceiver から最新データを取得
            ecg = self.receiver.getEcgData()
            temp = self.receiver.getTempData()
            acc_x = self.receiver.getAccXData()
            acc_y = self.receiver.getAccYData()
            acc_z = self.receiver.getAccZData()

            # CSV に書き込み
            with open(self.csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, ecg, temp, acc_x, acc_y, acc_z])

            time.sleep(self.log_interval)  # 指定秒数待機

    def start_logging(self):
        """CSV 記録を開始（別スレッドで実行）"""
        import threading
        self.logging_thread = threading.Thread(target=self.log_to_csv)
        self.logging_thread.daemon = True
        self.logging_thread.start()

    def stop_logging(self):
        """CSV 記録を停止"""
        self.logging = False
        self.logging_thread.join()
        print("データ記録を停止しました")

# メイン処理
if __name__ == "__main__":
    receiver = DataReceiver()  # rrd.py の DataReceiver を使用
    logger = CSVLogger(receiver, csv_filename="rrd_log.csv", log_interval=1)

    receiver.start()  # データ受信開始
    logger.start_logging()  # CSV 記録開始

    try:
        time.sleep(60)  # 10秒間記録
    except KeyboardInterrupt:
        pass  # Ctrl+C で停止

    logger.stop_logging()  # CSV 記録を停止
    receiver.stop()  # データ受信を停止
