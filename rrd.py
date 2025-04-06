

import os
import ctypes
from ctypes import *
import threading

# デバイス識別子
UTWS_RRD_1 = 0x00000001
UTWS_WHS_1 = 0x00000002

# RRD1受信データの要素
class RRD1EcgData(Structure):
    _fields_ = [
        ("ecg", c_ushort),
        ("temp", c_double),
        ("acc_x", c_double),
        ("acc_y", c_double),
        ("acc_z", c_double),
    ]

# RRD1受信データ
class RRD1Data(Structure):
    _fields_ = [
        ("year", c_ushort),
        ("month", c_ushort),
        ("day", c_ushort),
        ("hour", c_ushort),
        ("min", c_ushort),
        ("sec", c_ushort),
        ("msec", c_ushort),
        ("mode", c_ubyte),
        ("tempID", c_ubyte),
        ("sendedID", c_ubyte),
        ("ecg_mode", c_ubyte),
        ("acc_mode", c_ubyte),
        ("lowbattery", c_ubyte),
        ("sampling_freq", c_ubyte),
        ("data_count", c_ubyte),
        ("data", RRD1EcgData * 10),
    ]

class DataReceiver:
    # - `__init__`メソッドでは、各種変数の初期化、DLLのロード、関数プロトタイプの定義、RRD-1デバイスのオープン、ローカルアドレスの取得、WHS-1デバイスのオープンを行います。
    def __init__(self):
        self.latest_ecg_data = 0
        self.latest_temp_data = 0
        self.latest_acc_x_data = 0
        self.latest_acc_y_data = 0
        self.latest_acc_z_data = 0
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self._ecg_buffer = []
        self._temp = None
        self._acc = (None, None, None)
        
        self.utws_lib = self.load_utws_dll()
        self.define_function_prototypes()
        self.device_handle_void_p = self.open_rrd1_device()
        
        if self.device_handle_void_p:
            local_address = self.get_rrd1_local_address()
            self.open_whs1_devices(local_address)
    # DLLのロード
    def load_utws_dll(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        dll_path = os.path.join(current_dir, 'sdk', 'UTWS.dll')

        try:
            utws_lib = ctypes.CDLL(dll_path)
            print(f"{dll_path} ロード成功。")
            return utws_lib
        except OSError as e:
            print(f"DLLロード時にエラー発生: {e}")
            exit(1)
    # DLLの関数プロトタイプの定義
    def define_function_prototypes(self):
        self.utws_lib.UTWSOpenDevice.argtypes = [ctypes.c_uint, ctypes.c_uint]
        self.utws_lib.UTWSOpenDevice.restype = ctypes.c_void_p

        self.utws_lib.UTWSRRD1GetLocalAddress.argtypes = [c_void_p, ctypes.c_char_p]
        self.utws_lib.UTWSRRD1GetLocalAddress.restype = c_uint

        self.utws_lib.UTWSRRD1IsOpen.restype = c_bool

        self.utws_lib.UTWSWHS1SetDestinationAddress.argtypes = [c_void_p, c_char_p]
        self.utws_lib.UTWSWHS1SetDestinationAddress.restype = c_bool

        self.utws_lib.UTWSWHS1CountConnected.argtypes = [c_uint, c_uint]
        self.utws_lib.UTWSWHS1CountConnected.restype = c_uint

        self.utws_lib.UTWSRRD1StartReceiving.argtypes = [c_void_p, c_void_p, CFUNCTYPE(None, c_void_p, c_void_p), c_void_p]
        self.utws_lib.UTWSRRD1StartReceiving.restype = c_bool

        self.utws_lib.UTWSRRD1StopReceiving.argtypes = [c_void_p]
        self.utws_lib.UTWSRRD1StopReceiving.restype = c_bool

        self.utws_lib.UTWSRRD1GetDataEx.argtypes = [c_void_p, POINTER(RRD1Data)]
        self.utws_lib.UTWSRRD1GetDataEx.restype = c_bool
    # RRD-1デバイスのオープン
    def open_rrd1_device(self):
        device_handle = self.utws_lib.UTWSOpenDevice(UTWS_RRD_1, 0)
        if device_handle == 0 or device_handle == -1:
            print("RRD-1デバイスを開けません、またはデバイスハンドルが無効です。")
            return None
        else:
            device_handle_void_p = ctypes.c_void_p(device_handle)
            print(f"RRD-1デバイスハンドルが有効です:{device_handle_void_p.value}")

            if self.utws_lib.UTWSRRD1IsOpen(device_handle_void_p):
                print("RRD-1は開いています")
            else:
                print("開いていません")
            return device_handle_void_p
    # ローカルアドレスの取得
    def get_rrd1_local_address(self):
        local_address = create_string_buffer(11)

        if self.utws_lib.UTWSRRD1GetLocalAddress(self.device_handle_void_p, local_address):
            print("RRD-1のワイヤレスアドレス:", local_address.value.decode('ansi'))
            return local_address
        else:
            print("RRD-1のワイヤレスアドレスを読み取れません。")
            return None
    # WHS-1デバイスのオープン
    def open_whs1_devices(self, local_address):
        whs_connected = self.utws_lib.UTWSWHS1CountConnected(1, 10000)
        if whs_connected > 0:
            for i in range(whs_connected):
                whs_handle = self.utws_lib.UTWSOpenDevice(UTWS_WHS_1, i)
                if whs_handle:
                    whs_handle_void_p = ctypes.c_void_p(whs_handle)
                    self.utws_lib.UTWSWHS1SetDestinationAddress(whs_handle_void_p, local_address)
                    print(f"WHS-1 {i} の宛先アドレス設定に成功しました")
                else:
                    print(f"WHS-1デバイス {i} を開けません")
        else:
            print("WHS-1デバイスが見つかりませんでした")
    # データ受信コールバック
    def rrd1_callback(self, arg1, arg2):
        rrd1_data = RRD1Data()
        if self.utws_lib.UTWSRRD1GetDataEx(self.device_handle_void_p, byref(rrd1_data)):
            if rrd1_data.data_count > 0:
                with self.lock:

                    self.latest_ecg_data = rrd1_data.data[rrd1_data.data_count - 1].ecg
                    self.latest_temp_data = rrd1_data.data[rrd1_data.data_count - 1].temp
                    self.latest_acc_x_data = rrd1_data.data[rrd1_data.data_count - 1].acc_x
                    self.latest_acc_y_data = rrd1_data.data[rrd1_data.data_count - 1].acc_y
                    self.latest_acc_z_data = rrd1_data.data[rrd1_data.data_count - 1].acc_z

    
    def receive_data(self):
        rrd1_callback_func = CFUNCTYPE(None, c_void_p, c_void_p)(self.rrd1_callback)
        if self.utws_lib.UTWSRRD1StartReceiving(self.device_handle_void_p, None, rrd1_callback_func, None):
            print("RRD-1の受信を開始しました")
        else:
            print("RRD-1の受信開始に失敗しました")
        
        while not self.stop_event.is_set():
            pass
        
    # 開始と停止
    def start(self):
        self.receive_thread = threading.Thread(target=self.receive_data)
        self.receive_thread.start()

    def stop(self):
        self.stop_event.set()
        self.utws_lib.UTWSRRD1StopReceiving(self.device_handle_void_p)
        self.receive_thread.join()
        print("RRD-1の受信を停止しました")

    # データ取得と設定
    def getEcgData(self):
        return self.latest_ecg_data
    
    def getTempData(self):
        return self.latest_temp_data
    
    def getAccXData(self):
        return self.latest_acc_x_data
    
    def getAccYData(self):
        return self.latest_acc_y_data
    
    def getAccZData(self): 
        return self.latest_acc_z_data
 
    def setEcgData(self,ecg_data): 
        self.latest_ecg_data=ecg_data
    
    def setTempData(self,temp_data):
        self.latest_temp_data=temp_data
    
    def setAccXData(self,acc_x_data):
        self.latest_acc_x_data=acc_x_data
    
    def setAccYData(self,acc_y_data):
        self.latest_acc_y_data=acc_y_data
    
    def setAccZData(self,acc_z_data): 
        self.latest_acc_z_data=acc_z_data

    def _data_callback(self, data):
        for i in range(data.data_count):
            ecg_sample = data.data[i].ecg

            self._ecg_buffer.append(ecg_sample)
            if len(self._ecg_buffer) > 2000:  # 適切なバッファサイズ
                self._ecg_buffer.pop(0)

    def getLatestEcgBuffer(self, length=400):
        return self._ecg_buffer[-length:] if len(self._ecg_buffer) >= length else []

    def getLatestTemp(self):
        return self._temp

    def getLatestAcc(self):
        return self._acc



