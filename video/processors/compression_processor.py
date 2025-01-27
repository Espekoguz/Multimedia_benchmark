import cv2
import numpy as np
from typing import Dict, Any
import os
from .base_processor import BaseVideoProcessor
import subprocess
import re
from PyQt5.QtCore import QObject, pyqtSignal

class CompressionProcessor(BaseVideoProcessor, QObject):
    """Video sıkıştırma işlemcisi."""
    
    progress_update = pyqtSignal(int)  # İlerleme güncelleme sinyali
    
    SUPPORTED_CODECS = {
        "H.264": {
            "fourcc": "avc1",
            "crf_range": (0, 51)  # 0: en iyi kalite, 51: en düşük kalite
        },
        "HEVC": {
            "fourcc": "hvc1",
            "crf_range": (0, 51)
        },
        "VP9": {
            "fourcc": "vp09",
            "crf_range": (0, 63)
        }
    }
    
    def __init__(self):
        """Sınıfı başlatır."""
        BaseVideoProcessor.__init__(self)
        QObject.__init__(self)
        self.input_path = None
        self.output_path = None
        self.codec = "libx264"
        self.crf = 23
        self._video_info = None
        
    @property
    def video_info(self):
        """Video bilgilerini döndürür."""
        if self._video_info is None and self.input_path is not None:
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                return None
            self._video_info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
            cap.release()
        return self._video_info
    
    def set_parameters(self, input_path: str, output_path: str, codec: str = "H.264", crf: int = 23) -> None:
        """Sıkıştırma parametrelerini ayarlar.
        
        Args:
            input_path: Girdi video dosyası
            output_path: Çıktı video dosyası
            codec: Kullanılacak codec
            crf: CRF değeri
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError("Girdi dosyası bulunamadı")
            
        if codec not in self.SUPPORTED_CODECS:
            raise ValueError(f"Desteklenmeyen codec: {codec}")
            
        if not isinstance(crf, int) or crf < 0 or crf > 51:
            raise ValueError("CRF değeri 0-51 aralığında olmalı")
            
        self.input_path = input_path
        self.output_path = output_path
        self.codec = self.SUPPORTED_CODECS[codec]
        self.crf = crf
        self._video_info = None
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Kareyi işler ve sıkıştırır.
        
        Args:
            frame: İşlenecek kare
            **kwargs: Ek parametreler
            
        Returns:
            Dict[str, Any]: İşlem sonuçları
        """
        if not self.validate_frame(frame):
            raise ValueError("Geçersiz kare formatı")
        
        # Video writer'ı başlat
        if self.writer is None:
            self._initialize_writer(frame.shape)
        
        # Kareyi yaz
        self.writer.write(frame)
        
        return {"frame_processed": True}
    
    def _initialize_writer(self, frame_shape: tuple) -> None:
        """Video writer'ı başlatır.
        
        Args:
            frame_shape: Kare boyutları
        """
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.SUPPORTED_CODECS[self.codec]["fourcc"])
        
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )
    
    def close(self) -> None:
        """Kaynakları serbest bırakır."""
        super().close()
        if self.writer is not None:
            self.writer.release()
    
    def calculate_compression_ratio(self, compressed_size: int) -> float:
        """Sıkıştırma oranını hesaplar.
        
        Args:
            compressed_size: Sıkıştırılmış dosya boyutu
            
        Returns:
            float: Sıkıştırma oranı (%)
        """
        if not hasattr(self, "original_size"):
            raise ValueError("Orijinal dosya boyutu bilinmiyor")
            
        if compressed_size <= 0:
            raise ValueError("Geçersiz sıkıştırılmış dosya boyutu")
            
        return (self.original_size - compressed_size) / self.original_size * 100
        
    def start(self):
        """Sıkıştırma işlemini başlatır."""
        if self.input_path is None or self.output_path is None:
            raise ValueError("Girdi ve çıktı dosya yolları ayarlanmamış")
            
        # Başlangıç ilerleme değerini gönder
        self.progress_update.emit(0)
            
        # FFmpeg komutunu oluştur
        command = [
            "ffmpeg",
            "-i", str(self.input_path),
            "-c:v", "libx264" if self.codec["fourcc"] == "avc1" else "libx265" if self.codec["fourcc"] == "hvc1" else "libvpx-vp9",
            "-crf", str(self.crf),
            "-y",  # Varolan dosyanın üzerine yaz
            str(self.output_path)
        ]
        
        try:
            # FFmpeg işlemini başlat
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # İlerlemeyi takip et
            total_frames = self.video_info["frame_count"]
            current_frame = 0
            
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                    
                # İlerleme bilgisini güncelle
                if "frame=" in line:
                    try:
                        frame_match = re.search(r"frame=\s*(\d+)", line)
                        if frame_match:
                            current_frame = int(frame_match.group(1))
                            progress = int((current_frame / total_frames) * 100)
                            self.progress_update.emit(progress)
                    except (ValueError, ZeroDivisionError):
                        pass
                        
            # İşlem tamamlandı
            returncode = process.wait()
            if returncode != 0:
                stderr_output = process.stderr.read()
                raise RuntimeError(f"FFmpeg işlemi başarısız oldu: {stderr_output}")
                
        except Exception as e:
            raise RuntimeError(f"Sıkıştırma işlemi başarısız: {str(e)}")
            
        finally:
            # İşlem tamamlandı sinyali gönder
            self.progress_update.emit(100)
    
    def _parse_progress(self, output: str) -> float:
        """FFmpeg çıktısından ilerleme yüzdesini ayıklar.
        
        Args:
            output: FFmpeg çıktı satırı
            
        Returns:
            float: İlerleme yüzdesi (0-100)
        """
        # FFmpeg çıktısından süre bilgisini ayıkla
        time_match = re.search(r"time=(\d+:\d+:\d+.\d+)", output)
        if time_match:
            time_str = time_match.group(1)
            # Süreyi saniyeye çevir
            h, m, s = map(float, time_str.split(":"))
            current_time = h * 3600 + m * 60 + s
            # İlerleme yüzdesini hesapla
            if hasattr(self, "duration"):
                return min(100, (current_time / self.duration) * 100)
        return 0.0
    
    @classmethod
    def get_supported_codecs(cls) -> Dict[str, Dict[str, Any]]:
        """Desteklenen codec'leri döndürür.
        
        Returns:
            Dict[str, Dict[str, Any]]: Codec bilgileri
        """
        return cls.SUPPORTED_CODECS 
    
    def wait(self) -> None:
        """İşlemin tamamlanmasını bekler."""
        # Bu metod şu anda bir şey yapmıyor çünkü start() metodu zaten senkron çalışıyor
        pass 