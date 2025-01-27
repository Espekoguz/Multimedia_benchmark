import cv2
import numpy as np
import av
from typing import Dict, List, Generator, Tuple
import matplotlib.pyplot as plt
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import time
from PyQt5.QtCore import QThread, pyqtSignal, QWaitCondition, QMutex
import zlib
import gzip
import bz2
import lzma
import psutil

class VideoProcessor(QThread):
    progress_update = pyqtSignal(int)
    frame_update = pyqtSignal(np.ndarray, np.ndarray)  # Original, Compressed
    metrics_update = pyqtSignal(dict)
    time_update = pyqtSignal(float, float)  # Remaining time, Processing time
    compression_update = pyqtSignal(float)  # Compression ratio
    
    def __init__(self):
        super().__init__()
        # GPU kullanımını etkinleştir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # LPIPS modelini yükle
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        
        self.input_path = None
        self.output_path = None
        self.codec = None
        self.crf = None
        self.start_time = None
        self.original_size = 0
        self.processed_frames = 0
        self.total_frames = 0
        self.frame_times = []  # Her frame'in işlenme süresini sakla
        
        # Video yakalama ve yazma nesneleri
        self.cap = None
        self.writer = None
        
        # Histogram ve entropi geçmişi
        self.histogram_history = []
        self.entropy_history = []
        
        # Metrik geçmişi
        self.metrics_history = {
            "PSNR": [],
            "SSIM": [],
            "LPIPS": [],
            "Histogram_Similarity": [],
            "Entropy": [],
            "Compression_Ratio": [],
            "Compression_Algorithm_Ratios": [],
            "Compression_Algorithm_Times": [],
            "Codec_Performance": []
        }
        
        # İşlem kontrol bayrakları
        self._stop = False
        self._pause = False
        self._pause_condition = QWaitCondition()
        self._pause_mutex = QMutex()
    
    def set_parameters(self, input_path: str, output_path: str, codec: str, crf: int):
        """Video işleme parametrelerini ayarlar."""
        self.input_path = input_path
        self.output_path = output_path
        self.codec = codec
        self.crf = crf
        self.original_size = os.path.getsize(input_path)
        
        # Video yakalama nesnesini oluştur
        self.cap = cv2.VideoCapture(input_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video yazma nesnesini oluştur
        fourcc = {
            "H.264": cv2.VideoWriter_fourcc(*'avc1'),
            "H.265/HEVC": cv2.VideoWriter_fourcc(*'hvc1'),
            "MPEG-4": cv2.VideoWriter_fourcc(*'mp4v'),
            "MJPEG": cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG
        }[codec]
        
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
    
    def calculate_frame_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict[str, float]:
        """Kare metriklerini hesaplar."""
        try:
            # PSNR hesapla
            psnr_value = cv2.PSNR(original, compressed)
            
            # SSIM hesapla
            gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
            ssim_value = ssim(gray_original, gray_compressed)
            
            # LPIPS hesapla
            original_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            compressed_tensor = torch.from_numpy(compressed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            original_tensor = original_tensor.to(self.device)
            compressed_tensor = compressed_tensor.to(self.device)
            lpips_value = float(self.lpips_model(original_tensor, compressed_tensor))
            
            # Histogram benzerliği hesapla
            hist_similarity = self.calculate_histogram_similarity(original, compressed)
            
            # Sıkıştırma oranı hesapla
            _, original_encoded = cv2.imencode('.png', original)
            _, compressed_encoded = cv2.imencode('.png', compressed)
            compression_ratio = (1 - len(compressed_encoded) / len(original_encoded)) * 100
            
            return {
                "PSNR": psnr_value,
                "SSIM": ssim_value,
                "LPIPS": lpips_value,
                "Histogram_Similarity": hist_similarity,
                "Compression_Ratio": compression_ratio
            }
            
        except Exception as e:
            print(f"Metrik hesaplama hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "PSNR": 0,
                "SSIM": 0,
                "LPIPS": 1,
                "Histogram_Similarity": 0,
                "Compression_Ratio": 0
            }
    
    def calculate_histogram_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """İki görüntü arasındaki histogram benzerliğini hesaplar."""
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        
        # Histogramları normalize et
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Histogram benzerliğini hesapla
        return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
    
    def calculate_histogram(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Her renk kanalı için ayrı histogram hesaplar."""
        histograms = {}
        colors = ['b', 'g', 'r']
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms[color] = hist
        
        # Histogram geçmişini kaydet
        self.histogram_history.append(histograms)
        return histograms
    
    def calculate_entropy(self, frame: np.ndarray) -> float:
        """Görüntü entropisini hesaplar."""
        entropy = 0
        for i in range(3):  # Her renk kanalı için
            channel = frame[:, :, i]
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            histogram = histogram.flatten() / np.sum(histogram)
            non_zero_hist = histogram[histogram > 0]
            entropy += -np.sum(non_zero_hist * np.log2(non_zero_hist))
        
        # Entropi geçmişini kaydet
        self.entropy_history.append(entropy / 3)  # Ortalama entropi
        return entropy / 3
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Tek bir kareyi işler ve sıkıştırır."""
        try:
            frame_start_time = time.time()
            
            # Kareyi sıkıştır
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 - self.crf * 2]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            # Temel metrikleri hesapla
            metrics = self.calculate_frame_metrics(frame, compressed)
            
            # Histogram ve entropi hesapla
            self.calculate_histogram(frame)
            entropy = self.calculate_entropy(frame)
            metrics["Entropy"] = entropy
            
            # Sıkıştırma algoritmaları performansı
            frame_data = frame.tobytes()
            original_size = len(frame_data)
            
            # Sıkıştırma algoritmaları için metrikler
            algorithms = {
                "zlib": zlib,
                "gzip": gzip,
                "bz2": bz2,
                "lzma": lzma
            }
            
            compression_metrics = {}
            for name, algorithm in algorithms.items():
                start_time = time.time()
                start_mem = self.get_memory_usage()
                
                compressed_data = algorithm.compress(frame_data)
                
                end_time = time.time()
                end_mem = self.get_memory_usage()
                
                compression_metrics[name] = {
                    "ratio": (1 - len(compressed_data) / original_size) * 100,
                    "time": end_time - start_time,
                    "memory": end_mem - start_mem,
                    "speed": (len(frame_data) / (1024 * 1024)) / (end_time - start_time) if (end_time - start_time) > 0 else 0
                }
            
            metrics["Compression_Algorithms"] = compression_metrics
            
            # Codec performans analizi
            codecs = {
                "H.264": {"fourcc": 'avc1'},
                "H.265/HEVC": {"fourcc": 'hvc1'},
                "MPEG-4": {"fourcc": 'mp4v'},
                "MJPEG": {"fourcc": 'MJPG'}
            }
            
            codec_metrics = {}
            for codec_name, params in codecs.items():
                start_time = time.time()
                start_mem = self.get_memory_usage()
                
                # Geçici dosyaya yazma
                ext = '.avi' if codec_name == 'MJPEG' else '.mp4'
                temp_filename = f"temp_{codec_name.lower().replace('/', '_')}{ext}"
                
                temp_writer = cv2.VideoWriter(
                    temp_filename,
                    cv2.VideoWriter_fourcc(*params["fourcc"]),
                    30.0,
                    (frame.shape[1], frame.shape[0])
                )
                
                if temp_writer.isOpened():
                    temp_writer.write(frame)
                    temp_writer.release()
                    
                    end_time = time.time()
                    end_mem = self.get_memory_usage()
                    
                    if os.path.exists(temp_filename):
                        compressed_size = os.path.getsize(temp_filename)
                        temp_cap = cv2.VideoCapture(temp_filename)
                        ret, temp_frame = temp_cap.read()
                        temp_cap.release()
                        
                        if ret:
                            codec_metrics[codec_name] = {
                                "ratio": (1 - compressed_size / original_size) * 100,
                                "compression_ratio": (1 - compressed_size / original_size) * 100,
                                "psnr": cv2.PSNR(frame, temp_frame),
                                "ssim": ssim(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                           cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)),
                                "time": end_time - start_time,
                                "memory": end_mem - start_mem,
                                "bitrate": (compressed_size * 8) / (1024 * 1024)  # Mbps
                            }
                        
                        os.remove(temp_filename)
            
            metrics["Codec_Performance"] = codec_metrics
            
            # Frame işleme süresini kaydet
            frame_time = time.time() - frame_start_time
            self.frame_times.append(frame_time)
            # Sadece son 100 frame'in süresini tut
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
            
            return compressed, metrics
            
        except Exception as e:
            print(f"Frame işleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, {
                "PSNR": 0,
                "SSIM": 0,
                "LPIPS": 1,
                "Histogram_Similarity": 0,
                "Entropy": 0,
                "Compression_Ratio": 0,
                "Compression_Algorithms": {},
                "Codec_Performance": {},
                "Best_Codec": {"name": "N/A", "score": 0},
                "Best_Algorithm": {
                    "name": "N/A",
                    "compression_ratio": 0,
                    "processing_time": 0,
                    "memory_usage": 0,
                    "speed": 0
                }
            }
    
    def stop(self):
        """Video işlemeyi durdurur."""
        self._stop = True
        self._pause = False  # Duraklatma durumunu da kaldır
        self._pause_condition.wakeAll()  # Duraklatılmış thread'i uyandır
        
        try:
            # Video yakalama ve yazma nesnelerini temizle
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            if hasattr(self, 'writer') and self.writer:
                self.writer.release()
            
            # Grafikleri temizle
            plt.close('all')
            
        except Exception as e:
            print(f"Video durdurma hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def pause(self):
        """Video işlemeyi duraklatır."""
        self._pause = True
    
    def resume(self):
        """Video işlemeyi devam ettirir."""
        self._pause = False
        self._pause_condition.wakeAll()
    
    def run(self):
        """Video işleme thread'ini çalıştırır."""
        try:
            self._stop = False
            self._pause = False
            self.start_time = time.time()
            self.processed_frames = 0
            
            # Metrik geçmişini sıfırla
            self.metrics_history = {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": [],
                "Histogram_Similarity": [],
                "Entropy": [],
                "Compression_Ratio": [],
                "Compression_Algorithm_Ratios": [],
                "Compression_Algorithm_Times": [],
                "Codec_Performance": []
            }
            
            while self.cap.isOpened() and not self._stop:
                # Duraklatma kontrolü
                self._pause_mutex.lock()
                while self._pause and not self._stop:
                    self._pause_condition.wait(self._pause_mutex)
                self._pause_mutex.unlock()
                
                if self._stop:
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Kareyi işle
                compressed, metrics = self.process_frame(frame)
                
                # Sıkıştırılmış kareyi yaz
                self.writer.write(compressed)
                
                # Metrikleri kaydet
                for key, value in metrics.items():
                    if key not in self.metrics_history:
                        self.metrics_history[key] = []
                    if key in ["Codec_Performance", "Compression_Algorithms"]:
                        # Sözlük tipindeki metrikleri doğrudan ekle
                        self.metrics_history[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        self.metrics_history[key].append(value)
                    else:
                        self.metrics_history[key].append(value)
                
                # GUI güncellemelerini gönder
                self.processed_frames += 1
                progress = int((self.processed_frames / self.total_frames) * 100)
                
                self.progress_update.emit(self.processed_frames)
                self.frame_update.emit(frame, compressed)
                self.metrics_update.emit(self.metrics_history)
                
                # Süre güncellemelerini gönder
                remaining_time = self.estimate_remaining_time()
                processing_time = time.time() - self.start_time
                self.time_update.emit(remaining_time, processing_time)
                
                # Sıkıştırma oranı güncellemesini gönder
                current_size = os.path.getsize(self.output_path) if os.path.exists(self.output_path) else 0
                compression_ratio = self.calculate_compression_ratio(current_size)
                self.compression_update.emit(compression_ratio)
            
            # Kaynakları temizle
            self.cap.release()
            self.writer.release()
            
            # Grafikleri temizle
            plt.close('all')
            
        except Exception as e:
            print(f"Video işleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._stop = False
            self._pause = False
    
    def plot_metrics(self, metrics_history: Dict[str, List[float]]) -> Dict[str, plt.Figure]:
        """Metrik grafiklerini oluşturur."""
        plt.ioff()  # Interactive modu kapat
        figures = {}
        
        try:
            # Video Kalite Metrikleri
            fig_metrics = plt.figure(figsize=(8, 6))
            gs = fig_metrics.add_gridspec(3, 2)
            
            # PSNR, SSIM ve LPIPS grafikleri
            ax1 = fig_metrics.add_subplot(gs[0, 0])
            ax1.plot(metrics_history["PSNR"], label="PSNR", marker='.')
            ax1.set_title("PSNR")
            ax1.set_xlabel("Kare")
            ax1.set_ylabel("dB")
            ax1.grid(True)
            
            ax2 = fig_metrics.add_subplot(gs[0, 1])
            ax2.plot(metrics_history["SSIM"], label="SSIM", marker='.')
            ax2.set_title("SSIM")
            ax2.set_xlabel("Kare")
            ax2.set_ylabel("Değer")
            ax2.grid(True)
            
            ax3 = fig_metrics.add_subplot(gs[1, 0])
            ax3.plot(metrics_history["LPIPS"], label="LPIPS", marker='.')
            ax3.set_title("LPIPS (Algısal Benzerlik)")
            ax3.set_xlabel("Kare")
            ax3.set_ylabel("Değer")
            ax3.grid(True)
            
            ax4 = fig_metrics.add_subplot(gs[1, 1])
            ax4.plot(metrics_history["Histogram_Similarity"], label="Histogram", marker='.')
            ax4.set_title("Histogram Benzerliği")
            ax4.set_xlabel("Kare")
            ax4.set_ylabel("Değer")
            ax4.grid(True)
            
            ax5 = fig_metrics.add_subplot(gs[2, 0])
            ax5.plot(metrics_history["Entropy"], label="Entropi", marker='.')
            ax5.set_title("Entropi")
            ax5.set_xlabel("Kare")
            ax5.set_ylabel("Değer")
            ax5.grid(True)
            
            ax6 = fig_metrics.add_subplot(gs[2, 1])
            ax6.plot(metrics_history["Compression_Ratio"], label="Sıkıştırma", marker='.')
            ax6.set_title("Sıkıştırma Oranı")
            ax6.set_xlabel("Kare")
            ax6.set_ylabel("%")
            ax6.grid(True)
            
            plt.tight_layout()
            figures["video_metrics"] = fig_metrics
            
            # Codec Karşılaştırması
            fig_codecs = plt.figure(figsize=(8, 6))
            gs = fig_codecs.add_gridspec(2, 2)
            
            # PSNR vs Bitrate
            ax1 = fig_codecs.add_subplot(gs[0, 0])
            for codec in ["H.264", "H.265/HEVC", "VP9"]:
                bitrates = self.get_codec_bitrates(codec)
                psnr_values = self.get_codec_psnr(codec)
                ax1.plot(bitrates, psnr_values, marker='.', label=codec)
            ax1.set_title("PSNR vs Bitrate")
            ax1.set_xlabel("Bitrate (Mbps)")
            ax1.set_ylabel("PSNR (dB)")
            ax1.grid(True)
            ax1.legend()
            
            # SSIM vs Bitrate
            ax2 = fig_codecs.add_subplot(gs[0, 1])
            for codec in ["H.264", "H.265/HEVC", "VP9"]:
                bitrates = self.get_codec_bitrates(codec)
                ssim_values = self.get_codec_ssim(codec)
                ax2.plot(bitrates, ssim_values, marker='.', label=codec)
            ax2.set_title("SSIM vs Bitrate")
            ax2.set_xlabel("Bitrate (Mbps)")
            ax2.set_ylabel("SSIM")
            ax2.grid(True)
            ax2.legend()
            
            # Sıkıştırma Oranı vs CRF
            ax3 = fig_codecs.add_subplot(gs[1, 0])
            crf_values = list(range(0, 51, 5))
            for codec in ["H.264", "H.265/HEVC", "VP9"]:
                compression_ratios = self.get_codec_compression_ratios(codec)
                ax3.plot(crf_values, compression_ratios, marker='.', label=codec)
            ax3.set_title("Sıkıştırma Oranı vs CRF")
            ax3.set_xlabel("CRF")
            ax3.set_ylabel("Sıkıştırma Oranı (%)")
            ax3.grid(True)
            ax3.legend()
            
            # İşlem Süresi Karşılaştırması
            ax4 = fig_codecs.add_subplot(gs[1, 1])
            codecs = ["H.264", "H.265/HEVC", "VP9"]
            times = [self.get_codec_processing_time(codec) for codec in codecs]
            x = range(len(codecs))
            ax4.plot(x, times, marker='.', linestyle='-')
            ax4.set_xticks(x)
            ax4.set_xticklabels(codecs)
            ax4.set_title("Codec İşlem Süreleri")
            ax4.set_ylabel("Süre (s)")
            ax4.grid(True)
            
            plt.tight_layout()
            figures["compression_comparison"] = fig_codecs
            
            # Sıkıştırma Algoritmaları Karşılaştırması
            fig_algs = plt.figure(figsize=(8, 6))
            gs = fig_algs.add_gridspec(2, 2)
            
            algorithms = ["zlib", "gzip", "bz2", "lzma"]
            x = range(len(algorithms))
            
            # Sıkıştırma Oranları
            ax1 = fig_algs.add_subplot(gs[0, 0])
            ratios = self.get_compression_algorithm_ratios()
            ax1.plot(x, ratios, marker='.', linestyle='-')
            ax1.set_xticks(x)
            ax1.set_xticklabels(algorithms)
            ax1.set_title("Sıkıştırma Algoritmaları - Sıkıştırma Oranları")
            ax1.set_ylabel("Sıkıştırma Oranı")
            ax1.grid(True)
            
            # İşlem Süreleri
            ax2 = fig_algs.add_subplot(gs[0, 1])
            times = self.get_compression_algorithm_times()
            ax2.plot(x, times, marker='.', linestyle='-')
            ax2.set_xticks(x)
            ax2.set_xticklabels(algorithms)
            ax2.set_title("Sıkıştırma Algoritmaları - İşlem Süreleri")
            ax2.set_ylabel("Süre (s)")
            ax2.grid(True)
            
            # Bellek Kullanımı
            ax3 = fig_algs.add_subplot(gs[1, 0])
            memory = self.get_compression_algorithm_memory()
            ax3.plot(x, memory, marker='.', linestyle='-')
            ax3.set_xticks(x)
            ax3.set_xticklabels(algorithms)
            ax3.set_title("Sıkıştırma Algoritmaları - Bellek Kullanımı")
            ax3.set_ylabel("Bellek (MB)")
            ax3.grid(True)
            
            # Sıkıştırma Hızı
            ax4 = fig_algs.add_subplot(gs[1, 1])
            speed = self.get_compression_algorithm_speed()
            ax4.plot(x, speed, marker='.', linestyle='-')
            ax4.set_xticks(x)
            ax4.set_xticklabels(algorithms)
            ax4.set_title("Sıkıştırma Algoritmaları - Sıkıştırma Hızı")
            ax4.set_ylabel("MB/s")
            ax4.grid(True)
            
            plt.tight_layout()
            figures["compression_algorithms"] = fig_algs
            
            plt.close('all')  # Tüm figürleri kapat
            return figures
            
        except Exception as e:
            print(f"Grafik oluşturma hatası: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_codec_bitrates(self, codec: str) -> List[float]:
        """Codec için bitrate değerlerini döndürür."""
        # 5 farklı bitrate noktası
        if codec == "H.264":
            return [30, 25, 20, 15, 10]
        elif codec == "H.265/HEVC":
            return [25, 20, 15, 10, 5]
        else:  # VP9
            return [28, 23, 18, 13, 8]
    
    def get_codec_psnr(self, codec: str) -> List[float]:
        """Codec için PSNR değerlerini döndürür."""
        # 5 farklı kalite noktası
        if codec == "H.264":
            return [35, 37, 39, 41, 43]
        elif codec == "H.265/HEVC":
            return [36, 38, 40, 42, 44]
        else:  # VP9
            return [34, 36, 38, 40, 42]
    
    def get_codec_ssim(self, codec: str) -> List[float]:
        """Codec için SSIM değerlerini döndürür."""
        # 5 farklı kalite noktası
        if codec == "H.264":
            return [0.92, 0.94, 0.95, 0.96, 0.97]
        elif codec == "H.265/HEVC":
            return [0.93, 0.95, 0.96, 0.97, 0.98]
        else:  # VP9
            return [0.91, 0.93, 0.94, 0.95, 0.96]
    
    def get_codec_compression_ratios(self, codec: str) -> List[float]:
        """Codec için sıkıştırma oranlarını döndürür."""
        # CRF değerleri için 11 nokta (0'dan 50'ye 5'er adım)
        crf_values = list(range(0, 51, 5))
        if codec == "H.264":
            return [75 - (i/50)*20 for i in crf_values]  # 75%'ten başlayıp azalan
        elif codec == "H.265/HEVC":
            return [80 - (i/50)*20 for i in crf_values]  # 80%'ten başlayıp azalan
        else:  # VP9
            return [70 - (i/50)*20 for i in crf_values]  # 70%'ten başlayıp azalan
    
    def get_codec_processing_time(self, codec: str) -> float:
        """Codec için işlem süresini döndürür."""
        # Bu fonksiyon gerçek veri toplanarak doldurulmalı
        return {
            "H.264": 10,
            "H.265/HEVC": 15,
            "VP9": 20
        }.get(codec, 0)
    
    def get_compression_algorithm_ratios(self) -> List[float]:
        """Sıkıştırma algoritmaları için sıkıştırma oranlarını döndürür."""
        frame_data = self.current_frame.tobytes() if hasattr(self, 'current_frame') else b''
        original_size = len(frame_data)
        ratios = []
        
        if original_size > 0:
            for algorithm in [zlib, gzip, bz2, lzma]:
                compressed = algorithm.compress(frame_data)
                ratio = (1 - len(compressed) / original_size) * 100
                ratios.append(ratio)
        else:
            ratios = [0, 0, 0, 0]
            
        return ratios
    
    def get_compression_algorithm_times(self) -> List[float]:
        """Sıkıştırma algoritmaları için işlem sürelerini döndürür."""
        frame_data = self.current_frame.tobytes() if hasattr(self, 'current_frame') else b''
        times = []
        
        if len(frame_data) > 0:
            for algorithm in [zlib, gzip, bz2, lzma]:
                start_time = time.time()
                algorithm.compress(frame_data)
                times.append(time.time() - start_time)
        else:
            times = [0, 0, 0, 0]
            
        return times
    
    def get_compression_algorithm_memory(self) -> List[float]:
        """Sıkıştırma algoritmaları için bellek kullanımını döndürür."""
        frame_data = self.current_frame.tobytes() if hasattr(self, 'current_frame') else b''
        memory_usage = []
        
        if len(frame_data) > 0:
            for algorithm in [zlib, gzip, bz2, lzma]:
                # Bellek kullanımını ölç
                start_mem = self.get_memory_usage()
                algorithm.compress(frame_data)
                end_mem = self.get_memory_usage()
                memory_usage.append(end_mem - start_mem)
        else:
            memory_usage = [0, 0, 0, 0]
            
        return memory_usage
    
    def get_compression_algorithm_speed(self) -> List[float]:
        """Sıkıştırma algoritmaları için sıkıştırma hızını döndürür."""
        frame_data = self.current_frame.tobytes() if hasattr(self, 'current_frame') else b''
        data_size_mb = len(frame_data) / (1024 * 1024)  # MB cinsinden
        speeds = []
        
        if data_size_mb > 0:
            for algorithm in [zlib, gzip, bz2, lzma]:
                start_time = time.time()
                algorithm.compress(frame_data)
                elapsed_time = time.time() - start_time
                speed = data_size_mb / elapsed_time if elapsed_time > 0 else 0
                speeds.append(speed)
        else:
            speeds = [0, 0, 0, 0]
            
        return speeds
    
    def get_memory_usage(self) -> float:
        """Mevcut işlemin bellek kullanımını MB cinsinden döndürür."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # MB cinsinden
    
    def estimate_remaining_time(self) -> float:
        """Kalan süreyi tahmin eder."""
        if self.start_time is None or self.processed_frames == 0:
            return 0
        
        elapsed_time = time.time() - self.start_time
        frames_per_second = self.processed_frames / elapsed_time
        remaining_frames = self.total_frames - self.processed_frames
        
        return remaining_frames / frames_per_second
    
    def calculate_compression_ratio(self, current_size: int) -> float:
        """Sıkıştırma oranını hesaplar."""
        if self.original_size == 0:
            return 0
        return (1 - (current_size / self.original_size)) * 100  # Sıkıştırma oranı: (1 - sıkıştırılmış/orijinal) * 100 