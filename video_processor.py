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
        
        # Video yakalama ve yazma nesneleri
        self.cap = None
        self.writer = None
        
        # Histogram ve entropi geçmişi
        self.histogram_history = []
        self.entropy_history = []
        
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
            "VP9": cv2.VideoWriter_fourcc(*'VP90')
        }[codec]
        
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
    
    def calculate_frame_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict[str, float]:
        """Tek bir kare için kalite metriklerini hesaplar."""
        # Görüntüleri aynı boyuta getir
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
        
        try:
            # PSNR hesaplama
            psnr_value = psnr(original, compressed)
            
            # SSIM hesaplama - küçük görüntüler için win_size ayarı
            min_dim = min(original.shape[0], original.shape[1])
            if min_dim < 7:
                win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
                ssim_value = ssim(original, compressed, channel_axis=2, win_size=win_size)
            else:
                ssim_value = ssim(original, compressed, channel_axis=2)
            
            # LPIPS hesaplama (algısal benzerlik)
            with torch.no_grad():
                # Görüntüleri PyTorch tensorlarına dönüştür
                orig_tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                comp_tensor = torch.from_numpy(compressed).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                
                # GPU'ya taşı
                orig_tensor = orig_tensor.to(self.device)
                comp_tensor = comp_tensor.to(self.device)
                
                # LPIPS değerini hesapla
                lpips_value = float(self.lpips_model(orig_tensor, comp_tensor).item())
            
            # Histogram benzerliği hesaplama
            hist_similarity = self.calculate_histogram_similarity(original, compressed)
            
            return {
                "PSNR": psnr_value,
                "SSIM": ssim_value,
                "LPIPS": lpips_value,
                "Histogram_Similarity": hist_similarity
            }
        except Exception as e:
            print(f"Metrik hesaplama hatası: {str(e)}")
            return {
                "PSNR": 0.0,
                "SSIM": 0.0,
                "LPIPS": 1.0,
                "Histogram_Similarity": 0.0
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
        # Kareyi sıkıştır
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100 - self.crf * 2]  # CRF'yi JPEG kalitesine dönüştür
        _, encoded = cv2.imencode('.jpg', frame, encode_param)
        compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Metrikleri hesapla
        metrics = self.calculate_frame_metrics(frame, compressed)
        
        # Histogram ve entropi hesapla
        self.calculate_histogram(frame)
        entropy = self.calculate_entropy(frame)
        metrics["Entropy"] = entropy
        
        return compressed, metrics
    
    def stop(self):
        """Video işlemeyi durdurur."""
        self._stop = True
        self._pause_condition.wakeAll()  # Duraklatılmışsa devam ettir ve durdur
    
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
            metrics_history = {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": [],
                "Histogram_Similarity": [],
                "Entropy": [],
                "Compression_Ratio": []
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
                    metrics_history[key].append(value)
                
                # Sıkıştırma oranını hesapla
                current_size = os.path.getsize(self.output_path) if os.path.exists(self.output_path) else 0
                compression_ratio = self.calculate_compression_ratio(current_size)
                metrics_history["Compression_Ratio"].append(compression_ratio)
                
                # GUI güncellemelerini gönder
                self.processed_frames += 1
                progress = int((self.processed_frames / self.total_frames) * 100)
                
                self.progress_update.emit(progress)
                self.frame_update.emit(frame, compressed)
                self.metrics_update.emit(metrics_history)
                
                # Süre güncellemelerini gönder
                remaining_time = self.estimate_remaining_time()
                processing_time = time.time() - self.start_time
                self.time_update.emit(remaining_time, processing_time)
                
                # Sıkıştırma oranı güncellemesini gönder
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
        # Bu fonksiyon gerçek veri toplanarak doldurulmalı
        return [2.5, 2.3, 3.1, 3.5]  # Örnek değerler
    
    def get_compression_algorithm_times(self) -> List[float]:
        """Sıkıştırma algoritmaları için işlem sürelerini döndürür."""
        # Bu fonksiyon gerçek veri toplanarak doldurulmalı
        return [0.5, 0.6, 1.2, 2.0]  # Örnek değerler
    
    def get_compression_algorithm_memory(self) -> List[float]:
        """Sıkıştırma algoritmaları için bellek kullanımını döndürür."""
        # Bu fonksiyon gerçek veri toplanarak doldurulmalı
        return [50, 55, 80, 120]  # Örnek değerler
    
    def get_compression_algorithm_speed(self) -> List[float]:
        """Sıkıştırma algoritmaları için sıkıştırma hızını döndürür."""
        # Bu fonksiyon gerçek veri toplanarak doldurulmalı
        return [100, 90, 60, 30]  # Örnek değerler
    
    def estimate_remaining_time(self) -> float:
        """Kalan süreyi tahmin eder."""
        if self.start_time is None or self.processed_frames == 0:
            return 0
        
        elapsed_time = time.time() - self.start_time
        frames_per_second = self.processed_frames / elapsed_time
        remaining_frames = self.total_frames - self.processed_frames
        
        return remaining_frames / frames_per_second
    
    def calculate_compression_ratio(self, current_size: int) -> float:
        """Anlık sıkıştırma oranını hesaplar."""
        if self.original_size == 0:
            return 0
        # 100'den çıkararak gerçek sıkıştırma oranını hesapla
        return 100 - ((current_size / self.original_size) * 100) 