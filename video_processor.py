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
from PyQt5.QtCore import QThread, pyqtSignal

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
        
        # PSNR ve SSIM hesaplama
        psnr_value = psnr(original, compressed)
        ssim_value = ssim(original, compressed, multichannel=True)
        
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
    
    def run(self):
        """Video işleme thread'ini çalıştırır."""
        try:
            self.start_time = time.time()
            metrics_history = {
                "PSNR": [],
                "SSIM": [],
                "LPIPS": [],
                "Histogram_Similarity": [],
                "Entropy": [],
                "Compression_Ratio": []
            }
            
            while self.cap.isOpened():
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
            
        except Exception as e:
            print(f"Video işleme hatası: {str(e)}")
    
    def plot_metrics(self, metrics_history: Dict[str, List[float]]) -> plt.Figure:
        """Metrik grafiklerini oluşturur."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # PSNR, SSIM ve LPIPS grafikleri
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics_history["PSNR"], label="PSNR")
        ax1.set_title("PSNR")
        ax1.set_xlabel("Kare")
        ax1.set_ylabel("dB")
        ax1.grid(True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics_history["SSIM"], label="SSIM")
        ax2.set_title("SSIM")
        ax2.set_xlabel("Kare")
        ax2.set_ylabel("Değer")
        ax2.grid(True)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(metrics_history["LPIPS"], label="LPIPS")
        ax3.set_title("LPIPS (Algısal Benzerlik)")
        ax3.set_xlabel("Kare")
        ax3.set_ylabel("Değer")
        ax3.grid(True)
        
        # Histogram benzerliği ve entropi grafikleri
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(metrics_history["Histogram_Similarity"], label="Histogram Benzerliği")
        ax4.set_title("Histogram Benzerliği")
        ax4.set_xlabel("Kare")
        ax4.set_ylabel("Değer")
        ax4.grid(True)
        
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(metrics_history["Entropy"], label="Entropi")
        ax5.set_title("Entropi")
        ax5.set_xlabel("Kare")
        ax5.set_ylabel("Değer")
        ax5.grid(True)
        
        # Sıkıştırma oranı grafiği
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(metrics_history["Compression_Ratio"], label="Sıkıştırma Oranı")
        ax6.set_title("Sıkıştırma Oranı")
        ax6.set_xlabel("Kare")
        ax6.set_ylabel("%")
        ax6.grid(True)
        
        plt.tight_layout()
        return fig
    
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
        return (self.original_size - current_size) / self.original_size * 100 