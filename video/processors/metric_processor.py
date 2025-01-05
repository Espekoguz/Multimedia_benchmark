import cv2
import numpy as np
from typing import Dict, Any, List
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from .base_processor import BaseVideoProcessor
from PyQt6.QtCore import QObject, pyqtSignal
import os

class MetricProcessor(BaseVideoProcessor, QObject):
    """Video kalite metrikleri işlemcisi."""
    
    metrics_update = pyqtSignal(dict)  # Metrik güncelleme sinyali
    
    def __init__(self):
        """Sınıfı başlatır."""
        BaseVideoProcessor.__init__(self)
        QObject.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.reference_cap = None
        self.metrics = ["psnr", "ssim", "lpips", "mse"]  # MSE eklendi
        self.metrics_history = []
        
    def set_parameters(self, video_path: str, reference_path: str, metrics: List[str] = None) -> None:
        """Metrik hesaplama parametrelerini ayarlar.
        
        Args:
            video_path: Test video dosyası
            reference_path: Referans video dosyası
            metrics: Hesaplanacak metrikler listesi
        """
        # Test videosunu aç
        if not os.path.exists(video_path):
            raise FileNotFoundError("Test videosu bulunamadı")
            
        if not self.open(video_path):
            raise RuntimeError("Test videosu açılamadı")
        
        # Referans videoyu aç
        if not os.path.exists(reference_path):
            raise FileNotFoundError("Referans video bulunamadı")
            
        self.reference_cap = cv2.VideoCapture(reference_path)
        if not self.reference_cap.isOpened():
            raise RuntimeError("Referans video açılamadı")
        
        # Kare sayılarını kontrol et
        ref_frame_count = int(self.reference_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if ref_frame_count != self.frame_count:
            raise ValueError("Video kare sayıları eşleşmiyor")
        
        # Metrikleri ayarla
        if metrics is not None:
            self.metrics = metrics
        self.metrics_history = []
        
    def process_frame(self, frame, reference_frame):
        """Bir kare için metrikleri hesaplar."""
        if frame is None or reference_frame is None:
            return None
            
        # Karelerin boyutlarını kontrol et ve gerekirse yeniden boyutlandır
        if frame.shape != reference_frame.shape:
            reference_frame = cv2.resize(reference_frame, (frame.shape[1], frame.shape[0]))
            
        results = {}
        for metric_name in self.metrics:
            if metric_name == "psnr":
                results[metric_name] = [float(cv2.PSNR(frame, reference_frame))]
            elif metric_name == "ssim":
                results[metric_name] = [float(ssim(frame, reference_frame, channel_axis=2, win_size=7))]
            elif metric_name == "lpips":
                frame_tensor = self._preprocess_frame(frame)
                ref_tensor = self._preprocess_frame(reference_frame)
                results[metric_name] = [float(self.lpips_model(frame_tensor, ref_tensor).item())]
            elif metric_name == "mse":
                results[metric_name] = [float(self.calculate_mse(frame, reference_frame))]
                
        self.metrics_history.append(results)
        self.metrics_update.emit(results)
        return results
        
    def calculate_mse(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """MSE (Mean Squared Error) hesaplar.
        
        Args:
            frame: Test karesi
            reference: Referans kare
            
        Returns:
            float: MSE değeri
        """
        return np.mean((frame.astype(float) - reference.astype(float)) ** 2)
    
    def calculate_psnr(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """PSNR (Peak Signal-to-Noise Ratio) hesaplar.
        
        Args:
            frame: Test karesi
            reference: Referans kare
            
        Returns:
            float: PSNR değeri (dB)
        """
        return psnr(reference, frame)
    
    def calculate_ssim(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """SSIM (Structural Similarity Index) hesaplar.
        
        Args:
            frame: Test karesi
            reference: Referans kare
            
        Returns:
            float: SSIM değeri [0, 1]
        """
        return ssim(reference, frame, multichannel=True)
    
    def calculate_lpips(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """LPIPS (Learned Perceptual Image Patch Similarity) hesaplar.
        
        Args:
            frame: Test karesi
            reference: Referans kare
            
        Returns:
            float: LPIPS değeri [0, 1]
        """
        with torch.no_grad():
            # Görüntüleri PyTorch tensorlarına dönüştür
            img1 = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2 = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # GPU'ya taşı
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # LPIPS değerini hesapla
            return float(self.lpips_model(img1, img2).item())
    
    def close(self) -> None:
        """Kaynakları serbest bırakır."""
        super().close()
        if self.reference_cap is not None:
            self.reference_cap.release()
    
    @property
    def available_metrics(self) -> List[str]:
        """Kullanılabilir metrikleri döndürür.
        
        Returns:
            List[str]: Metrik isimleri
        """
        return ["mse", "psnr", "ssim", "lpips", "entropy"]
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Kareyi LPIPS için hazırlar.
        
        Args:
            frame: İşlenecek kare
            
        Returns:
            torch.Tensor: PyTorch tensoru
        """
        # BGR'den RGB'ye dönüştür
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tensora dönüştür ve normalize et
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # GPU'ya taşı
        return tensor.to(self.device) 
    
    def start(self) -> None:
        """Metrik hesaplama işlemini başlatır."""
        if self.cap is None or self.reference_cap is None:
            raise RuntimeError("Video dosyaları açılmamış")
        
        # Metrik sonuçlarını tutacak sözlük
        accumulated_results = {metric: [] for metric in self.metrics}
        
        while True:
            ret, frame = self.cap.read()
            ret_ref, ref_frame = self.reference_cap.read()
            
            if not ret or not ret_ref:
                break
                
            results = self.process_frame(frame, ref_frame)
            if results:
                # Her metrik için sonuçları birleştir
                for metric in self.metrics:
                    accumulated_results[metric].extend(results[metric])
                self.metrics_update.emit(accumulated_results)
    
    def wait(self) -> None:
        """İşlemin tamamlanmasını bekler."""
        # Bu metod şu anda bir şey yapmıyor çünkü start() metodu zaten senkron çalışıyor
        pass 