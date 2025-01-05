import cv2
import numpy as np
from typing import Dict, Any
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from .base_processor import BaseImageProcessor

class MetricProcessor(BaseImageProcessor):
    """Görüntü kalite metrikleri işlemcisi."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
    
    def process(self, image: np.ndarray, reference: np.ndarray = None, 
               metrics: list = None) -> Dict[str, float]:
        """Görüntü kalite metriklerini hesaplar.
        
        Args:
            image: Değerlendirilecek görüntü
            reference: Referans görüntü (karşılaştırmalı metrikler için)
            metrics: Hesaplanacak metrikler listesi
            
        Returns:
            Dict[str, float]: Metrik sonuçları
        """
        if not self.validate_image(image):
            raise ValueError("Geçersiz görüntü formatı")
        
        if reference is not None and not self.validate_image(reference):
            raise ValueError("Geçersiz referans görüntü formatı")
        
        # Varsayılan metrikler
        if metrics is None:
            metrics = ["MSE", "PSNR", "SSIM", "LPIPS"] if reference is not None else ["Entropy"]
        
        results = {}
        
        # Karşılaştırmalı metrikler
        if reference is not None:
            # Görüntüleri aynı boyuta getir
            if image.shape != reference.shape:
                image = cv2.resize(image, (reference.shape[1], reference.shape[0]))
            
            if "MSE" in metrics:
                results["MSE"] = self.calculate_mse(image, reference)
            
            if "PSNR" in metrics:
                results["PSNR"] = self.calculate_psnr(image, reference)
            
            if "SSIM" in metrics:
                results["SSIM"] = self.calculate_ssim(image, reference)
            
            if "LPIPS" in metrics:
                results["LPIPS"] = self.calculate_lpips(image, reference)
        
        # Bağımsız metrikler
        if "Entropy" in metrics:
            results["Entropy"] = self.calculate_entropy(image)
        
        return results
    
    def calculate_mse(self, image: np.ndarray, reference: np.ndarray) -> float:
        """MSE (Mean Squared Error) hesaplar.
        
        Args:
            image: Değerlendirilecek görüntü
            reference: Referans görüntü
            
        Returns:
            float: MSE değeri
        """
        return np.mean((image.astype(float) - reference.astype(float)) ** 2)
    
    def calculate_psnr(self, image: np.ndarray, reference: np.ndarray) -> float:
        """PSNR (Peak Signal-to-Noise Ratio) hesaplar.
        
        Args:
            image: Değerlendirilecek görüntü
            reference: Referans görüntü
            
        Returns:
            float: PSNR değeri (dB)
        """
        return psnr(reference, image)
    
    def calculate_ssim(self, image: np.ndarray, reference: np.ndarray) -> float:
        """SSIM (Structural Similarity Index) hesaplar.
        
        Args:
            image: Değerlendirilecek görüntü
            reference: Referans görüntü
            
        Returns:
            float: SSIM değeri [0, 1]
        """
        # Görüntü boyutunu kontrol et ve gerekirse yeniden boyutlandır
        min_size = 7  # SSIM için minimum boyut
        if image.shape[0] < min_size or image.shape[1] < min_size:
            scale_factor = max(min_size / image.shape[0], min_size / image.shape[1])
            new_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
            image = cv2.resize(image, new_size)
            reference = cv2.resize(reference, new_size)
        
        # channel_axis=2 parametresi, renk kanallarının son boyutta olduğunu belirtir
        return ssim(reference, image, channel_axis=2)
    
    def calculate_lpips(self, image: np.ndarray, reference: np.ndarray) -> float:
        """LPIPS (Learned Perceptual Image Patch Similarity) hesaplar.
        
        Args:
            image: Değerlendirilecek görüntü
            reference: Referans görüntü
            
        Returns:
            float: LPIPS değeri [0, 1]
        """
        with torch.no_grad():
            # Görüntüleri PyTorch tensorlarına dönüştür
            img1 = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img2 = torch.from_numpy(reference).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            
            # GPU'ya taşı
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            # LPIPS değerini hesapla
            return float(self.lpips_model(img1, img2).item())
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """Görüntü entropisini hesaplar.
        
        Args:
            image: Değerlendirilecek görüntü
            
        Returns:
            float: Entropi değeri (bits)
        """
        entropy = 0
        for i in range(3):  # Her renk kanalı için
            channel = image[:, :, i]
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            histogram = histogram.flatten() / np.sum(histogram)
            non_zero_hist = histogram[histogram > 0]
            entropy += -np.sum(non_zero_hist * np.log2(non_zero_hist))
        
        return entropy / 3  # Ortalama entropi
    
    @property
    def available_metrics(self) -> list:
        """Kullanılabilir metrikleri döndürür.
        
        Returns:
            list: Metrik isimleri
        """
        return ["MSE", "PSNR", "SSIM", "LPIPS", "Entropy"] 