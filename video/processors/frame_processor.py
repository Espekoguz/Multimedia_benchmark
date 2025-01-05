import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseVideoProcessor

class FrameProcessor(BaseVideoProcessor):
    """Video kare işleme sınıfı."""
    
    COLOR_SPACES = {
        "BGR": {
            "code": None,
            "channels": ["B", "G", "R"]
        },
        "HSV": {
            "code": cv2.COLOR_BGR2HSV,
            "channels": ["H", "S", "V"]
        },
        "YCrCb": {
            "code": cv2.COLOR_BGR2YCrCb,
            "channels": ["Y", "Cr", "Cb"]
        }
    }
    
    def process_frame(self, frame: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Kareyi analiz eder ve işler.
        
        Args:
            frame: İşlenecek kare
            **kwargs: Ek parametreler
            
        Returns:
            Dict[str, Any]: İşlem sonuçları
        """
        if not self.validate_frame(frame):
            raise ValueError("Geçersiz kare formatı")
        
        results = {}
        
        # Histogram analizi
        results["histograms"] = self.calculate_histogram(frame)
        
        # Entropi hesaplama
        results["entropy"] = self.calculate_entropy(frame)
        
        # Kanal istatistikleri
        results["statistics"] = self.calculate_statistics(frame)
        
        return results
    
    def calculate_histogram(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Her renk kanalı için histogram hesaplar.
        
        Args:
            frame: Görüntü karesi
            
        Returns:
            Dict[str, np.ndarray]: Kanal histogramları
        """
        histograms = {}
        for i, channel in enumerate(["b", "g", "r"]):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            histograms[channel] = hist.flatten()
        return histograms
    
    def calculate_entropy(self, frame: np.ndarray) -> float:
        """Görüntü entropisini hesaplar.
        
        Args:
            frame: Görüntü karesi
            
        Returns:
            float: Entropi değeri (bits)
        """
        entropy = 0
        for i in range(3):  # Her renk kanalı için
            channel = frame[:, :, i]
            histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
            histogram = histogram.flatten() / np.sum(histogram)
            non_zero_hist = histogram[histogram > 0]
            entropy += -np.sum(non_zero_hist * np.log2(non_zero_hist))
        
        return entropy / 3  # Ortalama entropi
    
    def calculate_statistics(self, frame: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Her kanal için istatistiksel değerleri hesaplar.
        
        Args:
            frame: Görüntü karesi
            
        Returns:
            Dict[str, Dict[str, float]]: Kanal istatistikleri
        """
        stats = {}
        channels = cv2.split(frame)
        
        for channel, name in zip(channels, ["b", "g", "r"]):
            stats[name] = {
                "mean": float(np.mean(channel)),
                "std": float(np.std(channel)),
                "min": float(np.min(channel)),
                "max": float(np.max(channel))
            }
        
        return stats
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Kareye ön işleme uygular (gürültü azaltma, keskinleştirme vb.).
        
        Args:
            frame: İşlenecek kare
            
        Returns:
            np.ndarray: İşlenmiş kare
        """
        # Gürültü azaltma
        denoised = cv2.fastNlMeansDenoisingColored(frame)
        
        # Yumuşak keskinleştirme
        kernel = np.array([[-0.5,-0.5,-0.5],
                         [-0.5, 5.0,-0.5],
                         [-0.5,-0.5,-0.5]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Gaussian bulanıklaştırma
        blurred = cv2.GaussianBlur(sharpened, (3, 3), 0.5)
        
        return blurred
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Kareyi yeniden boyutlandırır.
        
        Args:
            frame: Boyutlandırılacak kare
            target_size: Hedef boyut (genişlik, yükseklik)
            
        Returns:
            np.ndarray: Boyutlandırılmış kare
        """
        return cv2.resize(frame, target_size)
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Kareyi normalize eder.
        
        Args:
            frame: Normalize edilecek kare
            
        Returns:
            np.ndarray: Normalize edilmiş kare
        """
        return frame.astype(float) / 255.0
    
    def convert_color_space(self, frame: np.ndarray, target_space: str) -> np.ndarray:
        """Kareyi hedef renk uzayına dönüştürür.
        
        Args:
            frame: Dönüştürülecek kare
            target_space: Hedef renk uzayı
            
        Returns:
            np.ndarray: Dönüştürülmüş kare
        """
        if target_space not in self.COLOR_SPACES:
            raise ValueError(f"Desteklenmeyen renk uzayı: {target_space}")
        
        if target_space == "BGR":
            return frame
        
        return cv2.cvtColor(frame, self.COLOR_SPACES[target_space]["code"])
    
    def equalize_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Histogram eşitleme uygular.
        
        Args:
            frame: İşlenecek kare
            
        Returns:
            np.ndarray: Histogram eşitlenmiş kare
        """
        # YUV uzayına dönüştür
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Y kanalına histogram eşitleme uygula
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        
        # BGR'ye geri dönüştür
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR) 
    
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Kareyi işler ve sonuçları döndürür.
        
        Args:
            frame: İşlenecek kare
            
        Returns:
            Dict[str, Any]: İşlem sonuçları
        """
        if not self.validate_frame(frame):
            raise ValueError("Geçersiz kare formatı")
            
        results = {}
        
        # Kare istatistiklerini hesapla
        results["statistics"] = self.calculate_statistics(frame)
        
        # Histogram hesapla
        results["histograms"] = self.calculate_histogram(frame)
        
        # Entropi hesapla
        results["entropy"] = self.calculate_entropy(frame)
        
        # Kareyi normalize et
        results["normalized_frame"] = self.normalize_frame(frame)
        
        # Histogram eşitleme
        results["histogram_equalized"] = self.equalize_histogram(frame)
        
        return results 