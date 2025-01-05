import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .base_processor import BaseImageProcessor

class ColorProcessor(BaseImageProcessor):
    """Renk uzayı dönüşümleri ve renk analizi işlemcisi."""
    
    COLOR_SPACES = {
        "RGB": {
            "code": None,
            "channels": ["R", "G", "B"]
        },
        "BGR": {
            "code": None,
            "channels": ["B", "G", "R"]
        },
        "HSV": {
            "code": cv2.COLOR_BGR2HSV,
            "channels": ["H", "S", "V"]
        },
        "LAB": {
            "code": cv2.COLOR_BGR2LAB,
            "channels": ["L", "A", "B"]
        },
        "YUV": {
            "code": cv2.COLOR_BGR2YUV,
            "channels": ["Y", "U", "V"]
        },
        "YCrCb": {
            "code": cv2.COLOR_BGR2YCrCb,
            "channels": ["Y", "Cr", "Cb"]
        }
    }
    
    def process(self, image: np.ndarray, target_space: str = "RGB", 
               analyze: bool = True) -> Dict[str, Any]:
        """Renk uzayı dönüşümü yapar ve renk analizini gerçekleştirir.
        
        Args:
            image: İşlenecek görüntü
            target_space: Hedef renk uzayı
            analyze: Renk analizi yapılsın mı
            
        Returns:
            Dict[str, Any]: İşlem sonuçları
        """
        if not self.validate_image(image):
            raise ValueError("Geçersiz görüntü formatı")
        
        if target_space not in self.COLOR_SPACES:
            raise ValueError(f"Desteklenmeyen renk uzayı: {target_space}")
        
        # Renk uzayı dönüşümü
        converted = self.convert_color_space(image, target_space)
        
        results = {
            "converted_image": converted,
            "color_space": target_space,
            "channels": self.COLOR_SPACES[target_space]["channels"]
        }
        
        # Renk analizi
        if analyze:
            # Analiz için orijinal görüntüyü kullan (BGR formatında)
            results.update(self.analyze_colors(image, "BGR"))
        
        return results
    
    def convert_color_space(self, image: np.ndarray, target_space: str) -> np.ndarray:
        """Görüntüyü hedef renk uzayına dönüştürür.
        
        Args:
            image: Dönüştürülecek görüntü
            target_space: Hedef renk uzayı
            
        Returns:
            np.ndarray: Dönüştürülmüş görüntü
        """
        # BGR'den hedef uzaya dönüşüm
        if target_space == "BGR":
            return image
        elif target_space == "RGB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(image, self.COLOR_SPACES[target_space]["code"])
    
    def analyze_colors(self, image: np.ndarray, color_space: str) -> Dict[str, Any]:
        """Renk analizi yapar.
        
        Args:
            image: Analiz edilecek görüntü
            color_space: Görüntünün renk uzayı
            
        Returns:
            Dict[str, Any]: Analiz sonuçları
        """
        results = {}
        
        # Kanal histogramları
        histograms = self.calculate_channel_histograms(image)
        results["histograms"] = histograms
        
        # Kanal istatistikleri
        stats = self.calculate_channel_statistics(image)
        results["statistics"] = stats
        
        # Renk korelasyonu
        if color_space in ["RGB", "BGR"]:
            correlation = self.calculate_color_correlation(image)
            results["correlation"] = correlation
        
        return results
    
    def calculate_channel_histograms(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Her kanal için histogram hesaplar.
        
        Args:
            image: Görüntü
            
        Returns:
            Dict[str, np.ndarray]: Kanal histogramları
        """
        histograms = {}
        for i, channel in enumerate(self.COLOR_SPACES[self.get_color_space(image)]["channels"]):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            histograms[channel] = hist.flatten()
        return histograms
    
    def calculate_channel_statistics(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Her kanal için istatistiksel değerleri hesaplar.
        
        Args:
            image: Görüntü
            
        Returns:
            Dict[str, Dict[str, float]]: Kanal istatistikleri
        """
        stats = {}
        color_space = self.get_color_space(image)
        channels = cv2.split(image)
        channel_names = self.COLOR_SPACES[color_space]["channels"]
        
        # BGR formatından diğer formatlara dönüşüm için kanal sıralamasını ayarla
        if color_space == "BGR":
            # OpenCV BGR formatında: channels[0] = B, channels[1] = G, channels[2] = R
            channel_mapping = {
                "B": channels[0],
                "G": channels[1],
                "R": channels[2]
            }
        elif color_space == "RGB":
            # RGB formatında: channels[0] = B, channels[1] = G, channels[2] = R -> R, G, B olarak değiştir
            channel_mapping = {
                "R": channels[2],
                "G": channels[1],
                "B": channels[0]
            }
        else:
            # Diğer renk uzayları için sıralı eşleştirme
            channel_mapping = dict(zip(channel_names, channels))
        
        # Her kanal için istatistikleri hesapla
        for name in channel_names:
            channel = channel_mapping[name]
            stats[name] = {
                "mean": float(np.mean(channel)),
                "std": float(np.std(channel)),
                "min": float(np.min(channel)),
                "max": float(np.max(channel)),
                "median": float(np.median(channel))
            }
        
        return stats
    
    def calculate_color_correlation(self, image: np.ndarray) -> Dict[str, float]:
        """Renk kanalları arasındaki korelasyonu hesaplar.
        
        Args:
            image: RGB/BGR görüntü
            
        Returns:
            Dict[str, float]: Kanal korelasyonları
        """
        channels = cv2.split(image)
        channel_names = self.COLOR_SPACES[self.get_color_space(image)]["channels"]
        
        correlations = {}
        for i in range(3):
            for j in range(i + 1, 3):
                key = f"{channel_names[i]}-{channel_names[j]}"
                corr = np.corrcoef(channels[i].flatten(), channels[j].flatten())[0, 1]
                correlations[key] = float(corr)
        
        return correlations
    
    def get_color_space(self, image: np.ndarray) -> str:
        """Görüntünün renk uzayını tahmin eder.
        
        Args:
            image: Görüntü
            
        Returns:
            str: Renk uzayı
        """
        # Bu basit bir tahmin, gerçek uygulamada daha karmaşık olabilir
        if len(image.shape) != 3:
            raise ValueError("Geçersiz görüntü formatı")
        
        if image.shape[2] != 3:
            raise ValueError("3 kanallı görüntü değil")
        
        return "BGR"  # OpenCV varsayılan olarak BGR kullanır
    
    @classmethod
    def get_supported_color_spaces(cls) -> Dict[str, Dict[str, Any]]:
        """Desteklenen renk uzaylarını döndürür.
        
        Returns:
            Dict[str, Dict[str, Any]]: Renk uzayı bilgileri
        """
        return cls.COLOR_SPACES 