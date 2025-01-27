import cv2
import numpy as np
from typing import Dict, Any, Tuple
import os
from .base_processor import BaseImageProcessor

class CompressionProcessor(BaseImageProcessor):
    """Görüntü sıkıştırma işlemcisi."""
    
    SUPPORTED_METHODS = {
        "JPEG": {
            "extension": ".jpg",
            "quality_range": (0, 100)
        },
        "JPEG2000": {
            "extension": ".jp2",
            "quality_range": (0, 100)
        },
        "PNG": {
            "extension": ".png",
            "quality_range": (0, 9)
        },
        "WEBP": {
            "extension": ".webp",
            "quality_range": (0, 100)
        }
    }
    
    def __init__(self):
        super().__init__()
        self.original_size = 0
        self._temp_files = []
    
    def initialize(self) -> None:
        """Initialize compression processor."""
        pass  # No initialization needed
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.log_error(e, "cleanup")
        self._temp_files.clear()
    
    def process(self, image: np.ndarray, method: str = "JPEG", quality: int = 75, 
               save_path: str = None) -> Dict[str, Any]:
        """Görüntüyü sıkıştırır.
        
        Args:
            image: Sıkıştırılacak görüntü
            method: Sıkıştırma yöntemi
            quality: Kalite faktörü
            save_path: Kaydedilecek dosya yolu (opsiyonel)
            
        Returns:
            Dict[str, Any]: Sıkıştırma sonuçları
        """
        # Validate input image
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        self.validate_input(image)
        
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Desteklenmeyen sıkıştırma yöntemi: {method}")
        
        # Kalite değerini kontrol et
        quality_range = self.SUPPORTED_METHODS[method]["quality_range"]
        if not quality_range[0] <= quality <= quality_range[1]:
            raise ValueError(f"Geçersiz kalite değeri. Aralık: {quality_range}")
        
        # Geçici dosya yolu oluştur
        temp_path = "temp_compressed" + self.SUPPORTED_METHODS[method]["extension"]
        
        # Görüntüyü sıkıştır
        compressed_image, file_size = self._compress_image(image, method, quality, temp_path)
        
        # İsteğe bağlı olarak kaydet
        if save_path:
            cv2.imwrite(save_path, compressed_image)
        
        # Sonuçları hazırla
        results = {
            "compressed_image": compressed_image,
            "original_size": self.original_size,
            "compressed_size": file_size,
            "compression_ratio": self.calculate_compression_ratio(file_size),
            "method": method,
            "quality": quality
        }
        
        return results
    
    def _compress_image(self, image: np.ndarray, method: str, quality: int, 
                       temp_path: str) -> Tuple[np.ndarray, int]:
        """Görüntüyü belirtilen yöntemle sıkıştırır.
        
        Args:
            image: Sıkıştırılacak görüntü
            method: Sıkıştırma yöntemi
            quality: Kalite faktörü
            temp_path: Geçici dosya yolu
            
        Returns:
            Tuple[np.ndarray, int]: Sıkıştırılmış görüntü ve dosya boyutu
        """
        # Orijinal boyutu hesapla
        self.original_size = image.nbytes
        
        # Yönteme göre sıkıştırma parametrelerini ayarla
        if method == "JPEG":
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif method == "JPEG2000":
            params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality * 10]
        elif method == "PNG":
            params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
        elif method == "WEBP":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        
        # Görüntüyü sıkıştır ve kaydet
        cv2.imwrite(temp_path, image, params)
        self._temp_files.append(temp_path)  # Geçici dosyayı izle
        
        # Sıkıştırılmış görüntüyü oku ve boyutunu al
        compressed = cv2.imread(temp_path)
        file_size = os.path.getsize(temp_path)
        
        return compressed, file_size
    
    def calculate_compression_ratio(self, compressed_size: int) -> float:
        """Sıkıştırma oranını hesaplar.
        
        Args:
            compressed_size: Sıkıştırılmış dosya boyutu
            
        Returns:
            float: Sıkıştırma oranı (%)
        """
        if self.original_size == 0 or compressed_size == 0:
            return 0 if self.original_size == 0 else 100
        
        # Sıkıştırma oranı = (1 - sıkıştırılmış_boyut/orijinal_boyut) * 100
        return (1 - compressed_size / self.original_size) * 100
    
    @classmethod
    def get_supported_methods(cls) -> Dict[str, Dict[str, Any]]:
        """Desteklenen sıkıştırma yöntemlerini döndürür.
        
        Returns:
            Dict[str, Dict[str, Any]]: Yöntem bilgileri
        """
        return cls.SUPPORTED_METHODS 