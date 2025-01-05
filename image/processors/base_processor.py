from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict

class BaseImageProcessor(ABC):
    """Temel görüntü işleme sınıfı."""
    
    def __init__(self):
        self.image = None
        self.metadata = {}
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Görüntü işleme metodunu uygular.
        
        Args:
            image: İşlenecek görüntü
            **kwargs: Ek parametreler
            
        Returns:
            Dict[str, Any]: İşlem sonuçları
        """
        pass
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Görüntünün geçerliliğini kontrol eder.
        
        Args:
            image: Kontrol edilecek görüntü
            
        Returns:
            bool: Görüntü geçerli mi
        """
        if image is None:
            return False
        if not isinstance(image, np.ndarray):
            return False
        if len(image.shape) != 3:
            return False
        if image.shape[2] != 3:  # BGR/RGB kontrolü
            return False
        return True
    
    def update_metadata(self, key: str, value: Any):
        """Metadata'ya yeni bilgi ekler.
        
        Args:
            key: Metadata anahtarı
            value: Metadata değeri
        """
        self.metadata[key] = value 