import numpy as np
from typing import Any, Dict, Generator, Tuple
import cv2

class BaseVideoProcessor:
    """Video işleme temel sınıfı."""
    
    def __init__(self):
        """Sınıfı başlatır."""
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.duration = 0
        
    def open(self, video_path: str) -> bool:
        """Video dosyasını açar.
        
        Args:
            video_path: Video dosyası yolu
            
        Returns:
            bool: Başarılı ise True
        """
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return False
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps
        
        return True
    
    def close(self) -> None:
        """Kaynakları serbest bırakır."""
        if self.cap is not None:
            self.cap.release()
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        """Kare formatını doğrular.
        
        Args:
            frame: Doğrulanacak kare
            
        Returns:
            bool: Geçerli ise True
        """
        if frame is None:
            return False
        
        if len(frame.shape) != 3:
            return False
        
        if frame.shape[2] != 3:
            return False
        
        return True 