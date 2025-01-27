from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple

class BaseProcessor(ABC):
    """Base class for all processors in the multimedia benchmark tool."""
    
    def __init__(self):
        self._device = None
        self._initialized = False
    
    @property
    def device(self):
        """Get the current device (CPU/GPU)."""
        if self._device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the processor with necessary models and resources."""
        pass
    
    @abstractmethod
    def process(self, data: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process the input data and return processed data with metrics."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the processor."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def validate_input(self, data: np.ndarray) -> bool:
        """Validate input data."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if len(data.shape) not in [2, 3]:
            raise ValueError("Input must be 2D or 3D array")
        return True
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log an error with context."""
        import traceback
        print(f"Error in {self.__class__.__name__} - {context}: {str(error)}")
        traceback.print_exc() 

class BaseImageProcessor(BaseProcessor):
    """Base class for image processors."""
    
    def validate_input(self, data: np.ndarray) -> bool:
        """Validate image input data."""
        super().validate_input(data)
        if len(data.shape) == 2:  # Grayscale image
            return True
        if len(data.shape) == 3 and data.shape[2] in [1, 3, 4]:  # RGB or RGBA
            return True
        raise ValueError("Invalid image format. Must be grayscale, RGB, or RGBA")