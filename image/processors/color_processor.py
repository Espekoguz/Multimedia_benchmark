import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
from .base_processor import BaseImageProcessor
from sklearn.cluster import KMeans
from scipy.stats import skew, kurtosis

class ColorProcessor(BaseImageProcessor):
    """Processor for color-related operations and analysis."""
    
    COLOR_SPACES = {
        "BGR": {
            "code": None,  # No conversion needed
            "channels": ["B", "G", "R"]
        },
        "RGB": {
            "code": cv2.COLOR_BGR2RGB,
            "channels": ["R", "G", "B"]
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
    
    def __init__(self):
        super().__init__()
        self._current_color_space = "BGR"
    
    def initialize(self) -> None:
        """Initialize color processor."""
        pass  # No initialization needed
    
    def get_supported_color_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Return supported color spaces."""
        return self.COLOR_SPACES.copy()
    
    def process(self, image: np.ndarray, target_space: str = "BGR", analyze: bool = False) -> Dict[str, Any]:
        """Process image color space and analyze color properties.
        
        Args:
            image: Input image in BGR format
            target_space: Target color space
            analyze: Whether to perform color analysis
            
        Returns:
            Dictionary containing processed image and metrics
        """
        # Input validation
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input must be a 3-channel color image")
        if image.dtype != np.uint8:
            raise ValueError("Input must be an 8-bit image")
        self.validate_input(image)
        
        result = {}
        
        # Convert color space if needed
        if target_space != self._current_color_space:
            if target_space not in self.COLOR_SPACES:
                raise ValueError(f"Unsupported color space: {target_space}")
            
            # Convert to target space
            if target_space == "BGR":
                # Convert back to BGR from current space
                inverse_conversion = getattr(cv2, f"COLOR_{self._current_color_space}2BGR")
                processed = cv2.cvtColor(image, inverse_conversion)
            else:
                conversion_code = self.COLOR_SPACES[target_space]["code"]
                processed = cv2.cvtColor(image, conversion_code)
            
            self._current_color_space = target_space
        else:
            processed = image.copy()
        
        # Split channels
        channels = cv2.split(processed)
        channel_names = self.COLOR_SPACES[self._current_color_space]["channels"]
        
        result.update({
            "converted_image": processed,
            "color_space": self._current_color_space,
            "channels": dict(zip(channel_names, channels))
        })
        
        # Perform color analysis if requested
        if analyze:
            result.update(self._analyze_colors(processed))
        
        return result
    
    def _analyze_colors(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze color properties of the image."""
        try:
            metrics = {}
            channels = cv2.split(image)
            channel_names = self.COLOR_SPACES[self._current_color_space]["channels"]
            
            # Calculate histograms
            histograms = {}
            for i, channel in enumerate(channel_names):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms[channel] = hist.flatten().tolist()
            metrics["histograms"] = histograms
            
            # Calculate statistics
            stats = {}
            for i, channel in enumerate(channel_names):
                channel_data = channels[i]
                stats[channel] = {
                    "mean": float(np.mean(channel_data)),
                    "std": float(np.std(channel_data)),
                    "min": float(np.min(channel_data)),
                    "max": float(np.max(channel_data)),
                    "median": float(np.median(channel_data))
                }
            metrics["statistics"] = stats
            
            # Calculate color correlation
            correlations = {}
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    key = f"{channel_names[i]}-{channel_names[j]}"
                    corr = float(np.corrcoef(channels[i].flatten(), channels[j].flatten())[0, 1])
                    correlations[key] = corr
            metrics["correlation"] = correlations
            
            return metrics
            
        except Exception as e:
            self.log_error(e, "color analysis")
            return {}
    
    def _calculate_color_correlation(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate correlation between color channels."""
        try:
            b, g, r = cv2.split(image)
            
            corr_rg = float(np.corrcoef(r.flatten(), g.flatten())[0, 1])
            corr_rb = float(np.corrcoef(r.flatten(), b.flatten())[0, 1])
            corr_gb = float(np.corrcoef(g.flatten(), b.flatten())[0, 1])
            
            return {
                "R-G": corr_rg,
                "R-B": corr_rb,
                "G-B": corr_gb
            }
            
        except Exception as e:
            self.log_error(e, "color correlation")
            return {"R-G": 0.0, "R-B": 0.0, "G-B": 0.0}
    
    def _calculate_color_moments(self, image: np.ndarray) -> Dict[str, List[float]]:
        """Calculate color moments (mean, std, skewness, kurtosis)."""
        try:
            moments = {}
            
            for i, channel in enumerate(self.COLOR_SPACES[self._current_color_space]["channels"]):
                pixels = image[:, :, i].astype(float)
                mean = float(np.mean(pixels))
                std = float(np.std(pixels))
                skewness = float(skew(pixels.flatten()))
                kurtosis = float(kurtosis(pixels.flatten()))
                moments[channel] = [mean, std, skewness, kurtosis]
            
            return moments
            
        except Exception as e:
            self.log_error(e, "color moments")
            return {
                "First": [0.0, 0.0, 0.0, 0.0],
                "Second": [0.0, 0.0, 0.0, 0.0],
                "Third": [0.0, 0.0, 0.0, 0.0]
            }
    
    def _find_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[List[int]]:
        """Find dominant colors using k-means clustering."""
        try:
            pixels = image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()
            
        except Exception as e:
            self.log_error(e, "dominant colors")
            return [[0, 0, 0]] * n_colors
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass  # No cleanup needed 