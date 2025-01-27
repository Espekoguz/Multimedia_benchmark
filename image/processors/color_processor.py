import numpy as np
import cv2
from typing import Dict, Any, Tuple, List
from .base_processor import BaseImageProcessor

class ColorProcessor(BaseImageProcessor):
    """Processor for color-related operations and analysis."""
    
    COLOR_SPACES = {
        "BGR": cv2.COLOR_BGR2BGR,  # No conversion
        "RGB": cv2.COLOR_BGR2RGB,
        "HSV": cv2.COLOR_BGR2HSV,
        "LAB": cv2.COLOR_BGR2LAB,
        "YUV": cv2.COLOR_BGR2YUV,
        "YCrCb": cv2.COLOR_BGR2YCrCb,
        "GRAY": cv2.COLOR_BGR2GRAY
    }
    
    def __init__(self):
        super().__init__()
        self._current_color_space = "BGR"
    
    def initialize(self) -> None:
        """Initialize color processor."""
        pass  # No initialization needed
    
    def get_supported_color_spaces(self) -> Dict[str, int]:
        """Return supported color spaces."""
        return self.COLOR_SPACES.copy()
    
    def process(self, image: np.ndarray, target_space: str = "BGR", 
                analyze: bool = False) -> Dict[str, Any]:
        """Process image color space and analyze color properties.
        
        Args:
            image: Input image in BGR format
            target_space: Target color space
            analyze: Whether to perform color analysis
            
        Returns:
            Dictionary containing processed image and metrics
        """
        try:
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
                    conversion_code = self.COLOR_SPACES[target_space]
                    processed = cv2.cvtColor(image, conversion_code)
                
                self._current_color_space = target_space
            else:
                processed = image.copy()
            
            result["converted_image"] = processed
            
            # Perform color analysis if requested
            if analyze:
                result.update(self._analyze_colors(processed, target_space))
            
            return result
            
        except Exception as e:
            self.log_error(e, "color processing")
            return {"converted_image": image.copy()}
    
    def _analyze_colors(self, image: np.ndarray, color_space: str) -> Dict[str, Any]:
        """Analyze color properties of the image."""
        try:
            metrics = {}
            
            # Calculate color distribution
            if color_space in ["BGR", "RGB"]:
                channels = ["Blue", "Green", "Red"] if color_space == "BGR" else ["Red", "Green", "Blue"]
                for i, channel in enumerate(channels):
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                    metrics[f"{channel}_Mean"] = float(np.mean(image[:, :, i]))
                    metrics[f"{channel}_Std"] = float(np.std(image[:, :, i]))
                    metrics[f"{channel}_Distribution"] = hist.flatten().tolist()
            
            # Calculate color correlation
            if color_space in ["BGR", "RGB"]:
                correlations = self._calculate_color_correlation(image)
                metrics["Color_Correlations"] = correlations
            
            # Calculate color moments
            moments = self._calculate_color_moments(image)
            metrics["Color_Moments"] = moments
            
            # Calculate dominant colors
            dominant_colors = self._find_dominant_colors(image, n_colors=5)
            metrics["Dominant_Colors"] = dominant_colors
            
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
            
            for i, channel in enumerate(["First", "Second", "Third"]):
                pixels = image[:, :, i].flatten()
                
                mean = float(np.mean(pixels))
                std = float(np.std(pixels))
                skewness = float(np.mean(((pixels - mean) / std) ** 3)) if std != 0 else 0
                kurtosis = float(np.mean(((pixels - mean) / std) ** 4)) if std != 0 else 0
                
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
            # Reshape image
            pixels = image.reshape(-1, 3)
            
            # Convert to float32
            pixels = np.float32(pixels)
            
            # Define criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Apply k-means
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, 
                                          cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to integers
            centers = np.uint8(centers)
            
            # Count occurrences of each label
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Sort colors by frequency
            sorted_indices = np.argsort(-counts)
            sorted_colors = centers[sorted_indices].tolist()
            
            return sorted_colors
            
        except Exception as e:
            self.log_error(e, "dominant colors")
            return [[0, 0, 0]] * n_colors
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass  # No cleanup needed 