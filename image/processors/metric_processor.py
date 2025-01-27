import numpy as np
import cv2
from typing import Dict, Any, Optional, List
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
from .base_processor import BaseImageProcessor

class MetricProcessor(BaseImageProcessor):
    """Processor for calculating various quality metrics."""
    
    AVAILABLE_METRICS = {
        "basic": ["MSE", "PSNR", "SSIM", "LPIPS"],
        "advanced": ["MS-SSIM", "FSIM", "VIF"]
    }
    
    def __init__(self):
        super().__init__()
        self._lpips_model = None
        self._initialized = False
    
    @property
    def available_metrics(self) -> Dict[str, List[str]]:
        """Return available metrics."""
        return self.AVAILABLE_METRICS.copy()
    
    def initialize(self) -> None:
        """Initialize LPIPS model and other resources."""
        if not self._initialized:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self._lpips_model = lpips.LPIPS(net='alex', verbose=False)
                    if torch.cuda.is_available():
                        self._lpips_model = self._lpips_model.cuda()
                self._initialized = True
                print("LPIPS modeli başarıyla yüklendi.")
            except Exception as e:
                self.log_error(e, "LPIPS initialization")
    
    def process(self, compressed: np.ndarray, reference: Optional[np.ndarray] = None, 
                metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Calculate metrics between reference and compressed images.
        
        Args:
            compressed: Compressed/processed image
            reference: Reference image (optional)
            metrics: List of metrics to calculate (optional)
            
        Returns:
            Dictionary containing calculated metrics
        """
        try:
            self.validate_input(compressed)
            if reference is not None:
                self.validate_input(reference)
                
                # Ensure same size
                if reference.shape != compressed.shape:
                    compressed = cv2.resize(compressed, (reference.shape[1], reference.shape[0]))
            
            result = {}
            
            # Determine which metrics to calculate
            if metrics is None:
                metrics = self.AVAILABLE_METRICS["basic"]
            
            # Calculate basic metrics if reference is available
            if reference is not None:
                if "MSE" in metrics:
                    mse = np.mean((reference.astype(float) - compressed.astype(float)) ** 2)
                    result["MSE"] = float(mse)
                
                if "PSNR" in metrics:
                    if "MSE" in result:
                        mse = result["MSE"]
                    else:
                        mse = np.mean((reference.astype(float) - compressed.astype(float)) ** 2)
                    result["PSNR"] = float('inf') if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
                
                if "SSIM" in metrics:
                    result["SSIM"] = float(ssim(reference, compressed, channel_axis=2, data_range=255))
                
                if "LPIPS" in metrics:
                    if self._initialized and self._lpips_model is not None:
                        result["LPIPS"] = self._calculate_lpips(reference, compressed)
                    else:
                        self.initialize()
                        if self._initialized:
                            result["LPIPS"] = self._calculate_lpips(reference, compressed)
                        else:
                            result["LPIPS"] = 1.0
            
            # Calculate advanced metrics if requested and reference is available
            advanced_metrics = set(metrics) & set(self.AVAILABLE_METRICS["advanced"])
            if advanced_metrics and reference is not None:
                result.update(self._calculate_advanced_metrics(reference, compressed))
            
            # Calculate reference-free metrics
            if "Entropy" in metrics:
                result["Entropy"] = self.calculate_entropy(compressed)
            
            return result
            
        except Exception as e:
            self.log_error(e, "metric calculation")
            return {}
    
    def _calculate_lpips(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """Calculate LPIPS perceptual similarity."""
        try:
            # Convert BGR to RGB
            orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            comp_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
            
            # Normalize to [-1, 1]
            orig_tensor = torch.from_numpy(orig_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
            comp_tensor = torch.from_numpy(comp_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
            
            # Move to GPU if available
            if torch.cuda.is_available():
                orig_tensor = orig_tensor.cuda()
                comp_tensor = comp_tensor.cuda()
            
            # Calculate LPIPS
            with torch.no_grad():
                lpips_value = float(self._lpips_model(orig_tensor, comp_tensor).item())
            
            return lpips_value
            
        except Exception as e:
            self.log_error(e, "LPIPS calculation")
            return 1.0
    
    def _calculate_advanced_metrics(self, original: np.ndarray, 
                                  compressed: np.ndarray) -> Dict[str, float]:
        """Calculate advanced image quality metrics."""
        try:
            metrics = {}
            
            # MS-SSIM
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            comp_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
            metrics["MS-SSIM"] = float(ssim(orig_gray, comp_gray, data_range=255))
            
            # FSIM (Feature Similarity)
            orig_edges = cv2.Canny(orig_gray, 100, 200)
            comp_edges = cv2.Canny(comp_gray, 100, 200)
            metrics["FSIM"] = float(ssim(orig_edges, comp_edges, data_range=255))
            
            # VIF (Visual Information Fidelity)
            orig_gray = orig_gray.astype(np.float32)
            comp_gray = comp_gray.astype(np.float32)
            diff = orig_gray - comp_gray
            var_orig = np.var(orig_gray)
            var_diff = np.var(diff)
            vif = var_orig/(var_diff + 1e-8)
            metrics["VIF"] = float(1.0/(1.0 + 1.0/vif))
            
            return metrics
            
        except Exception as e:
            self.log_error(e, "advanced metrics calculation")
            return {
                "MS-SSIM": 0.0,
                "FSIM": 0.0,
                "VIF": 0.0
            }
    
    def calculate_histogram_similarity(self, img1: np.ndarray, 
                                    img2: np.ndarray) -> float:
        """Calculate histogram similarity between two images."""
        try:
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
            
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            
        except Exception as e:
            self.log_error(e, "histogram similarity calculation")
            return 0.0
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy."""
        try:
            entropy = 0
            for i in range(3):  # For each color channel
                channel = image[:, :, i]
                histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
                histogram = histogram.flatten() / np.sum(histogram)
                non_zero_hist = histogram[histogram > 0]
                entropy += -np.sum(non_zero_hist * np.log2(non_zero_hist))
            
            return float(entropy / 3)  # Average entropy
            
        except Exception as e:
            self.log_error(e, "entropy calculation")
            return 0.0
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._lpips_model is not None:
            del self._lpips_model
            self._lpips_model = None
        self._initialized = False 