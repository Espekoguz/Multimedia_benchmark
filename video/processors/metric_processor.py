import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import time
import psutil
from image.processors.metric_processor import MetricProcessor as ImageMetricProcessor
from .base_processor import BaseVideoProcessor
from .signal import Signal
import warnings
import os

class MetricProcessor(BaseVideoProcessor):
    """Video kalite metriklerini hesaplayan işlemci."""
    
    def __init__(self):
        super().__init__()
        self._device = None
        self._lpips_model = None
        self.initialize_lpips()
        self._parameters = {}
        self.available_metrics = ["psnr", "ssim", "lpips", "mse", "entropy"]
        self.metrics_update = Signal()
        self._is_running = False
        self._current_frame = 0
    
    def initialize_lpips(self):
        """LPIPS modelini yükler."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self._lpips_model = lpips.LPIPS(net='alex', verbose=False)
                if torch.cuda.is_available():
                    self._lpips_model = self._lpips_model.cuda()
        except Exception as e:
            self.log_error(e, "LPIPS initialization")
            self._lpips_model = None
    
    def set_parameters(self, test_video_path: str, reference_video_path: str, metrics: List[str] = None) -> None:
        """Set processing parameters.
        
        Args:
            test_video_path: Path to test video
            reference_video_path: Path to reference video
            metrics: List of metrics to calculate
        """
        if not os.path.exists(test_video_path):
            raise FileNotFoundError(f"Test video not found: {test_video_path}")
        if not os.path.exists(reference_video_path):
            raise FileNotFoundError(f"Reference video not found: {reference_video_path}")
        
        self._parameters = {
            "test_video_path": test_video_path,
            "reference_video_path": reference_video_path,
            "metrics": metrics or self.available_metrics.copy()
        }
    
    def start(self) -> None:
        """İşlemi başlatır."""
        if not self._parameters:
            raise ValueError("Parameters not set. Call set_parameters first.")
        self._is_running = True
        self._current_frame = 0
        self._process_video()
    
    def wait(self) -> None:
        """İşlemin bitmesini bekler."""
        while self._is_running:
            time.sleep(0.1)
    
    def _process_video(self) -> None:
        """Video işleme döngüsü."""
        try:
            test_cap = cv2.VideoCapture(self._parameters["test_video_path"])
            ref_cap = cv2.VideoCapture(self._parameters["reference_video_path"])
            
            while self._is_running:
                ret1, test_frame = test_cap.read()
                ret2, ref_frame = ref_cap.read()
                
                if not ret1 or not ret2:
                    break
                
                metrics = self.process_frame(test_frame, ref_frame)
                self.metrics_update.emit(metrics)
                self._current_frame += 1
            
            test_cap.release()
            ref_cap.release()
            
        except Exception as e:
            self.log_error(e, "video processing")
        finally:
            self._is_running = False
    
    def process_frame(self, frame: np.ndarray, reference: np.ndarray = None) -> Dict[str, Any]:
        """Process a single frame and calculate metrics.
        
        Args:
            frame: Input frame
            reference: Reference frame
            
        Returns:
            Dictionary containing calculated metrics
        """
        return self.process(frame, reference)
    
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    def process(self, frame: np.ndarray, reference: np.ndarray = None, metrics: List[str] = None) -> Dict[str, Any]:
        """Video karesi için metrikleri hesaplar.
        
        Args:
            frame: Test karesi
            reference: Referans kare (opsiyonel)
            metrics: Hesaplanacak metrikler listesi (opsiyonel)
            
        Returns:
            Dict[str, Any]: Hesaplanan metrikler
        """
        if not self.validate_frame(frame):
            raise ValueError("Geçersiz kare formatı")
        
        if reference is not None and not self.validate_frame(reference):
            raise ValueError("Geçersiz referans kare formatı")
        
        if metrics is None:
            metrics = self._parameters.get("metrics", self.available_metrics.copy())
        
        results = {}
        
        if reference is not None:
            if "mse" in metrics:
                results["mse"] = [self.calculate_mse(frame, reference)]
            
            if "psnr" in metrics:
                results["psnr"] = [self.calculate_psnr(frame, reference)]
            
            if "ssim" in metrics:
                results["ssim"] = [self.calculate_ssim(frame, reference)]
            
            if "lpips" in metrics and self._lpips_model is not None:
                results["lpips"] = [self.calculate_lpips(frame, reference)]
        
        # Entropi her zaman hesaplanabilir
        if "entropy" in metrics:
            results["entropy"] = [self.calculate_entropy(frame)]
        
        # Sonuçları yayınla
        self.metrics_update.emit(results)
        
        return results
    
    def _resize_if_needed(self, frame: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Gerekirse görüntüleri aynı boyuta getirir."""
        if frame.shape != reference.shape:
            # Referans görüntünün boyutlarına göre yeniden boyutlandır
            frame = cv2.resize(frame, (reference.shape[1], reference.shape[0]))
        return frame, reference
    
    def calculate_mse(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """MSE (Mean Squared Error) hesaplar."""
        try:
            frame, reference = self._resize_if_needed(frame, reference)
            return float(np.mean((frame.astype(float) - reference.astype(float)) ** 2))
        except Exception as e:
            self.log_error(e, "MSE calculation")
            return float('inf')
    
    def calculate_psnr(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """PSNR (Peak Signal-to-Noise Ratio) hesaplar."""
        try:
            frame, reference = self._resize_if_needed(frame, reference)
            mse = np.mean((frame.astype(float) - reference.astype(float)) ** 2)
            if mse == 0:
                return float('inf')
            return float(20 * np.log10(255.0) - 10 * np.log10(mse))
        except Exception as e:
            self.log_error(e, "PSNR calculation")
            return 0.0
    
    def calculate_ssim(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """SSIM (Structural Similarity Index) hesaplar."""
        try:
            frame, reference = self._resize_if_needed(frame, reference)
            return float(ssim(frame, reference, channel_axis=2))
        except Exception as e:
            self.log_error(e, "SSIM calculation")
            return 0.0
    
    def calculate_lpips(self, frame: np.ndarray, reference: np.ndarray) -> float:
        """LPIPS (Learned Perceptual Image Patch Similarity) hesaplar."""
        try:
            frame, reference = self._resize_if_needed(frame, reference)
            
            # Görüntüleri normalize et ve tensöre dönüştür
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            reference_tensor = torch.from_numpy(reference).float() / 255.0
            
            # Boyutları düzenle: (H, W, C) -> (1, C, H, W)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
            reference_tensor = reference_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # GPU'ya taşı
            frame_tensor = frame_tensor.to(self.device)
            reference_tensor = reference_tensor.to(self.device)
            
            # LPIPS hesapla
            with torch.no_grad():
                lpips_value = float(self._lpips_model(frame_tensor, reference_tensor))
            
            return lpips_value
            
        except Exception as e:
            self.log_error(e, "LPIPS calculation")
            return 0.0
    
    def calculate_entropy(self, frame: np.ndarray) -> float:
        """Görüntü entropisini hesaplar."""
        histogram = cv2.calcHist([frame], [0], None, [256], [0, 256])
        histogram = histogram.flatten() / np.sum(histogram)
        non_zero_hist = histogram[histogram > 0]
        return float(-np.sum(non_zero_hist * np.log2(non_zero_hist)))
    
    def stop(self) -> None:
        """İşlemi durdurur."""
        self._is_running = False
    
    def cleanup(self) -> None:
        """Kaynakları temizler."""
        if self._lpips_model is not None:
            del self._lpips_model
            self._lpips_model = None
    
    def log_error(self, error: Exception, context: str) -> None:
        """Hata mesajını loglar.
        
        Args:
            error: Hata nesnesi
            context: Hatanın oluştuğu bağlam
        """
        print(f"Error in {context}: {str(error)}")
        import traceback
        traceback.print_exc()

class VideoMetricProcessor(ImageMetricProcessor):
    """Processor for video quality metrics and analysis."""
    
    def __init__(self):
        super().__init__()
        self.metrics_history = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
            "mse": [],
            "entropy": [],
            "Histogram_Similarity": [],
            "Compression_Ratio": [],
            "Frame_Processing_Time": [],
            "Memory_Usage": []
        }
        self.frame_count = 0
        self.start_time = None
    
    def initialize(self) -> None:
        """Initialize video metric processor."""
        super().initialize()
        self.start_time = time.time()
        self.frame_count = 0
    
    def process(self, original_frame: np.ndarray, compressed_frame: np.ndarray, 
                frame_number: int) -> Dict[str, Any]:
        """Process a video frame and calculate metrics.
        
        Args:
            original_frame: Original video frame
            compressed_frame: Compressed video frame
            frame_number: Current frame number
            
        Returns:
            Dictionary containing calculated metrics
        """
        try:
            # Start frame processing time
            frame_start_time = time.time()
            
            # Calculate basic metrics using parent class
            metrics = super().process(original_frame, compressed_frame)
            
            # Add frame-specific metrics
            metrics.update({
                "Frame_Number": frame_number,
                "Frame_Processing_Time": time.time() - frame_start_time,
                "Memory_Usage": self.get_memory_usage(),
                "Total_Processing_Time": time.time() - self.start_time
            })
            
            # Update metrics history
            self._update_metrics_history(metrics)
            
            # Calculate frame rate
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            metrics["FPS"] = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            return metrics
            
        except Exception as e:
            self.log_error(e, f"frame {frame_number} processing")
            return {
                "Frame_Number": frame_number,
                "Error": str(e)
            }
    
    def process_frame(self, frame: np.ndarray, reference: np.ndarray = None) -> Dict[str, Any]:
        """Process a single frame and calculate metrics.
        
        Args:
            frame: Input frame
            reference: Reference frame
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = super().process(frame, reference)
        self.frame_count += 1
        
        # Metrikleri geçmişe ekle
        for metric_name, value in metrics.items():
            if metric_name in self.metrics_history:
                self.metrics_history[metric_name].extend(value)
        
        return metrics
    
    def _update_metrics_history(self, metrics: Dict[str, Any]) -> None:
        """Update metrics history with new values."""
        try:
            for key in self.metrics_history.keys():
                if key in metrics:
                    self.metrics_history[key].append(metrics[key])
        except Exception as e:
            self.log_error(e, "metrics history update")
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics."""
        try:
            summary = {}
            
            for metric_name, values in self.metrics_history.items():
                if values:  # Only process non-empty lists
                    values_array = np.array(values)
                    summary[metric_name] = {
                        "mean": float(np.mean(values_array)),
                        "std": float(np.std(values_array)),
                        "min": float(np.min(values_array)),
                        "max": float(np.max(values_array)),
                        "median": float(np.median(values_array))
                    }
            
            # Add overall statistics
            summary["Overall"] = {
                "Total_Frames": self.frame_count,
                "Total_Time": time.time() - self.start_time if self.start_time else 0,
                "Average_FPS": self.frame_count / (time.time() - self.start_time) 
                              if self.start_time else 0
            }
            
            return summary
            
        except Exception as e:
            self.log_error(e, "metrics summary")
            return {}
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get the complete metrics history."""
        return self.metrics_history
    
    def reset_metrics(self) -> None:
        """Reset all metrics history."""
        self.metrics_history = {key: [] for key in self.metrics_history.keys()}
        self.frame_count = 0
        self.start_time = time.time()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self.reset_metrics() 