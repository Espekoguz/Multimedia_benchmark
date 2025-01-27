import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import time
import psutil
from image.processors.metric_processor import MetricProcessor as ImageMetricProcessor

class VideoMetricProcessor(ImageMetricProcessor):
    """Processor for video quality metrics and analysis."""
    
    def __init__(self):
        super().__init__()
        self.metrics_history = {
            "PSNR": [],
            "SSIM": [],
            "LPIPS": [],
            "Histogram_Similarity": [],
            "Entropy": [],
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
    
    def _update_metrics_history(self, metrics: Dict[str, Any]) -> None:
        """Update metrics history with new values."""
        try:
            for key in self.metrics_history.keys():
                if key in metrics:
                    self.metrics_history[key].append(metrics[key])
        except Exception as e:
            self.log_error(e, "metrics history update")
    
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