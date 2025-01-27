import pytest
import numpy as np
import cv2
from video.processors.metric_processor import VideoMetricProcessor

@pytest.fixture
def test_frames():
    """Create test video frames."""
    frames = []
    for _ in range(10):  # Create 10 test frames
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        frames.append(frame)
    return frames

@pytest.fixture
def video_processor():
    """Create a video processor instance."""
    processor = VideoMetricProcessor()
    processor.initialize()
    return processor

def test_video_processor_initialization(video_processor):
    """Test video processor initialization."""
    assert video_processor._initialized
    assert video_processor._lpips_model is not None
    assert video_processor.start_time is not None
    assert video_processor.frame_count == 0

def test_frame_processing(video_processor, test_frames):
    """Test frame processing."""
    original = test_frames[0]
    compressed = cv2.GaussianBlur(original.copy(), (5, 5), 0)
    
    metrics = video_processor.process(original, compressed, frame_number=0)
    
    # Check basic metrics
    assert "Frame_Number" in metrics
    assert "Frame_Processing_Time" in metrics
    assert "Memory_Usage" in metrics
    assert "Total_Processing_Time" in metrics
    assert "FPS" in metrics
    
    # Check inherited metrics
    assert "PSNR" in metrics
    assert "SSIM" in metrics
    assert "LPIPS" in metrics
    
    # Check values
    assert metrics["Frame_Number"] == 0
    assert metrics["Frame_Processing_Time"] > 0
    assert metrics["Memory_Usage"] > 0
    assert metrics["Total_Processing_Time"] > 0
    assert metrics["FPS"] >= 0

def test_metrics_history(video_processor, test_frames):
    """Test metrics history tracking."""
    for i, frame in enumerate(test_frames):
        compressed = cv2.GaussianBlur(frame.copy(), (5, 5), 0)
        video_processor.process(frame, compressed, frame_number=i)
    
    history = video_processor.get_metrics_history()
    
    # Check history structure
    assert "PSNR" in history
    assert "SSIM" in history
    assert "LPIPS" in history
    assert "Frame_Processing_Time" in history
    
    # Check history length
    assert len(history["PSNR"]) == len(test_frames)
    assert len(history["SSIM"]) == len(test_frames)
    assert len(history["LPIPS"]) == len(test_frames)

def test_metrics_summary(video_processor, test_frames):
    """Test metrics summary calculation."""
    for i, frame in enumerate(test_frames):
        compressed = cv2.GaussianBlur(frame.copy(), (5, 5), 0)
        video_processor.process(frame, compressed, frame_number=i)
    
    summary = video_processor.get_metrics_summary()
    
    # Check summary structure
    for metric_name in video_processor.metrics_history.keys():
        if metric_name in summary:
            assert "mean" in summary[metric_name]
            assert "std" in summary[metric_name]
            assert "min" in summary[metric_name]
            assert "max" in summary[metric_name]
            assert "median" in summary[metric_name]
    
    # Check overall statistics
    assert "Overall" in summary
    assert "Total_Frames" in summary["Overall"]
    assert "Total_Time" in summary["Overall"]
    assert "Average_FPS" in summary["Overall"]
    
    assert summary["Overall"]["Total_Frames"] == len(test_frames)
    assert summary["Overall"]["Total_Time"] > 0
    assert summary["Overall"]["Average_FPS"] > 0

def test_reset_metrics(video_processor, test_frames):
    """Test metrics reset functionality."""
    # Process some frames
    for i, frame in enumerate(test_frames):
        compressed = cv2.GaussianBlur(frame.copy(), (5, 5), 0)
        video_processor.process(frame, compressed, frame_number=i)
    
    # Reset metrics
    video_processor.reset_metrics()
    
    # Check if metrics are reset
    history = video_processor.get_metrics_history()
    for metric_list in history.values():
        assert len(metric_list) == 0
    
    assert video_processor.frame_count == 0
    assert video_processor.start_time is not None

def test_cleanup(video_processor):
    """Test cleanup functionality."""
    video_processor.cleanup()
    
    assert not video_processor._initialized
    assert video_processor._lpips_model is None
    assert len(video_processor.metrics_history["PSNR"]) == 0
    assert video_processor.frame_count == 0 