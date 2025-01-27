import pytest
import numpy as np
import cv2
from image.processors.metric_processor import MetricProcessor

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def metric_processor():
    """Create a metric processor instance."""
    processor = MetricProcessor()
    processor.initialize()
    return processor

def test_metric_processor_initialization(metric_processor):
    """Test metric processor initialization."""
    assert metric_processor._initialized
    assert metric_processor._lpips_model is not None

def test_basic_metrics_calculation(metric_processor, test_image):
    """Test basic metrics calculation."""
    # Create a slightly modified version of the test image
    compressed = test_image.copy()
    compressed = cv2.GaussianBlur(compressed, (5, 5), 0)
    
    metrics = metric_processor.process(test_image, compressed)
    
    assert "MSE" in metrics
    assert "PSNR" in metrics
    assert "SSIM" in metrics
    assert "LPIPS" in metrics
    
    assert metrics["MSE"] >= 0
    assert metrics["PSNR"] >= 0
    assert 0 <= metrics["SSIM"] <= 1
    assert 0 <= metrics["LPIPS"] <= 1

def test_advanced_metrics_calculation(metric_processor, test_image):
    """Test advanced metrics calculation."""
    compressed = test_image.copy()
    compressed = cv2.GaussianBlur(compressed, (5, 5), 0)
    
    metrics = metric_processor.process(test_image, compressed, advanced=True)
    
    assert "MS-SSIM" in metrics
    assert "FSIM" in metrics
    assert "VIF" in metrics
    
    assert 0 <= metrics["MS-SSIM"] <= 1
    assert 0 <= metrics["FSIM"] <= 1
    assert metrics["VIF"] >= 0

def test_histogram_similarity(metric_processor, test_image):
    """Test histogram similarity calculation."""
    compressed = test_image.copy()
    compressed = cv2.GaussianBlur(compressed, (5, 5), 0)
    
    similarity = metric_processor.calculate_histogram_similarity(test_image, compressed)
    
    assert -1 <= similarity <= 1

def test_entropy_calculation(metric_processor, test_image):
    """Test entropy calculation."""
    entropy = metric_processor.calculate_entropy(test_image)
    
    assert entropy >= 0
    assert entropy <= 8  # Maximum entropy for 8-bit image

def test_error_handling(metric_processor):
    """Test error handling with invalid inputs."""
    invalid_image = np.random.rand(100, 100)  # 2D array instead of 3D
    
    with pytest.raises(ValueError):
        metric_processor.process(invalid_image, invalid_image)

def test_cleanup(metric_processor):
    """Test cleanup functionality."""
    metric_processor.cleanup()
    
    assert not metric_processor._initialized
    assert metric_processor._lpips_model is None 