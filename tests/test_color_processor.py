import pytest
import numpy as np
import cv2
from image.processors.color_processor import ColorProcessor

@pytest.fixture
def test_image():
    """Create a test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def color_processor():
    """Create a color processor instance."""
    processor = ColorProcessor()
    processor.initialize()
    return processor

def test_color_space_conversion(color_processor, test_image):
    """Test color space conversion."""
    for target_space in ColorProcessor.COLOR_SPACES.keys():
        processed, metrics = color_processor.process(test_image, target_space=target_space)
        
        assert processed is not None
        assert processed.shape == test_image.shape
        if target_space != "GRAY":  # GRAY will have different shape
            assert processed.shape[2] == 3

def test_color_analysis(color_processor, test_image):
    """Test color analysis functionality."""
    processed, metrics = color_processor.process(test_image, analyze=True)
    
    # Check color distribution metrics
    for channel in ["Blue", "Green", "Red"]:
        assert f"{channel}_Mean" in metrics
        assert f"{channel}_Std" in metrics
        assert f"{channel}_Distribution" in metrics
        
        assert 0 <= metrics[f"{channel}_Mean"] <= 255
        assert metrics[f"{channel}_Std"] >= 0
        assert len(metrics[f"{channel}_Distribution"]) == 256

def test_color_correlation(color_processor, test_image):
    """Test color correlation calculation."""
    processed, metrics = color_processor.process(test_image, analyze=True)
    
    correlations = metrics["Color_Correlations"]
    assert "R-G" in correlations
    assert "R-B" in correlations
    assert "G-B" in correlations
    
    for value in correlations.values():
        assert -1 <= value <= 1

def test_color_moments(color_processor, test_image):
    """Test color moments calculation."""
    processed, metrics = color_processor.process(test_image, analyze=True)
    
    moments = metrics["Color_Moments"]
    for channel in ["First", "Second", "Third"]:
        assert channel in moments
        assert len(moments[channel]) == 4  # mean, std, skewness, kurtosis

def test_dominant_colors(color_processor, test_image):
    """Test dominant colors extraction."""
    processed, metrics = color_processor.process(test_image, analyze=True)
    
    dominant_colors = metrics["Dominant_Colors"]
    assert len(dominant_colors) == 5  # Default n_colors
    
    for color in dominant_colors:
        assert len(color) == 3  # RGB values
        assert all(0 <= c <= 255 for c in color)

def test_error_handling(color_processor):
    """Test error handling with invalid inputs."""
    invalid_image = np.random.rand(100, 100)  # 2D array instead of 3D
    
    with pytest.raises(ValueError):
        color_processor.process(invalid_image)
    
    with pytest.raises(ValueError):
        color_processor.process(np.random.rand(100, 100, 3), target_space="INVALID")

def test_cleanup(color_processor):
    """Test cleanup functionality."""
    color_processor.cleanup()
    # No specific cleanup needed for ColorProcessor, but should not raise errors 