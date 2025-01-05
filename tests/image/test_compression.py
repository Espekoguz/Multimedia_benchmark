import unittest
import numpy as np
import cv2
import os
from image.processors.compression_processor import CompressionProcessor

class TestCompressionProcessor(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlık."""
        self.processor = CompressionProcessor()
        
        # Test görüntüsü oluştur
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_supported_methods(self):
        """Desteklenen sıkıştırma yöntemlerini test eder."""
        methods = self.processor.get_supported_methods()
        
        self.assertIsInstance(methods, dict)
        self.assertGreater(len(methods), 0)
        
        for method, info in methods.items():
            self.assertIn("extension", info)
            self.assertIn("quality_range", info)
            self.assertEqual(len(info["quality_range"]), 2)
    
    def test_compression_jpeg(self):
        """JPEG sıkıştırmasını test eder."""
        results = self.processor.process(
            self.test_image,
            method="JPEG",
            quality=75
        )
        
        self.assertIn("compressed_image", results)
        self.assertIn("original_size", results)
        self.assertIn("compressed_size", results)
        self.assertIn("compression_ratio", results)
        
        self.assertIsInstance(results["compressed_image"], np.ndarray)
        self.assertEqual(results["compressed_image"].shape, self.test_image.shape)
        self.assertGreater(results["compression_ratio"], 0)
    
    def test_compression_with_save(self):
        """Sıkıştırılmış görüntüyü kaydetmeyi test eder."""
        save_path = "test_compressed.jpg"
        
        try:
            results = self.processor.process(
                self.test_image,
                method="JPEG",
                quality=75,
                save_path=save_path
            )
            
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)
            
        finally:
            # Test dosyasını temizle
            if os.path.exists(save_path):
                os.remove(save_path)
    
    def test_invalid_method(self):
        """Geçersiz sıkıştırma yöntemini test eder."""
        with self.assertRaises(ValueError):
            self.processor.process(
                self.test_image,
                method="INVALID_METHOD"
            )
    
    def test_invalid_quality(self):
        """Geçersiz kalite değerini test eder."""
        with self.assertRaises(ValueError):
            self.processor.process(
                self.test_image,
                method="JPEG",
                quality=101
            )
    
    def test_invalid_image(self):
        """Geçersiz görüntü formatını test eder."""
        invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # 2D array
        
        with self.assertRaises(ValueError):
            self.processor.process(invalid_image)
    
    def test_compression_ratio_calculation(self):
        """Sıkıştırma oranı hesaplamasını test eder."""
        # Test için boyutları ayarla
        self.processor.original_size = 1000
        compressed_size = 500
        
        ratio = self.processor.calculate_compression_ratio(compressed_size)
        self.assertEqual(ratio, 50.0)
        
        # Sıfır boyutlu dosya
        ratio = self.processor.calculate_compression_ratio(0)
        self.assertEqual(ratio, 100.0)
        
        # Orijinal boyut sıfır
        self.processor.original_size = 0
        ratio = self.processor.calculate_compression_ratio(100)
        self.assertEqual(ratio, 0.0)

if __name__ == '__main__':
    unittest.main() 