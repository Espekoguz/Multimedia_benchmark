import unittest
import numpy as np
import cv2
from image.processors.color_processor import ColorProcessor

class TestColorProcessor(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlık."""
        self.processor = ColorProcessor()
        
        # Test görüntüsü oluştur
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_supported_color_spaces(self):
        """Desteklenen renk uzaylarını test eder."""
        color_spaces = self.processor.get_supported_color_spaces()
        
        self.assertIsInstance(color_spaces, dict)
        self.assertGreater(len(color_spaces), 0)
        
        for space, info in color_spaces.items():
            self.assertIn("code", info)
            self.assertIn("channels", info)
            self.assertEqual(len(info["channels"]), 3)
    
    def test_color_space_conversion(self):
        """Renk uzayı dönüşümlerini test eder."""
        for target_space in ["RGB", "HSV", "LAB", "YUV", "YCrCb"]:
            results = self.processor.process(
                self.test_image,
                target_space=target_space
            )
            
            self.assertIn("converted_image", results)
            self.assertIn("color_space", results)
            self.assertIn("channels", results)
            
            converted = results["converted_image"]
            self.assertEqual(converted.shape, self.test_image.shape)
            self.assertEqual(results["color_space"], target_space)
    
    def test_color_analysis(self):
        """Renk analizini test eder."""
        results = self.processor.process(self.test_image, analyze=True)
        
        # Histogram analizi
        self.assertIn("histograms", results)
        histograms = results["histograms"]
        self.assertEqual(len(histograms), 3)  # 3 kanal
        for hist in histograms.values():
            self.assertEqual(len(hist), 256)  # 256 bin
        
        # İstatistikler
        self.assertIn("statistics", results)
        stats = results["statistics"]
        self.assertEqual(len(stats), 3)  # 3 kanal
        for channel_stats in stats.values():
            self.assertIn("mean", channel_stats)
            self.assertIn("std", channel_stats)
            self.assertIn("min", channel_stats)
            self.assertIn("max", channel_stats)
            self.assertIn("median", channel_stats)
        
        # Korelasyon (RGB/BGR için)
        self.assertIn("correlation", results)
        correlation = results["correlation"]
        self.assertEqual(len(correlation), 3)  # 3 kanal çifti
    
    def test_no_analysis(self):
        """Analiz yapılmadan işlemi test eder."""
        results = self.processor.process(self.test_image, analyze=False)
        
        self.assertNotIn("histograms", results)
        self.assertNotIn("statistics", results)
        self.assertNotIn("correlation", results)
    
    def test_invalid_color_space(self):
        """Geçersiz renk uzayını test eder."""
        with self.assertRaises(ValueError):
            self.processor.process(
                self.test_image,
                target_space="INVALID_SPACE"
            )
    
    def test_invalid_image(self):
        """Geçersiz görüntü formatını test eder."""
        invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # 2D array
        
        with self.assertRaises(ValueError):
            self.processor.process(invalid_image)
    
    def test_channel_statistics(self):
        """Kanal istatistiklerinin hesaplanmasını test eder."""
        # Bilinen değerlere sahip test görüntüsü
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        test_image[:, :, 0] = 100  # B kanalı
        test_image[:, :, 1] = 150  # G kanalı
        test_image[:, :, 2] = 200  # R kanalı
        
        results = self.processor.process(test_image, analyze=True)
        stats = results["statistics"]
        
        # B kanalı
        self.assertEqual(stats["B"]["mean"], 100)
        self.assertEqual(stats["B"]["min"], 100)
        self.assertEqual(stats["B"]["max"], 100)
        self.assertEqual(stats["B"]["std"], 0)
        
        # G kanalı
        self.assertEqual(stats["G"]["mean"], 150)
        self.assertEqual(stats["G"]["min"], 150)
        self.assertEqual(stats["G"]["max"], 150)
        self.assertEqual(stats["G"]["std"], 0)
        
        # R kanalı
        self.assertEqual(stats["R"]["mean"], 200)
        self.assertEqual(stats["R"]["min"], 200)
        self.assertEqual(stats["R"]["max"], 200)
        self.assertEqual(stats["R"]["std"], 0)
    
    def test_color_correlation(self):
        """Renk korelasyonu hesaplamasını test eder."""
        # Tamamen korelasyonlu kanallar
        test_image = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(10):
            test_image[i, :, :] = i * 25
        
        results = self.processor.process(test_image, analyze=True)
        correlation = results["correlation"]
        
        # Tüm kanallar arasında tam korelasyon olmalı
        for corr in correlation.values():
            self.assertAlmostEqual(abs(corr), 1.0, places=5)

if __name__ == '__main__':
    unittest.main() 