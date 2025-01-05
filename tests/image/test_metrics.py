import unittest
import numpy as np
import torch
from image.processors.metric_processor import MetricProcessor

class TestMetricProcessor(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlık."""
        self.processor = MetricProcessor()
        
        # Test görüntüleri oluştur
        self.reference_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_image = self.reference_image.copy()
        
        # Test görüntüsüne gürültü ekle
        noise = np.random.normal(0, 25, self.test_image.shape).astype(np.uint8)
        self.test_image = np.clip(self.test_image + noise, 0, 255).astype(np.uint8)
    
    def test_available_metrics(self):
        """Kullanılabilir metrikleri test eder."""
        metrics = self.processor.available_metrics
        
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)
        self.assertIn("PSNR", metrics)
        self.assertIn("SSIM", metrics)
        self.assertIn("LPIPS", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("Entropy", metrics)
    
    def test_metrics_calculation(self):
        """Metrik hesaplamalarını test eder."""
        results = self.processor.process(
            self.test_image,
            reference=self.reference_image
        )
        
        # Tüm metrikler hesaplanmış mı
        self.assertIn("MSE", results)
        self.assertIn("PSNR", results)
        self.assertIn("SSIM", results)
        self.assertIn("LPIPS", results)
        
        # Değer aralıkları doğru mu
        self.assertGreaterEqual(results["MSE"], 0)
        self.assertGreaterEqual(results["PSNR"], 0)
        self.assertGreaterEqual(results["SSIM"], 0)
        self.assertLessEqual(results["SSIM"], 1)
        self.assertGreaterEqual(results["LPIPS"], 0)
    
    def test_metrics_without_reference(self):
        """Referanssız metrikleri test eder."""
        results = self.processor.process(self.test_image)
        
        # Sadece entropi hesaplanmalı
        self.assertIn("Entropy", results)
        self.assertNotIn("PSNR", results)
        self.assertNotIn("SSIM", results)
        self.assertNotIn("LPIPS", results)
        
        # Entropi değeri geçerli mi
        self.assertGreaterEqual(results["Entropy"], 0)
    
    def test_specific_metrics(self):
        """Belirli metriklerin hesaplanmasını test eder."""
        metrics = ["PSNR", "SSIM"]
        results = self.processor.process(
            self.test_image,
            reference=self.reference_image,
            metrics=metrics
        )
        
        # Sadece istenen metrikler hesaplanmış mı
        self.assertEqual(set(results.keys()), set(metrics))
    
    def test_invalid_image(self):
        """Geçersiz görüntü formatını test eder."""
        invalid_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # 2D array
        
        with self.assertRaises(ValueError):
            self.processor.process(invalid_image)
    
    def test_invalid_reference(self):
        """Geçersiz referans görüntüsünü test eder."""
        invalid_reference = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # 2D array
        
        with self.assertRaises(ValueError):
            self.processor.process(self.test_image, reference=invalid_reference)
    
    def test_different_sizes(self):
        """Farklı boyutlu görüntüleri test eder."""
        larger_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        results = self.processor.process(
            larger_image,
            reference=self.reference_image
        )
        
        # Metrikler hesaplanabilmeli
        self.assertIn("PSNR", results)
        self.assertIn("SSIM", results)
    
    def test_perfect_match(self):
        """Aynı görüntüler için metrikleri test eder."""
        results = self.processor.process(
            self.reference_image,
            reference=self.reference_image
        )
        
        # MSE sıfır olmalı
        self.assertEqual(results["MSE"], 0)
        
        # PSNR sonsuz olmalı (veya çok büyük)
        self.assertGreater(results["PSNR"], 50)
        
        # SSIM 1 olmalı
        self.assertAlmostEqual(results["SSIM"], 1, places=5)
        
        # LPIPS 0'a yakın olmalı
        self.assertLess(results["LPIPS"], 0.1)

if __name__ == '__main__':
    unittest.main() 