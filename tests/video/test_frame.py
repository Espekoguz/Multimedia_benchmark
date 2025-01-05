import unittest
import numpy as np
import cv2
from video.processors.frame_processor import FrameProcessor

class TestFrameProcessor(unittest.TestCase):
    def setUp(self):
        """Her test öncesi hazırlık."""
        self.processor = FrameProcessor()
        
        # Test karesi oluştur
        self.test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    def test_histogram_calculation(self):
        """Histogram hesaplamasını test eder."""
        histograms = self.processor.calculate_histogram(self.test_frame)
        
        # Her renk kanalı için histogram var mı
        self.assertIn("b", histograms)
        self.assertIn("g", histograms)
        self.assertIn("r", histograms)
        
        # Histogram boyutları doğru mu
        for hist in histograms.values():
            self.assertEqual(len(hist), 256)  # 256 bin
            self.assertTrue(np.all(hist >= 0))  # Negatif değer olmamalı
    
    def test_entropy_calculation(self):
        """Entropi hesaplamasını test eder."""
        entropy = self.processor.calculate_entropy(self.test_frame)
        
        # Entropi değeri geçerli aralıkta mı
        self.assertGreaterEqual(entropy, 0)
        self.assertLessEqual(entropy, 8)  # 8-bit görüntü için maksimum entropi
    
    def test_frame_analysis(self):
        """Kare analizini test eder."""
        results = self.processor.process(self.test_frame)
        
        # Tüm analiz sonuçları var mı
        self.assertIn("histograms", results)
        self.assertIn("entropy", results)
        self.assertIn("statistics", results)
        
        # İstatistikler doğru mu
        stats = results["statistics"]
        for channel in ["b", "g", "r"]:
            self.assertIn(channel, stats)
            channel_stats = stats[channel]
            self.assertIn("mean", channel_stats)
            self.assertIn("std", channel_stats)
            self.assertIn("min", channel_stats)
            self.assertIn("max", channel_stats)
    
    def test_frame_preprocessing(self):
        """Kare ön işlemeyi test eder."""
        # Gürültülü kare oluştur
        noisy_frame = self.test_frame.copy()
        noise = np.random.normal(0, 25, noisy_frame.shape).astype(np.uint8)
        noisy_frame = cv2.add(noisy_frame, noise)
        
        processed = self.processor.preprocess_frame(noisy_frame)
        
        # Ön işleme sonrası boyut aynı kalmalı
        self.assertEqual(processed.shape, noisy_frame.shape)
        
        # Gürültü azalmış olmalı
        original_std = np.std(noisy_frame)
        processed_std = np.std(processed)
        self.assertLess(processed_std, original_std)
    
    def test_frame_resize(self):
        """Kare boyutlandırmayı test eder."""
        target_size = (160, 120)  # Yarı boyut
        resized = self.processor.resize_frame(self.test_frame, target_size)
        
        self.assertEqual(resized.shape[:2], target_size[::-1])
    
    def test_frame_normalization(self):
        """Kare normalizasyonunu test eder."""
        normalized = self.processor.normalize_frame(self.test_frame)
        
        # Değerler [0, 1] aralığında olmalı
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
    
    def test_color_space_conversion(self):
        """Renk uzayı dönüşümünü test eder."""
        # BGR -> HSV
        hsv = self.processor.convert_color_space(self.test_frame, "HSV")
        self.assertEqual(hsv.shape, self.test_frame.shape)
        
        # BGR -> YCrCb
        ycrcb = self.processor.convert_color_space(self.test_frame, "YCrCb")
        self.assertEqual(ycrcb.shape, self.test_frame.shape)
    
    def test_invalid_frame(self):
        """Geçersiz kare formatını test eder."""
        # 2D array (tek kanallı)
        invalid_frame = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            self.processor.process(invalid_frame)
    
    def test_frame_statistics(self):
        """Kare istatistiklerini test eder."""
        # Bilinen değerlere sahip test karesi
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        test_frame[:, :, 0] = 100  # B kanalı
        test_frame[:, :, 1] = 150  # G kanalı
        test_frame[:, :, 2] = 200  # R kanalı
        
        results = self.processor.process(test_frame)
        stats = results["statistics"]
        
        # B kanalı
        self.assertEqual(stats["b"]["mean"], 100)
        self.assertEqual(stats["b"]["min"], 100)
        self.assertEqual(stats["b"]["max"], 100)
        self.assertEqual(stats["b"]["std"], 0)
        
        # G kanalı
        self.assertEqual(stats["g"]["mean"], 150)
        self.assertEqual(stats["g"]["min"], 150)
        self.assertEqual(stats["g"]["max"], 150)
        self.assertEqual(stats["g"]["std"], 0)
        
        # R kanalı
        self.assertEqual(stats["r"]["mean"], 200)
        self.assertEqual(stats["r"]["min"], 200)
        self.assertEqual(stats["r"]["max"], 200)
        self.assertEqual(stats["r"]["std"], 0)
    
    def test_frame_histogram_normalization(self):
        """Histogram normalizasyonunu test eder."""
        # Düşük kontrastlı kare oluştur
        low_contrast = np.uint8(np.random.normal(128, 10, self.test_frame.shape))
        
        # Histogram eşitleme uygula
        equalized = self.processor.equalize_histogram(low_contrast)
        
        # Kontrast artmış olmalı
        original_std = np.std(low_contrast)
        equalized_std = np.std(equalized)
        self.assertGreater(equalized_std, original_std)

if __name__ == '__main__':
    unittest.main() 