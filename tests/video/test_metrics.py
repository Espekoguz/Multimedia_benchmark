import unittest
import numpy as np
import cv2
import os
from video.processors.metric_processor import MetricProcessor

class TestVideoMetricProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Test sınıfı için hazırlık."""
        # Test videoları oluştur
        cls.reference_video_path = "reference_video.mp4"
        cls.test_video_path = "test_video.mp4"
        
        # Referans video
        cls.create_test_video(cls.reference_video_path)
        
        # Test videosu (gürültülü)
        cls.create_test_video(cls.test_video_path, add_noise=True)
    
    @classmethod
    def tearDownClass(cls):
        """Test sınıfı sonrası temizlik."""
        for path in [cls.reference_video_path, cls.test_video_path]:
            if os.path.exists(path):
                os.remove(path)
    
    @classmethod
    def create_test_video(cls, path: str, frames: int = 30, add_noise: bool = False):
        """Test videosu oluşturur.
        
        Args:
            path: Video dosya yolu
            frames: Kare sayısı
            add_noise: Gürültü eklensin mi
        """
        width, height = 320, 240  # Minimum 7x7 için yeterli büyüklük
        fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPEG codec kullan
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Video writer başlatılamadı")
        
        for _ in range(frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            if add_noise:
                noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
                frame = cv2.add(frame, noise)
            
            out.write(frame)
        
        out.release()
    
    def setUp(self):
        """Her test öncesi hazırlık."""
        self.processor = MetricProcessor()
    
    def test_available_metrics(self):
        """Kullanılabilir metrikleri test eder."""
        metrics = self.processor.available_metrics
        
        self.assertIsInstance(metrics, list)
        self.assertGreater(len(metrics), 0)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertIn("lpips", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("entropy", metrics)
    
    def test_frame_metrics(self):
        """Kare metriklerini test eder."""
        # Test kareleri oluştur (minimum 7x7 boyutunda)
        height, width = 240, 320
        reference_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        test_frame = reference_frame.copy()
        
        # Test karesine gürültü ekle
        noise = np.random.normal(0, 25, test_frame.shape).astype(np.uint8)
        test_frame = cv2.add(test_frame, noise)
        
        results = self.processor.process_frame(test_frame, reference_frame)
        
        # Tüm metrikler hesaplanmış mı
        self.assertIn("psnr", results)
        self.assertIn("ssim", results)
        self.assertIn("lpips", results)
        self.assertIn("mse", results)
        
        # Değer aralıkları doğru mu
        self.assertGreaterEqual(results["psnr"][0], 0)
        self.assertGreaterEqual(results["ssim"][0], 0)
        self.assertLessEqual(results["ssim"][0], 1)
        self.assertGreaterEqual(results["lpips"][0], 0)
        self.assertGreaterEqual(results["mse"][0], 0)
        
    def test_video_metrics(self):
        """Video metriklerini test eder."""
        self.processor.set_parameters(
            self.test_video_path,
            self.reference_video_path
        )
        
        metrics_history = []
        
        def metrics_callback(metrics):
            metrics_history.append(metrics)
        
        self.processor.metrics_update.connect(metrics_callback)
        
        # İşlemi başlat
        self.processor.start()
        self.processor.wait()
        
        # Metrikler hesaplanmış mı
        self.assertGreater(len(metrics_history), 0)
        
        last_metrics = metrics_history[-1]
        for metric in ["psnr", "ssim", "lpips", "mse"]:
            self.assertIn(metric, last_metrics)
            self.assertGreater(len(last_metrics[metric]), 0)
    
    def test_specific_metrics(self):
        """Belirli metriklerin hesaplanmasını test eder."""
        metrics = ["psnr", "ssim"]
        
        self.processor.set_parameters(
            self.test_video_path,
            self.reference_video_path,
            metrics=metrics
        )
        
        metrics_history = []
        
        def metrics_callback(metrics):
            metrics_history.append(metrics)
        
        self.processor.metrics_update.connect(metrics_callback)
        
        self.processor.start()
        self.processor.wait()
        
        # Sadece istenen metrikler hesaplanmış mı
        last_metrics = metrics_history[-1]
        self.assertEqual(set(last_metrics.keys()), set(metrics))
    
    def test_invalid_video(self):
        """Geçersiz video dosyasını test eder."""
        with self.assertRaises(FileNotFoundError):
            self.processor.set_parameters(
                "nonexistent_video.mp4",
                self.reference_video_path
            )
    
    def test_different_video_sizes(self):
        """Farklı boyutlu videoları test eder."""
        # Farklı boyutta test videosu oluştur
        different_size_path = "different_size.mp4"
        width, height = 640, 480  # Farklı boyut
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(different_size_path, fourcc, 30, (width, height))
        
        for _ in range(30):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        try:
            self.processor.set_parameters(
                different_size_path,
                self.reference_video_path
            )
            
            metrics_history = []
            
            def metrics_callback(metrics):
                metrics_history.append(metrics)
            
            self.processor.metrics_update.connect(metrics_callback)
            
            self.processor.start()
            self.processor.wait()
            
            # Metrikler hesaplanabilmeli
            self.assertGreater(len(metrics_history), 0)
            
        finally:
            if os.path.exists(different_size_path):
                os.remove(different_size_path)
    
    def test_perfect_match(self):
        """Aynı videolar için metrikleri test eder."""
        self.processor.set_parameters(
            self.reference_video_path,
            self.reference_video_path
        )
        
        metrics_history = []
        
        def metrics_callback(metrics):
            metrics_history.append(metrics)
        
        self.processor.metrics_update.connect(metrics_callback)
        
        # İşlemi başlat
        self.processor.start()
        self.processor.wait()
        
        # Metrikler hesaplanmış mı
        self.assertGreater(len(metrics_history), 0)
        
        last_metrics = metrics_history[-1]
        
        # MSE sıfır olmalı (aynı video)
        self.assertEqual(np.mean(last_metrics["mse"]), 0)
        
        # PSNR sonsuz olmalı (aynı video)
        self.assertGreater(np.mean(last_metrics["psnr"]), 50)
        
        # SSIM 1 olmalı (aynı video)
        self.assertAlmostEqual(np.mean(last_metrics["ssim"]), 1.0, places=5)
        
        # LPIPS 0 olmalı (aynı video)
        self.assertAlmostEqual(np.mean(last_metrics["lpips"]), 0.0, places=5)

if __name__ == '__main__':
    unittest.main() 