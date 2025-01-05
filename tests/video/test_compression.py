import unittest
import numpy as np
import cv2
import os
from video.processors.compression_processor import CompressionProcessor

class TestVideoCompressionProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Test sınıfı için hazırlık."""
        # Test videosu oluştur
        cls.test_video_path = "test_video.mp4"
        cls.create_test_video(cls.test_video_path)
    
    @classmethod
    def tearDownClass(cls):
        """Test sınıfı sonrası temizlik."""
        if os.path.exists(cls.test_video_path):
            os.remove(cls.test_video_path)
    
    @classmethod
    def create_test_video(cls, path: str, frames: int = 30):
        """Test videosu oluşturur.
        
        Args:
            path: Video dosya yolu
            frames: Kare sayısı
        """
        width, height = 320, 240
        fps = 30
        
        # MJPEG codec'i daha yaygın destekleniyor
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Video writer başlatılamadı")
        
        for _ in range(frames):
            # Daha büyük boyutlu test karesi oluştur (SSIM için minimum 7x7)
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
    
    def setUp(self):
        """Her test öncesi hazırlık."""
        self.processor = CompressionProcessor()
        self.output_path = "compressed_test_video.mp4"
    
    def tearDown(self):
        """Her test sonrası temizlik."""
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
    
    def test_supported_codecs(self):
        """Desteklenen codec'leri test eder."""
        codecs = self.processor.get_supported_codecs()
        
        self.assertIsInstance(codecs, dict)
        self.assertGreater(len(codecs), 0)
        
        for codec, info in codecs.items():
            self.assertIn("fourcc", info)
            self.assertIn("crf_range", info)
            self.assertEqual(len(info["crf_range"]), 2)
    
    def test_video_compression(self):
        """Video sıkıştırmayı test eder."""
        self.processor.set_parameters(
            self.test_video_path,
            self.output_path,
            codec="H.264",
            crf=23
        )
        
        # İşlemi başlat
        self.processor.start()
        self.processor.wait()  # İşlem bitene kadar bekle
        
        # Çıktı dosyası oluşturulmuş mu
        self.assertTrue(os.path.exists(self.output_path))
        
        # Dosya boyutu azalmış mı
        original_size = os.path.getsize(self.test_video_path)
        compressed_size = os.path.getsize(self.output_path)
        self.assertLess(compressed_size, original_size)
    
    def test_invalid_codec(self):
        """Geçersiz codec'i test eder."""
        with self.assertRaises(ValueError):
            self.processor.set_parameters(
                self.test_video_path,
                self.output_path,
                codec="INVALID_CODEC",
                crf=23
            )
    
    def test_invalid_crf(self):
        """Geçersiz CRF değerini test eder."""
        with self.assertRaises(ValueError):
            self.processor.set_parameters(
                self.test_video_path,
                self.output_path,
                codec="H.264",
                crf=100  # Geçersiz değer
            )
    
    def test_invalid_input_path(self):
        """Geçersiz girdi dosyasını test eder."""
        with self.assertRaises(FileNotFoundError):
            self.processor.set_parameters(
                "nonexistent_video.mp4",
                self.output_path,
                codec="H.264",
                crf=23
            )
    
    def test_compression_progress(self):
        """Sıkıştırma ilerlemesini test eder."""
        progress_values = []
        
        def progress_callback(value):
            progress_values.append(value)
        
        self.processor.progress_update.connect(progress_callback)
        
        self.processor.set_parameters(
            self.test_video_path,
            self.output_path,
            codec="H.264",
            crf=23
        )
        
        self.processor.start()
        self.processor.wait()
        
        # İlerleme değerleri kontrol
        self.assertGreater(len(progress_values), 0)
        self.assertEqual(min(progress_values), 0)
        self.assertEqual(max(progress_values), 100)
        self.assertEqual(progress_values[-1], 100)
    
    def test_compression_ratio(self):
        """Sıkıştırma oranı hesaplamasını test eder."""
        original_size = 1000
        compressed_size = 500
        
        self.processor.original_size = original_size
        ratio = self.processor.calculate_compression_ratio(compressed_size)
        
        self.assertEqual(ratio, 50.0)
    
    def test_video_info(self):
        """Video bilgilerini test eder."""
        self.processor.set_parameters(
            self.test_video_path,
            self.output_path,
            codec="H.264",
            crf=23
        )
        
        info = self.processor.video_info
        
        self.assertIn("frame_count", info)
        self.assertIn("fps", info)
        self.assertIn("width", info)
        self.assertIn("height", info)
        self.assertGreater(info["frame_count"], 0)
        self.assertGreater(info["fps"], 0)
        self.assertGreater(info["width"], 0)
        self.assertGreater(info["height"], 0)

if __name__ == '__main__':
    unittest.main() 