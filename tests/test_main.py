import unittest
from PyQt5.QtWidgets import QApplication
from main import MultimediaBenchmark, PlotWindow
import sys

class TestMultimediaBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Test sınıfı başlatılmadan önce çağrılır."""
        # QApplication örneği oluştur
        if not QApplication.instance():
            cls.app = QApplication(sys.argv)
    
    def setUp(self):
        """Her test metodundan önce çağrılır."""
        self.window = MultimediaBenchmark()
    
    def test_initialization(self):
        """Başlangıç durumunun doğru olduğunu kontrol eder."""
        # Temel özellikler
        self.assertIsNotNone(self.window)
        self.assertEqual(self.window.windowTitle(), "Multimedia Benchmark Tool")
        
        # İşlemci nesneleri
        self.assertIsNotNone(self.window.image_processor)
        self.assertIsNotNone(self.window.video_processor)
        self.assertIsNotNone(self.window.plot_manager)
        
        # Durum değişkenleri
        self.assertFalse(self.window.is_video_processing)
        self.assertFalse(self.window.is_video_paused)
        
        # UI bileşenleri
        self.assertIsNotNone(self.window.original_image_label)
        self.assertIsNotNone(self.window.compressed_image_label)
        self.assertIsNotNone(self.window.metrics_label)
        self.assertIsNotNone(self.window.recommendation_label)
        self.assertIsNotNone(self.window.progress_bar)
        self.assertIsNotNone(self.window.time_label)
    
    def test_required_methods_exist(self):
        """Gerekli metodların var olduğunu kontrol eder."""
        required_methods = [
            'show_plots',
            'select_image_file',
            'select_video_file',
            'display_image',
            'compress_and_analyze',
            'update_metrics_label',
            'update_recommendation_label',
            'update_time',
            'update_compression',
            'update_progress',
            'update_frames',
            'update_video_metrics',
            'compress_and_analyze_video',
            'toggle_fullscreen',
            'update_current_best_label',
            'on_video_processing_finished',
            'setup_video_controls',
            'toggle_video_processing',
            'toggle_video_pause'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(self.window, method), f"'{method}' metodu eksik")
            self.assertTrue(callable(getattr(self.window, method)), f"'{method}' çağrılabilir değil")
    
    def test_plot_window_initialization(self):
        """Plot penceresinin doğru başlatıldığını kontrol eder."""
        # Plot penceresi başlangıçta None olmalı
        self.assertIsNone(self.window.plot_window)
        
        # Plot penceresini göster
        self.window.show_plots()
        
        # Plot penceresi oluşturulmuş olmalı
        self.assertIsNotNone(self.window.plot_window)
        self.assertIsInstance(self.window.plot_window, PlotWindow)
        
        # Plot manager aktarılmış olmalı
        self.assertIsNotNone(self.window.plot_window.plot_manager)
        self.assertEqual(self.window.plot_window.plot_manager, self.window.plot_manager)
    
    def test_plot_window_required_methods_exist(self):
        """Plot penceresinin gerekli metodlarının var olduğunu kontrol eder."""
        self.window.show_plots()  # Plot penceresini oluştur
        
        required_methods = [
            'setup_plot_tabs',
            'update_plots'
        ]
        
        for method in required_methods:
            self.assertTrue(hasattr(self.window.plot_window, method), 
                          f"PlotWindow'da '{method}' metodu eksik")
            self.assertTrue(callable(getattr(self.window.plot_window, method)), 
                          f"PlotWindow'da '{method}' çağrılabilir değil")
    
    def test_signal_connections(self):
        """Video işlemci sinyallerinin doğru bağlandığını kontrol eder."""
        # Slot metodlarının varlığını kontrol et
        required_slots = {
            'update_progress': self.window.video_processor.progress_update,
            'update_frames': self.window.video_processor.frame_update,
            'update_video_metrics': self.window.video_processor.metrics_update,
            'update_time': self.window.video_processor.time_update,
            'update_compression': self.window.video_processor.compression_update,
            'on_video_processing_finished': self.window.video_processor.finished
        }
        
        for slot_name, signal in required_slots.items():
            # Metodun varlığını kontrol et
            self.assertTrue(
                hasattr(self.window, slot_name),
                f"{slot_name} metodu bulunamadı"
            )
            
            # Metodun çağrılabilir olduğunu kontrol et
            slot = getattr(self.window, slot_name)
            self.assertTrue(
                callable(slot),
                f"{slot_name} çağrılabilir değil"
            )
            
            # Sinyalin bağlı olduğunu kontrol et
            try:
                signal.disconnect(slot)  # Eğer bağlantı varsa, ayrılabilir
                signal.connect(slot)     # Bağlantıyı tekrar kur
                self.assertTrue(True)    # Buraya ulaşıldıysa bağlantı vardı
            except TypeError:
                self.fail(f"{slot_name} sinyale bağlı değil")
    
    def tearDown(self):
        """Her test metodundan sonra çağrılır."""
        self.window.close()
    
    @classmethod
    def tearDownClass(cls):
        """Test sınıfı sonlandırılmadan önce çağrılır."""
        if hasattr(cls, 'app'):
            cls.app.quit()

if __name__ == '__main__':
    unittest.main() 