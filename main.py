import matplotlib
matplotlib.use('Qt5Agg')

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QComboBox, QSpinBox, QTabWidget, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from image_processor import ImageProcessor
from video_processor import VideoProcessor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class PlotManager:
    def __init__(self):
        self._plt = None
        self.FigureCanvas = FigureCanvasQTAgg
        
    @property
    def plt(self):
        if self._plt is None:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            self._plt = plt
        return self._plt
    
    def create_canvas(self, figsize=(8, 5)):
        figure = Figure(figsize=figsize)
        canvas = self.FigureCanvas(figure)
        return canvas, figure

class PlotWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Grafik Analizi")
        self.setGeometry(200, 200, 1000, 800)
        
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Plot sekmelerini oluştur
        self.plot_tabs = QTabWidget()
        layout.addWidget(self.plot_tabs)
        
        # Plot canvas'ları
        self.plot_canvases = {}
        
        # Plot manager
        self.plot_manager = None
    
    def setup_plot_tabs(self):
        """Grafik sekmelerini oluşturur."""
        if not self.plot_manager:
            return
            
        # Mevcut sekmeleri temizle
        while self.plot_tabs.count() > 0:
            self.plot_tabs.removeTab(0)
        
        # Görüntü grafiklerini ekle
        plot_tabs = {
            "compression": "Sıkıştırma Analizi",
            "psnr": "PSNR Analizi",
            "ssim": "SSIM Analizi",
            "time": "Süre Analizi"
        }
        
        for key, title in plot_tabs.items():
            container = QWidget()
            layout = QVBoxLayout(container)
            
            # Canvas ve figure oluştur
            canvas, figure = self.plot_manager.create_canvas()
            layout.addWidget(canvas)
            
            self.plot_tabs.addTab(container, title)
            self.plot_canvases[key] = canvas
    
    def update_plots(self, results: dict):
        """Grafikleri günceller."""
        plt = self.plot_manager.plt
        plt.ioff()  # Interactive modu kapat
        
        try:
            # Grafikleri temizle ve güncelle
            for canvas in self.plot_canvases.values():
                canvas.figure.clear()
            
            # Sıkıştırma Oranı vs Kalite
            ax1 = self.plot_canvases["compression"].figure.add_subplot(111)
            for method in results:
                ax1.plot(results[method]["qualities"], 
                        results[method]["compression_ratios"], 
                        marker='.', label=method)
            ax1.set_xlabel("Kalite Faktörü")
            ax1.set_ylabel("Sıkıştırma Oranı (%)")
            ax1.set_title("Sıkıştırma Oranı vs Kalite")
            ax1.legend()
            ax1.grid(True)
            self.plot_canvases["compression"].figure.tight_layout()
            
            # PSNR vs Kalite
            ax2 = self.plot_canvases["psnr"].figure.add_subplot(111)
            for method in results:
                ax2.plot(results[method]["qualities"], 
                        results[method]["psnr_values"], 
                        marker='.', label=method)
            ax2.set_xlabel("Kalite Faktörü")
            ax2.set_ylabel("PSNR (dB)")
            ax2.set_title("PSNR vs Kalite")
            ax2.legend()
            ax2.grid(True)
            self.plot_canvases["psnr"].figure.tight_layout()
            
            # SSIM vs Kalite
            ax3 = self.plot_canvases["ssim"].figure.add_subplot(111)
            for method in results:
                ax3.plot(results[method]["qualities"], 
                        results[method]["ssim_values"], 
                        marker='.', label=method)
            ax3.set_xlabel("Kalite Faktörü")
            ax3.set_ylabel("SSIM")
            ax3.set_title("SSIM vs Kalite")
            ax3.legend()
            ax3.grid(True)
            self.plot_canvases["ssim"].figure.tight_layout()
            
            # İşlem Süresi vs Kalite
            ax4 = self.plot_canvases["time"].figure.add_subplot(111)
            for method in results:
                ax4.plot(results[method]["qualities"], 
                        results[method]["processing_times"], 
                        marker='.', label=method)
            ax4.set_xlabel("Kalite Faktörü")
            ax4.set_ylabel("İşlem Süresi (s)")
            ax4.set_title("İşlem Süresi vs Kalite")
            ax4.legend()
            ax4.grid(True)
            self.plot_canvases["time"].figure.tight_layout()
            
            # Canvas'ları güncelle
            for canvas in self.plot_canvases.values():
                canvas.draw()
            
            plt.close('all')  # Tüm figürleri kapat
            
        except Exception as e:
            print(f"Grafik güncelleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()

class MultimediaBenchmark(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Benchmark Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Video işleme durumu
        self.is_video_processing = False
        self.is_video_paused = False
        
        # Plot manager ve işlemci nesneleri
        self.plot_manager = PlotManager()
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        
        # Plot penceresi
        self.plot_window = None
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Üst panel - Kontroller
        top_panel = QHBoxLayout()
        
        # Sol panel - Dosya seçimi ve kontroller
        left_panel = QVBoxLayout()
        
        # Dosya seçimi
        file_layout = QVBoxLayout()
        self.image_path_label = QLabel("Dosya seçilmedi")
        select_image_btn = QPushButton("Görüntü Seç")
        select_video_btn = QPushButton("Video Seç")
        select_image_btn.clicked.connect(self.select_image_file)
        select_video_btn.clicked.connect(self.select_video_file)
        file_layout.addWidget(select_image_btn)
        file_layout.addWidget(select_video_btn)
        file_layout.addWidget(self.image_path_label)
        left_panel.addLayout(file_layout)
        
        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)
        
        # Kalan süre etiketi
        self.time_label = QLabel()
        self.time_label.setVisible(False)
        left_panel.addWidget(self.time_label)
        
        top_panel.addLayout(left_panel)
        
        # Sağ panel - Sıkıştırma ayarları
        compression_layout = QVBoxLayout()
        
        # Görüntü sıkıştırma ayarları
        self.image_settings = QWidget()
        image_layout = QVBoxLayout(self.image_settings)
        image_layout.addWidget(QLabel("Görüntü Sıkıştırma:"))
        self.image_compression_combo = QComboBox()
        self.image_compression_combo.addItems(["JPEG", "JPEG2000", "HEVC"])
        image_layout.addWidget(self.image_compression_combo)
        image_layout.addWidget(QLabel("Kalite:"))
        self.image_quality_spin = QSpinBox()
        self.image_quality_spin.setRange(1, 100)
        self.image_quality_spin.setValue(75)
        image_layout.addWidget(self.image_quality_spin)
        
        # Sıkıştırma butonu
        self.compress_btn = QPushButton("Sıkıştır ve Analiz Et")
        self.compress_btn.clicked.connect(self.compress_and_analyze)
        self.compress_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        image_layout.addWidget(self.compress_btn)
        compression_layout.addWidget(self.image_settings)
        
        # Video sıkıştırma ayarları
        self.video_settings = QWidget()
        video_layout = QVBoxLayout(self.video_settings)
        video_layout.addWidget(QLabel("Video Codec:"))
        self.video_codec_combo = QComboBox()
        self.video_codec_combo.addItems(["H.264", "H.265/HEVC", "VP9"])
        video_layout.addWidget(self.video_codec_combo)
        video_layout.addWidget(QLabel("CRF (Kalite):"))
        self.video_quality_spin = QSpinBox()
        self.video_quality_spin.setRange(0, 51)
        self.video_quality_spin.setValue(23)
        video_layout.addWidget(self.video_quality_spin)
        video_layout.addLayout(self.setup_video_controls())
        compression_layout.addWidget(self.video_settings)
        
        # Başlangıçta video ayarlarını gizle
        self.video_settings.hide()
        
        top_panel.addLayout(compression_layout)
        
        self.layout.addLayout(top_panel)
        
        # Orta panel - Görüntüler ve Metrikler
        middle_panel = QHBoxLayout()
        
        # Sol panel - Görüntüler
        images_panel = QVBoxLayout()
        
        # Orijinal görüntü
        original_layout = QVBoxLayout()
        original_layout.addWidget(QLabel("Orijinal Görüntü"))
        self.original_image_label = QLabel()
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet("border: 1px solid #666;")
        original_layout.addWidget(self.original_image_label)
        images_panel.addLayout(original_layout)
        
        # Sıkıştırılmış görüntü
        compressed_layout = QVBoxLayout()
        compressed_layout.addWidget(QLabel("Sıkıştırılmış Görüntü"))
        self.compressed_image_label = QLabel()
        self.compressed_image_label.setMinimumSize(400, 300)
        self.compressed_image_label.setStyleSheet("border: 1px solid #666;")
        compressed_layout.addWidget(self.compressed_image_label)
        images_panel.addLayout(compressed_layout)
        
        middle_panel.addLayout(images_panel)
        
        # Sağ panel - Metrikler
        metrics_panel = QVBoxLayout()
        
        # Metrikler etiketi
        self.metrics_label = QLabel()
        self.metrics_label.setStyleSheet("""
            QLabel { 
                background-color: #2C3E50; 
                color: #ECF0F1;
                padding: 10px; 
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        metrics_panel.addWidget(QLabel("Anlık Metrikler:"))
        metrics_panel.addWidget(self.metrics_label)
        
        # Anlık en iyi değerler etiketi
        self.current_best_label_title = QLabel("Anlık En İyi Değerler:")
        self.current_best_label = QLabel()
        self.current_best_label.setStyleSheet("""
            QLabel { 
                background-color: #8E44AD; 
                color: #FFFFFF;
                padding: 10px; 
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        metrics_panel.addWidget(self.current_best_label_title)
        metrics_panel.addWidget(self.current_best_label)
        # Başlangıçta gizle
        self.current_best_label_title.setVisible(False)
        self.current_best_label.setVisible(False)
        
        # Önerilen yöntem etiketi
        self.recommendation_label = QLabel()
        self.recommendation_label.setStyleSheet("""
            QLabel { 
                background-color: #27AE60; 
                color: #FFFFFF;
                padding: 10px; 
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        metrics_panel.addWidget(QLabel("Önerilen Yöntem:"))
        metrics_panel.addWidget(self.recommendation_label)
        
        middle_panel.addLayout(metrics_panel)
        
        self.layout.addLayout(middle_panel)
        
        # Buton layout'u
        button_layout = QHBoxLayout()
        
        # Grafik göster butonu
        show_plots_btn = QPushButton("Grafikleri Göster")
        show_plots_btn.clicked.connect(self.show_plots)
        show_plots_btn.setStyleSheet("""
            QPushButton {
                background-color: #8e44ad;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9b59b6;
            }
        """)
        
        # Tam ekran butonu
        fullscreen_btn = QPushButton("Tam Ekran")
        fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2c3e50;
            }
        """)
        
        button_layout.addWidget(show_plots_btn)
        button_layout.addWidget(fullscreen_btn)
        button_layout.addStretch()
        self.layout.addLayout(button_layout)
        
        # Video işleme sinyallerini bağla
        self.video_processor.progress_update.connect(self.update_progress)
        self.video_processor.frame_update.connect(self.update_frames)
        self.video_processor.metrics_update.connect(self.update_video_metrics)
        self.video_processor.time_update.connect(self.update_time)
        self.video_processor.compression_update.connect(self.update_compression)
        self.video_processor.finished.connect(self.on_video_processing_finished)
        
        self.show()

    def show_plots(self):
        """Grafik penceresini gösterir."""
        if self.plot_window is None:
            self.plot_window = PlotWindow(self)
            self.plot_window.plot_manager = self.plot_manager
            self.plot_window.setup_plot_tabs()  # Plot sekmelerini oluştur
            
            # Eğer orijinal görüntü yüklenmişse, tüm yöntemleri analiz et ve grafikleri güncelle
            if hasattr(self, 'original_image') and self.original_image is not None:
                try:
                    results = self.image_processor.analyze_all_methods(self.original_image)
                    self.plot_window.update_plots(results)
                except Exception as e:
                    print(f"Grafik güncelleme hatası: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
        self.plot_window.show()
    
    def select_image_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Görüntü Dosyası Seç",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.bmp)"
        )
        if file_name:
            self.image_path_label.setText(file_name)
            self.original_image = self.image_processor.load_image(file_name)
            self.display_image(self.original_image, self.original_image_label)
            # Görüntü ayarlarını göster, video ayarlarını gizle
            self.image_settings.show()
            self.video_settings.hide()
            # İlerleme çubuğunu, süre etiketini ve en iyi değerler etiketini gizle
            self.progress_bar.setVisible(False)
            self.time_label.setVisible(False)
            self.current_best_label.setVisible(False)
            self.current_best_label_title.setVisible(False)
            # Görüntü grafiklerini göster
            self.show_image_plots()
    
    def select_video_file(self):
        """Video dosyası seçme işlemini gerçekleştirir."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Video Dosyası Seç",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if file_name:
            self.video_path = file_name
            self.image_path_label.setText(file_name)
            # Video ayarlarını göster, görüntü ayarlarını gizle
            self.video_settings.show()
            self.image_settings.hide()
            # İlerleme çubuğunu, süre etiketini ve en iyi değerler etiketini göster
            self.progress_bar.setVisible(True)
            self.time_label.setVisible(True)
            self.current_best_label.setVisible(True)
            self.current_best_label_title.setVisible(True)
            
            # Video işleme parametrelerini ayarla
            output_path = file_name.rsplit('.', 1)[0] + '_compressed.' + file_name.rsplit('.', 1)[1]
            codec = self.video_codec_combo.currentText()
            crf = self.video_quality_spin.value()
            
            # Video işleme nesnesini hazırla ama başlatma
            self.video_processor.set_parameters(file_name, output_path, codec, crf)
    
    def show_image_plots(self):
        """Görüntü grafiklerini gösterir."""
        if hasattr(self, 'original_image') and self.original_image is not None:
            try:
                # Plot penceresini oluştur veya mevcut pencereyi kullan
                if self.plot_window is None:
                    self.plot_window = PlotWindow(self)
                    self.plot_window.plot_manager = self.plot_manager
                    self.plot_window.setup_plot_tabs()
                
                # Analiz yap ve grafikleri güncelle
                results = self.image_processor.analyze_all_methods(self.original_image)
                self.plot_window.update_plots(results)
                
            except Exception as e:
                print(f"Grafik güncelleme hatası: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def display_image(self, image: np.ndarray, label: QLabel):
        """Numpy dizisini QLabel'da görüntüler."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))
    
    def compress_and_analyze(self):
        """Görüntüyü sıkıştırır ve metrikleri hesaplar."""
        print("Sıkıştırma ve analiz başlatılıyor...")
        
        if not hasattr(self, 'original_image'):
            print("Hata: Görüntü yüklenmemiş!")
            return
        
        if self.original_image is None:
            print("Hata: Görüntü geçersiz!")
            return
            
        try:
            print("Sıkıştırma parametreleri alınıyor...")
            # Sıkıştırma parametreleri
            method = self.image_compression_combo.currentText()
            quality = self.image_quality_spin.value()
            
            print(f"Görüntü sıkıştırılıyor... (Yöntem: {method}, Kalite: {quality})")
            # Görüntüyü sıkıştır
            compressed, file_size = self.image_processor.compress_image(
                self.original_image, method, quality)
            
            if compressed is None:
                print("Hata: Sıkıştırma başarısız!")
                return
            
            print("Metrikler hesaplanıyor...")
            # Metrikleri hesapla
            metrics = self.image_processor.calculate_metrics(
                self.original_image, compressed)
            
            # Sıkıştırma oranını hesapla
            compression_ratio = self.image_processor.calculate_compression_ratio(
                self.image_processor.original_size, file_size)
            
            # Metrikleri güncelle
            metrics.update({
                "compression_ratio": compression_ratio,
                "file_size": file_size,
                "method": method,
                "quality": quality
            })
            
            print("Arayüz güncelleniyor...")
            # Metrikleri göster
            self.update_metrics_label(metrics)
            
            # Sıkıştırılmış görüntüyü göster
            self.display_image(compressed, self.compressed_image_label)
            
            print("Tüm yöntemler analiz ediliyor...")
            # Tüm yöntemleri analiz et
            results = self.image_processor.analyze_all_methods(self.original_image)
            
            print("En iyi yöntem belirleniyor...")
            # En iyi yöntemi bul
            optimal = self.image_processor.find_optimal_method(results)
            
            # Önerilen yöntemi güncelle
            self.update_recommendation_label(optimal)
            
            # Plot penceresini güncelle
            if self.plot_window is not None:
                self.plot_window.update_plots(results)
            
            print("İşlem tamamlandı!")
            
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            self.metrics_label.setText(f"İşlem sırasında hata oluştu: {str(e)}")
            self.recommendation_label.setText("İşlem sırasında hata oluştu")
    
    def update_metrics_label(self, metrics: dict):
        """Metrik etiketini günceller."""
        text = f"""
        <table style='color: #ECF0F1;'>
            <tr><td><b>Yöntem:</b></td><td>{metrics.get('method', 'N/A')}</td></tr>
            <tr><td><b>Kalite:</b></td><td>{metrics.get('quality', 0)}</td></tr>
            <tr><td><b>Sıkıştırma Oranı:</b></td><td>{metrics.get('compression_ratio', 0):.2f}%</td></tr>
            <tr><td><b>Dosya Boyutu:</b></td><td>{metrics.get('file_size', 0)/1024:.2f} KB</td></tr>
            <tr><td><b>PSNR:</b></td><td>{metrics.get('PSNR', 0):.2f} dB</td></tr>
            <tr><td><b>SSIM:</b></td><td>{metrics.get('SSIM', 0):.4f}</td></tr>
            <tr><td><b>LPIPS:</b></td><td>{metrics.get('LPIPS', 1):.4f}</td></tr>
        </table>
        """
        self.metrics_label.setText(text)
    
    def update_recommendation_label(self, optimal: dict):
        """Öneri etiketini günceller."""
        # En iyi sıkıştırma oranına sahip yöntem
        best_compression = optimal.get('best_compression', {})
        # En iyi kaliteye sahip yöntem (PSNR, SSIM ve LPIPS'e göre)
        best_quality = optimal.get('best_quality', {})
        # En iyi genel skora sahip yöntem
        best_overall = optimal.get('best_overall', {})

        text = f"""
        <table style='color: #FFFFFF;'>
            <tr><td colspan='2'><b>En İyi Genel Performans:</b></td></tr>
            <tr><td>Yöntem:</td><td>{best_overall.get('method', 'N/A')}</td></tr>
            <tr><td>Kalite:</td><td>{best_overall.get('quality', 0)}</td></tr>
            <tr><td>Genel Skor:</td><td>{best_overall.get('score', 0):.2f}</td></tr>
            <tr><td>Sıkıştırma Oranı:</td><td>{best_overall.get('compression_ratio', 0):.2f}%</td></tr>
            <tr><td>PSNR:</td><td>{best_overall.get('psnr', 0):.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{best_overall.get('ssim', 0):.4f}</td></tr>
            <tr><td>LPIPS:</td><td>{best_overall.get('lpips', 1):.4f}</td></tr>

            <tr><td colspan='2'><b>En İyi Sıkıştırma:</b></td></tr>
            <tr><td>Yöntem:</td><td>{best_compression.get('method', 'N/A')}</td></tr>
            <tr><td>Kalite:</td><td>{best_compression.get('quality', 0)}</td></tr>
            <tr><td>Sıkıştırma Oranı:</td><td>{best_compression.get('compression_ratio', 0):.2f}%</td></tr>

            <tr><td colspan='2'><b>En İyi Kalite:</b></td></tr>
            <tr><td>Yöntem:</td><td>{best_quality.get('method', 'N/A')}</td></tr>
            <tr><td>Kalite:</td><td>{best_quality.get('quality', 0)}</td></tr>
            <tr><td>PSNR:</td><td>{best_quality.get('psnr', 0):.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{best_quality.get('ssim', 0):.4f}</td></tr>
            <tr><td>LPIPS:</td><td>{best_quality.get('lpips', 1):.4f}</td></tr>
        </table>
        """
        self.recommendation_label.setText(text)
    
    def update_time(self, remaining: float, elapsed: float):
        """Kalan ve geçen süreyi günceller."""
        self.time_label.setText(
            f"Kalan Süre: {remaining:.1f}s | "
            f"Geçen Süre: {elapsed:.1f}s"
        )
    
    def update_compression(self, ratio: float):
        """Sıkıştırma oranını günceller."""
        text = f"""
        <table style='color: #ECF0F1;'>
            <tr><td><b>Anlık Sıkıştırma Oranı:</b></td><td>{ratio:.2f}%</td></tr>
        </table>
        """
        self.metrics_label.setText(text)
    
    def update_progress(self, value: int):
        """İlerleme çubuğunu günceller."""
        self.progress_bar.setValue(value)
    
    def update_frames(self, original: np.ndarray, compressed: np.ndarray):
        """Orijinal ve sıkıştırılmış kareleri günceller."""
        try:
            # Görüntüleri yeniden boyutlandır
            height = self.original_image_label.height()
            width = self.original_image_label.width()
            
            # Orijinal frame'i göster
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            height_o, width_o = original_rgb.shape[:2]
            scale = min(width/width_o, height/height_o)
            new_width = int(width_o * scale)
            new_height = int(height_o * scale)
            original_resized = cv2.resize(original_rgb, (new_width, new_height))
            
            # Sıkıştırılmış frame'i göster
            compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
            compressed_resized = cv2.resize(compressed_rgb, (new_width, new_height))
            
            # QImage ve QPixmap oluştur
            bytes_per_line = 3 * new_width
            
            # Orijinal görüntü
            q_image_original = QImage(original_resized.data, new_width, new_height, 
                                    bytes_per_line, QImage.Format_RGB888)
            pixmap_original = QPixmap.fromImage(q_image_original)
            self.original_image_label.setPixmap(pixmap_original)
            
            # Sıkıştırılmış görüntü
            q_image_compressed = QImage(compressed_resized.data, new_width, new_height, 
                                      bytes_per_line, QImage.Format_RGB888)
            pixmap_compressed = QPixmap.fromImage(q_image_compressed)
            self.compressed_image_label.setPixmap(pixmap_compressed)
            
        except Exception as e:
            print(f"Frame güncelleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_video_metrics(self, metrics: dict):
        """Video metriklerini günceller."""
        try:
            # Mevcut durumu güncelle
            current_metrics = {
                "PSNR": metrics["PSNR"][-1] if metrics["PSNR"] else 0,
                "SSIM": metrics["SSIM"][-1] if metrics["SSIM"] else 0,
                "LPIPS": metrics["LPIPS"][-1] if metrics["LPIPS"] else 1,
                "Histogram_Similarity": metrics["Histogram_Similarity"][-1] if metrics["Histogram_Similarity"] else 0,
                "Entropy": metrics["Entropy"][-1] if metrics["Entropy"] else 0,
                "Compression_Ratio": metrics["Compression_Ratio"][-1] if metrics["Compression_Ratio"] else 0
            }
            
            # Metrikleri göster
            text = f"""
            <table style='color: #ECF0F1;'>
                <tr><td colspan='2'><b>Mevcut Durum:</b></td></tr>
                <tr><td>Codec:</td><td>{self.video_codec_combo.currentText()}</td></tr>
                <tr><td>CRF:</td><td>{self.video_quality_spin.value()}</td></tr>
                <tr><td>PSNR:</td><td>{current_metrics["PSNR"]:.2f} dB</td></tr>
                <tr><td>SSIM:</td><td>{current_metrics["SSIM"]:.4f}</td></tr>
                <tr><td>LPIPS:</td><td>{current_metrics["LPIPS"]:.4f}</td></tr>
                <tr><td>Histogram Benzerliği:</td><td>{current_metrics["Histogram_Similarity"]:.4f}</td></tr>
                <tr><td>Entropi:</td><td>{current_metrics["Entropy"]:.4f}</td></tr>
                <tr><td>Sıkıştırma Oranı:</td><td>{current_metrics["Compression_Ratio"]:.2f}%</td></tr>
            </table>
            """
            self.metrics_label.setText(text)
            
            # En iyi değerleri bul
            best_metrics = {
                "best_quality": {
                    "frame_number": len(metrics["PSNR"]),
                    "PSNR": max(metrics["PSNR"]) if metrics["PSNR"] else 0,
                    "SSIM": max(metrics["SSIM"]) if metrics["SSIM"] else 0,
                    "LPIPS": min(metrics["LPIPS"]) if metrics["LPIPS"] else 1,
                },
                "best_compression": {
                    "frame_number": len(metrics["Compression_Ratio"]),
                    "ratio": max(metrics["Compression_Ratio"]) if metrics["Compression_Ratio"] else 0,
                    "PSNR": metrics["PSNR"][metrics["Compression_Ratio"].index(max(metrics["Compression_Ratio"]))] if metrics["Compression_Ratio"] else 0,
                    "SSIM": metrics["SSIM"][metrics["Compression_Ratio"].index(max(metrics["Compression_Ratio"]))] if metrics["Compression_Ratio"] else 0,
                }
            }
            
            # En iyi değerleri göster
            self.update_current_best_label(best_metrics)
            
        except Exception as e:
            print(f"Video metrikleri güncelleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def compress_and_analyze_video(self):
        """Video sıkıştırma ve analiz işlemini başlatır."""
        try:
            # Video dosyası seçili mi kontrol et
            if not hasattr(self, 'video_path') or not self.video_path:
                print("Hata: Video dosyası seçilmemiş!")
                return
            
            # Video işleme parametrelerini ayarla
            output_path = self.video_path.rsplit('.', 1)[0] + '_compressed.' + self.video_path.rsplit('.', 1)[1]
            codec = self.video_codec_combo.currentText()
            crf = self.video_quality_spin.value()
            
            # Video işlemeyi başlat
            self.video_processor.set_parameters(self.video_path, output_path, codec, crf)
            self.video_processor.start()
            
        except Exception as e:
            print(f"Video işleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def toggle_fullscreen(self):
        """Tam ekran modunu açıp kapatır."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def resizeEvent(self, event):
        """Pencere boyutu değiştiğinde çağrılır."""
        super().resizeEvent(event)
        # Frame'leri yeniden boyutlandır
        if hasattr(self, 'original_image'):
            self.display_image(self.original_image, self.original_image_label)
        if hasattr(self, 'compressed_image'):
            self.display_image(self.compressed_image, self.compressed_image_label)
    
    def update_current_best_label(self, current_best: dict):
        """Anlık en iyi değerleri günceller."""
        text = f"""
        <table style='color: #FFFFFF;'>
            <tr><td colspan='2'><b>En İyi Kalite (Kare {current_best['best_quality']['frame_number']}):</b></td></tr>
            <tr><td>PSNR:</td><td>{current_best['best_quality']['PSNR']:.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{current_best['best_quality']['SSIM']:.4f}</td></tr>
            <tr><td>LPIPS:</td><td>{current_best['best_quality']['LPIPS']:.4f}</td></tr>
            
            <tr><td colspan='2'><b>En İyi Sıkıştırma (Kare {current_best['best_compression']['frame_number']}):</b></td></tr>
            <tr><td>Sıkıştırma Oranı:</td><td>{current_best['best_compression']['ratio']:.2f}%</td></tr>
            <tr><td>PSNR:</td><td>{current_best['best_compression']['PSNR']:.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{current_best['best_compression']['SSIM']:.4f}</td></tr>
        </table>
        """
        self.current_best_label.setText(text)
    
    def on_video_processing_finished(self):
        """Video işleme tamamlandığında çağrılır."""
        self.is_video_processing = False
        self.video_start_stop_btn.setText("Başlat")
        self.video_start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.video_pause_resume_btn.setEnabled(False)
        self.video_pause_resume_btn.setText("Duraklat")
    
    def setup_video_controls(self):
        """Video kontrol butonlarını oluşturur."""
        video_controls = QHBoxLayout()
        
        # Başlat/Durdur butonu
        self.video_start_stop_btn = QPushButton("Başlat")
        self.video_start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.video_start_stop_btn.clicked.connect(self.toggle_video_processing)
        
        # Duraklat/Devam Et butonu
        self.video_pause_resume_btn = QPushButton("Duraklat")
        self.video_pause_resume_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #f1c40f;
            }
        """)
        self.video_pause_resume_btn.clicked.connect(self.toggle_video_pause)
        self.video_pause_resume_btn.setEnabled(False)
        
        video_controls.addWidget(self.video_start_stop_btn)
        video_controls.addWidget(self.video_pause_resume_btn)
        
        return video_controls
    
    def toggle_video_processing(self):
        """Video işlemeyi başlatır veya durdurur."""
        if not self.is_video_processing:
            # Video işlemeyi başlat
            self.is_video_processing = True
            self.video_start_stop_btn.setText("Durdur")
            self.video_start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #c0392b;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #e74c3c;
                }
            """)
            self.video_pause_resume_btn.setEnabled(True)
            self.compress_and_analyze_video()
        else:
            # Video işlemeyi durdur
            self.is_video_processing = False
            self.video_processor.stop()
            self.video_start_stop_btn.setText("Başlat")
            self.video_start_stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 100px;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                }
            """)
            self.video_pause_resume_btn.setEnabled(False)
            self.video_pause_resume_btn.setText("Duraklat")
    
    def toggle_video_pause(self):
        """Video işlemeyi duraklatır veya devam ettirir."""
        if not self.is_video_paused:
            # Video işlemeyi duraklat
            self.is_video_paused = True
            self.video_processor.pause()
            self.video_pause_resume_btn.setText("Devam Et")
        else:
            # Video işlemeye devam et
            self.is_video_paused = False
            self.video_processor.resume()
            self.video_pause_resume_btn.setText("Duraklat")
    
    def find_optimal_method(self, results: dict) -> dict:
        """En iyi sıkıştırma yöntemini ve parametrelerini bulur."""
        best_overall = {'score': -float('inf')}
        best_compression = {'compression_ratio': -float('inf')}  # En yüksek sıkıştırma oranı en iyidir
        best_quality = {'quality_score': -float('inf')}

        for method, data in results.items():
            for i, quality in enumerate(data['qualities']):
                # Kalite metrikleri
                psnr = data['psnr_values'][i]
                ssim = data['ssim_values'][i]
                lpips = data['lpips_values'][i]
                compression_ratio = data['compression_ratios'][i]
                
                # Kalite skoru hesaplama (PSNR, SSIM ve LPIPS'e göre)
                quality_score = (psnr / 50.0) * 0.3 + ssim * 0.4 + (1 - lpips) * 0.3
                
                # Genel skor hesaplama (kalite ve sıkıştırma oranına göre)
                compression_score = compression_ratio / 100.0  # Normalize edilmiş sıkıştırma oranı
                overall_score = quality_score * 0.7 + compression_score * 0.3

                # En iyi genel performans
                if overall_score > best_overall['score']:
                    best_overall = {
                        'method': method,
                        'quality': quality,
                        'score': overall_score,
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips,
                        'compression_ratio': compression_ratio
                    }

                # En iyi sıkıştırma (en yüksek sıkıştırma oranı)
                if compression_ratio > best_compression['compression_ratio']:
                    best_compression = {
                        'method': method,
                        'quality': quality,
                        'compression_ratio': compression_ratio,
                        'score': overall_score
                    }

                # En iyi kalite
                if quality_score > best_quality['quality_score']:
                    best_quality = {
                        'method': method,
                        'quality': quality,
                        'quality_score': quality_score,
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips
                    }

        return {
            'best_overall': best_overall,
            'best_compression': best_compression,
            'best_quality': best_quality
        }

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MultimediaBenchmark()
    sys.exit(app.exec_())
