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

# Bu modüllerin de proje içerisinde tanımlı olduğu varsayılıyor
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
        
        # Video metrikleri için geçmiş veriler
        self.metrics_history = {
            "PSNR": [],
            "SSIM": [],
            "LPIPS": [],
            "Histogram_Similarity": [],
            "Entropy": [],
            "Compression_Ratio": [],
            "Compression_Algorithm_Ratios": [],
            "Compression_Algorithm_Times": [],
            "Codec_Performance": []
        }
        
        # Görüntü/Video modu
        self.mode = None  # 'image' veya 'video'
    
    def set_mode(self, mode: str):
        """Görüntü veya video modunu ayarlar ve ilgili sekmeleri gösterir."""
        self.mode = mode
        self.setup_plot_tabs()
    
    def setup_plot_tabs(self):
        """Grafik sekmelerini oluşturur."""
        if not self.plot_manager:
            self.plot_manager = PlotManager()
            
        # Mevcut sekmeleri temizle
        while self.plot_tabs.count() > 0:
            self.plot_tabs.removeTab(0)
        
        # Plot canvas'larını temizle
        self.plot_canvases.clear()
        
        # Mod'a göre sekmeleri ayarla
        if self.mode == 'image':
            plot_tabs = {
                "image_compression": "Sıkıştırma Analizi",
                "image_psnr": "PSNR Analizi",
                "image_ssim": "SSIM Analizi",
                "image_time": "Süre Analizi"
            }
        elif self.mode == 'video':
            plot_tabs = {
                "video_metrics": "Video Kalite Metrikleri",
                "entropy": "Entropi Analizi",
                "compression_algorithms": "Sıkıştırma Algoritmaları",
                "codec_comparison": "Codec Karşılaştırması"
            }
        else:
            return
        
        for key, title in plot_tabs.items():
            container = QWidget()
            layout = QVBoxLayout(container)
            
            # Canvas ve figure oluştur
            canvas, figure = self.plot_manager.create_canvas()
            layout.addWidget(canvas)
            
            self.plot_tabs.addTab(container, title)
            self.plot_canvases[key] = canvas
    
    def update_video_plots(self, metrics: dict):
        """Video metriklerini günceller ve grafikleri yeniler."""
        try:
            # Video kalite metrikleri grafiği
            if "video_metrics" in self.plot_canvases:
                canvas = self.plot_canvases["video_metrics"]
                canvas.figure.clear()
                
                # 2x3 grid oluştur
                gs = canvas.figure.add_gridspec(2, 3)
                
                # PSNR grafiği
                ax1 = canvas.figure.add_subplot(gs[0, 0])
                ax1.plot(metrics["PSNR"], label="PSNR", color='blue')
                ax1.set_title("PSNR Değişimi")
                ax1.set_xlabel("Frame")
                ax1.set_ylabel("PSNR (dB)")
                ax1.grid(True)
                
                # SSIM grafiği
                ax2 = canvas.figure.add_subplot(gs[0, 1])
                ax2.plot(metrics["SSIM"], label="SSIM", color='green')
                ax2.set_title("SSIM Değişimi")
                ax2.set_xlabel("Frame")
                ax2.set_ylabel("SSIM")
                ax2.grid(True)
                
                # LPIPS grafiği
                ax3 = canvas.figure.add_subplot(gs[0, 2])
                ax3.plot(metrics["LPIPS"], label="LPIPS", color='red')
                ax3.set_title("LPIPS Değişimi")
                ax3.set_xlabel("Frame")
                ax3.set_ylabel("LPIPS")
                ax3.grid(True)
                
                # Histogram Benzerliği grafiği
                ax4 = canvas.figure.add_subplot(gs[1, 0])
                ax4.plot(metrics["Histogram_Similarity"], label="Histogram", color='purple')
                ax4.set_title("Histogram Benzerliği")
                ax4.set_xlabel("Frame")
                ax4.set_ylabel("Benzerlik")
                ax4.grid(True)
                
                # Entropi grafiği
                ax5 = canvas.figure.add_subplot(gs[1, 1])
                ax5.plot(metrics["Entropy"], label="Entropi", color='orange')
                ax5.set_title("Entropi Değişimi")
                ax5.set_xlabel("Frame")
                ax5.set_ylabel("Entropi")
                ax5.grid(True)
                
                # Sıkıştırma Oranı grafiği
                ax6 = canvas.figure.add_subplot(gs[1, 2])
                ax6.plot(metrics["Compression_Ratio"], label="Sıkıştırma", color='brown')
                ax6.set_title("Sıkıştırma Oranı")
                ax6.set_xlabel("Frame")
                ax6.set_ylabel("Oran (%)")
                ax6.grid(True)
                
                canvas.figure.tight_layout()
                canvas.draw()
            
            # Entropi analizi grafiği
            if "entropy" in self.plot_canvases:
                canvas = self.plot_canvases["entropy"]
                canvas.figure.clear()
                
                ax = canvas.figure.add_subplot(111)
                ax.plot(metrics["Entropy"], label="Entropi", color='orange', linewidth=2)
                ax.plot(metrics["Compression_Ratio"], label="Sıkıştırma Oranı", color='brown', linewidth=2)
                ax.set_title("Entropi ve Sıkıştırma Oranı Karşılaştırması")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Değer")
                ax.legend()
                ax.grid(True)
                
                canvas.figure.tight_layout()
                canvas.draw()
            
            # Sıkıştırma algoritmaları karşılaştırması
            if "compression_algorithms" in self.plot_canvases:
                canvas = self.plot_canvases["compression_algorithms"]
                canvas.figure.clear()
                
                gs = canvas.figure.add_gridspec(2, 2)
                
                algorithms = ["zlib", "gzip", "bz2", "lzma"]
                x = range(len(algorithms))
                
                compression_metrics = metrics.get("Compression_Algorithms", {})
                if compression_metrics and isinstance(compression_metrics, dict):
                    # Sıkıştırma Oranları
                    ax1 = canvas.figure.add_subplot(gs[0, 0])
                    ratios = [compression_metrics.get(algo, {}).get("ratio", 0) for algo in algorithms]
                    bars = ax1.bar(x, ratios, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(algorithms)
                    ax1.set_title("Sıkıştırma Algoritmaları - Sıkıştırma Oranları")
                    ax1.set_ylabel("Sıkıştırma Oranı (%)")
                    ax1.grid(True, alpha=0.3)
                    
                    # İşlem Süreleri
                    ax2 = canvas.figure.add_subplot(gs[0, 1])
                    times = [compression_metrics.get(algo, {}).get("time", 0) for algo in algorithms]
                    bars = ax2.bar(x, times, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}s', ha='center', va='bottom')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(algorithms)
                    ax2.set_title("Sıkıştırma Algoritmaları - İşlem Süreleri")
                    ax2.set_ylabel("Süre (s)")
                    ax2.grid(True, alpha=0.3)
                    
                    # Bellek Kullanımı
                    ax3 = canvas.figure.add_subplot(gs[1, 0])
                    memory = [compression_metrics.get(algo, {}).get("memory", 0) for algo in algorithms]
                    bars = ax3.bar(x, memory, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}MB', ha='center', va='bottom')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(algorithms)
                    ax3.set_title("Sıkıştırma Algoritmaları - Bellek Kullanımı")
                    ax3.set_ylabel("Bellek (MB)")
                    ax3.grid(True, alpha=0.3)
                    
                    # Sıkıştırma Hızı
                    ax4 = canvas.figure.add_subplot(gs[1, 1])
                    speed = [compression_metrics.get(algo, {}).get("speed", 0) for algo in algorithms]
                    bars = ax4.bar(x, speed, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}MB/s', ha='center', va='bottom')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(algorithms)
                    ax4.set_title("Sıkıştırma Algoritmaları - Sıkıştırma Hızı")
                    ax4.set_ylabel("MB/s")
                    ax4.grid(True, alpha=0.3)
                
                canvas.figure.tight_layout()
                canvas.draw()
            
            # Codec karşılaştırması
            if "codec_comparison" in self.plot_canvases:
                canvas = self.plot_canvases["codec_comparison"]
                canvas.figure.clear()
                
                gs = canvas.figure.add_gridspec(2, 2)
                
                codec_metrics = metrics.get("Codec_Performance", {})
                if codec_metrics and isinstance(codec_metrics, dict):
                    codecs = list(codec_metrics.keys())
                    x = range(len(codecs))
                    
                    # PSNR vs Bitrate
                    ax1 = canvas.figure.add_subplot(gs[0, 0])
                    for codec in codecs:
                        if codec in codec_metrics:
                            ax1.plot([codec_metrics[codec].get("bitrate", 0)], 
                                   [codec_metrics[codec].get("psnr", 0)], 
                                   marker='o', label=codec, linewidth=2)
                    ax1.set_title("PSNR vs Bitrate")
                    ax1.set_xlabel("Bitrate (Mbps)")
                    ax1.set_ylabel("PSNR (dB)")
                    ax1.grid(True)
                    ax1.legend()
                    
                    # SSIM vs Bitrate
                    ax2 = canvas.figure.add_subplot(gs[0, 1])
                    for codec in codecs:
                        if codec in codec_metrics:
                            ax2.plot([codec_metrics[codec].get("bitrate", 0)], 
                                   [codec_metrics[codec].get("ssim", 0)], 
                                   marker='o', label=codec, linewidth=2)
                    ax2.set_title("SSIM vs Bitrate")
                    ax2.set_xlabel("Bitrate (Mbps)")
                    ax2.set_ylabel("SSIM")
                    ax2.grid(True)
                    ax2.legend()
                    
                    # Sıkıştırma Oranı
                    ax3 = canvas.figure.add_subplot(gs[1, 0])
                    ratios = [codec_metrics[codec].get("ratio", 0) for codec in codecs]
                    bars = ax3.bar(x, ratios, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(codecs)
                    ax3.set_title("Codec Sıkıştırma Oranları")
                    ax3.set_ylabel("Sıkıştırma Oranı (%)")
                    ax3.grid(True)
                    
                    # İşlem Süresi
                    ax4 = canvas.figure.add_subplot(gs[1, 1])
                    times = [codec_metrics[codec].get("time", 0) for codec in codecs]
                    bars = ax4.bar(x, times, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}s', ha='center', va='bottom')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(codecs)
                    ax4.set_title("Codec İşlem Süreleri")
                    ax4.set_ylabel("Süre (s)")
                    ax4.grid(True)
                
                canvas.figure.tight_layout()
                canvas.draw()
            
        except Exception as e:
            print(f"Video grafikleri güncelleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_plots(self, results: dict):
        """Görüntü grafiklerini günceller."""
        plt = self.plot_manager.plt
        plt.ioff()  # Interactive modu kapat
        
        try:
            # Grafikleri temizle ve güncelle
            for canvas in self.plot_canvases.values():
                canvas.figure.clear()
            
            # "image_compression" sekmesi
            ax1 = self.plot_canvases["image_compression"].figure.add_subplot(111)
            for method in results:
                ax1.plot(results[method]["qualities"], 
                        results[method]["compression_ratios"], 
                        marker='.', label=method)
            ax1.set_xlabel("Kalite Faktörü")
            ax1.set_ylabel("Sıkıştırma Oranı (%)")
            ax1.set_title("Sıkıştırma Oranı vs Kalite")
            ax1.legend()
            ax1.grid(True)
            self.plot_canvases["image_compression"].figure.tight_layout()
            
            # "image_psnr" sekmesi
            ax2 = self.plot_canvases["image_psnr"].figure.add_subplot(111)
            for method in results:
                ax2.plot(results[method]["qualities"], 
                        results[method]["psnr_values"], 
                        marker='.', label=method)
            ax2.set_xlabel("Kalite Faktörü")
            ax2.set_ylabel("PSNR (dB)")
            ax2.set_title("PSNR vs Kalite")
            ax2.legend()
            ax2.grid(True)
            self.plot_canvases["image_psnr"].figure.tight_layout()
            
            # "image_ssim" sekmesi
            ax3 = self.plot_canvases["image_ssim"].figure.add_subplot(111)
            for method in results:
                ax3.plot(results[method]["qualities"], 
                        results[method]["ssim_values"], 
                        marker='.', label=method)
            ax3.set_xlabel("Kalite Faktörü")
            ax3.set_ylabel("SSIM")
            ax3.setTitle = ("SSIM vs Kalite")
            ax3.set_title("SSIM vs Kalite")
            ax3.legend()
            ax3.grid(True)
            self.plot_canvases["image_ssim"].figure.tight_layout()
            
            # "image_time" sekmesi
            ax4 = self.plot_canvases["image_time"].figure.add_subplot(111)
            for method in results:
                ax4.plot(results[method]["qualities"], 
                        results[method]["processing_times"], 
                        marker='.', label=method)
            ax4.set_xlabel("Kalite Faktörü")
            ax4.set_ylabel("İşlem Süresi (s)")
            ax4.set_title("İşlem Süresi vs Kalite")
            ax4.legend()
            ax4.grid(True)
            self.plot_canvases["image_time"].figure.tight_layout()
            
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
            
            # Mevcut modu belirle
            if hasattr(self, 'video_path') and self.video_path:
                self.plot_window.set_mode('video')
            elif hasattr(self, 'original_image') and self.original_image is not None:
                self.plot_window.set_mode('image')
            
            # Plot sekmelerini oluştur
            self.plot_window.setup_plot_tabs()
            
            # Eğer görüntü modundaysa ve orijinal görüntü yüklenmişse grafikleri güncelle
            if (self.plot_window.mode == 'image' and 
                hasattr(self, 'original_image') and 
                self.original_image is not None):
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
            
            # Eğer hali hazırda plot penceresi varsa modu 'image' olarak ayarla
            if self.plot_window is not None:
                self.plot_window.set_mode('image')
            
            # Görsel grafikleri göster
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
            
            # Eğer hali hazırda plot penceresi varsa modu 'video' olarak ayarla
            if self.plot_window is not None:
                self.plot_window.set_mode('video')
            
            # Video işleme parametrelerini ayarla (ancak henüz başlatma)
            output_path = file_name.rsplit('.', 1)[0] + '_compressed.' + file_name.rsplit('.', 1)[1]
            codec = self.video_codec_combo.currentText()
            crf = self.video_quality_spin.value()
            self.video_processor.set_parameters(file_name, output_path, codec, crf)
    
    def show_image_plots(self):
        """Görüntü grafiklerini oluşturup gösterir."""
        if hasattr(self, 'original_image') and self.original_image is not None:
            try:
                # PlotWindow nesnesi yoksa oluştur
                if self.plot_window is None:
                    self.plot_window = PlotWindow(self)
                    self.plot_window.plot_manager = self.plot_manager
                    # Önce modu ayarlıyoruz
                    self.plot_window.set_mode('image')
                    # Sonra sekmeleri oluşturuyoruz
                    self.plot_window.setup_plot_tabs()
                
                # Tüm yöntemleri analiz et
                results = self.image_processor.analyze_all_methods(self.original_image)
                # Grafikleri güncelle
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
        """Görüntüyü sıkıştırır ve analiz eder."""
        try:
            print("Sıkıştırma başlıyor...")
            method = self.image_compression_combo.currentText()
            quality = self.image_quality_spin.value()
            
            # Sıkıştırma ve analiz
            compressed, metrics = self.image_processor.compress_and_analyze(
                self.original_image, method, quality
            )
            
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
    
    def update_metrics_label(self, metrics: dict):
        """Metrik etiketini günceller."""
        text = f"""
        <table style='color: #ECF0F1;'>
            <tr><td colspan='2' style='text-align:center;'><b>Sıkıştırma Bilgileri</b></td></tr>
            <tr><td><b>Yöntem:</b></td><td>{metrics.get('method', 'N/A')}</td></tr>
            <tr><td><b>Kalite:</b></td><td>{metrics.get('quality', 0)}</td></tr>
            <tr><td colspan='2' style='text-align:center;'><b>Dosya Boyutları</b></td></tr>
            <tr><td><b>Orijinal:</b></td><td>{metrics.get('original_size', 0)/1024:.2f} KB</td></tr>
            <tr><td><b>Sıkıştırılmış:</b></td><td>{metrics.get('file_size', 0)/1024:.2f} KB</td></tr>
            <tr><td><b>Sıkıştırma Oranı:</b></td><td>{metrics.get('compression_ratio', 0):.2f}%</td></tr>
            <tr><td colspan='2' style='text-align:center;'><b>Kalite Metrikleri</b></td></tr>
            <tr><td><b>PSNR:</b></td><td>{metrics.get('PSNR', 0):.2f} dB</td></tr>
            <tr><td><b>SSIM:</b></td><td>{metrics.get('SSIM', 0):.4f}</td></tr>
            <tr><td><b>LPIPS:</b></td><td>{metrics.get('LPIPS', 0):.4f}</td></tr>
        </table>
        """
        self.metrics_label.setText(text)
    
    def update_recommendation_label(self, optimal: dict):
        """Öneri etiketini günceller."""
        # En iyi sıkıştırma oranına sahip yöntem
        best_compression = optimal.get('best_compression', {})
        # En iyi kaliteye sahip yöntem
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
            <tr><td>LPIPS:</td><td>{best_overall.get('lpips', 0):.4f}</td></tr>

            <tr><td colspan='2'><b>En İyi Sıkıştırma:</b></td></tr>
            <tr><td>Yöntem:</td><td>{best_compression.get('method', 'N/A')}</td></tr>
            <tr><td>Kalite:</td><td>{best_compression.get('quality', 0)}</td></tr>
            <tr><td>Sıkıştırma Oranı:</td><td>{best_compression.get('compression_ratio', 0):.2f}%</td></tr>

            <tr><td colspan='2'><b>En İyi Kalite:</b></td></tr>
            <tr><td>Yöntem:</td><td>{best_quality.get('method', 'N/A')}</td></tr>
            <tr><td>Kalite:</td><td>{best_quality.get('quality', 0)}</td></tr>
            <tr><td>PSNR:</td><td>{best_quality.get('psnr', 0):.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{best_quality.get('ssim', 0):.4f}</td></tr>
            <tr><td>LPIPS:</td><td>{best_quality.get('lpips', 0):.4f}</td></tr>
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
            height = self.original_image_label.height()
            width = self.original_image_label.width()
            
            # Orijinal frame
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            height_o, width_o = original_rgb.shape[:2]
            scale = min(width/width_o, height/height_o)
            new_width = int(width_o * scale)
            new_height = int(height_o * scale)
            original_resized = cv2.resize(original_rgb, (new_width, new_height))
            
            # Sıkıştırılmış frame
            compressed_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
            compressed_resized = cv2.resize(compressed_rgb, (new_width, new_height))
            
            bytes_per_line = 3 * new_width
            
            # Orijinal
            q_image_original = QImage(original_resized.data, new_width, new_height, 
                                      bytes_per_line, QImage.Format_RGB888)
            pixmap_original = QPixmap.fromImage(q_image_original)
            self.original_image_label.setPixmap(pixmap_original)
            
            # Sıkıştırılmış
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
            # Mevcut metrikleri al
            current_metrics = {
                "PSNR": float(metrics["PSNR"][-1]) if metrics["PSNR"] else 0,
                "SSIM": float(metrics["SSIM"][-1]) if metrics["SSIM"] else 0,
                "LPIPS": float(metrics["LPIPS"][-1]) if metrics["LPIPS"] else 0,
                "Histogram_Similarity": float(metrics["Histogram_Similarity"][-1]) if metrics["Histogram_Similarity"] else 0,
                "Entropy": float(metrics["Entropy"][-1]) if metrics["Entropy"] else 0,
                "Compression_Ratio": float(metrics["Compression_Ratio"][-1]) if metrics["Compression_Ratio"] else 0
            }
            
            # En iyi değerleri hesapla
            best_metrics = {
                "best_quality": {
                    "frame_number": len(metrics["PSNR"]),
                    "PSNR": max(metrics["PSNR"]) if metrics["PSNR"] else 0,
                    "SSIM": max(metrics["SSIM"]) if metrics["SSIM"] else 0,
                    "LPIPS": min(metrics["LPIPS"]) if metrics["LPIPS"] else 1,
                    "Entropy": max(metrics["Entropy"]) if metrics["Entropy"] else 0
                },
                "best_compression": {
                    "frame_number": len(metrics["Compression_Ratio"]),
                    "ratio": max(metrics["Compression_Ratio"]) if metrics["Compression_Ratio"] else 0,
                    "PSNR": metrics["PSNR"][metrics["Compression_Ratio"].index(max(metrics["Compression_Ratio"]))] if metrics["Compression_Ratio"] else 0,
                    "SSIM": metrics["SSIM"][metrics["Compression_Ratio"].index(max(metrics["Compression_Ratio"]))] if metrics["Compression_Ratio"] else 0,
                    "Entropy": metrics["Entropy"][metrics["Compression_Ratio"].index(max(metrics["Compression_Ratio"]))] if metrics["Compression_Ratio"] else 0
                }
            }
            
            # Codec performansını hesapla
            codec_metrics = metrics.get("Codec_Performance", {})
            if codec_metrics and isinstance(codec_metrics, dict):
                codec_scores = {}
                for codec, values in codec_metrics.items():
                    psnr_score = values["psnr"] / 50.0
                    ssim_score = values["ssim"]
                    # Use either compression_ratio or ratio, whichever is available
                    compression_ratio = values.get("compression_ratio", values.get("ratio", 0))
                    compression_score = compression_ratio / 100.0
                    total_score = psnr_score * 0.3 + ssim_score * 0.4 + compression_score * 0.3
                    codec_scores[codec] = {
                        "total_score": total_score,
                        "psnr_score": psnr_score,
                        "ssim_score": ssim_score,
                        "compression_score": compression_score
                    }
                best_codec, best_scores = max(codec_scores.items(), key=lambda x: x[1]["total_score"])
                best_metrics["best_quality"]["codec"] = best_codec
                best_metrics["best_quality"]["total_score"] = best_scores["total_score"]
            
            # Sıkıştırma algoritması performansını hesapla
            compression_metrics = metrics.get("Compression_Algorithms", {})
            if compression_metrics and isinstance(compression_metrics, dict):
                algo_scores = {}
                for algo, values in compression_metrics.items():
                    algo_scores[algo] = values["ratio"]
                best_algo = max(algo_scores.items(), key=lambda x: x[1])
                best_metrics["best_compression"]["algorithm"] = best_algo[0]
                best_metrics["best_compression"]["algorithm_score"] = best_algo[1]
            
            # Metrikleri güncelle
            text = f"""
            <table style='color: #ECF0F1;'>
                <tr><td colspan='2'><b>Mevcut Durum:</b></td></tr>
                <tr><td>Codec:</td><td>{self.video_codec_combo.currentText()}</td></tr>
                <tr><td>CRF:</td><td>{str(self.video_quality_spin.value())}</td></tr>
                <tr><td>PSNR:</td><td>{current_metrics["PSNR"]:.2f} dB</td></tr>
                <tr><td>SSIM:</td><td>{current_metrics["SSIM"]:.4f}</td></tr>
                <tr><td>LPIPS:</td><td>{current_metrics["LPIPS"]:.4f}</td></tr>
                <tr><td>Histogram Benzerliği:</td><td>{current_metrics["Histogram_Similarity"]:.4f}</td></tr>
                <tr><td>Entropi:</td><td>{current_metrics["Entropy"]:.4f}</td></tr>
                <tr><td>Sıkıştırma Oranı:</td><td>{current_metrics["Compression_Ratio"]:.2f}%</td></tr>
            </table>
            """
            self.metrics_label.setText(text)
            
            # En iyi değerleri güncelle
            self.update_current_best_label(best_metrics)
            
            # Grafikleri güncelle
            if self.plot_window is not None and self.plot_window.isVisible():
                self.plot_window.update_video_plots(metrics)
            
        except Exception as e:
            print(f"Video metrikleri güncelleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def compress_and_analyze_video(self):
        """Video sıkıştırma ve analiz işlemini başlatır."""
        try:
            if not hasattr(self, 'video_path') or not self.video_path:
                print("Hata: Video dosyası seçilmemiş!")
                return
            
            output_path = self.video_path.rsplit('.', 1)[0] + '_compressed.' + self.video_path.rsplit('.', 1)[1]
            codec = self.video_codec_combo.currentText()
            crf = self.video_quality_spin.value()
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
        if hasattr(self, 'original_image'):
            self.display_image(self.original_image, self.original_image_label)
        if hasattr(self, 'compressed_image'):
            self.display_image(self.compressed_image, self.compressed_image_label)
    
    def update_current_best_label(self, current_best: dict):
        """Anlık en iyi değerleri günceller."""
        text = f"""
        <table style='color: #FFFFFF;'>
            <tr><td colspan='2' style='text-align:center; background-color: #2c3e50;'><b>En İyi Kalite Değerleri</b></td></tr>
            <tr><td colspan='2'><b>Kare {current_best['best_quality']['frame_number']}:</b></td></tr>
            <tr><td>PSNR:</td><td>{current_best['best_quality']['PSNR']:.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{current_best['best_quality']['SSIM']:.4f}</td></tr>
            <tr><td>LPIPS:</td><td>{current_best['best_quality']['LPIPS']:.4f}</td></tr>
            <tr><td>Entropi:</td><td>{current_best['best_quality']['Entropy']:.4f}</td></tr>
            <tr><td colspan='2'><b>Önerilen Yöntem:</b></td></tr>
            <tr><td>Codec:</td><td>{current_best['best_quality'].get('codec', 'N/A')}</td></tr>
            <tr><td>Toplam Skor:</td><td>{current_best['best_quality'].get('total_score', 0.0):.4f}</td></tr>
            
            <tr><td colspan='2' style='text-align:center; background-color: #2c3e50;'><b>En İyi Sıkıştırma Değerleri</b></td></tr>
            <tr><td colspan='2'><b>Kare {current_best['best_compression']['frame_number']}:</b></td></tr>
            <tr><td>Sıkıştırma Oranı:</td><td>{current_best['best_compression']['ratio']:.2f}%</td></tr>
            <tr><td>PSNR:</td><td>{current_best['best_compression']['PSNR']:.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{current_best['best_compression']['SSIM']:.4f}</td></tr>
            <tr><td>Entropi:</td><td>{current_best['best_compression']['Entropy']:.4f}</td></tr>
            <tr><td colspan='2'><b>Önerilen Yöntem:</b></td></tr>
            <tr><td>Algoritma:</td><td>{current_best['best_compression'].get('algorithm', 'N/A')}</td></tr>
            <tr><td>Sıkıştırma Skoru:</td><td>{current_best['best_compression'].get('algorithm_score', 0.0):.2f}</td></tr>
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
            self.video_processor.stop()  # VideoProcessor'a durdurma sinyali gönder
            
            if hasattr(self.video_processor, 'cap') and self.video_processor.cap:
                self.video_processor.cap.release()
            if hasattr(self.video_processor, 'writer') and self.video_processor.writer:
                self.video_processor.writer.release()
            
            if self.video_processor.isRunning():
                self.video_processor.wait()
            
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
            self.is_video_paused = False
            
            self.progress_bar.setValue(0)
            self.time_label.setText("Kalan Süre: 0.0s | Geçen Süre: 0.0s")
    
    def toggle_video_pause(self):
        """Video işlemeyi duraklatır veya devam ettirir."""
        if not self.is_video_paused:
            self.is_video_paused = True
            self.video_processor.pause()
            self.video_pause_resume_btn.setText("Devam Et")
        else:
            self.is_video_paused = False
            self.video_processor.resume()
            self.video_pause_resume_btn.setText("Duraklat")
    
    def find_optimal_method(self, results: dict) -> dict:
        """En iyi sıkıştırma yöntemini ve parametrelerini bulur."""
        best_overall = {'score': -float('inf')}
        best_compression = {'compression_ratio': -float('inf')}
        best_quality = {'quality_score': -float('inf')}

        for method, data in results.items():
            for i, quality in enumerate(data['qualities']):
                psnr = data['psnr_values'][i]
                ssim = data['ssim_values'][i]
                lpips = data['lpips_values'][i]
                compression_ratio = data['compression_ratios'][i]
                
                # Kalite skoru (PSNR, SSIM, LPIPS)
                quality_score = (psnr / 50.0) * 0.3 + ssim * 0.4 + (1 - lpips) * 0.3
                
                # Genel skor (kalite %70, sıkıştırma %30 gibi)
                compression_score = compression_ratio / 100.0
                overall_score = quality_score * 0.7 + compression_score * 0.3

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
                if compression_ratio > best_compression['compression_ratio']:
                    best_compression = {
                        'method': method,
                        'quality': quality,
                        'compression_ratio': compression_ratio,
                        'score': overall_score
                    }
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
