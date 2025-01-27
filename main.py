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
        
        # Codec skorları
        self.codec_scores = {}
        
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
                    
                    # Sıkıştırma Hızı
                    ax3 = canvas.figure.add_subplot(gs[1, 0])
                    speed = [compression_metrics.get(algo, {}).get("speed", 0) for algo in algorithms]
                    bars = ax3.bar(x, speed, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}MB/s', ha='center', va='bottom')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(algorithms)
                    ax3.set_title("Sıkıştırma Algoritmaları - Sıkıştırma Hızı")
                    ax3.set_ylabel("MB/s")
                    ax3.grid(True, alpha=0.3)
                
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
                    
                    # PSNR Karşılaştırması
                    ax1 = canvas.figure.add_subplot(gs[0, 0])
                    psnr_values = [codec_metrics[codec].get("psnr", 0) for codec in codecs]
                    # En yüksek PSNR en iyi
                    max_psnr = max(psnr_values) if psnr_values else 1
                    psnr_scores = [v/max_psnr for v in psnr_values]  # Yüksek olan yüksek skor alır
                    bars = ax1.bar(x, psnr_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.2f}dB', ha='center', va='bottom')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(codecs)
                    ax1.set_title("Codec PSNR Karşılaştırması")
                    ax1.set_ylabel("PSNR (dB)")
                    ax1.grid(True, alpha=0.3)
                    
                    # SSIM Karşılaştırması
                    ax2 = canvas.figure.add_subplot(gs[0, 1])
                    ssim_values = [codec_metrics[codec].get("ssim", 0) for codec in codecs]
                    # En yüksek SSIM en iyi
                    max_ssim = max(ssim_values) if ssim_values else 1
                    ssim_scores = [v/max_ssim for v in ssim_values]  # Yüksek olan yüksek skor alır
                    bars = ax2.bar(x, ssim_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(codecs)
                    ax2.set_title("Codec SSIM Karşılaştırması")
                    ax2.set_ylabel("SSIM")
                    ax2.grid(True, alpha=0.3)
                    
                    # Sıkıştırma Oranı
                    ax3 = canvas.figure.add_subplot(gs[1, 0])
                    ratios = [codec_metrics[codec].get("ratio", 0) for codec in codecs]
                    # En düşük oran en iyi
                    min_ratio = min(ratios) if ratios else 1
                    ratio_scores = [min_ratio/v if v != 0 else 0 for v in ratios]  # Düşük olan yüksek skor alır
                    bars = ax3.bar(x, ratios, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom')
                    ax3.set_xticks(x)
                    ax3.set_xticklabels(codecs)
                    ax3.set_title("Codec Sıkıştırma Oranları")
                    ax3.set_ylabel("Sıkıştırma Oranı (%)")
                    ax3.grid(True, alpha=0.3)
                    
                    # İşlem Süresi
                    ax4 = canvas.figure.add_subplot(gs[1, 1])
                    times = [codec_metrics[codec].get("time", 0) for codec in codecs]
                    # En düşük süre en iyi
                    min_time = min(times) if times else 1
                    time_scores = [min_time/v if v != 0 else 0 for v in times]
                    bars = ax4.bar(x, times, color=['#3498db', '#2ecc71', '#e74c3c', '#f1c40f'])
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.3f}s', ha='center', va='bottom')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(codecs)
                    ax4.set_title("Codec İşlem Süreleri")
                    ax4.set_ylabel("Süre (s)")
                    ax4.grid(True, alpha=0.3)
                    
                    # Her codec için toplam skoru hesapla
                    total_scores = {}
                    individual_scores = {}
                    for i, codec in enumerate(codecs):
                        individual_scores[codec] = {
                            'PSNR': psnr_scores[i],
                            'SSIM': ssim_scores[i],
                            'Sıkıştırma': ratio_scores[i],
                            'Süre': time_scores[i]
                        }
                        individual_scores[codec]['Toplam'] = (psnr_scores[i] + ssim_scores[i] + ratio_scores[i] + time_scores[i]) / 4
                        total_scores[codec] = individual_scores[codec]['Toplam']
                    
                    # Codec skorlarını sakla
                    self.codec_scores = individual_scores
                    
                    # Skorları konsola yazdır
                    print("\nCodec Performans Skorları (1.0 en iyi):")
                    print("-" * 60)
                    print(f"{'Codec':<12} {'PSNR':<10} {'SSIM':<10} {'Sıkıştırma':<12} {'Süre':<10} {'Toplam':<10}")
                    print("-" * 60)
                    for codec in codecs:
                        scores = individual_scores[codec]
                        print(f"{codec:<12} {scores['PSNR']:.4f}    {scores['SSIM']:.4f}    {scores['Sıkıştırma']:.4f}      {scores['Süre']:.4f}    {scores['Toplam']:.4f}")
                    print("-" * 60)
                
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
                compression_ratios = [100 - ratio for ratio in results[method]["compression_ratios"]]  # Sıkıştırma oranını tersine çevir
                ax1.plot(results[method]["qualities"], 
                        compression_ratios, 
                        marker='.', label=method)
            ax1.set_xlabel("Kalite Faktörü")
            ax1.set_ylabel("Sıkıştırılmış Boyut (%)")
            ax1.set_title("Sıkıştırılmış Boyut vs Kalite")
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
        self.video_codec_combo.addItems(["H.264", "H.265/HEVC", "MPEG-4", "MJPEG"])
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
        self.original_image_label.setMaximumSize(400, 300)  # Maksimum boyutu sabitle
        self.original_image_label.setStyleSheet("border: 1px solid #666;")
        self.original_image_label.setAlignment(Qt.AlignCenter)  # Merkeze hizala
        original_layout.addWidget(self.original_image_label)
        images_panel.addLayout(original_layout)
        
        # Sıkıştırılmış görüntü
        compressed_layout = QVBoxLayout()
        compressed_layout.addWidget(QLabel("Sıkıştırılmış Görüntü"))
        self.compressed_image_label = QLabel()
        self.compressed_image_label.setMinimumSize(400, 300)
        self.compressed_image_label.setMaximumSize(400, 300)  # Maksimum boyutu sabitle
        self.compressed_image_label.setStyleSheet("border: 1px solid #666;")
        self.compressed_image_label.setAlignment(Qt.AlignCenter)  # Merkeze hizala
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
            
            # Eğer görüntü modundaysa ve sonuçlar mevcutsa grafikleri güncelle
            if (self.plot_window.mode == 'image' and 
                hasattr(self.image_processor, 'last_results') and 
                self.image_processor.last_results is not None):
                try:
                    self.plot_window.update_plots(self.image_processor.last_results)
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
            # Sadece görüntüyü yükle, analiz yapma
            self.original_image = cv2.imread(file_name)
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
        try:
            if image is None:
                return
                
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Label'ın sabit boyutlarını al
            label_width = 400  # Sabit genişlik
            label_height = 300  # Sabit yükseklik
            
            # Görüntünün orijinal boyutlarını al
            height, width = image.shape[:2]
            
            # En-boy oranını koru
            aspect_ratio = width / height
            
            # Label'ın en-boy oranı
            label_aspect_ratio = label_width / label_height
            
            # Yeni boyutları hesapla
            if aspect_ratio > label_aspect_ratio:
                # Görüntü daha geniş, genişliğe göre ölçekle
                new_width = label_width
                new_height = int(new_width / aspect_ratio)
                # Dikey boşlukları hesapla
                y_offset = (label_height - new_height) // 2
                x_offset = 0
            else:
                # Görüntü daha uzun, yüksekliğe göre ölçekle
                new_height = label_height
                new_width = int(new_height * aspect_ratio)
                # Yatay boşlukları hesapla
                x_offset = (label_width - new_width) // 2
                y_offset = 0
            
            # Görüntüyü yeniden boyutlandır
            scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Siyah arka plan oluştur
            background = np.zeros((label_height, label_width, 3), dtype=np.uint8)
            
            # Ölçeklendirilmiş görüntüyü arka plana yerleştir
            y_start = y_offset
            y_end = y_start + new_height
            x_start = x_offset
            x_end = x_start + new_width
            background[y_start:y_end, x_start:x_end] = scaled_image
            
            # QImage oluştur
            bytes_per_line = 3 * label_width
            q_image = QImage(background.data, label_width, label_height, bytes_per_line, QImage.Format_RGB888)
            
            # QPixmap oluştur ve label'a ayarla
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            
        except Exception as e:
            print(f"Görüntü gösterme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def format_size(self, size_in_bytes: int) -> str:
        """Bayt cinsinden boyutu okunaklı formata dönüştürür."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024.0:
                return f"{size_in_bytes:.2f} {unit}"
            size_in_bytes /= 1024.0
        return f"{size_in_bytes:.2f} TB"

    def compress_and_analyze(self):
        """Görüntüyü sıkıştırır ve analiz eder."""
        try:
            # Butonu devre dışı bırak ve metnini güncelle
            self.compress_btn.setEnabled(False)
            self.compress_btn.setText("İşleniyor...")
            self.compress_btn.repaint()  # Buton metnini hemen güncelle
            QApplication.processEvents()  # UI'ı hemen güncelle
            
            print("Sıkıştırma başlıyor...")
            method = self.image_compression_combo.currentText()
            quality = self.image_quality_spin.value()
            
            # Orijinal boyutu hesapla
            _, original_buffer = cv2.imencode('.png', self.original_image)
            original_size = len(original_buffer)
            self.image_processor.original_size = original_size
            
            # Sıkıştırma ve analiz
            compressed, metrics = self.image_processor.compress_and_analyze(
                self.original_image, method, quality
            )
            
            print("Arayüz güncelleniyor...")
            # Anlık metrikleri göster
            metrics_text = f"""
            <table style='color: #ECF0F1;'>
                <tr><td colspan='2'><b>Anlık Metrikler:</b></td></tr>
                <tr><td>Sıkıştırma Oranı:</td><td>{metrics['compression_ratio']:.2f}%</td></tr>
                <tr><td>Orijinal Boyut:</td><td>{self.format_size(original_size)}</td></tr>
                <tr><td>Sıkıştırılmış Boyut:</td><td>{self.format_size(metrics['file_size'])}</td></tr>
                <tr><td>PSNR:</td><td>{metrics['PSNR']:.2f} dB</td></tr>
                <tr><td>SSIM:</td><td>{metrics['SSIM']:.4f}</td></tr>
                <tr><td>LPIPS:</td><td>{metrics['LPIPS']:.4f}</td></tr>
            </table>
            """
            self.metrics_label.setText(metrics_text)
            
            # Sıkıştırılmış görüntüyü göster
            self.display_image(compressed, self.compressed_image_label)
            
            print("Tüm yöntemler analiz ediliyor...")
            # Tüm yöntemleri analiz et
            results = self.image_processor.analyze_all_methods(self.original_image)
            # Sonuçları sakla
            self.image_processor.last_results = results
            
            print("En iyi yöntem belirleniyor...")
            # En iyi yöntemi bul
            optimal = self.find_optimal_method(results)
            
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
            
        finally:
            # İşlem bittiğinde butonu tekrar aktif et ve metnini güncelle
            self.compress_btn.setEnabled(True)
            self.compress_btn.setText("Sıkıştır ve Analiz Et")
            self.compress_btn.repaint()  # Buton metnini hemen güncelle
            QApplication.processEvents()  # UI'ı hemen güncelle
    
    def update_metrics_label(self, metrics: dict):
        """Metrik etiketini günceller."""
        text = f"""
        <table style='color: #ECF0F1;'>
            <tr><td colspan='4' style='text-align:center; background-color: #2c3e50;'><b>Performans Skorları</b></td></tr>
            <tr><td colspan='4' style='text-align:center;'><i>(100 en iyi)</i></td></tr>
            <tr>
                <td colspan='2' style='text-align:center; background-color: #34495e;'><b>Codec Performansı</b></td>
                <td colspan='2' style='text-align:center; background-color: #34495e;'><b>Sıkıştırma Algoritmaları</b></td>
            </tr>
            <tr>
                <td colspan='2'>
                    <table style='width:100%; color: #FFFFFF;'>
                        <tr style='background-color: #2c3e50;'>
                            <td><b>Codec</b></td>
                            <td><b>PSNR</b></td>
                            <td><b>SSIM</b></td>
                            <td><b>Sıkış.</b></td>
                            <td><b>Süre</b></td>
                            <td><b>Toplam</b></td>
                        </tr>"""
        
        # Codec skorlarını ekle
        if hasattr(self, 'codec_scores'):
            for codec, scores in self.codec_scores.items():
                # Skorları 100 üzerinden hesapla
                psnr_100 = scores['PSNR'] * 100
                ssim_100 = scores['SSIM'] * 100
                comp_100 = scores['Sıkıştırma'] * 100
                time_100 = scores['Süre'] * 100
                total_100 = scores['Toplam'] * 100
                
                text += f"""
                        <tr>
                            <td>{codec}</td>
                            <td>{psnr_100:.1f}</td>
                            <td>{ssim_100:.1f}</td>
                            <td>{comp_100:.1f}</td>
                            <td>{time_100:.1f}</td>
                            <td>{total_100:.1f}</td>
                        </tr>"""
        
        text += """
                    </table>
                </td>
                <td colspan='2'>
                    <table style='width:100%; color: #FFFFFF;'>
                        <tr style='background-color: #2c3e50;'>
                            <td><b>Algoritma</b></td>
                            <td><b>Sıkış.</b></td>
                            <td><b>Süre</b></td>
                            <td><b>Hız</b></td>
                            <td><b>Toplam</b></td>
                        </tr>"""
        
        # Sıkıştırma algoritması skorlarını ekle
        if "Compression_Algorithms" in metrics:
            compression_metrics = metrics["Compression_Algorithms"]
            if compression_metrics and isinstance(compression_metrics, dict):
                # Maksimum değerleri bul
                max_ratio = max(m['ratio'] for m in compression_metrics.values())
                min_time = min(m['time'] for m in compression_metrics.values())
                max_speed = max(m['speed'] for m in compression_metrics.values())
                
                for algo, values in compression_metrics.items():
                    # Skorları 100 üzerinden hesapla
                    ratio_score = (values['ratio'] / max_ratio) * 100 if max_ratio > 0 else 0
                    time_score = (min_time / values['time']) * 100 if values['time'] > 0 else 0
                    speed_score = (values['speed'] / max_speed) * 100 if max_speed > 0 else 0
                    total_score = (ratio_score + time_score + speed_score) / 3
                    
                    text += f"""
                        <tr>
                            <td>{algo}</td>
                            <td>{ratio_score:.1f}</td>
                            <td>{time_score:.1f}</td>
                            <td>{speed_score:.1f}</td>
                            <td>{total_score:.1f}</td>
                        </tr>"""
        
        text += """
                    </table>
                </td>
            </tr>
        </table>
        """
        self.recommendation_label.setText(text)
    
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
        if hasattr(self.video_processor, 'total_frames'):
            current_frame = self.video_processor.processed_frames
            remaining_frames = self.video_processor.total_frames - current_frame
            
            # Son 10 frame'in ortalama işleme hızını hesapla
            if hasattr(self.video_processor, 'frame_times'):
                recent_times = self.video_processor.frame_times[-10:]  # Son 10 frame'in süreleri
                if recent_times:
                    avg_frame_time = sum(recent_times) / len(recent_times)
                    remaining = remaining_frames * avg_frame_time
            
            # Süre formatlaması
            def format_time(seconds):
                if seconds < 60:
                    return f"{seconds:.1f}s"
                elif seconds < 3600:
                    minutes = seconds // 60
                    seconds = seconds % 60
                    return f"{int(minutes)}dk {int(seconds)}s"
                elif seconds < 86400:  # 24 saat
                    hours = seconds // 3600
                    minutes = (seconds % 3600) // 60
                    return f"{int(hours)}sa {int(minutes)}dk"
                else:
                    days = seconds // 86400
                    hours = (seconds % 86400) // 3600
                    return f"{int(days)}gün {int(hours)}sa"
            
            remaining_str = format_time(remaining)
            elapsed_str = format_time(elapsed)
            
            self.time_label.setText(
                f"Kalan Süre: {remaining_str} | "
                f"Geçen Süre: {elapsed_str}"
            )
    
    def update_compression(self, ratio: float):
        """Sıkıştırma oranını günceller."""
        try:
            # Orijinal boyutu al
            original_size = self.video_processor.original_size
            
            # Frame sayısını al
            current_frame = self.video_processor.processed_frames
            total_frames = self.video_processor.total_frames
            
            # Her frame için ortalama boyut hesapla
            avg_frame_size = original_size / total_frames if total_frames > 0 else 0
            
            # Şu ana kadar işlenen frame'lerin toplam boyutu
            processed_size = avg_frame_size * current_frame
            
            # Sıkıştırılmış boyutu hesapla
            compressed_size = int(processed_size * (1 - ratio/100))
            
            # Yeni sıkıştırma oranı metni
            compression_text = f"""
            <table style='color: #ECF0F1;'>
                <tr><td colspan='2'><b>Anlık Metrikler:</b></td></tr>
                <tr><td>Sıkıştırma Oranı:</td><td>{ratio:.2f}%</td></tr>
                <tr><td>Orijinal Boyut:</td><td>{self.format_size(original_size)}</td></tr>
                <tr><td>İşlenen Boyut:</td><td>{self.format_size(processed_size)}</td></tr>
                <tr><td>Sıkıştırılmış Boyut:</td><td>{self.format_size(compressed_size)}</td></tr>
                <tr><td>LPIPS:</td><td>{self.video_processor.metrics_history["LPIPS"][-1] if self.video_processor.metrics_history["LPIPS"] else 0:.4f}</td></tr>
            </table>
            """
            
            # Metrik etiketini güncelle
            self.metrics_label.setText(compression_text)
            
            # UI'ı hemen güncelle
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Sıkıştırma oranı güncelleme hatası: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_progress(self, value: int):
        """İlerleme çubuğunu günceller."""
        if hasattr(self.video_processor, 'total_frames'):
            # Progress bar'ı güncelle
            self.progress_bar.setValue(value)
            
            # İşlem tamamlandıysa
            if value >= self.video_processor.total_frames - 1:  # Son frame'e ulaşıldığında
                self.video_processor.stop()
                self.on_video_processing_finished()
    
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
                    "LPIPS": min(metrics["LPIPS"]) if metrics["LPIPS"] else 0,
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

            # Sıkıştırma algoritması metriklerini ekle
            if "Compression_Algorithms" in metrics:
                best_metrics["Compression_Algorithms"] = metrics["Compression_Algorithms"]
            
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
            
            # Codec performansını hesapla ve önerilen yöntemi güncelle
            codec_metrics = metrics.get("Codec_Performance", {})
            if codec_metrics and isinstance(codec_metrics, dict):
                codecs = list(codec_metrics.keys())
                
                # PSNR skorları
                psnr_values = [codec_metrics[codec].get("psnr", 0) for codec in codecs]
                max_psnr = max(psnr_values) if psnr_values else 1
                psnr_scores = [v/max_psnr for v in psnr_values]
                
                # SSIM skorları
                ssim_values = [codec_metrics[codec].get("ssim", 0) for codec in codecs]
                max_ssim = max(ssim_values) if ssim_values else 1
                ssim_scores = [v/max_ssim for v in ssim_values]
                
                # Sıkıştırma oranı skorları
                ratios = [codec_metrics[codec].get("ratio", 0) for codec in codecs]
                min_ratio = min(ratios) if ratios else 1
                ratio_scores = [min_ratio/v if v != 0 else 0 for v in ratios]
                
                # İşlem süresi skorları
                times = [codec_metrics[codec].get("time", 0) for codec in codecs]
                min_time = min(times) if times else 1
                time_scores = [min_time/v if v != 0 else 0 for v in times]
                
                # Her codec için toplam skoru hesapla
                individual_scores = {}
                for i, codec in enumerate(codecs):
                    individual_scores[codec] = {
                        'PSNR': psnr_scores[i],
                        'SSIM': ssim_scores[i],
                        'Sıkıştırma': ratio_scores[i],
                        'Süre': time_scores[i],
                        'Toplam': (psnr_scores[i]*1.2 + ssim_scores[i]*1.2 + ratio_scores[i]*1.2 + time_scores[i]*0.4) / 4
                    }
                
                # Codec skorlarını sakla
                self.codec_scores = individual_scores
                
                # Önerilen yöntem etiketini güncelle
                self.update_metrics_label(metrics)
            
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
            
            # Video'nun toplam frame sayısını al
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            output_path = self.video_path.rsplit('.', 1)[0] + '_compressed.' + self.video_path.rsplit('.', 1)[1]
            codec = self.video_codec_combo.currentText()
            crf = self.video_quality_spin.value()
            
            # Progress bar'ı ayarla
            self.progress_bar.setMinimum(0)
            self.progress_bar.setMaximum(total_frames)
            self.progress_bar.setValue(0)
            
            # Video processor parametrelerini ayarla
            self.video_processor.total_frames = total_frames
            self.video_processor.fps = fps
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
        # En iyi codec'i bul
        best_codec = None
        best_codec_score = -1
        if hasattr(self, 'codec_scores'):
            for codec, scores in self.codec_scores.items():
                if scores['Toplam'] > best_codec_score:
                    best_codec_score = scores['Toplam']
                    best_codec = codec

        text = f"""
        <table style='color: #FFFFFF;'>
            <tr><td colspan='2' style='text-align:center; background-color: #2c3e50;'><b>En İyi Kalite Değerleri</b></td></tr>
            <tr><td colspan='2'><b>Kare {current_best['best_quality']['frame_number']}:</b></td></tr>
            <tr><td>PSNR:</td><td>{current_best['best_quality']['PSNR']:.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{current_best['best_quality']['SSIM']:.4f}</td></tr>
            <tr><td>LPIPS:</td><td>{current_best['best_quality']['LPIPS']:.4f}</td></tr>
            <tr><td>Entropi:</td><td>{current_best['best_quality']['Entropy']:.4f}</td></tr>
            <tr><td colspan='2'><b>Önerilen Codec:</b></td></tr>"""

        if best_codec and best_codec_score > 0:
            text += f"""
            <tr><td>Codec:</td><td>{best_codec}</td></tr>
            <tr><td>Toplam Skor:</td><td>{best_codec_score*100:.4f}%</td></tr>"""
        else:
            text += """
            <tr><td colspan='2'>Henüz yeterli veri yok</td></tr>"""
            
        text += f"""
            <tr><td colspan='2' style='text-align:center; background-color: #2c3e50;'><b>En İyi Sıkıştırma Değerleri</b></td></tr>
            <tr><td colspan='2'><b>Kare {current_best['best_compression']['frame_number']}:</b></td></tr>
            <tr><td>Sıkıştırma Oranı:</td><td>{current_best['best_compression']['ratio']:.2f}%</td></tr>
            <tr><td>PSNR:</td><td>{current_best['best_compression']['PSNR']:.2f} dB</td></tr>
            <tr><td>SSIM:</td><td>{current_best['best_compression']['SSIM']:.4f}</td></tr>
            <tr><td>Entropi:</td><td>{current_best['best_compression']['Entropy']:.4f}</td></tr>"""

        # En iyi sıkıştırma algoritmasını bul
        if 'Compression_Algorithms' in current_best:
            best_algo = None
            best_total_score = -1
            for algo, metrics in current_best['Compression_Algorithms'].items():
                # Normalize edilmiş metrikler (0-1 aralığında)
                ratio_score = metrics['ratio'] / max(m['ratio'] for m in current_best['Compression_Algorithms'].values()) * 100
                time_score = min(m['time'] for m in current_best['Compression_Algorithms'].values()) / metrics['time'] * 100
                speed_score = metrics['speed'] / max(m['speed'] for m in current_best['Compression_Algorithms'].values()) * 100
                total_score = (ratio_score + time_score + speed_score) / 3
                
                if total_score > best_total_score:
                    best_total_score = total_score
                    best_algo = algo
            
            if best_algo:
                text += f"""
                <tr><td colspan='2'><b>Önerilen Sıkıştırma Algoritması:</b></td></tr>
                <tr><td>Algoritma:</td><td>{best_algo}</td></tr>
                <tr><td>Toplam Skor:</td><td>{best_total_score:.1f}%</td></tr>"""

        text += """
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
        best_quality = {'psnr': -float('inf')}  # PSNR'a göre en iyi kaliteyi bul

        # Her yöntem için maksimum değerleri bul
        max_psnr = max(max(data['psnr_values']) for data in results.values())
        max_compression = max(max(data['compression_ratios']) for data in results.values())

        # Tüm skorları saklamak için dictionary
        all_scores = {}

        print("\nTüm Yöntemlerin Skorları:")
        print("-" * 100)
        print(f"{'Yöntem':<10} {'Kalite':<8} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10} {'Sıkıştırma':<12} {'Genel Skor':<12}")
        print("-" * 100)

        for method, data in results.items():
            method_scores = []
            for i, quality in enumerate(data['qualities']):
                # Metrik değerlerini al
                psnr = data['psnr_values'][i]
                ssim = data['ssim_values'][i]
                lpips = data['lpips_values'][i]
                compression_ratio = data['compression_ratios'][i]  # Direkt olarak kullan
                
                # Normalize edilmiş metrikler (0-1 aralığında)
                psnr_norm = psnr / max_psnr if max_psnr > 0 else 0
                compression_norm = compression_ratio / max_compression if max_compression > 0 else 0
                
                # Genel skor (sıkıştırma %30, PSNR %70)
                overall_score = (compression_norm * 0.3) + (psnr_norm * 0.7)
                overall_score = overall_score * 100  # 100 üzerinden göster
                
                # Skorları yazdır
                print(f"{method:<10} {quality:<8} {psnr:<10.2f} {ssim:<10.4f} {lpips:<10.4f} {compression_ratio:<12.2f} {overall_score:<12.2f}")
                
                # Skorları sakla
                method_scores.append({
                    'quality': quality,
                    'psnr': psnr,
                    'ssim': ssim,
                    'lpips': lpips,
                    'compression_ratio': compression_ratio,
                    'overall_score': overall_score
                })

                # En iyi genel performans (en yüksek skora sahip olan)
                if overall_score > best_overall['score']:
                    best_overall = {
                        'method': method,
                        'quality': quality,
                        'score': overall_score,  # Skor zaten 100 üzerinden
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips,
                        'compression_ratio': compression_ratio
                    }
                
                # En iyi sıkıştırma (en yüksek sıkıştırma oranına sahip olan)
                if compression_ratio > best_compression['compression_ratio']:
                    best_compression = {
                        'method': method,
                        'quality': quality,
                        'compression_ratio': compression_ratio
                    }
                
                # En iyi kalite (en yüksek PSNR değerine sahip olan)
                if psnr > best_quality['psnr']:
                    best_quality = {
                        'method': method,
                        'quality': quality,
                        'quality_score': psnr_norm * 100,  # PSNR'ı 100 üzerinden göster
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips
                    }
            
            all_scores[method] = method_scores

        print("-" * 100)
        print("\nEn İyi Sonuçlar:")
        print("-" * 100)
        print(f"En İyi Genel Performans: {best_overall['method']} (Kalite: {best_overall['quality']}, Skor: {best_overall['score']:.2f})")
        print(f"En İyi Sıkıştırma: {best_compression['method']} (Kalite: {best_compression['quality']}, Oran: {best_compression['compression_ratio']:.2f}%)")
        print(f"En İyi Kalite: {best_quality['method']} (Kalite: {best_quality['quality']}, PSNR: {best_quality['psnr']:.2f}dB)")
        print("-" * 100)

        return {
            'best_overall': best_overall,
            'best_compression': best_compression,
            'best_quality': best_quality
        }


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MultimediaBenchmark()
    sys.exit(app.exec_())
