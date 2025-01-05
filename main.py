import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog,
                           QComboBox, QSpinBox, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from image_processor import ImageProcessor

# Matplotlib'i lazy loading ile import et
class PlotManager:
    def __init__(self):
        self._plt = None
        self._FigureCanvas = None
        
    @property
    def plt(self):
        if self._plt is None:
            import matplotlib.pyplot as plt
            plt.style.use('default')
            self._plt = plt
        return self._plt
    
    @property
    def FigureCanvas(self):
        if self._FigureCanvas is None:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            self._FigureCanvas = FigureCanvasQTAgg
        return self._FigureCanvas

class MultimediaBenchmark(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Benchmark Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Plot manager ve işlemci nesnesi
        self.plot_manager = PlotManager()
        self.image_processor = ImageProcessor()
        
        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Üst panel - Kontroller
        top_panel = QHBoxLayout()
        
        # Dosya seçimi
        file_layout = QVBoxLayout()
        self.image_path_label = QLabel("Dosya seçilmedi")
        select_file_btn = QPushButton("Görüntü Seç")
        select_file_btn.clicked.connect(self.select_image_file)
        file_layout.addWidget(select_file_btn)
        file_layout.addWidget(self.image_path_label)
        top_panel.addLayout(file_layout)
        
        # Sıkıştırma ayarları
        compression_layout = QVBoxLayout()
        compression_layout.addWidget(QLabel("Sıkıştırma:"))
        self.compression_combo = QComboBox()
        self.compression_combo.addItems(["JPEG", "JPEG2000", "HEVC"])
        compression_layout.addWidget(self.compression_combo)
        
        # Kalite ayarı
        compression_layout.addWidget(QLabel("Kalite:"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(75)
        compression_layout.addWidget(self.quality_spin)
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
        original_layout.addWidget(self.original_image_label)
        images_panel.addLayout(original_layout)
        
        # Sıkıştırılmış görüntü
        compressed_layout = QVBoxLayout()
        compressed_layout.addWidget(QLabel("Sıkıştırılmış Görüntü"))
        self.compressed_image_label = QLabel()
        self.compressed_image_label.setMinimumSize(400, 300)
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
        
        # Sıkıştırma butonu
        compress_btn = QPushButton("Sıkıştır ve Analiz Et")
        compress_btn.clicked.connect(self.compress_and_analyze)
        metrics_panel.addWidget(compress_btn)
        
        middle_panel.addLayout(metrics_panel)
        
        self.layout.addLayout(middle_panel)
        
        # Alt panel - Grafikler
        self.plot_tabs = QTabWidget()
        self.plot_canvases = {}
        self.setup_plot_tabs()
        self.layout.addWidget(self.plot_tabs)
        
        self.show()
    
    def setup_plot_tabs(self):
        """Grafik sekmelerini oluşturur."""
        plot_titles = {
            "compression": "Sıkıştırma Analizi",
            "psnr": "PSNR Analizi",
            "ssim": "SSIM Analizi",
            "time": "Süre Analizi"
        }
        
        for key, title in plot_titles.items():
            container = QWidget()
            layout = QVBoxLayout(container)
            
            # Canvas'ı lazy loading ile oluştur
            canvas = self.plot_manager.FigureCanvas(self.plot_manager.plt.figure(figsize=(8, 5)))
            layout.addWidget(canvas)
            self.plot_tabs.addTab(container, title)
            
            # Canvas'ı sakla
            self.plot_canvases[key] = canvas
    
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
        if not hasattr(self, 'original_image'):
            return
            
        try:
            # Sıkıştırma parametreleri
            method = self.compression_combo.currentText()
            quality = self.quality_spin.value()
            
            # Görüntüyü sıkıştır
            compressed, file_size = self.image_processor.compress_image(
                self.original_image, method, quality)
            
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
            
            # Metrikleri göster
            self.update_metrics_label(metrics)
            
            # Sıkıştırılmış görüntüyü göster
            self.display_image(compressed, self.compressed_image_label)
            
            # Tüm yöntemleri analiz et
            results = self.image_processor.analyze_all_methods(self.original_image)
            
            # En iyi yöntemi bul
            optimal = self.image_processor.find_optimal_method(results)
            
            # Önerilen yöntemi güncelle
            self.update_recommendation_label(optimal)
            
            # Grafikleri güncelle
            self.update_plots(results)
            
        except Exception as e:
            print(f"Hata: {str(e)}")
            self.metrics_label.setText("İşlem sırasında hata oluştu")
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
    
    def update_plots(self, results: dict):
        """Grafikleri günceller."""
        plt = self.plot_manager.plt
        
        # Grafikleri temizle ve güncelle
        for canvas in self.plot_canvases.values():
            canvas.figure.clear()
        
        # Sıkıştırma Oranı vs Kalite
        ax1 = self.plot_canvases["compression"].figure.add_subplot(111)
        for method in results:
            ax1.plot(results[method]["qualities"], 
                    results[method]["compression_ratios"], 
                    marker='o', label=method)
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
                    marker='o', label=method)
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
                    marker='o', label=method)
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
                    marker='o', label=method)
        ax4.set_xlabel("Kalite Faktörü")
        ax4.set_ylabel("İşlem Süresi (s)")
        ax4.set_title("İşlem Süresi vs Kalite")
        ax4.legend()
        ax4.grid(True)
        self.plot_canvases["time"].figure.tight_layout()
        
        # Canvas'ları güncelle
        for canvas in self.plot_canvases.values():
            canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MultimediaBenchmark()
    sys.exit(app.exec_())
