import cv2
import numpy as np
from PIL import Image
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, List
import time
import warnings
import torchvision.models as models

class ImageProcessor:
    def __init__(self):
        self._device = None
        self._lpips_model = None
        self.original_size = 0
        self.compression_methods = ["JPEG", "JPEG2000", "HEVC"]
        self.quality_range = range(10, 101, 10)
        self.jp2_compression_rates = [4, 8, 16, 32, 64, 128, 256]
    
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device
    
    @property
    def lpips_model(self):
        if self._lpips_model is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                self._lpips_model = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        return self._lpips_model
    
    def load_image(self, path: str) -> np.ndarray:
        """Görüntüyü yükler ve BGR formatında döndürür."""
        self.original_size = os.path.getsize(path)
        return cv2.imread(path)
    
    def convert_color_space(self, image: np.ndarray, target_space: str) -> np.ndarray:
        """Görüntüyü hedef renk uzayına dönüştürür."""
        if target_space == "YUV":
            return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif target_space == "YCbCr":
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        return image  # BGR/RGB
    
    def calculate_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict[str, float]:
        """Kalite metriklerini hesaplar."""
        if original.shape != compressed.shape:
            compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
        
        try:
            metrics = {}
            
            # Temel metrikler (her zaman hesaplanır)
            mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
            metrics["MSE"] = mse
            
            metrics["PSNR"] = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            metrics["SSIM"] = ssim(original, compressed, channel_axis=2, data_range=255)
            
            # LPIPS sadece gerektiğinde hesaplanır
            if self._lpips_model is not None:
                if len(original.shape) == 3 and original.shape[2] == 3:
                    orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                    comp_rgb = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
                else:
                    orig_rgb = original
                    comp_rgb = compressed
                
                orig_tensor = torch.from_numpy(orig_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
                comp_tensor = torch.from_numpy(comp_rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
                orig_tensor = orig_tensor.to(self.device)
                comp_tensor = comp_tensor.to(self.device)
                
                metrics["LPIPS"] = float(self.lpips_model(orig_tensor, comp_tensor).item())
            else:
                metrics["LPIPS"] = 1.0
            
            return metrics
            
        except Exception as e:
            print(f"Metrik hesaplama hatası: {e}")
            return {
                "MSE": float('inf'),
                "PSNR": 0.0,
                "SSIM": 0.0,
                "LPIPS": 1.0
            }
    
    def calculate_advanced_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict[str, float]:
        """İleri seviye metrikleri hesaplar (sadece gerektiğinde)."""
        try:
            metrics = {}
            
            # MS-SSIM hesaplama
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            comp_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)
            metrics["MS-SSIM"] = ssim(orig_gray, comp_gray, data_range=255)
            
            # FSIM hesaplama
            orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 100, 200)
            comp_edges = cv2.Canny(cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY), 100, 200)
            metrics["FSIM"] = ssim(orig_edges, comp_edges, data_range=255)
            
            # VIF hesaplama
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
            comp_gray = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = orig_gray - comp_gray
            var_orig = np.var(orig_gray)
            var_diff = np.var(diff)
            vif = var_orig/(var_diff+1e-8)
            metrics["VIF"] = 1.0/(1.0+1.0/vif)
            
            return metrics
            
        except Exception as e:
            print(f"İleri seviye metrik hesaplama hatası: {e}")
            return {
                "MS-SSIM": 0.0,
                "FSIM": 0.0,
                "VIF": 0.0
            }
    
    def compress_image(self, image: np.ndarray, method: str, quality: int) -> Tuple[np.ndarray, int]:
        """Görüntüyü sıkıştırır ve sıkıştırılmış görüntü ile boyutunu döndürür."""
        import os
        
        # Geçici dosya yollarını oluştur
        temp_dir = "temp_files"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, "temp_compressed")
        
        try:
            if method == "JPEG":
                output_path = temp_path + ".jpg"
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif method == "JPEG2000":
                output_path = temp_path + ".jp2"
                try:
                    # Quality değerini compression rate'e dönüştür
                    rate_index = int((100 - quality) / 100.0 * (len(self.jp2_compression_rates) - 1))
                    compression_rate = self.jp2_compression_rates[rate_index]
                    
                    # OpenJPEG kullanarak JPEG2000 sıkıştırma
                    import subprocess
                    input_path = os.path.join(temp_dir, "temp_input.png")
                    cv2.imwrite(input_path, image)
                    
                    encode_cmd = [
                        "/opt/homebrew/bin/opj_compress",
                        "-i", input_path,
                        "-o", output_path,
                        "-r", str(compression_rate)
                    ]
                    
                    decode_cmd = [
                        "/opt/homebrew/bin/opj_decompress",
                        "-i", output_path,
                        "-o", input_path
                    ]
                    
                    # Sıkıştırma ve açma işlemleri
                    subprocess.run(encode_cmd, check=True, capture_output=True)
                    subprocess.run(decode_cmd, check=True, capture_output=True)
                    
                    # Sıkıştırılmış görüntüyü oku
                    compressed = cv2.imread(input_path)
                    if compressed is None:
                        raise Exception("JPEG2000 görüntü okunamadı")
                    
                    file_size = os.path.getsize(output_path)
                    
                    # Geçici dosyaları temizle
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    
                    return compressed, file_size
                    
                except Exception as e:
                    print(f"JPEG2000 sıkıştırma hatası: {e}")
                    output_path = os.path.join(temp_dir, "temp_compressed.jpg")
                    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            elif method == "HEVC":
                output_path = temp_path + ".hevc"
                
                # HEVC kalitesini CRF değerine dönüştür
                crf = int(51 - (quality / 100.0 * 51))
                
                # ffmpeg komutu
                import subprocess
                cmd = [
                    "/opt/homebrew/bin/ffmpeg",
                    "-y",
                    "-f", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{image.shape[1]}x{image.shape[0]}",
                    "-r", "1",
                    "-i", "-",
                    "-c:v", "libx265",
                    "-crf", str(crf),
                    "-preset", "medium",
                    output_path
                ]
                
                try:
                    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = process.communicate(input=image.tobytes())
                    
                    if process.returncode != 0:
                        raise Exception(f"FFmpeg hatası: {err.decode()}")
                    
                    # HEVC dosyasını tekrar görüntüye dönüştür
                    cmd_decode = [
                        "/opt/homebrew/bin/ffmpeg",
                        "-y",
                        "-i", output_path,
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-"
                    ]
                    
                    process = subprocess.Popen(cmd_decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = process.communicate()
                    
                    if process.returncode != 0:
                        raise Exception(f"FFmpeg decode hatası: {err.decode()}")
                    
                    compressed = np.frombuffer(out, dtype=np.uint8)
                    compressed = compressed.reshape(image.shape)
                    
                    file_size = os.path.getsize(output_path)
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    
                    return compressed, file_size
                    
                except Exception as e:
                    print(f"HEVC hatası: {str(e)}")
                    return image, 0
            
            # Sıkıştırılmış görüntüyü oku
            compressed = cv2.imread(output_path)
            if compressed is None:
                raise Exception(f"Görüntü okunamadı: {output_path}")
            
            file_size = os.path.getsize(output_path)
            
            # Geçici dosyayı temizle
            if os.path.exists(output_path):
                os.remove(output_path)
            
            return compressed, file_size
            
        except Exception as e:
            print(f"Sıkıştırma hatası: {str(e)}")
            return image, 0
        
        finally:
            # Geçici dizini temizle
            try:
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Geçici dizin temizleme hatası: {str(e)}")
    
    def calculate_color_correlation(self, image: np.ndarray) -> Dict[str, float]:
        """Renk kanalları arasındaki korelasyonu hesaplar."""
        b, g, r = cv2.split(image)
        
        corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        corr_rb = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        corr_gb = np.corrcoef(g.flatten(), b.flatten())[0, 1]
        
        return {
            "R-G": corr_rg,
            "R-B": corr_rb,
            "G-B": corr_gb
        }
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """Görüntü entropisini hesaplar."""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.flatten() / np.sum(histogram)
        non_zero_hist = histogram[histogram > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        return entropy
    
    def calculate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Sıkıştırma oranını hesaplar."""
        if original_size == 0:
            return 0
        # Sıkıştırma oranını doğrudan hesapla
        return ((compressed_size / original_size) * 100)
    
    def analyze_all_methods(self, image: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        """Tüm sıkıştırma yöntemlerini analiz eder."""
        results = {
            method: {
                "qualities": list(self.quality_range),
                "compression_ratios": [],
                "psnr_values": [],
                "ssim_values": [],
                "lpips_values": [],
                "file_sizes": [],
                "processing_times": []
            }
            for method in self.compression_methods
        }
        
        for method in self.compression_methods:
            for quality in self.quality_range:
                start_time = time.time()
                compressed, file_size = self.compress_image(image, method, quality)
                processing_time = time.time() - start_time
                
                if compressed is not None:
                    metrics = self.calculate_metrics(image, compressed)
                    compression_ratio = self.calculate_compression_ratio(self.original_size, file_size)
                    
                    results[method]["compression_ratios"].append(compression_ratio)
                    results[method]["psnr_values"].append(metrics["PSNR"])
                    results[method]["ssim_values"].append(metrics["SSIM"])
                    results[method]["lpips_values"].append(metrics["LPIPS"])
                    results[method]["file_sizes"].append(file_size)
                    results[method]["processing_times"].append(processing_time)
        
        return results
    
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
    
    def plot_compression_analysis(self, results: Dict[str, Dict[str, List[float]]]) -> List[plt.Figure]:
        """Sıkıştırma analiz grafiklerini oluşturur."""
        figures = []
        
        # Sıkıştırma Oranı vs Kalite
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for method in self.compression_methods:
            ax1.plot(results[method]["qualities"], 
                    results[method]["compression_ratios"], 
                    marker='o', label=method)
        ax1.set_xlabel("Kalite Faktörü")
        ax1.set_ylabel("Sıkıştırma Oranı (%)")
        ax1.set_title("Sıkıştırma Oranı vs Kalite")
        ax1.legend()
        ax1.grid(True)
        fig1.tight_layout()
        figures.append(fig1)
        
        # PSNR vs Kalite
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for method in self.compression_methods:
            ax2.plot(results[method]["qualities"], 
                    results[method]["psnr_values"], 
                    marker='o', label=method)
        ax2.set_xlabel("Kalite Faktörü")
        ax2.set_ylabel("PSNR (dB)")
        ax2.set_title("PSNR vs Kalite")
        ax2.legend()
        ax2.grid(True)
        fig2.tight_layout()
        figures.append(fig2)
        
        # SSIM vs Kalite
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        for method in self.compression_methods:
            ax3.plot(results[method]["qualities"], 
                    results[method]["ssim_values"], 
                    marker='o', label=method)
        ax3.set_xlabel("Kalite Faktörü")
        ax3.set_ylabel("SSIM")
        ax3.set_title("SSIM vs Kalite")
        ax3.legend()
        ax3.grid(True)
        fig3.tight_layout()
        figures.append(fig3)
        
        # İşlem Süresi vs Kalite
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        for method in self.compression_methods:
            ax4.plot(results[method]["qualities"], 
                    results[method]["processing_times"], 
                    marker='o', label=method)
        ax4.set_xlabel("Kalite Faktörü")
        ax4.set_ylabel("İşlem Süresi (s)")
        ax4.set_title("İşlem Süresi vs Kalite")
        ax4.legend()
        ax4.grid(True)
        fig4.tight_layout()
        figures.append(fig4)
        
        return figures
    
    def plot_metrics(self, qualities: List[int], metrics: Dict[str, List[float]], 
                    file_sizes: List[int]) -> Tuple[plt.Figure, plt.Figure]:
        """Metrik grafiklerini oluşturur."""
        # Kalite-Metrik grafiği
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for metric_name, values in metrics.items():
            ax1.plot(qualities, values, marker='o', label=metric_name)
        ax1.set_xlabel("Kalite Faktörü")
        ax1.set_ylabel("Metrik Değeri")
        ax1.set_title("Kalite Metrikleri")
        ax1.legend()
        ax1.grid(True)
        
        # Dosya Boyutu-Kalite grafiği
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(qualities, [size/1024 for size in file_sizes], marker='o')
        ax2.set_xlabel("Kalite Faktörü")
        ax2.set_ylabel("Dosya Boyutu (KB)")
        ax2.set_title("Dosya Boyutu vs Kalite")
        ax2.grid(True)
        
        return fig1, fig2 