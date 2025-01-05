# Multimedia Benchmark Tool

Bu uygulama, görüntü ve video dosyalarının kalitesini ve sıkıştırma performansını analiz eden kapsamlı bir benchmark aracıdır.

## Özellikler

### Görüntü İşleme
- Desteklenen formatlar: YUV, TIFF, JPG, JPEG, PNG, BPM, JP2
- Kalite metrikleri: PSNR, SSIM, MS-SSIM, LPIPS, MSE
- Renk uzayı dönüşümleri: RGB, YUV, YCbCr
- Sıkıştırma algoritmaları: JPEG, JPEG2000, HEVC
- Gerçek zamanlı metrik grafikleri
- Dosya boyutu analizi

### Video İşleme
- Desteklenen formatlar: MP4, AVI, WEBM, MOV, MKV
- Video codec'leri: H.264, H.265/HEVC, VP9
- Kare bazlı kalite analizi
- Renk histogramı ve entropi hesaplama
- Gerçek zamanlı kare görüntüleme
- İlerleme takibi

## Kurulum

1. Python 3.8 veya üstü sürümünün yüklü olduğundan emin olun.

2. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

3. FFmpeg'i yükleyin:
- macOS: `brew install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`
- Windows: [FFmpeg web sitesi](https://ffmpeg.org/download.html)

## Kullanım

1. Uygulamayı başlatın:
```bash
python main.py
```

2. Görüntü Analizi:
   - "Görüntü Analizi" sekmesini seçin
   - "Görüntü Seç" butonu ile bir görüntü dosyası seçin
   - Renk uzayı ve sıkıştırma ayarlarını yapın
   - "Analiz Başlat" butonuna tıklayın

3. Video Analizi:
   - "Video Analizi" sekmesini seçin
   - "Video Seç" butonu ile bir video dosyası seçin
   - Codec ve CRF değerlerini ayarlayın
   - "Video Analizi Başlat" butonuna tıklayın

## Geliştirme

Proje modüler bir yapıda tasarlanmıştır:
- `main.py`: Ana uygulama ve GUI
- `image_processor.py`: Görüntü işleme modülü
- `video_processor.py`: Video işleme modülü

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 