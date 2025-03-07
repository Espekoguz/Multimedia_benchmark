# Multimedia Benchmark Tool

A comprehensive benchmarking tool for analyzing and comparing image/video compression algorithms, codecs, and their performance metrics in real-time.

## Core Features

### Image Processing & Analysis
- **Supported Input Formats**: 
  - PNG, JPG, JPEG, TIFF, BMP
  - Raw image data support
  - High bit-depth image support (8/16-bit)

- **Compression Methods**: 
  - JPEG (with quality settings 1-100)
  - JPEG2000 (with various compression ratios)
  - HEVC (with CRF settings 0-51)

- **Quality Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio) for objective quality measurement
  - SSIM (Structural Similarity Index) for perceptual quality
  - LPIPS (Learned Perceptual Image Patch Similarity) for AI-based perceptual quality
  - Histogram Similarity for color distribution analysis
  - Entropy Analysis for information density measurement

### Video Processing & Analysis
- **Supported Input Formats**:
  - MP4, AVI, MKV, MOV
  - Various pixel formats (YUV420p, YUV444p, RGB24)
  - High bit-depth video support

- **Video Codecs with Configurable Parameters**:
  - H.264/AVC (with CRF 0-51, preset options)
  - H.265/HEVC (with CRF 0-51, preset options)
  - MPEG-4 (with quality settings)
  - MJPEG (with quality settings)

- **Compression Algorithms Analysis**:
  - zlib (with optimization levels)
  - gzip (with compression levels 1-9)
  - bz2 (with compression levels 1-9)
  - lzma (with preset compression modes)

### Real-time Analysis Features
- **Performance Monitoring**:
  - Frame-by-frame quality metrics calculation
  - Real-time compression ratio display
  - Processing speed measurement (MB/s)
  - Memory usage tracking
  - CPU/GPU utilization monitoring

- **Visual Analysis**:
  - Side-by-side comparison of original and compressed media
  - Interactive quality comparison tools
  - Real-time histogram visualization
  - Frame difference analysis

- **Comprehensive Metrics Display**:
  - Performance score tables (normalized to 100)
  - Codec comparison charts
  - Compression algorithm benchmarks
  - Quality vs. compression ratio graphs

### Advanced Features
- **Automated Analysis**:
  - Batch processing capability
  - Multiple compression method comparison
  - Best settings recommendation system
  - Optimal codec/algorithm selection

- **Performance Optimization**:
  - Multi-threaded video processing
  - GPU acceleration for LPIPS calculations
  - Efficient memory management
  - Progress tracking with ETA

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimedia_benchmark.git
cd multimedia_benchmark
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. GPU Support (Optional):
- CUDA-enabled GPU will be automatically utilized for LPIPS calculations
- Supports both CPU and GPU processing modes

## Usage Guide

### Basic Usage
1. Launch the application:
```bash
python main.py
```

2. For Image Analysis:
   - Click "Select Image" to load an image
   - Choose compression method (JPEG/JPEG2000/HEVC)
   - Adjust quality settings
   - Click "Compress and Analyze"
   - View real-time metrics and comparisons

3. For Video Analysis:
   - Click "Select Video" to load a video
   - Select codec and compression settings
   - Use playback controls (Start/Pause/Stop)
   - Monitor real-time metrics
   - View final analysis results

### Advanced Features
1. Metric Analysis:
   - Real-time quality metrics (PSNR, SSIM, LPIPS)
   - Compression performance metrics
   - Processing speed and memory usage
   - Detailed performance graphs

2. Comparison Tools:
   - Side-by-side visual comparison
   - Multiple codec performance comparison
   - Compression algorithm benchmarking
   - Historical data tracking

## Technical Details

### Quality Metrics
- **PSNR (Peak Signal-to-Noise Ratio)**:
  - Range: 30-50 dB (higher is better)
  - Objective quality measurement
  - Mathematical comparison of pixel values

- **SSIM (Structural Similarity Index)**:
  - Range: 0-1 (higher is better)
  - Perceptual quality measurement
  - Structure and luminance comparison

- **LPIPS (Learned Perceptual Image Patch Similarity)**:
  - Range: 0-1 (lower is better)
  - AI-based perceptual quality metric
  - Trained on human perceptual judgments

### Performance Metrics
- **Compression Ratio**:
  - Calculated as: (1 - compressed_size/original_size) * 100
  - Displayed as percentage of size reduction
  - Real-time updates during processing

- **Processing Speed**:
  - Measured in MB/s
  - Separate measurements for compression/decompression
  - Real-time speed monitoring

- **Resource Usage**:
  - Memory consumption tracking
  - CPU/GPU utilization monitoring
  - Thread usage statistics

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- Dual-core processor
- 1GB free disk space

### Recommended Specifications
- Python 3.10 or higher
- 8GB RAM
- Quad-core processor
- CUDA-capable GPU
- 5GB free disk space

### Required Libraries
- OpenCV (image/video processing)
- PyQt5 (GUI framework)
- PyTorch (LPIPS calculations)
- NumPy (numerical operations)
- Matplotlib (visualization)
- scikit-image (image metrics)
- LPIPS (perceptual metrics)
- psutil (system monitoring)

## Project Structure
```
multimedia_benchmark/
├── main.py                 # Main GUI application
├── image_processor.py      # Image processing module
├── video_processor.py      # Video processing module
├── image/                  # Image processing components
│   └── processors/        
│       ├── base_processor.py
│       ├── metric_processor.py
│       └── color_processor.py
├── video/                  # Video processing components
│   └── processors/
│       └── metric_processor.py
├── tests/                  # Test suite
└── requirements.txt        # Python dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for comprehensive image and video processing capabilities
- PyQt5 for robust GUI framework
- LPIPS for advanced perceptual similarity metrics
- FFmpeg for reliable video encoding/decoding
- OpenJPEG for JPEG2000 support
- The open-source community for various tools and libraries 