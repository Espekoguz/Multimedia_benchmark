from setuptools import setup, find_packages

setup(
    name="multimedia_benchmark",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "PyQt5>=5.15.0",
        "matplotlib>=3.4.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "lpips>=0.1.4",
        "scikit-image>=0.18.0",
        "av>=9.0.0",
        "psutil>=5.8.0",
        "pillow>=8.3.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-qt>=4.0.0",
            "flake8>=3.9.0",
            "black>=21.6b0"
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive multimedia analysis and benchmarking tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimedia_benchmark",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Multimedia :: Video",
    ],
) 