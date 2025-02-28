from setuptools import setup, find_packages

setup(
    name="whale-mil",
    version="0.1.0",
    description="Multiple Instance Learning for Whale Sound Detection",
    author="Ragib",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "matplotlib>=3.9.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.13.0",
        "tqdm>=4.66.0",
        "PyYAML>=6.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.13.0",
    ],
    extras_require={
        "dev": [
            "wandb>=0.17.0",
            "opencv-python>=4.10.0",
        ]
    },
)
