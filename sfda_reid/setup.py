from setuptools import setup, find_packages

setup(
    name="sfda_reid",
    version="1.0.0",
    description="Source-Free Domain Adaptive Person Re-Identification with Formal Learnability Guarantees",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/sfda_reid",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch==2.2.0",
        "torchvision==0.17.0",
        "numpy==1.26.4",
        "scipy==1.12.0",
        "scikit-learn",
        "faiss-gpu",
        "tqdm==4.66.2",
        "omegaconf==2.3.0",
        "matplotlib==3.8.3",
        "seaborn==0.13.2",
        "tensorboard==2.16.2",
        "einops==0.7.0",
        "timm==0.9.16"
    ],
    python_requires=">=3.10",
)
