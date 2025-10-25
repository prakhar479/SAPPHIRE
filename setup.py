#!/usr/bin/env python3
"""
Setup script for the SAPPHIRE music analysis pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'librosa>=0.9.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'nltk>=3.6.0',
        'sentence-transformers>=2.0.0'
    ]

setup(
    name="sapphire-music-analysis",
    version="1.0.0",
    author="SAPPHIRE Team",
    description="Semantic and Acoustic Perceptual Holistic Integration REtrieval for music analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sapphire-team/sapphire",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "advanced": ["crepe>=0.0.12"],
        "dev": ["pytest>=6.0.0", "jupyter>=1.0.0", "ipykernel>=6.0.0"],
        "all": ["crepe>=0.0.12", "pytest>=6.0.0", "jupyter>=1.0.0"]
    },
    entry_points={
        "console_scripts": [
            "sapphire-pipeline=example_usage:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pipeline": ["*.py"],
    },
    keywords="music analysis, mood classification, audio processing, machine learning, MIR",
    project_urls={
        "Bug Reports": "https://github.com/sapphire-team/sapphire/issues",
        "Source": "https://github.com/sapphire-team/sapphire",
        "Documentation": "https://github.com/sapphire-team/sapphire/wiki",
    },
)