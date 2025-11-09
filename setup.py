"""
GLEC DTG Edge AI SDK - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="glec-dtg-edge-ai",
    version="0.1.0",
    author="GLEC Engineering Team",
    author_email="engineering@glec.ai",
    description="Edge AI SDK for commercial vehicle telematics on STM32 + Snapdragon platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/glecdev/edgeai",
    packages=find_packages(exclude=["tests", "docs", "android-*", "stm32-*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9,<3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "pylint>=3.0.3",
            "mypy>=1.7.1",
        ],
        "docs": [
            "pdoc3>=0.10.0",
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "glec-train=ai_models.training.train_tcn:main",
            "glec-quantize=ai_models.optimization.quantize_model:main",
            "glec-export=ai_models.conversion.export_onnx:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)
