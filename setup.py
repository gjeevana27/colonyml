from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="colonyml",
    version="1.0.1",
    author="Jeevana Gogineni",
    description="Zero-config distributed ML training for CPU clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gjeevana27/colonyml",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "zeroconf>=0.128.0",
        "rich>=13.0.0",
        "click>=8.0.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "colonyml=colonyml.cli:cli",
        ],
    },
    python_requires=">=3.10",
)