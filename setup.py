from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="colonyml",
    version="0.1.0",
    author="Jeevana Gogineni",
    description="Zero-config distributed ML training for CPU clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gjeevana27/colonyml",
    packages=find_packages(),
    python_requires=">=3.10",
)