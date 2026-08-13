import torch
import pytest
from colonyml.compressor import GradientCompressor, CompressionLevel


def test_compress_decompress_none():
    compressor = GradientCompressor()
    gradient = torch.randn(100, 100)
    compressed = compressor.compress(gradient, CompressionLevel.NONE)
    decompressed = compressor.decompress(compressed)
    assert torch.allclose(gradient, decompressed)


def test_compress_decompress_medium():
    compressor = GradientCompressor()
    gradient = torch.randn(1000)
    compressed = compressor.compress(gradient, CompressionLevel.MEDIUM)
    decompressed = compressor.decompress(compressed)
    assert decompressed.shape == gradient.shape


def test_compress_decompress_high():
    compressor = GradientCompressor()
    gradient = torch.randn(1000)
    compressed = compressor.compress(gradient, CompressionLevel.HIGH)
    decompressed = compressor.decompress(compressed)
    assert decompressed.shape == gradient.shape


def test_compression_ratio_none():
    compressor = GradientCompressor()
    gradient = torch.randn(100, 100)
    ratio = compressor.compression_ratio(gradient, CompressionLevel.NONE)
    assert ratio == 1.0


def test_compression_ratio_medium():
    compressor = GradientCompressor()
    gradient = torch.randn(1000, 1000)
    ratio = compressor.compression_ratio(gradient, CompressionLevel.MEDIUM)
    assert ratio > 2.0


def test_compression_ratio_high():
    compressor = GradientCompressor()
    gradient = torch.randn(1000, 1000)
    ratio = compressor.compression_ratio(gradient, CompressionLevel.HIGH)
    assert ratio > 20.0


def test_auto_select_wifi():
    compressor = GradientCompressor()
    level = compressor.auto_select_level(10)
    assert level == CompressionLevel.HIGH


def test_auto_select_ethernet():
    compressor = GradientCompressor()
    level = compressor.auto_select_level(100)
    assert level == CompressionLevel.MEDIUM


def test_auto_select_gigabit():
    compressor = GradientCompressor()
    level = compressor.auto_select_level(1001)
    assert level == CompressionLevel.NONE


def test_compress_preserves_shape():
    compressor = GradientCompressor()
    gradient = torch.randn(32, 64, 3)
    for level in CompressionLevel:
        compressed = compressor.compress(gradient, level)
        decompressed = compressor.decompress(compressed)
        assert decompressed.shape == gradient.shape