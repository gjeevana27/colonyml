import numpy as np
import torch
from enum import Enum


class CompressionLevel(Enum):
    NONE = "none"      # No compression — fast network / cable
    MEDIUM = "medium"  # Top 10% of gradients — ethernet
    HIGH = "high"      # Top 1% of gradients — wifi


class GradientCompressor:
    """
    Compresses gradients before sending over the network.
    Sends only the most important gradient values.

    Why this works: Most gradient values are near zero
    and don't meaningfully affect training. We only
    send the ones that matter most (TopK compression).

    Usage:
        compressor = GradientCompressor()
        compressed = compressor.compress(gradient, CompressionLevel.MEDIUM)
        decompressed = compressor.decompress(compressed)
    """

    def compress(
        self,
        gradient: torch.Tensor,
        level: CompressionLevel = CompressionLevel.MEDIUM
    ) -> dict:

        if level == CompressionLevel.NONE:
            return {
                "type": "full",
                "data": gradient.numpy().tobytes(),
                "shape": list(gradient.shape),
                "dtype": str(gradient.dtype)
            }

        flat = gradient.flatten().float()
        n_total = len(flat)

        if level == CompressionLevel.MEDIUM:
            k = max(1, int(n_total * 0.1))
        else:
            k = max(1, int(n_total * 0.01))

        topk_values, topk_indices = torch.topk(flat.abs(), k)
        actual_values = flat[topk_indices]

        return {
            "type": "topk",
            "values": actual_values.numpy().tobytes(),
            "indices": topk_indices.numpy().tobytes(),
            "shape": list(gradient.shape),
            "n_total": n_total,
            "k": k,
            "dtype": "torch.float32"
        }

    def decompress(self, compressed: dict) -> torch.Tensor:

        shape = compressed["shape"]

        if compressed["type"] == "full":
            data = np.frombuffer(
                compressed["data"], dtype=np.float32
            ).copy()
            return torch.tensor(data).reshape(shape)

        values = torch.tensor(
            np.frombuffer(
                compressed["values"], dtype=np.float32
            ).copy()
        )
        indices = torch.tensor(
            np.frombuffer(
                compressed["indices"], dtype=np.int64
            ).copy()
        )
        n_total = compressed["n_total"]

        flat = torch.zeros(n_total)
        flat[indices] = values

        return flat.reshape(shape)

    def compression_ratio(
        self,
        gradient: torch.Tensor,
        level: CompressionLevel
    ) -> float:

        if level == CompressionLevel.NONE:
            return 1.0

        compressed = self.compress(gradient, level)
        original_bytes = gradient.numel() * 4
        compressed_bytes = (
            len(compressed["values"]) +
            len(compressed["indices"])
        )

        return round(original_bytes / compressed_bytes, 1)

    def auto_select_level(
        self,
        network_speed_mbps: float
    ) -> CompressionLevel:
        """
        Auto-select compression based on network speed.
        > 1000 Mbps → no compression
        > 100 Mbps  → medium
        <= 100 Mbps → high
        """
        if network_speed_mbps > 1000:
            return CompressionLevel.NONE
        elif network_speed_mbps >= 100:
            return CompressionLevel.MEDIUM
        else:
            return CompressionLevel.HIGH


if __name__ == "__main__":
    compressor = GradientCompressor()

    print("Gradient Compression Test")
    print("=" * 50)

    gradient = torch.randn(1000, 1000)
    original_mb = gradient.numel() * 4 / (1024 * 1024)

    for level in CompressionLevel:
        ratio = compressor.compression_ratio(gradient, level)
        compressed = compressor.compress(gradient, level)
        decompressed = compressor.decompress(compressed)
        error = (gradient - decompressed).abs().mean().item()
        compressed_mb = original_mb / ratio if ratio > 0 else original_mb

        print(f"\n{level.value.upper():8s}:")
        print(f"  Original:    {original_mb:.2f} MB")
        print(f"  Compressed:  {compressed_mb:.2f} MB")
        print(f"  Ratio:       {ratio}x smaller")
        print(f"  Avg error:   {error:.6f}")

    print("\n" + "=" * 50)
    print("Auto-selection by network speed:")
    for speed in [10, 100, 500, 1000]:
        level = compressor.auto_select_level(speed)
        print(f"  {speed:5d} Mbps → {level.value}")