#!/usr/bin/env python3
"""Check a factored orthogonal transform for non-power-of-two head sizes."""

import numpy as np


def hadamard(values):
    output = np.asarray(values, dtype=np.float32).copy()
    width = output.shape[-1]
    stride = 1
    while stride < width:
        blocks = output.reshape(-1, width // (2 * stride), 2 * stride)
        left = blocks[..., :stride].copy()
        right = blocks[..., stride:].copy()
        blocks[..., :stride] = left + right
        blocks[..., stride:] = left - right
        stride *= 2
    output *= np.float32(1.0 / np.sqrt(width))
    return output


def factored_transform(values):
    """Apply H_p tensor (I - 2/m 11^T), where D = p*m and p is a power of two."""
    source = np.asarray(values, dtype=np.float32)
    dimension = source.shape[-1]
    if dimension < 2 or dimension % 2:
        raise ValueError("dimension must be even")
    power = dimension & -dimension
    odd = dimension // power
    output = hadamard(source.reshape(*source.shape[:-1], odd, power))
    if odd > 1:
        column_sum = output.sum(axis=-2, keepdims=True, dtype=np.float32)
        output -= np.float32(2.0 / odd) * column_sum
    return output.reshape(source.shape)


def block_hadamard_96(values):
    source = np.asarray(values, dtype=np.float32)
    return np.concatenate((hadamard(source[..., :64]), hadamard(source[..., 64:])), axis=-1)


def relative_l2(left, right):
    return float(np.linalg.norm(left - right) / np.linalg.norm(left))


def operation_counts(dimension):
    power = dimension & -dimension
    odd = dimension // power
    stages = power.bit_length() - 1
    additions = dimension * stages
    multiplies = dimension
    if odd > 1:
        additions += 2 * dimension - power
        multiplies += power
    return additions, multiplies


def main():
    rng = np.random.default_rng(7)
    print("dim  add/sub  mul  energy_drift  roundtrip_l2  fp16_boundary_l2  outlier_peak/rms")
    for dimension in (64, 96, 128):
        source = rng.standard_t(3, size=(256, dimension)).astype(np.float32)
        transformed = factored_transform(source)
        restored = factored_transform(transformed)
        stored = factored_transform(source.astype(np.float16).astype(np.float32))
        stored = stored.astype(np.float16).astype(np.float32)
        half_restored = factored_transform(stored)
        energy = abs(float(np.sum(transformed * transformed, dtype=np.float64)
                           / np.sum(source * source, dtype=np.float64) - 1.0))
        outlier = np.zeros(dimension, dtype=np.float32)
        outlier[dimension // 2] = 1.0
        mixed = factored_transform(outlier)
        peak_rms = float(np.max(np.abs(mixed)) / np.sqrt(np.mean(mixed * mixed)))
        additions, multiplies = operation_counts(dimension)
        error = relative_l2(source, restored)
        half_error = relative_l2(source, half_restored)
        print(f"{dimension:3d}  {additions:7d}  {multiplies:3d}  "
              f"{energy:12.3e}  {error:12.3e}  {half_error:14.3e}  "
              f"{peak_rms:16.6f}")
        assert energy < 2e-6
        assert error < 2e-6

    source = rng.standard_normal((32, 64), dtype=np.float32)
    assert np.array_equal(factored_transform(source), hadamard(source))

    for dimension in range(2, 513, 2):
        source = rng.standard_normal(dimension, dtype=np.float32)
        assert relative_l2(source, factored_transform(factored_transform(source))) < 2e-6

    outlier = np.zeros(96, dtype=np.float32)
    outlier[80] = 1.0
    factored_peak = np.max(np.abs(factored_transform(outlier)))
    blocked_peak = np.max(np.abs(block_hadamard_96(outlier)))
    print(f"D96 outlier peak: factored={factored_peak:.6f}, "
          f"H64+H32={blocked_peak:.6f}")
    print("D96 operation references: H64+H32=544 add/sub + 96 mul; "
          "padded H128=896 add/sub + 128 mul + 32 stored values")
    assert factored_peak < blocked_peak


if __name__ == "__main__":
    main()
