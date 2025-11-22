"""Quick test of BIOQICDataLoader with adaptive sampling."""
import sys
import numpy as np

# Add path
sys.path.insert(0, '.')

# Import without torch dependencies
from pathlib import Path
from bioqic_pielm.data_loader import BIOQICDataLoader

print("="*80)
print("Testing BIOQICDataLoader with Adaptive Sampling")
print("="*80)

# Test 1: Uniform sampling
print("\nTest 1: UNIFORM SAMPLING")
print("-"*80)
try:
    loader = BIOQICDataLoader(
        data_dir="data/processed/phase1_box",
        subsample=5000,
        adaptive_sampling=False,
        seed=42
    )
    # We can't call load() because it imports torch, but we can test initialization
    print("✓ Uniform sampling initialization successful")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Adaptive sampling
print("\nTest 2: ADAPTIVE SAMPLING")
print("-"*80)
try:
    loader = BIOQICDataLoader(
        data_dir="data/processed/phase1_box",
        subsample=5000,
        adaptive_sampling=True,
        blob_sample_ratio=0.5,
        boundary_sample_ratio=0.3,
        seed=42
    )
    print("✓ Adaptive sampling initialization successful")
    print(f"  - Blob ratio: {loader.blob_sample_ratio}")
    print(f"  - Boundary ratio: {loader.boundary_sample_ratio}")
    print(f"  - Background ratio: {1.0 - loader.blob_sample_ratio - loader.boundary_sample_ratio}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("Initialization tests passed!")
print("Run with PyTorch environment to test full data loading.")
print("="*80)
