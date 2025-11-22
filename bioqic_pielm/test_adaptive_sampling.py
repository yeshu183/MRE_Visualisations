"""
Test adaptive sampling in data loader.
"""
import sys
sys.path.insert(0, '.')

from bioqic_pielm.data_loader import BIOQICDataLoader
import torch

def test_uniform_sampling():
    """Test uniform random sampling (baseline)."""
    print("\n" + "="*80)
    print("TEST 1: UNIFORM RANDOM SAMPLING")
    print("="*80)

    loader = BIOQICDataLoader(
        data_dir="data/processed/phase1_box",
        subsample=5000,
        adaptive_sampling=False,
        seed=42
    )
    data = loader.load()

    print(f"\nSampled {len(data['coords'])} points")
    print(f"Stiffness range: [{data['mu_raw'].min():.0f}, {data['mu_raw'].max():.0f}] Pa")

    # Count samples in each region
    blob_count = (data['mu_raw'] > 8000).sum()
    background_count = (data['mu_raw'] <= 8000).sum()
    print(f"Blob samples: {blob_count} ({100*blob_count/len(data['coords']):.1f}%)")
    print(f"Background samples: {background_count} ({100*background_count/len(data['coords']):.1f}%)")


def test_adaptive_sampling():
    """Test adaptive sampling with more blob/boundary points."""
    print("\n" + "="*80)
    print("TEST 2: ADAPTIVE SAMPLING (50% blob, 30% boundary, 20% background)")
    print("="*80)

    loader = BIOQICDataLoader(
        data_dir="data/processed/phase1_box",
        subsample=5000,
        adaptive_sampling=True,
        blob_sample_ratio=0.5,
        boundary_sample_ratio=0.3,
        seed=42
    )
    data = loader.load()

    print(f"\nSampled {len(data['coords'])} points")
    print(f"Stiffness range: [{data['mu_raw'].min():.0f}, {data['mu_raw'].max():.0f}] Pa")

    # Count samples in each region
    blob_count = (data['mu_raw'] > 8000).sum()
    background_count = (data['mu_raw'] <= 8000).sum()
    print(f"Blob samples: {blob_count} ({100*blob_count/len(data['coords']):.1f}%)")
    print(f"Background samples: {background_count} ({100*background_count/len(data['coords']):.1f}%)")


if __name__ == "__main__":
    test_uniform_sampling()
    test_adaptive_sampling()
    print("\n" + "="*80)
    print("TESTS COMPLETED")
    print("="*80)
