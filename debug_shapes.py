import numpy as np
from pathlib import Path

data_dir = Path("data/processed/phase1_box")
coords = np.load(data_dir / "coordinates.npy")
displacement = np.load(data_dir / "displacement.npy")
stiffness = np.load(data_dir / "stiffness_ground_truth.npy")

print("Shape inspection:")
print(f"coords shape: {coords.shape}")
print(f"displacement shape: {displacement.shape}")
print(f"displacement dtype: {displacement.dtype}")
print(f"stiffness shape: {stiffness.shape}")
print(f"stiffness dtype: {stiffness.dtype}")

# Extract z-component
if np.iscomplexobj(displacement):
    u_z = displacement[:, 2].real
else:
    u_z = displacement[:, 2]

print(f"\nu_z shape: {u_z.shape}")
print(f"u_z dtype: {u_z.dtype}")

mu_real = stiffness.real
print(f"mu_real shape: {mu_real.shape}")

is_blob = (mu_real > 8000.0)
print(f"is_blob shape: {is_blob.shape}")
print(f"is_blob dtype: {is_blob.dtype}")
print(f"is_blob type: {type(is_blob)}")
print(f"is_blob sample: {is_blob[:10]}")
