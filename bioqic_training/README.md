# BIOQIC Training Framework - Modular MRE Inversion

**Step-by-step modular implementation for MRE inverse problems**

This folder contains a clean, modular framework for training MRE inversions on BIOQIC data, addressing the key issues identified in the data exploration phase.

---

## 🎯 Key Improvements Over Previous Implementations

### **1. Physics-Informed Boundary Detection**
- ✅ **Correct actuator identification**: Top Y-face (based on BIOQIC FEM documentation)
- ✅ **Three strategies**: Actuator-based, minimal anchoring, interior weighting
- ✅ **Addresses**: Previous 10% tolerance issue that over-constrained the problem

### **2. Flexible Displacement Handling**
- ✅ **Three modes**: Magnitude, Z-component, Full 3-component vector
- ✅ **Addresses**: Previous magnitude-only approach lost directional information
- ✅ **Z-dominance**: Accounts for vertical excitation (16× larger than X/Y)

### **3. Correct Stiffness Bounds**
- ✅ **Configurable ranges**: Matches normalized BIOQIC [0.3, 1.0]
- ✅ **Three strategies**: Direct sigmoid, log-scale, softplus
- ✅ **Addresses**: Previous hardcoded [0.5, 5.0] bounds incompatible with data

### **4. Physics Scaling Options**
- ✅ **Two modes**: Physical (ρω²=142M) vs Effective (ρω²=400)
- ✅ **Coordinate rescaling**: Proper Laplacian scaling for physical mode
- ✅ **Addresses**: Previous 3500× mismatch in physics parameters

---

## 📦 Module Overview

### **1. `boundary_detection.py`**
**Purpose**: Identify boundary conditions for MRE inverse problems

**Strategies**:
- `'actuator'`: Physics-based detection of top Y-face (traction force location)
- `'minimal'`: 2-3 anchor points for uniqueness (data-driven approach)
- `'weighted'`: Interior weighting (no hard BCs, soft constraints)

**Key Insight**: BIOQIC box has "traction force applied on top x-z plane" → Top Y-face is the actuator

**Usage**:
```python
from boundary_detection import BoundaryDetector

detector = BoundaryDetector(strategy='actuator')
bc_indices, u_bc_vals, info = detector.detect(
    coords, coords_norm, u_meas, device, subsample=5
)
```

**Test**: `python boundary_detection.py`

---

### **2. `data_loader.py`**
**Purpose**: Load and preprocess BIOQIC Phase 1 data with flexible options

**Displacement Modes**:
- `'magnitude'`: √(|u_x|² + |u_y|² + |u_z|²) - Scalar field
- `'z_component'`: |u_z| only - Dominant direction (97.2% of energy)
- `'3_components'`: [u_x, u_y, u_z] - Full vector field

**Features**:
- Subsampling for faster iteration
- Normalization (displacement + stiffness)
- Physical parameters (ω=377 rad/s, ρ=1000 kg/m³)
- Both physical (142M) and effective (400) ρω² values
- Torch tensor conversion

**Usage**:
```python
from data_loader import BIOQICDataLoader

loader = BIOQICDataLoader(
    displacement_mode='z_component',
    subsample=1000,
    physics_mode='effective'
)
data = loader.load()

x = data['x']  # (N, 3) normalized coordinates
u_meas = data['u_meas']  # (N, 1) or (N, 3) displacement
mu_true = data['mu_true']  # (N, 1) ground truth stiffness
```

**Test**: `python data_loader.py`

---

### **3. `stiffness_network.py`**
**Purpose**: Neural network to parameterize heterogeneous stiffness field

**Output Strategies**:
- `'direct'`: Sigmoid scaled to [μ_min, μ_max]
- `'log'`: Predict log(μ) for wide ranges (e.g., [0.1, 10.0])
- `'softplus'`: Smooth ReLU-like activation

**Features**:
- Random Fourier features for better spatial representation
- Configurable architecture (hidden dims, layers)
- Proper Xavier initialization
- Guaranteed bounds enforcement

**Usage**:
```python
from stiffness_network import FlexibleStiffnessNetwork

mu_net = FlexibleStiffnessNetwork(
    input_dim=3,
    mu_min=0.3,  # Match normalized BIOQIC range
    mu_max=1.0,
    strategy='direct',
    hidden_dim=64,
    n_layers=3,
    n_fourier=10
)

mu_pred = mu_net(x)  # (N, 1) stiffness prediction
```

**Test**: `python stiffness_network.py`

---

### **4. `forward_model.py`**
**Purpose**: PIELM-based forward MRE solver for displacement prediction

**Physics Modes**:
- `'effective'`: ρω²=400 (tuned for stable inversion)
- `'physical'`: ρω²=142M (true physics with coordinate rescaling)

**Features**:
- Random wave basis functions φ(x) = sin(ω·x)
- Laplacian computation: ∇²φ = -‖ω‖² φ
- PIELM system: Minimize ‖-μ∇²u - ρω²u‖²
- Boundary condition support
- Data constraint support
- Differentiable solver (autograd through QR/SVD)

**Helmholtz Equation**:
```
∇·(μ∇u) + ρω²u = 0
```

**Usage**:
```python
from forward_model import ForwardMREModel

model = ForwardMREModel(
    n_wave_neurons=100,
    input_dim=3,
    mu_network=mu_net,
    physics_mode='effective',
    seed=42
)

u_pred, mu_pred = model(
    x, bc_indices, u_bc_vals,
    rho_omega2=400.0,
    bc_weight=50.0,
    u_data=u_meas,
    data_weight=50.0
)
```

**Test**: `python forward_model.py`

---

## 🔬 Data Insights (from exploration)

### **BIOQIC Phase 1 Box Phantom**
- **Grid**: 100×80×10 voxels (1mm isotropic)
- **Domain**: 79×99×9 mm (thin slab!)
- **Frequency**: 60 Hz (ω = 377 rad/s)
- **Stiffness**: Background 3 kPa, 4 targets at 10 kPa
- **Excitation**: Traction force on top Y-face (y=99mm)
- **Motion**: Primarily Z-direction (97.2% of energy)

### **Key Findings**
1. **Z-component dominant**: 10,931 μm vs 640 μm for X/Y
2. **Vertical excitation**: Top face actuator → vertical waves
3. **Thin geometry**: Only 9mm thick → essentially 2D+thin
4. **Clean FEM data**: 2 exact stiffness values, no noise
5. **Strong X-correlation**: 0.448 (wave propagation direction)

---

## 🚀 Next Steps

### **Step 6: Trainer Module** (In Progress)
Create `trainer.py` with:
- Training loop with loss tracking
- Multiple loss terms (data, TV, boundary)
- Learning rate scheduling
- Early stopping
- Comprehensive visualization
- Experiment tracking

### **Step 7: Main Training Script**
Create `train_bioqic.py` with:
- Experiment configuration
- Component orchestration
- Multiple training runs
- Results comparison
- Best practices from lessons learned

---

## 📊 Recommended Experiment Sequence

### **Experiment 1: Baseline (Simplest)**
- Displacement: `z_component`
- Boundary: `minimal` (3 anchor points)
- Stiffness: `direct` strategy, [0.3, 1.0]
- Physics: `effective` (ρω²=400)
- Weights: `data_weight=100`, `bc_weight=10`

### **Experiment 2: Physics-Informed**
- Displacement: `z_component`
- Boundary: `actuator` (top Y-face)
- Stiffness: `direct` strategy, [0.3, 1.0]
- Physics: `effective` (ρω²=400)
- Weights: `data_weight=50`, `bc_weight=50`

### **Experiment 3: Full Vector**
- Displacement: `3_components`
- Boundary: `actuator`
- Stiffness: `direct` strategy, [0.3, 1.0]
- Physics: `effective` (ρω²=400)
- Weights: `data_weight=50`, `bc_weight=50`

### **Experiment 4: Physical Scaling**
- Displacement: `z_component`
- Boundary: `actuator`
- Stiffness: `direct` strategy, [0.3, 1.0]
- Physics: `physical` (ρω²=142M, rescaled Laplacian)
- Weights: `data_weight=50`, `bc_weight=50`

---

## 📝 Lessons Applied

### **From Previous Training Failures**

1. ❌ **Data loss plateau at 0.21**
   - Root cause: Over-constrained boundaries (10% tolerance)
   - ✅ Fix: Physics-informed actuator detection (2% of points)

2. ❌ **Stiffness collapse to narrow range**
   - Root cause: Network bounds [0.5, 5.0] incompatible with [0.3, 1.0]
   - ✅ Fix: Configurable bounds matching data

3. ❌ **93-376% reconstruction error**
   - Root cause: Wrong physics scaling (ρω²=400 vs 142M mismatch)
   - ✅ Fix: Both modes available with proper coordinate rescaling

4. ❌ **Loss of directional information**
   - Root cause: Using magnitude only
   - ✅ Fix: Three modes including full 3-component vector

---

## 🧪 Testing

Each module has built-in tests. Run independently:

```bash
cd bioqic_training

# Test boundary detection
python boundary_detection.py

# Test data loader
python data_loader.py

# Test stiffness network
python stiffness_network.py

# Test forward model
python forward_model.py
```

All tests should pass with informative output showing:
- Module functionality
- Parameter ranges
- Shape validation
- Visual outputs (where applicable)

---

## 📚 References

**BIOQIC Documentation**:
- Box phantom: "Traction force applied on top x-z plane"
- Material: Voigt model (μ = μ' + iωη)
- Background: 3 kPa, Targets: 10 kPa, Viscosity: 1 Pa·s

**Key Papers**:
- Barnhill et al., 2017 (BIOQIC Box Phantom)
- MRE physics: ∇·(μ∇u) + ρω²u = 0

---

## ✅ Status

- [x] Boundary detection module (3 strategies)
- [x] Data loader (3 displacement modes)
- [x] Stiffness network (3 output strategies)
- [x] Forward model (2 physics modes)
- [ ] Trainer with logging/visualization
- [ ] Main training script with experiments

**Ready for training loop implementation!**
