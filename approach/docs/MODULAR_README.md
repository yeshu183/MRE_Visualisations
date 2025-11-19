# Modular PIELM-MRE Framework

## 📁 New Structure

```
approach/
├── core/                           # ✨ NEW: Reusable components
│   ├── __init__.py                # Module exports
│   ├── data_generators.py         # Synthetic data generation
│   ├── solver.py                  # Training & evaluation
│   └── visualization.py           # Plotting utilities
│
├── models.py                       # Neural network architectures
├── pielm_solver.py                # Differentiable PIELM solver
├── config_forward.json            # Main configuration file
│
├── example_gaussian_bump.py       # ✨ Example 1: Single inclusion
├── example_multiple_inclusions.py # ✨ Example 2: Two peaks
├── example_step_function.py       # ✨ Example 3: Sharp transition
├── run_all_examples.py            # ✨ Run all tests
│
└── [OLD FILES]                    # Can be removed after testing
    ├── main_mre.py
    ├── test_multiple_inclusions.py
    └── test_step_function.py
```

## 🚀 Quick Start

### Run All Examples
```bash
python approach/run_all_examples.py
```

### Run Individual Examples
```bash
python approach/example_gaussian_bump.py
python approach/example_multiple_inclusions.py
python approach/example_step_function.py
```

## 📝 Creating New Examples

### Method 1: Use Existing Generators

```python
import torch
import json
from core import (
    generate_gaussian_bump,  # or generate_multiple_inclusions, generate_step_function
    train_inverse_problem,
    evaluate_reconstruction,
    plot_results
)

# Load config
with open('approach/config_forward.json', 'r') as f:
    config = json.load(f)

# Customize if needed
config['tv_weight'] = 0.001
config['lr'] = 0.01

# Generate data
x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_gaussian_bump(
    n_points=100,
    n_wave_neurons=60,
    device='cpu',
    seed=0
)

# Train
model, history = train_inverse_problem(
    x, u_meas, mu_true, bc_indices, u_bc_vals,
    config, device='cpu'
)

# Evaluate and visualize
# ... (see examples)
```

### Method 2: Create Custom Data Generator

```python
from core import generate_synthetic_data

# Define your custom stiffness distribution
x = torch.linspace(0, 1, 100).reshape(-1, 1)
mu_custom = 1.0 + 0.5 * torch.sin(10 * np.pi * x)  # Oscillatory pattern

# Generate synthetic measurements
x, u_meas, mu_true, u_true, bc_indices, u_bc_vals = generate_synthetic_data(
    x, mu_custom,
    n_wave_neurons=60,
    device='cpu',
    seed=123,
    rho_omega2=400.0,
    bc_weight=200.0,
    noise_std=0.001
)

# Then train using train_inverse_problem() as above
```

## ⚙️ Configuration

### Main Config File: `config_forward.json`

```json
{
  "n_points": 100,              // Number of spatial points
  "n_wave_neurons": 60,         // Wave basis functions
  "iterations": 5000,           // Training iterations
  "lr": 0.01,                   // Learning rate
  "lr_decay_step": 1500,        // LR decay frequency
  "lr_decay_gamma": 0.9,        // LR decay factor
  "rho_omega2": 400.0,          // PDE parameter
  "noise_std": 0.001,           // Measurement noise
  "bc_weight": 200.0,           // Boundary condition weight
  "tv_weight": 0.0,             // Total variation regularization
  "seed": 0,                    // Random seed
  "early_stopping_patience": 1500,
  "grad_clip_max_norm": 1.0,
  "mu_min": 0.7,
  "mu_max": 6.0
}
```

### Override Config Per Example

```python
# Load base config
with open('config_forward.json', 'r') as f:
    config = json.load(f)

# Override specific parameters
config['tv_weight'] = 0.002  # For sharp edges
config['lr'] = 0.015         // Different learning rate
config['seed'] = 42          // Different random seed
```

## 📊 Core Modules

### 1. **data_generators.py**
- `generate_gaussian_bump()`: Single Gaussian inclusion
- `generate_multiple_inclusions()`: Two Gaussian peaks
- `generate_step_function()`: Sharp layer boundary
- `generate_synthetic_data()`: Generic data generation from μ(x)

### 2. **solver.py**
- `train_inverse_problem()`: Main training loop with:
  - Configurable hyperparameters
  - Early stopping
  - Learning rate scheduling
  - TV regularization
  - Gradient clipping
  - Comprehensive metrics tracking
  
- `evaluate_reconstruction()`: Compute metrics:
  - MSE, MAE, max error
  - Relative errors
  - Range statistics

### 3. **visualization.py**
- `plot_results()`: 6-panel comprehensive plot:
  - Wave field reconstruction
  - Stiffness reconstruction
  - Pointwise error
  - Loss curves
  - Gradient/TV monitoring
  - Mu statistics over training
  
- `create_loss_plots()`: Detailed training diagnostics
- `plot_comparison_1d()`: Simple comparison plots

## 🎯 Benefits of New Structure

### ✅ Easy to Add New Examples
Just create a new `example_*.py` file using the core functions.

### ✅ Consistent Visualization
All examples use the same plotting format.

### ✅ Centralized Configuration
Change `config_forward.json` once, affects all examples.

### ✅ Per-Example Customization
Override specific config values in each example.

### ✅ Code Reuse
No duplicated training loops or plotting code.

### ✅ Easy Maintenance
Bug fixes in `core/` automatically fix all examples.

## 🧪 Testing Workflow

1. **Modify config**: Edit `config_forward.json`
2. **Run tests**: `python approach/run_all_examples.py`
3. **Compare results**: Check generated `.png` files
4. **Iterate**: Adjust config based on results

## 📈 Expected Results

### Gaussian Bump
- **Loss**: < 1e-4
- **Relative MSE**: < 0.1
- **Characteristics**: Smooth, single peak

### Multiple Inclusions
- **Loss**: < 1e-3
- **Relative MSE**: < 0.2
- **Characteristics**: Two separated peaks

### Step Function
- **Loss**: < 5e-3 (relaxed)
- **Relative MSE**: < 0.5 (relaxed)
- **Characteristics**: Sharp transition (hard for smooth networks)

## 🔧 Advanced Usage

### Custom Evaluation Metrics

```python
from core.solver import evaluate_reconstruction

# Get standard metrics
metrics = evaluate_reconstruction(mu_pred, mu_true, final_loss)

# Add custom metrics
metrics['custom_score'] = compute_my_metric(mu_pred, mu_true)
```

### Custom Training Callbacks

```python
# Modify core/solver.py train_inverse_problem()
# Add callback hooks at key points:
# - After each iteration
# - Before early stopping
# - On convergence
```

### Multi-Resolution Training

```python
# Train on coarse grid first
config['n_points'] = 50
model_coarse, _ = train_inverse_problem(...)

# Transfer to fine grid
config['n_points'] = 200
# Initialize with coarse solution
# ... (implementation needed)
```

## 📚 Migration from Old Code

Old code stil works but is deprecated:
- `main_mre.py` → Use `example_gaussian_bump.py`
- `test_multiple_inclusions.py` → Use `example_multiple_inclusions.py`
- `test_step_function.py` → Use `example_step_function.py`

Old files can be removed after verifying new examples work.

## 🐛 Troubleshooting

### Import Errors
Make sure to run from workspace root or add to path:
```python
import sys
sys.path.append('path/to/MRE_Visualisations')
```

### Config Not Found
Check that `config_forward.json` exists in `approach/` folder.

### Module Not Found
Verify `core/__init__.py` exists and exports functions.

## 📄 License & Citation

[Your license and citation info here]
