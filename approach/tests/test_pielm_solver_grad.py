"""Finite difference gradient check for pielm_solve.
Run directly: python test_pielm_solver_grad.py
Prints max absolute/relative error for one element perturbation.
"""

import torch
import math
import sys
from pathlib import Path

# Ensure approach package importable
ROOT = Path(__file__).parent
sys.path.append(str(ROOT / 'approach'))

from pielm_solver import pielm_solve  # type: ignore


def gradient_check(eps: float = 1e-4):
    torch.manual_seed(123)
    H = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    b = torch.randn(5, 1, dtype=torch.double, requires_grad=True)

    C = pielm_solve(H, b)  # (3,1)
    objective = C.sum()
    objective.backward()

    autograd_grad_H = H.grad.clone()

    # Finite difference on one element (0,0)
    i, j = 0, 0
    orig = H[i, j].item()
    H_plus = H.clone().detach(); H_plus[i, j] = orig + eps
    H_minus = H.clone().detach(); H_minus[i, j] = orig - eps

    C_plus = pielm_solve(H_plus, b.detach())
    C_minus = pielm_solve(H_minus, b.detach())
    obj_plus = C_plus.sum().item()
    obj_minus = C_minus.sum().item()

    fd_grad = (obj_plus - obj_minus) / (2 * eps)
    auto_grad = autograd_grad_H[i, j].item()

    abs_err = abs(fd_grad - auto_grad)
    rel_err = abs_err / (abs(auto_grad) + 1e-12)

    print(f"Finite difference grad (H[0,0]): {fd_grad:.6e}")
    print(f"Autograd grad        (H[0,0]): {auto_grad:.6e}")
    print(f"Absolute error: {abs_err:.3e}")
    print(f"Relative error: {rel_err:.3e}")
    tol = 1e-3
    if rel_err < tol:
        print("PASS: Gradient check within tolerance.")
        return True
    else:
        print("FAIL: Gradient check exceeded tolerance.")
        return False


if __name__ == '__main__':
    ok = gradient_check()
    if not ok:
        sys.exit(1)
