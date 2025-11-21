"""
Differentiable PIELM Solver
===========================

Custom autograd function for solving least-squares with analytical gradients.
Enables backpropagation through the linear solver for stiffness optimization.

Key Formula (Backward Pass):
    dC = -(H^T H)^{-1} [dH^T r + H^T dH C]
    where r = HC - b (residual)
"""

import torch


class DifferentiablePIELM(torch.autograd.Function):
    """Differentiable least-squares solver with analytical backward pass.

    Solves: (H^T H + λI) C = H^T b via Cholesky decomposition.

    Backward implements:
        grad_H = -(H v C^T + r v^T)
        grad_b = H v
    where v = (H^T H + λI)^{-1} grad_C and r = H C - b.
    """

    @staticmethod
    def forward(ctx, H, b, regularization=1e-3, max_attempts=8):
        """Solve least-squares with adaptive regularization.

        Args:
            H: (N, M) design matrix
            b: (N, 1) target vector
            regularization: Initial regularization strength
            max_attempts: Max escalation attempts before failure

        Returns:
            C: (M, 1) coefficient vector
        """
        H_T = H.t()
        A = H_T @ H
        eye = torch.eye(A.shape[0], device=H.device, dtype=H.dtype)
        reg = regularization
        last_error = None

        for attempt in range(max_attempts):
            A_reg = A + reg * eye
            try:
                L = torch.linalg.cholesky(A_reg)
                rhs = H_T @ b
                C = torch.cholesky_solve(rhs, L)
                r = (H @ C) - b
                ctx.save_for_backward(H, C, L, r)
                ctx.reg = reg
                return C
            except RuntimeError as e:
                last_error = e
                reg *= 5.0  # Escalate regularization

        raise RuntimeError(f"Cholesky failed after {max_attempts} attempts: {last_error}")

    @staticmethod
    def backward(ctx, grad_C):
        """Analytical backward pass through solver.

        Implements the gradient formula derived from normal equations.
        """
        H, C, L, r = ctx.saved_tensors
        v = torch.cholesky_solve(grad_C, L)
        grad_b = H @ v
        term1 = (H @ v) @ C.t()
        term2 = r @ v.t()
        grad_H = -(term1 + term2)
        return grad_H, grad_b, None, None


def pielm_solve(H, b, reg=1e-2, try_fallback=True, verbose=False):
    """Public solver interface with fallback to lstsq.

    Args:
        H: (N, M) design matrix
        b: (N, 1) targets
        reg: Initial regularization strength
        try_fallback: Enable lstsq fallback if Cholesky fails
        verbose: Print solver path used

    Returns:
        C: (M, 1) coefficient vector
    """
    try:
        C = DifferentiablePIELM.apply(H, b, reg)
        if verbose:
            print("  [Solver: Custom Cholesky with analytical backward]")
        return C
    except RuntimeError as e:
        if not try_fallback:
            raise
        if verbose:
            print(f"  [Solver: Fallback to QR/SVD]")
        lstsq_out = torch.linalg.lstsq(H, b, rcond=None, driver='gelsd')
        return lstsq_out.solution
