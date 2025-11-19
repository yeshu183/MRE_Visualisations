import torch

class DifferentiablePIELM(torch.autograd.Function):
    """Differentiable least-squares solver using normal equations with adaptive regularization.

    Solves (H^T H + λI) C = H^T b via Cholesky. If Cholesky fails, caller should fall back.
    Backward implements analytic gradient:
        grad_H = -(H v C^T + r v^T)
        grad_b = H v
    where v = (H^T H + λI)^{-1} grad_C and r = H C - b.
    """

    @staticmethod
    def forward(ctx, H, b, regularization=1e-3, max_attempts=8):
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
                reg *= 5.0  # escalate regularization more aggressively
        # If we reach here, all attempts failed.
        raise RuntimeError(f"Adaptive Cholesky failed after {max_attempts} attempts; last error: {last_error}")

    @staticmethod
    def backward(ctx, grad_C):
        H, C, L, r = ctx.saved_tensors
        v = torch.cholesky_solve(grad_C, L)
        grad_b = H @ v
        term1 = (H @ v) @ C.t()
        term2 = r @ v.t()
        grad_H = -(term1 + term2)
        return grad_H, grad_b, None, None  # None for regularization & max_attempts


def pielm_solve(H, b, reg=1e-2, try_fallback=True, verbose=False):
    """Public solver interface with fallback to torch.linalg.lstsq.

    Attempts adaptive Cholesky first. Fallback uses standard torch autograd which
    DOES support gradients wrt H (just slower than our custom implementation).

    Args:
        H: (N,M) design matrix
        b: (N,1) targets
        reg: initial regularization strength (increased default for stability)
        try_fallback: enable lstsq fallback if normal equations fail
        verbose: print which solver path is used
    Returns:
        C: (M,1) coefficient vector
    """
    try:
        C = DifferentiablePIELM.apply(H, b, reg)
        if verbose:
            print("  [Solver: Custom Cholesky with analytical backward ✓]")
        return C
    except RuntimeError as e:
        if not try_fallback:
            raise
        if verbose:
            print(f"  [Solver: Fallback to QR/SVD (torch autograd, slower but stable)]")
        # Fallback: QR-based least-squares with automatic differentiation
        lstsq_out = torch.linalg.lstsq(H, b, rcond=None, driver='gelsd')
        C = lstsq_out.solution
        return C
        return C