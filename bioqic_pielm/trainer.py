"""
MRE Trainer
===========

Training loop for MRE stiffness reconstruction with:
- Data fitting loss
- Total variation regularization
- Learning rate scheduling
- Early stopping
- Progress visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List

from .forward_model import ForwardMREModel


class MRETrainer:
    """Trainer for MRE inverse problem."""

    def __init__(
        self,
        model: ForwardMREModel,
        device: torch.device,
        output_dir: str = "./outputs"
    ):
        """
        Initialize trainer.

        Args:
            model: ForwardMREModel instance
            device: Torch device
            output_dir: Directory for saving results
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            'data_loss': [],
            'tv_loss': [],
            'total_loss': [],
            'grad_norm': [],
            'mu_mse': [],
            'mu_min': [],
            'mu_max': [],
        }

    def train(
        self,
        x: torch.Tensor,
        u_meas: torch.Tensor,
        mu_true: torch.Tensor,
        bc_indices: torch.Tensor,
        u_bc_vals: torch.Tensor,
        rho_omega2: float,
        iterations: int = 3000,
        lr: float = 0.005,
        bc_weight: float = 200.0,
        data_weight: float = 0.0,
        tv_weight: float = 0.001,
        lr_decay_step: int = 1000,
        lr_decay_gamma: float = 0.9,
        grad_clip: float = 1.0,
        early_stopping: int = 1500,
        log_interval: int = 500,
        save_interval: int = 1000
    ) -> Dict:
        """
        Train stiffness reconstruction.

        Args:
            x: (N, dim) coordinates
            u_meas: (N, 1) measured displacement
            mu_true: (N, 1) ground truth stiffness (for monitoring)
            bc_indices: (K,) boundary indices
            u_bc_vals: (K, 1) boundary values
            rho_omega2: Physics parameter
            iterations: Max training iterations
            lr: Learning rate
            bc_weight: BC enforcement weight (200 recommended)
            data_weight: Data constraint weight (0 for BC-only)
            tv_weight: Total variation regularization
            lr_decay_step: LR decay schedule step
            lr_decay_gamma: LR decay factor
            grad_clip: Gradient clipping threshold
            early_stopping: Patience for early stopping
            log_interval: Iterations between log prints
            save_interval: Iterations between saving plots

        Returns:
            Dictionary with final results
        """
        print("=" * 60)
        print("MRE Training")
        print("=" * 60)
        print(f"  Points: {len(x)}")
        print(f"  BC points: {len(bc_indices)}")
        print(f"  Iterations: {iterations}")
        print(f"  LR: {lr}, decay every {lr_decay_step} by {lr_decay_gamma}")
        print(f"  Weights: bc={bc_weight}, data={data_weight}, tv={tv_weight}")
        print("=" * 60)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.mu_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma
        )

        # Early stopping
        best_loss = float('inf')
        patience_counter = 0

        # Training loop
        for i in range(iterations):
            optimizer.zero_grad()

            # Forward pass
            u_pred, mu_pred = self.model(
                x, bc_indices, u_bc_vals, rho_omega2,
                bc_weight=bc_weight,
                u_data=u_meas,
                data_weight=data_weight,
                verbose=(i == 0)
            )

            # Data loss
            loss_data = torch.mean((u_pred - u_meas) ** 2)

            # TV regularization (promotes smooth/piecewise-constant)
            # Sort by first coordinate for meaningful TV
            if self.model.input_dim == 1:
                loss_tv = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))
            else:
                # For 3D, use neighbor differences (simplified)
                loss_tv = torch.mean(torch.abs(mu_pred[1:] - mu_pred[:-1]))

            # Total loss
            loss_total = loss_data + tv_weight * loss_tv

            # Backward pass
            loss_total.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.mu_net.parameters(), max_norm=grad_clip
                )

            optimizer.step()
            scheduler.step()

            # Compute metrics
            with torch.no_grad():
                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in self.model.mu_net.parameters()
                    if p.grad is not None
                ) ** 0.5
                mu_mse = torch.mean((mu_pred - mu_true) ** 2).item()

            # Record history
            self.history['data_loss'].append(loss_data.item())
            self.history['tv_loss'].append(loss_tv.item())
            self.history['total_loss'].append(loss_total.item())
            self.history['grad_norm'].append(grad_norm)
            self.history['mu_mse'].append(mu_mse)
            self.history['mu_min'].append(mu_pred.min().item())
            self.history['mu_max'].append(mu_pred.max().item())

            # Logging
            if i % log_interval == 0 or i == iterations - 1:
                print(f"Iter {i:5d}: Loss={loss_data.item():.4e}, "
                      f"MuMSE={mu_mse:.4e}, Grad={grad_norm:.4e}")
                print(f"           mu=[{mu_pred.min().item():.3f}, "
                      f"{mu_pred.max().item():.3f}], "
                      f"true=[{mu_true.min().item():.3f}, {mu_true.max().item():.3f}]")

            # Save progress plots
            if save_interval > 0 and i > 0 and i % save_interval == 0:
                self._save_progress_plot(i, x, mu_pred, mu_true, u_pred, u_meas)

            # Early stopping
            if loss_data.item() < best_loss:
                best_loss = loss_data.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"\nEarly stopping at iteration {i}")
                    break

        # Final evaluation
        with torch.no_grad():
            u_pred, mu_pred = self.model(
                x, bc_indices, u_bc_vals, rho_omega2,
                bc_weight=bc_weight
            )

        results = {
            'u_pred': u_pred,
            'mu_pred': mu_pred,
            'final_loss': self.history['data_loss'][-1],
            'final_mu_mse': self.history['mu_mse'][-1],
            'history': self.history
        }

        # Save final results
        self._save_final_results(x, mu_pred, mu_true, u_pred, u_meas)

        return results

    def _save_progress_plot(
        self,
        iteration: int,
        x: torch.Tensor,
        mu_pred: torch.Tensor,
        mu_true: torch.Tensor,
        u_pred: torch.Tensor,
        u_meas: torch.Tensor
    ):
        """Save progress visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curve
        ax = axes[0, 0]
        ax.semilogy(self.history['data_loss'], label='Data Loss')
        ax.semilogy(self.history['mu_mse'], label='Mu MSE')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training Loss')
        ax.grid(True)

        # Gradient norm
        ax = axes[0, 1]
        ax.plot(self.history['grad_norm'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True)

        # Mu comparison (scatter for 3D)
        ax = axes[1, 0]
        mu_p = mu_pred.detach().cpu().numpy().flatten()
        mu_t = mu_true.detach().cpu().numpy().flatten()
        ax.scatter(mu_t, mu_p, alpha=0.3, s=1)
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect')
        ax.set_xlabel('True μ')
        ax.set_ylabel('Predicted μ')
        ax.set_title(f'Stiffness Comparison (iter {iteration})')
        ax.legend()
        ax.grid(True)

        # Displacement comparison
        ax = axes[1, 1]
        u_p = u_pred.detach().cpu().numpy().flatten()
        u_m = u_meas.detach().cpu().numpy().flatten()
        ax.scatter(u_m, u_p, alpha=0.3, s=1)
        ax.plot([u_m.min(), u_m.max()], [u_m.min(), u_m.max()], 'r--')
        ax.set_xlabel('Measured u')
        ax.set_ylabel('Predicted u')
        ax.set_title('Displacement Fit')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'progress_iter_{iteration:05d}.png', dpi=150)
        plt.close()

    def _save_final_results(
        self,
        x: torch.Tensor,
        mu_pred: torch.Tensor,
        mu_true: torch.Tensor,
        u_pred: torch.Tensor,
        u_meas: torch.Tensor
    ):
        """Save final results visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Training curves
        ax = axes[0, 0]
        ax.semilogy(self.history['data_loss'], label='Data Loss')
        ax.semilogy(self.history['total_loss'], label='Total Loss', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title('Training Loss')
        ax.grid(True)

        ax = axes[0, 1]
        ax.semilogy(self.history['mu_mse'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title('Stiffness MSE')
        ax.grid(True)

        ax = axes[0, 2]
        ax.plot(self.history['grad_norm'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Norm')
        ax.set_title('Gradient Norm')
        ax.grid(True)

        # Stiffness scatter
        ax = axes[1, 0]
        mu_p = mu_pred.detach().cpu().numpy().flatten()
        mu_t = mu_true.detach().cpu().numpy().flatten()
        ax.scatter(mu_t, mu_p, alpha=0.3, s=1)
        ax.plot([0, 1], [0, 1], 'r--', lw=2)
        ax.set_xlabel('True μ (normalized)')
        ax.set_ylabel('Predicted μ')
        ax.set_title('Stiffness Reconstruction')
        ax.grid(True)

        # Displacement scatter
        ax = axes[1, 1]
        u_p = u_pred.detach().cpu().numpy().flatten()
        u_m = u_meas.detach().cpu().numpy().flatten()
        ax.scatter(u_m, u_p, alpha=0.3, s=1)
        lims = [min(u_m.min(), u_p.min()), max(u_m.max(), u_p.max())]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlabel('Measured u')
        ax.set_ylabel('Predicted u')
        ax.set_title('Displacement Fit')
        ax.grid(True)

        # Stiffness range evolution
        ax = axes[1, 2]
        ax.fill_between(
            range(len(self.history['mu_min'])),
            self.history['mu_min'],
            self.history['mu_max'],
            alpha=0.3
        )
        ax.axhline(mu_t.min(), color='r', linestyle='--', label='True min')
        ax.axhline(mu_t.max(), color='r', linestyle='--', label='True max')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('μ value')
        ax.set_title('Stiffness Range Evolution')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'final_results.png', dpi=150)
        plt.close()

        # Save history
        np.save(self.output_dir / 'training_history.npy', self.history)
        print(f"\nResults saved to {self.output_dir}")
