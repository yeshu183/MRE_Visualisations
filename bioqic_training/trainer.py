"""
MRE Trainer Module
==================

Comprehensive training loop for MRE inverse problems with:
- Multiple loss terms (data, physics, boundary, regularization)
- Learning rate scheduling
- Early stopping
- Extensive visualization
- Detailed logging

Theory: Same as approach folder
--------------------------------
We use the SAME gradient-based optimization through differentiable PIELM:
1. Forward model: u = Φ @ C_u where C_u = solve(H, b)
2. Gradient flows through the linear solver using autograd
3. Loss = data_loss + TV_loss (+ optional physics/BC penalties)
4. Optimizer: Adam with learning rate decay
5. Physics: Helmholtz equation ∇·(μ∇u) + ρω²u = 0

Key differences from approach folder:
- Better boundary detection (actuator-based vs tolerance)
- Flexible stiffness network bounds (configurable)
- Multiple displacement modes (magnitude/component/vector)
- Physics scaling options (physical vs effective)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Callable
import time
import json


class MRETrainer:
    """Trainer for MRE inverse problems with comprehensive logging."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device('cpu'),
        output_dir: str = 'outputs'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Forward MRE model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler (optional)
            device: Compute device
            output_dir: Directory for outputs
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # History tracking
        self.history = {
            'iteration': [],
            'loss_total': [],
            'loss_data': [],
            'loss_bc': [],
            'loss_tv': [],
            'loss_l2': [],
            'grad_norm': [],
            'lr': [],
            'mu_min': [],
            'mu_max': [],
            'mu_mean': [],
            'mu_std': [],
            'mu_range': [],
            'u_mse': [],
            'time_per_iter': []
        }
        
        self.best_loss = float('inf')
        self.best_iteration = 0
        self.patience_counter = 0
    
    def compute_losses(
        self,
        u_pred: torch.Tensor,
        u_meas: torch.Tensor,
        mu_pred: torch.Tensor,
        bc_indices: Optional[torch.Tensor],
        u_bc_vals: Optional[torch.Tensor],
        weights: Optional[torch.Tensor] = None,
        data_weight: float = 1.0,
        bc_weight: float = 1.0,
        tv_weight: float = 0.0,
        l2_weight: float = 0.0,
        mu_prior_mean: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss terms.
        
        Args:
            u_pred: Predicted displacement
            u_meas: Measured displacement
            mu_pred: Predicted stiffness
            bc_indices: Boundary condition indices (can be None)
            u_bc_vals: Boundary values (can be None)
            weights: Point-wise weights for data loss (optional)
            data_weight: Weight for data fitting term
            bc_weight: Weight for boundary conditions
            tv_weight: Weight for Total Variation regularization
            l2_weight: Weight for L2 regularization toward prior
            mu_prior_mean: Prior mean for L2 regularization
            
        Returns:
            Dictionary of loss terms
        """
        losses = {}
        
        # 1. Data loss (MSE between predicted and measured displacement)
        if weights is not None:
            # Weighted data loss (for interior weighting strategy)
            loss_data = torch.mean(weights.unsqueeze(-1) * (u_pred - u_meas) ** 2)
        else:
            # Standard MSE
            loss_data = torch.mean((u_pred - u_meas) ** 2)
        
        losses['data'] = loss_data
        
        # 2. Boundary condition loss
        if bc_indices is not None and len(bc_indices) > 0 and u_bc_vals is not None:
            u_bc_pred = u_pred[bc_indices]
            loss_bc = torch.mean((u_bc_pred - u_bc_vals) ** 2)
        else:
            loss_bc = torch.tensor(0.0, device=self.device)
        
        losses['bc'] = loss_bc
        
        # 3. Total Variation (TV) regularization
        # Promotes piecewise constant/smooth solutions
        if tv_weight > 0:
            # Simple gradient-based TV (finite difference)
            dmu = mu_pred[1:] - mu_pred[:-1]
            loss_tv = torch.mean(torch.abs(dmu))
        else:
            loss_tv = torch.tensor(0.0, device=self.device)
        
        losses['tv'] = loss_tv
        
        # 4. L2 regularization toward prior
        if l2_weight > 0:
            loss_l2 = torch.mean((mu_pred - mu_prior_mean) ** 2)
        else:
            loss_l2 = torch.tensor(0.0, device=self.device)
        
        losses['l2'] = loss_l2
        
        # 5. Total loss
        loss_total = (
            data_weight * loss_data +
            bc_weight * loss_bc +
            tv_weight * loss_tv +
            l2_weight * loss_l2
        )
        
        losses['total'] = loss_total
        
        return losses
    
    def train_step(
        self,
        x: torch.Tensor,
        u_meas: torch.Tensor,
        bc_indices: Optional[torch.Tensor],
        u_bc_vals: Optional[torch.Tensor],
        rho_omega2: float,
        weights: Optional[torch.Tensor] = None,
        data_weight: float = 1.0,
        bc_weight: float = 1.0,
        tv_weight: float = 0.0,
        l2_weight: float = 0.0,
        mu_prior_mean: float = 0.5,
        grad_clip_norm: float = 1.0,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Returns:
            Dictionary of metrics for this iteration
        """
        start_time = time.time()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        u_pred, mu_pred = self.model(
            x, rho_omega2,
            bc_indices=bc_indices,
            u_bc_vals=u_bc_vals,
            bc_weight=bc_weight if bc_indices is not None else 0.0,
            u_data=u_meas,
            data_weight=data_weight,
            verbose=verbose
        )
        
        # Compute losses
        losses = self.compute_losses(
            u_pred, u_meas, mu_pred,
            bc_indices, u_bc_vals, weights,
            data_weight, bc_weight, tv_weight, l2_weight, mu_prior_mean
        )
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.mu_network.parameters(),
            max_norm=grad_clip_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Learning rate step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = {
                'loss_total': losses['total'].item(),
                'loss_data': losses['data'].item(),
                'loss_bc': losses['bc'].item(),
                'loss_tv': losses['tv'].item(),
                'loss_l2': losses['l2'].item(),
                'grad_norm': grad_norm.item(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'mu_min': mu_pred.min().item(),
                'mu_max': mu_pred.max().item(),
                'mu_mean': mu_pred.mean().item(),
                'mu_std': mu_pred.std().item(),
                'mu_range': (mu_pred.max() - mu_pred.min()).item(),
                'u_mse': torch.mean((u_pred - u_meas) ** 2).item(),
                'time_per_iter': time.time() - start_time
            }
        
        return metrics
    
    def train(
        self,
        x: torch.Tensor,
        u_meas: torch.Tensor,
        mu_true: torch.Tensor,
        bc_indices: Optional[torch.Tensor],
        u_bc_vals: Optional[torch.Tensor],
        rho_omega2: float,
        scales: Dict[str, float],
        n_iterations: int = 5000,
        weights: Optional[torch.Tensor] = None,
        data_weight: float = 1.0,
        bc_weight: float = 1.0,
        tv_weight: float = 0.0,
        l2_weight: float = 0.0,
        mu_prior_mean: float = 0.5,
        grad_clip_norm: float = 1.0,
        log_interval: int = 100,
        plot_interval: int = 500,
        early_stopping_patience: Optional[int] = None,
        save_best: bool = True
    ):
        """
        Full training loop.
        
        Args:
            x: Collocation points (N, 3)
            u_meas: Measured displacement (N, 1) or (N, 3)
            mu_true: Ground truth stiffness (N, 1)
            bc_indices: Boundary indices
            u_bc_vals: Boundary values
            rho_omega2: Physics parameter
            scales: Normalization scales
            n_iterations: Number of training iterations
            weights: Point-wise weights (optional)
            data_weight: Data loss weight
            bc_weight: BC loss weight
            tv_weight: TV regularization weight
            l2_weight: L2 regularization weight
            mu_prior_mean: Prior mean for L2
            grad_clip_norm: Gradient clipping threshold
            log_interval: Print progress every N iterations
            plot_interval: Generate plots every N iterations
            early_stopping_patience: Stop if no improvement for N iters
            save_best: Save best model checkpoint
        """
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING")
        print("="*80)
        
        print(f"\n📊 Training Configuration:")
        print(f"   Iterations: {n_iterations:,}")
        print(f"   Data points: {len(x):,}")
        print(f"   BC points: {len(bc_indices) if bc_indices is not None else 0}")
        print(f"   Data weight: {data_weight}")
        print(f"   BC weight: {bc_weight}")
        print(f"   TV weight: {tv_weight}")
        print(f"   L2 weight: {l2_weight}")
        print(f"   Grad clip: {grad_clip_norm}")
        print(f"   ρω²: {rho_omega2}")
        print(f"   Device: {self.device}")
        
        # Training loop
        for iteration in range(n_iterations):
            verbose = (iteration == 0)  # Verbose on first iteration
            
            metrics = self.train_step(
                x, u_meas, bc_indices, u_bc_vals, rho_omega2,
                weights, data_weight, bc_weight, tv_weight, l2_weight,
                mu_prior_mean, grad_clip_norm, verbose
            )
            
            # Log metrics
            self.history['iteration'].append(iteration)
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Print progress
            if iteration % log_interval == 0 or iteration < 3:
                self._print_progress(iteration, metrics, scales)
            
            # Generate plots
            if iteration % plot_interval == 0 and iteration > 0:
                self._plot_training_progress(x, u_meas, mu_true, rho_omega2, scales, iteration)
            
            # Early stopping check
            if early_stopping_patience is not None:
                if metrics['loss_total'] < self.best_loss:
                    self.best_loss = metrics['loss_total']
                    self.best_iteration = iteration
                    self.patience_counter = 0
                    
                    if save_best:
                        self._save_checkpoint('best_model.pt')
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= early_stopping_patience:
                        print(f"\n⚠️  Early stopping at iteration {iteration}")
                        print(f"   Best loss: {self.best_loss:.6e} at iteration {self.best_iteration}")
                        break
        
        # Final evaluation and visualization
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE")
        print("="*80)
        
        self._final_evaluation(x, u_meas, mu_true, rho_omega2, scales)
        self._save_training_history()
    
    def _print_progress(self, iteration: int, metrics: Dict, scales: Dict):
        """Print training progress."""
        # Denormalize stiffness for display
        mu_min_real = metrics['mu_min'] * scales['mu_scale']
        mu_max_real = metrics['mu_max'] * scales['mu_scale']
        mu_mean_real = metrics['mu_mean'] * scales['mu_scale']
        mu_std_real = metrics['mu_std'] * scales['mu_scale']
        
        print(f"\n{'─'*80}")
        print(f"Iter {iteration:5d} | Time: {metrics['time_per_iter']:.3f}s | LR: {metrics['lr']:.2e}")
        print(f"{'─'*80}")
        print(f"  Losses:")
        print(f"    Total:  {metrics['loss_total']:.6e}")
        print(f"    Data:   {metrics['loss_data']:.6e} (MSE)")
        print(f"    BC:     {metrics['loss_bc']:.6e}")
        print(f"    TV:     {metrics['loss_tv']:.6e}")
        print(f"    L2:     {metrics['loss_l2']:.6e}")
        print(f"  Optimization:")
        print(f"    Grad norm: {metrics['grad_norm']:.3e}")
        print(f"    u MSE:     {metrics['u_mse']:.6e}")
        print(f"  Stiffness (normalized):")
        print(f"    Range: [{metrics['mu_min']:.4f}, {metrics['mu_max']:.4f}]")
        print(f"    Mean:  {metrics['mu_mean']:.4f} ± {metrics['mu_std']:.4f}")
        print(f"  Stiffness (Pa):")
        print(f"    Range: [{mu_min_real:.0f}, {mu_max_real:.0f}] Pa")
        print(f"    Mean:  {mu_mean_real:.0f} ± {mu_std_real:.0f} Pa")
        print(f"    (Target: [3000, 10000] Pa)")
    
    def _plot_training_progress(
        self,
        x: torch.Tensor,
        u_meas: torch.Tensor,
        mu_true: torch.Tensor,
        rho_omega2: float,
        scales: Dict,
        iteration: int
    ):
        """Generate comprehensive progress plots."""
        with torch.no_grad():
            # Get current predictions
            u_pred, mu_pred = self.model(
                x, rho_omega2,
                bc_indices=None, u_bc_vals=None,
                bc_weight=0.0, u_data=None, data_weight=0.0
            )
            
            # Convert to numpy
            x_np = x.cpu().numpy()
            u_meas_np = u_meas.cpu().numpy()
            u_pred_np = u_pred.cpu().numpy()
            mu_true_np = (mu_true.cpu().numpy() * scales['mu_scale']).flatten()
            mu_pred_np = (mu_pred.cpu().numpy() * scales['mu_scale']).flatten()
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Row 1: Loss curves (4 plots)
        ax1 = fig.add_subplot(3, 4, 1)
        ax1.semilogy(self.history['iteration'], self.history['loss_total'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Loss (log)')
        ax1.set_title('Total Loss Evolution')
        ax1.grid(alpha=0.3)
        
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.semilogy(self.history['iteration'], self.history['loss_data'], 'r-', linewidth=2, label='Data')
        ax2.semilogy(self.history['iteration'], self.history['loss_bc'], 'g-', linewidth=2, label='BC')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (log)')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.semilogy(self.history['iteration'], self.history['grad_norm'], 'purple', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Norm (log)')
        ax3.set_title('Gradient Magnitude')
        ax3.grid(alpha=0.3)
        
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(self.history['iteration'], self.history['lr'], 'orange', linewidth=2)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('LR Schedule')
        ax4.grid(alpha=0.3)
        
        # Row 2: Stiffness evolution (4 plots)
        mu_min_hist = np.array(self.history['mu_min']) * scales['mu_scale']
        mu_max_hist = np.array(self.history['mu_max']) * scales['mu_scale']
        mu_mean_hist = np.array(self.history['mu_mean']) * scales['mu_scale']
        mu_std_hist = np.array(self.history['mu_std']) * scales['mu_scale']
        
        ax5 = fig.add_subplot(3, 4, 5)
        ax5.plot(self.history['iteration'], mu_min_hist, 'b-', linewidth=2, label='Min')
        ax5.plot(self.history['iteration'], mu_max_hist, 'r-', linewidth=2, label='Max')
        ax5.plot(self.history['iteration'], mu_mean_hist, 'g-', linewidth=2, label='Mean')
        ax5.axhline(3000, color='b', linestyle='--', alpha=0.5, label='True min')
        ax5.axhline(10000, color='r', linestyle='--', alpha=0.5, label='True max')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Stiffness (Pa)')
        ax5.set_title('μ Range Evolution')
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.3)
        
        ax6 = fig.add_subplot(3, 4, 6)
        ax6.plot(self.history['iteration'], mu_std_hist, 'purple', linewidth=2)
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Stiffness Std Dev (Pa)')
        ax6.set_title('μ Variability')
        ax6.grid(alpha=0.3)
        
        ax7 = fig.add_subplot(3, 4, 7)
        ax7.hist(mu_true_np, bins=50, alpha=0.6, label='Ground Truth', color='blue', edgecolor='black')
        ax7.hist(mu_pred_np, bins=50, alpha=0.6, label='Predicted', color='red', edgecolor='black')
        ax7.set_xlabel('Stiffness (Pa)')
        ax7.set_ylabel('Frequency')
        ax7.set_title(f'μ Distribution (Iter {iteration})')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.scatter(mu_true_np, mu_pred_np, alpha=0.3, s=1)
        ax8.plot([3000, 10000], [3000, 10000], 'r--', linewidth=2)
        ax8.set_xlabel('True μ (Pa)')
        ax8.set_ylabel('Predicted μ (Pa)')
        ax8.set_title('μ Reconstruction')
        ax8.grid(alpha=0.3)
        
        # Row 3: Displacement fit (4 plots)
        ax9 = fig.add_subplot(3, 4, 9, projection='3d')
        scatter9 = ax9.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                              c=u_meas_np.flatten(), cmap='viridis', s=1)
        ax9.set_title('Measured u')
        ax9.set_xlabel('X'); ax9.set_ylabel('Y'); ax9.set_zlabel('Z')
        plt.colorbar(scatter9, ax=ax9, shrink=0.5, pad=0.1)
        
        ax10 = fig.add_subplot(3, 4, 10, projection='3d')
        scatter10 = ax10.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                                c=u_pred_np.flatten(), cmap='viridis', s=1)
        ax10.set_title('Predicted u')
        ax10.set_xlabel('X'); ax10.set_ylabel('Y'); ax10.set_zlabel('Z')
        plt.colorbar(scatter10, ax=ax10, shrink=0.5, pad=0.1)
        
        u_error = np.abs(u_pred_np - u_meas_np).flatten()
        ax11 = fig.add_subplot(3, 4, 11, projection='3d')
        scatter11 = ax11.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2],
                                c=u_error, cmap='hot', s=1)
        ax11.set_title('Displacement Error')
        ax11.set_xlabel('X'); ax11.set_ylabel('Y'); ax11.set_zlabel('Z')
        plt.colorbar(scatter11, ax=ax11, shrink=0.5, pad=0.1)
        
        ax12 = fig.add_subplot(3, 4, 12)
        ax12.scatter(u_meas_np, u_pred_np, alpha=0.3, s=1)
        ax12.plot([u_meas_np.min(), u_meas_np.max()],
                 [u_meas_np.min(), u_meas_np.max()], 'r--', linewidth=2)
        ax12.set_xlabel('Measured u')
        ax12.set_ylabel('Predicted u')
        ax12.set_title('Displacement Fit')
        ax12.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / f'progress_iter_{iteration:05d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  📊 Saved progress plot: {save_path}")
        plt.close()
    
    def _final_evaluation(self, x, u_meas, mu_true, rho_omega2, scales):
        """Comprehensive final evaluation."""
        print(f"\n📈 Final Metrics:")
        print(f"   Best iteration: {self.best_iteration}")
        print(f"   Best loss: {self.best_loss:.6e}")
        print(f"   Final loss: {self.history['loss_total'][-1]:.6e}")
        
        with torch.no_grad():
            u_pred, mu_pred = self.model(
                x, rho_omega2,
                bc_indices=None, u_bc_vals=None,
                bc_weight=0.0, u_data=None, data_weight=0.0
            )
            
            mu_pred_np = (mu_pred.cpu().numpy() * scales['mu_scale']).flatten()
            mu_true_np = (mu_true.cpu().numpy() * scales['mu_scale']).flatten()
            
            mu_error = np.abs(mu_pred_np - mu_true_np)
            mu_rel_error = mu_error / (mu_true_np + 1e-10) * 100
            
            print(f"\n🎯 Stiffness Reconstruction:")
            print(f"   Predicted range: [{mu_pred_np.min():.0f}, {mu_pred_np.max():.0f}] Pa")
            print(f"   True range: [{mu_true_np.min():.0f}, {mu_true_np.max():.0f}] Pa")
            print(f"   MAE: {mu_error.mean():.1f} Pa")
            print(f"   RMSE: {np.sqrt(np.mean(mu_error**2)):.1f} Pa")
            print(f"   Rel Error: {mu_rel_error.mean():.1f}% ± {mu_rel_error.std():.1f}%")
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_loss': self.best_loss,
            'best_iteration': self.best_iteration
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / 'training_history.json'
        
        # Convert to serializable format
        history_serializable = {}
        for key, values in self.history.items():
            if isinstance(values[0], (int, float)):
                history_serializable[key] = values
            else:
                history_serializable[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"\n  💾 Saved training history: {history_path}")


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("TRAINER MODULE - Example Usage")
    print("="*80)
    print("\nThis module provides comprehensive training loop for MRE inversion.")
    print("\nKey features:")
    print("  ✓ Multiple loss terms (data, BC, TV, L2)")
    print("  ✓ Learning rate scheduling")
    print("  ✓ Early stopping")
    print("  ✓ Extensive visualization (every N iterations)")
    print("  ✓ Detailed logging and metrics tracking")
    print("  ✓ Model checkpointing")
    print("\nSee train_bioqic.py for full integration example.")
    print("="*80)
