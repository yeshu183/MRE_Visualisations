"""Visualization utilities for MRE inverse problems.

Provides standardized plotting functions for results and training diagnostics.
"""

import matplotlib.pyplot as plt
import torch


def plot_results(x, u_meas, u_pred, u_true, mu_true, mu_pred, history, 
                save_path, title_suffix=""):
    """Create comprehensive 6-panel visualization.
    
    Args:
        x: Spatial coordinates (N, 1)
        u_meas: Measured wave field (N, 1)
        u_pred: Predicted wave field (N, 1)
        u_true: True wave field (N, 1)
        mu_true: Ground truth stiffness (N, 1)
        mu_pred: Predicted stiffness (N, 1)
        history: Training history dictionary
        save_path: Path to save the figure
        title_suffix: Optional suffix for plot titles
    """
    x_np = x.cpu().numpy()
    
    fig = plt.figure(figsize=(20, 10))
    
    # ========== Row 1: Reconstruction Results ==========
    
    # Wave field
    plt.subplot(2, 3, 1)
    title = f"Wave Field Reconstruction{title_suffix}"
    plt.title(title, fontsize=12, fontweight='bold')
    plt.plot(x_np, u_true.cpu().numpy(), 'k', label='True (Clean)', linewidth=2)
    plt.plot(x_np, u_meas.cpu().numpy(), 'k--', label='Measured (Noisy)', alpha=0.7)
    plt.plot(x_np, u_pred.cpu().numpy(), 'r', label='Predicted', linewidth=2)
    plt.xlabel('Position x')
    plt.ylabel('Displacement u')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Stiffness reconstruction
    plt.subplot(2, 3, 2)
    title = f"Stiffness Reconstruction{title_suffix}"
    plt.title(title, fontsize=12, fontweight='bold')
    plt.plot(x_np, mu_true.cpu().numpy(), 'k', label='Ground Truth', linewidth=2)
    plt.plot(x_np, mu_pred.cpu().numpy(), 'b', label='Recovered', linewidth=2)
    plt.xlabel('Position x')
    plt.ylabel('Stiffness μ')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Pointwise error
    plt.subplot(2, 3, 3)
    plt.title("Pointwise Stiffness Error", fontsize=12, fontweight='bold')
    error = (mu_pred.cpu().numpy() - mu_true.cpu().numpy())
    plt.plot(x_np, error, 'r', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Position x')
    plt.ylabel('Error (μ_pred - μ_true)')
    plt.grid(alpha=0.3)
    
    # ========== Row 2: Training Diagnostics ==========
    
    iterations = range(len(history['data_loss']))
    
    # Data loss
    plt.subplot(2, 3, 4)
    plt.title("Data Loss (MSE)", fontsize=12, fontweight='bold')
    plt.semilogy(iterations, history['data_loss'], 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.grid(alpha=0.3)
    
    # Gradient norm & TV loss
    plt.subplot(2, 3, 5)
    has_tv = max(history['tv_loss']) > 1e-10
    if has_tv:
        title = "TV Loss & Gradient Norm"
    else:
        title = "Gradient Norm"
    plt.title(title, fontsize=12, fontweight='bold')
    
    ax1 = plt.gca()
    ax1.semilogy(iterations, history['grad_norm'], 'g-', linewidth=2, label='Grad Norm')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Gradient Norm (log scale)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(alpha=0.3)
    
    if has_tv:
        ax2 = ax1.twinx()
        ax2.semilogy(iterations, history['tv_loss'], 'orange', linewidth=2, label='TV Loss')
        ax2.set_ylabel('TV Loss (log scale)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
    
    # Mu statistics
    plt.subplot(2, 3, 6)
    plt.title("Mu Statistics Over Training", fontsize=12, fontweight='bold')
    plt.plot(iterations, history['mu_min'], 'b-', linewidth=2, label='Min μ')
    plt.plot(iterations, history['mu_max'], 'r-', linewidth=2, label='Max μ')
    plt.plot(iterations, history['mu_mean'], 'g-', linewidth=2, label='Mean μ')
    plt.axhline(mu_true.min().item(), color='b', linestyle='--', alpha=0.5, label='True min')
    plt.axhline(mu_true.max().item(), color='r', linestyle='--', alpha=0.5, label='True max')
    plt.xlabel('Iteration')
    plt.ylabel('Stiffness μ')
    plt.legend(loc='best', fontsize=8)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"\n  Plot saved: {save_path}")
    plt.close()


def create_loss_plots(history, save_path):
    """Create detailed loss and metric plots.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    iterations = range(len(history['data_loss']))
    
    # Data loss
    axes[0, 0].semilogy(iterations, history['data_loss'], 'b-', linewidth=2)
    axes[0, 0].set_title("Data Loss (MSE)", fontweight='bold')
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss (log scale)")
    axes[0, 0].grid(alpha=0.3)
    
    # Gradient norm
    axes[0, 1].semilogy(iterations, history['grad_norm'], 'g-', linewidth=2)
    axes[0, 1].set_title("Gradient Norm", fontweight='bold')
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Norm (log scale)")
    axes[0, 1].grid(alpha=0.3)
    
    # Mu MSE (reconstruction error)
    axes[1, 0].semilogy(iterations, history['mu_mse'], 'm-', linewidth=2)
    axes[1, 0].set_title("Mu Reconstruction MSE", fontweight='bold')
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("MSE (log scale)")
    axes[1, 0].grid(alpha=0.3)
    
    # Mu range over time
    axes[1, 1].plot(iterations, history['mu_min'], 'b-', linewidth=2, label='Min')
    axes[1, 1].plot(iterations, history['mu_max'], 'r-', linewidth=2, label='Max')
    axes[1, 1].plot(iterations, history['mu_mean'], 'g-', linewidth=2, label='Mean')
    axes[1, 1].set_title("Mu Statistics", fontweight='bold')
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Stiffness μ")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"  Loss plots saved: {save_path}")
    plt.close()


def plot_comparison_1d(x, y_true, y_pred, ylabel, title, save_path=None):
    """Simple 1D comparison plot.
    
    Args:
        x: Spatial coordinates
        y_true: Ground truth values
        y_pred: Predicted values
        ylabel: Y-axis label
        title: Plot title
        save_path: Optional path to save figure
    """
    x_np = x.cpu().numpy() if torch.is_tensor(x) else x
    y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_np, y_true_np, 'k-', label='Ground Truth', linewidth=2)
    plt.plot(x_np, y_pred_np, 'r--', label='Predicted', linewidth=2)
    plt.xlabel('Position x')
    plt.ylabel(ylabel)
    plt.title(title, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"  Plot saved: {save_path}")
    else:
        plt.show()
    
    plt.close()
