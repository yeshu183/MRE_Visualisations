"""Core utilities for PIELM-MRE framework.

This package provides reusable components for:
- Forward and inverse solving
- Visualization
- Data generation
"""

from .solver import train_inverse_problem, evaluate_reconstruction
from .visualization import plot_results, create_loss_plots
from .data_generators import (
    generate_gaussian_bump,
    generate_multiple_inclusions,
    generate_step_function,
    generate_synthetic_data
)

__all__ = [
    'train_inverse_problem',
    'evaluate_reconstruction',
    'plot_results',
    'create_loss_plots',
    'generate_gaussian_bump',
    'generate_multiple_inclusions',
    'generate_step_function',
    'generate_synthetic_data'
]
