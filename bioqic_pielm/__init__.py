"""
BIOQIC-PIELM: Physics-Informed Extreme Learning Machine for MRE Inversion
==========================================================================

Solves the inverse problem in Magnetic Resonance Elastography using
differentiable PIELM with custom analytical gradients.

Tailored for BIOQIC FEM phantom data with viscoelastic physics support.
"""

from .pielm_solver import pielm_solve, DifferentiablePIELM
from .data_loader import BIOQICDataLoader
from .stiffness_network import StiffnessNetwork
from .forward_model import ForwardMREModel
from .trainer import MRETrainer

__all__ = [
    'pielm_solve',
    'DifferentiablePIELM',
    'BIOQICDataLoader',
    'StiffnessNetwork',
    'ForwardMREModel',
    'MRETrainer',
]
