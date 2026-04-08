# solver package
# This makes the solver directory a proper Python package
"""
Solver package for nuclear shell model calculations.

Contains modules for:
- Hamiltonian construction and TBME handling
- Model space definition and quantum state management
- Angular momentum coupling calculations
- Slater determinant operations
"""

# Explicitly import key classes for convenient access
from .hamiltonian import Hamiltonian
from .modelspace import SingleParticleBasis, build_model_space
from .tbme import TBME
from .angmomcoupling import clebsch, wigner
from .slater_determinant import SlaterDeterminant

__all__ = ['Hamiltonian', 'SingleParticleBasis', 'build_model_space', 
           'TBME', 'clebsch', 'wigner', 'SlaterDeterminant']