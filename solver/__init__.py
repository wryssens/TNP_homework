# solver package
# This makes the solver directory a proper Python package
"""
Solver package for nuclear shell model calculations.

Contains modules for:
- Hamiltonian construction and TBME handling
- Single-particle basis definition and quantum state management
- Angular momentum coupling calculations
- Slater determinant operations
"""

# Explicitly import key classes for convenient access
from .hamiltonian import Hamiltonian
from .singleparticlebasis import SingleParticleBasis, build_single_particle_basis
from .tbme import TBME
from .angmomcoupling import clebsch, wigner
from .slater_determinant import SlaterDeterminant

__all__ = ['Hamiltonian', 'SingleParticleBasis', 'build_single_particle_basis', 
           'TBME', 'clebsch', 'wigner', 'SlaterDeterminant']