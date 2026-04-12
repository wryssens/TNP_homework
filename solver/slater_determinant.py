#!/usr/bin/env python3
"""
Slater Determinant class for nuclear Hartree-Fock calculations.

This module defines a SlaterDeterminant class that represents a Slater determinant
state in nuclear physics, including the associated Hartree-Fock basis, density matrix,
mean-field, and single-particle Hamiltonian.
"""

import numpy as np
from numba import njit

class SlaterDeterminant:
    """
    Represents a Slater determinant state in nuclear Hartree-Fock calculations.
    
    A Slater determinant is defined by:
    - A (computional) single-particle basis
    - A one-body density matrix ρ in the computational basis
    - A mean-field Γ (from two-body interactions)
    - A single-particle Hamiltonian h = h0 + Γ
    - The corresponding Hartree-Fock basis that diagonalizes h
       (Note: this is stored as a unitary transformation matrix linking the
              HF basis to the original computational basis)
    - Occupation factors in the Hartree-Fock basis

    Attributes:
        basis (SingleParticleBasis)   : The computational single-particle basis
        rho (numpy.ndarray)           : The one-body density matrix expressed in the computational basis
        Gamma (numpy.ndarray)         : Mean-field potential in the computational basis
        h (numpy.ndarray)             : Single-particle Hamiltonian matrix in the computational basis
        hf_sp_energies (numpy.ndarray): Single-particle energies, i.e. the eigenvalues of h
        hf_basis (numpy.ndarray)      : The unitary transformation linking the computational and
                                        Hartree-Fock basis, i.e. the eigenvectors of h
        occupations (numpy.ndarray)   : Occupation numbers for each state in the HF basis
        N_valence (int)               : Number of valence neutrons
        Z_valence (int)               : Number of valence protons
        hf_parity (numpy.ndarray)     : Parity quantum numbers in HF basis
        hf_isospin (numpy.ndarray)    : Isospin quantum numbers in HF basis
        hf_two_m (numpy.ndarray)      : Magnetic quantum numbers (2m) in HF basis

    Methods:
        __init__(basis, hf_energies, N_valence, Z_valence): Initialize Slater determinant
        diagonalize(): Diagonalize single-particle Hamiltonian with symmetry constraints
        calculate_energy(H): Calculate total energy
        calculate_quadrupole_expectation() : Calculate the expectation value of Q20
        particle_number(): Calculate total particle number
        neutron_number(): Calculate neutron number
        proton_number(): Calculate proton number
        build_density(): Build density matrix from HF basis and occupations
        build_single_particle_hamiltonian(H): Build h = h0 + Γ
        build_gamma(H): Build mean-field Γ from TBMEs and mean-field density
        set_occupations(): Set occupations based on HF energies and valence numbers
        print_hf_basis(): Print HF basis information
    """
    
    def __init__(self, basis, hf_energies: np.ndarray, N_valence: int, Z_valence: int, shape: str):
        """
        Initialize a Slater determinant.

        Careful: the initialisation builds a Slater determinant by occupying the lowest-lying
                 single-particle states in the computational basis. "Lowest-lying" is defined
                 by the ordering of the hf_energies passed with a (small) added tweak!

                     \epsilon_{\mu} = hf_energies[\mu] + a * | m_\mu |

                 where
                    - |m_{\mu}| is the absolute value of the projection of the angular momentum
                    - the parameter a is
                          * +1 for prolate shapes.
                          * -1 for oblate shapes.
                          -  0 for spherical shapes.

        Note that "spherical" initialisation is NOT GUARANTEED to result in a spherical configuration;
         this will only be the case for nuclei whose neutron and proton number both correspond to
         a closed shell.

        Parameters:
            basis (SingleParticleBasis): Single-particle basis object
            hf_energies (np.ndarray): Initial guess for Hartree-Fock single-particle energies
            N_valence (int): Number of valence neutrons to occupy
            Z_valence (int): Number of valence protons to occupy
            shape(str)     : The type of shape to initialize

        Initializes:
            - Zero mean-field and Hamiltonian matrices
            - Identity HF basis transformation
            - Quantum number tracking in HF basis
            - Automatic occupation setting based on valence numbers
            - Density matrix construction from occupations
        
        Example:
            >>> basis = build_single_particle_basis("basis_def.txt", None)
            >>> initial_energies = np.random.rand(basis.size)
            >>> sd = SlaterDeterminant(basis, initial_energies, 4, 4, 'oblate')
            >>> print(f"Created Slater determinant with {sd.particle_number()} particles")
        """
        self.basis           = basis

        # Apply shape deformation to the energies
        if shape == 'prolate':
            sign = +1
        elif shape == 'oblate':
            sign = -1
        elif shape == 'spherical':
            sign = 0
        else:
            raise ValueError(f"Unknown shape: {shape}")

        changed_energies = np.zeros(basis.size)
        for i in range(basis.size):
            changed_energies[i] = hf_energies[i] +  sign * np.abs(basis.two_m[i])

        self.hf_sp_energies  = changed_energies
        self.N_valence       = N_valence
        self.Z_valence       = Z_valence

        self.Gamma           = np.zeros((basis.size,basis.size))
        self.h               = np.zeros((basis.size,basis.size))

        # Trivial HF transformation = identity matrix
        self.hf_basis        = np.eye(basis.size, basis.size)
        # Track quantum numbers in HF basis
        self.hf_parity       = self.basis.parity
        self.hf_isospin      = self.basis.isospin
        self.hf_two_m        = self.basis.two_m

        # Set occupations automatically based on valence numbers using initial energies
        self.set_occupations()
        # Build density matrix from occupations
        self.rho = np.diag(self.occupations)

    def diagonalize(self):
        """
        Diagonalize the single-particle Hamiltonian, respecting

         1. Axial symmetry, i.e. the 'm' single-particle quantum number
         2. Parity, i.e. the 'p' single-particle quantum number
         3. Isospin, i.e. the proton/neutron nature of single-particle states

        Returns:
            tuple: (hf_sp_energies, hf_basis) where:
                hf_sp_energies (np.ndarray): Single-particle energies in HF basis
                hf_basis (np.ndarray): Transformation matrix from computational to HF basis

        Notes:
            - Tracks quantum numbers for each HF basis state
            - Stores mapping between computational basis and HF basis
            - Sets hf_basis_constructed flag to True upon completion
            - The HF basis states maintain the same quantum numbers as the original basis
        
        Example:
            >>> energies, basis = sd.diagonalize()
            >>> print(f"HF energies: {energies[:5]}")  # First 5 energies
            >>> print(f"Energy range: {energies.min():.2f} to {energies.max():.2f} MeV")
        """
        # Block-diagonal diagonalization
        N = self.h.shape[0]
        self.hf_sp_energies = np.zeros(N)
        self.hf_basis       = np.zeros((N, N))

        # Initialize quantum number arrays
        self.hf_parity       = np.zeros(N, dtype=int)
        self.hf_isospin      = np.zeros(N, dtype=int)
        self.hf_two_m        = np.zeros(N, dtype=int)

        col = 0  # column index in global eigenvectors
        for key, idx in self.basis.one_body_blocks.items():
            idx = np.array(idx, dtype=int)

            # Extract block
            h_block = self.h[np.ix_(idx, idx)]

            # Store quantum numbers for this block (before diagonalization)
            for i, state_idx in enumerate(idx):
                self.hf_parity[col+i] = int(self.basis.parity[state_idx])
                self.hf_isospin[col+i] = int(self.basis.isospin[state_idx])
                self.hf_two_m[col+i] = int(self.basis.two_m[state_idx])

            # Diagonalize
            e, c = np.linalg.eigh(h_block)

            n = len(idx)

            self.hf_sp_energies[col:col+n] = e
            self.hf_basis[np.ix_(idx, range(col, col+n))] = c

            col += n

        self.hf_basis_constructed = True

        return self.hf_sp_energies, self.hf_basis
    
    def calculate_energy(self, H) -> tuple:
        """
        Calculate the total energy of the Slater determinant.
        
        The total energy is computed as:
            E_total = E_sp + E_int
        where:
            E_sp  = trace(SPE * rho)  (single-particle energy)
            E_int = 0.5 * trace(Gamma * rho)  (interaction energy)
        
        Parameters:
            H (Hamiltonian): Hamiltonian object containing SPEs and TBMEs
            
        Returns:
            tuple: (total_energy, single_particle_energy, interaction_energy)
                where all energies are floats in MeV
        
        Example:
            >>> total_E, sp_E, int_E = sd.calculate_energy(hamiltonian)
            >>> print(f"Total energy: {total_E:.4f} MeV")
            >>> print(f"  Single-particle: {sp_E:.4f} MeV ({sp_E/total_E*100:.1f}%)")
            >>> print(f"  Interaction: {int_E:.4f} MeV ({int_E/total_E*100:.1f}%)")
        """
        # Calculate single-particle energy: trace(SPE * rho)
        E_sp = np.sum(H.SPE * np.diag(self.rho))
        
        # Calculate interaction energy: 0.5 * trace(Gamma * rho)
        E_int = 0.5 * np.trace(self.Gamma @ self.rho)

        # Total energy
        E_total = E_sp + E_int

        return (E_total, E_sp, E_int)
    

    
    def particle_number(self) -> float:
        """
        Calculate the total particle number (trace of density matrix).
        
        Returns:
            float: Total particle number
        """
        return np.trace(self.rho)
    
    def neutron_number(self) -> float:
        """
        Calculate the neutron number.
        
        Returns:
            float: Neutron number
        """
        neutron_mask = self.hf_isospin == -1
        return np.trace(self.rho[np.ix_(neutron_mask, neutron_mask)])
    
    def proton_number(self) -> float:
        """
        Calculate the proton number.
        
        Returns:
            float: Proton number
        """
        proton_mask = self.hf_isospin == 1
        return np.trace(self.rho[np.ix_(proton_mask, proton_mask)])
    
    def build_density(self) -> np.ndarray:
        """
        Calculate the one-body density matrix of a Slater determinant.

        This function computes the density matrix for a Slater determinant given
        the Hartree-Fock eigenvectors and occupation numbers. The density matrix
        is constructed as ρ = C * occ * C^T, where C contains the single-particle
        wavefunctions and occ specifies which states are occupied.

        Returns
        -------
        numpy.ndarray
            The one-body density matrix of shape (N_sp, N_sp), representing the
            density of a Slater determinant constructed from the occupied states.

        Example
        -------
        >>> rho = sd.build_density()
        >>> print(f"Density matrix shape: {rho.shape}")
        >>> print(f"Particle number (trace): {np.trace(rho):.2f}")
        >>> # Check idempotency for pure state
        >>> idempotency_error = np.linalg.norm(rho @ rho - rho)
        >>> print(f"Idempotency error: {idempotency_error:.2e}")
        """
        return (self.hf_basis * self.occupations) @ self.hf_basis.T

    def build_single_particle_hamiltonian(self, H):
        """
        Build the single-particle Hamiltonian matrix h.

        The single-particle Hamiltonian h is constructed as:
                      h = h0 + Γ
        where h0 is the one-body part of the Hamiltonian and Γ is the
        mean-field contribution from the two-body interaction.

        Note: this assumes that the mean-field Gamma has been precalculated!

        Parameters
        ----------
        H : Hamiltonian
            Hamiltonian object containing SPEs, TBMEs, and basis information
            rho : numpy.ndarray
            Density matrix of shape (N_sp, N_sp)

        Returns
        -------
        None
            Updates the self.h attribute with the single-particle Hamiltonian matrix

        Example
        -------
        >>> sd.build_gamma(hamiltonian)  # First build Gamma
        >>> sd.build_single_particle_hamiltonian(hamiltonian)
        >>> print(f"Hamiltonian matrix shape: {sd.h.shape}")
        >>> print(f"Hamiltonian diagonal (first 5): {np.diag(sd.h)[:5]}")
        """
        N = H.basis.n.size
        # Initialize single-particle Hamiltonian matrix
        self.h = np.zeros((N, N))

        # Add one-body part (single-particle energies) - diagonal matrix
        self.h += np.diag(H.SPE)

        # Add mean-field contribution from two-body interaction (Gamma matrix)
        self.h += self.Gamma

    def build_gamma(self, H):
        """
        Build the Gamma matrix for the Hartree-Fock calculation.

        This optimized version iterates only over fundamental TBMEs and applies
        antisymmetry relations explicitly, reducing numerical complexity significantly.
        The Gamma matrix represents the mean-field potential from two-body interactions:
            Γ_ij = sum_{kl} \bar{v}_{ikjl} * ρ_{kl}
        where \bar{v}_{ikjl} are antisymmetrized two-body matrix elements.

        Parameters
        ----------
        H : Hamiltonian
            Hamiltonian object containing TBMEs and basis information

        Result:
        -------
        None
            Updates the self.Gamma attribute with the mean-field matrix

        Example
        -------
        >>> sd.build_gamma(hamiltonian)
        >>> print(f"Gamma matrix shape: {sd.Gamma.shape}")
        >>> print(f"Gamma matrix norm: {np.linalg.norm(sd.Gamma):.4f}")
        """

        # Pre-convert TBME data to Numba-compatible arrays
        max_items = len(H.TBME._data)
        tbme_keys = np.zeros((max_items, 4), dtype=np.int32)
        tbme_vals = np.zeros(max_items, dtype=np.float64)

        # Fill arrays from dictionary
        for i, ((a, b, c, d), val) in enumerate(H.TBME._data.items()):
            tbme_keys[i, 0] = a
            tbme_keys[i, 1] = b
            tbme_keys[i, 2] = c
            tbme_keys[i, 3] = d
            tbme_vals[i] = val

        # Call the Numba-compiled function
        self.Gamma = _build_gamma_numba_core(tbme_keys, tbme_vals, H.basis.size, self.rho)

    def set_occupations(self):
        """
        Set occupation numbers for neutron and proton states based on the single-particle
         energies in the Hartree-Fock basis.

        This method separates neutron and proton states using their isospin values,
        sorts them by HF basis energy, and occupies the lowest energy states up to the
        specified valence numbers for neutrons and protons.

        Example
        -------
        >>> sd.set_occupations()
        >>> print(f"Total occupied states: {np.sum(sd.occupations)}")
        >>> print(f"Neutron occupations: {np.sum(sd.occupations[sd.hf_isospin == -1])}")
        >>> print(f"Proton occupations: {np.sum(sd.occupations[sd.hf_isospin == 1])}")
        
        Raises
        ------
        ValueError
            If insufficient neutron or proton states are available for the specified
            valence numbers (N_valence or Z_valence)
        """

        # Use HF energies for occupation determination
        energies = self.hf_sp_energies

        # Separate neutron and proton states
        neutron_mask = self.hf_isospin == -1  # neutrons isospin = -1
        proton_mask  = self.hf_isospin == 1   # protons  isospin = +1

        # Obtain separate proton and neutron energies
        neutron_energies = energies[neutron_mask]
        proton_energies  = energies[proton_mask]

        neutron_states = np.where(neutron_mask)[0]
        proton_states  = np.where(proton_mask)[0]

        # Get indices sorted by HF energy
        sorted_neutron_indices = neutron_states[np.argsort(neutron_energies)]
        sorted_proton_indices  = proton_states[np.argsort(proton_energies)]

        # Initialize occupation array
        self.occupations = np.zeros(self.basis.size)

        if self.N_valence <= len(sorted_neutron_indices):
            # Occupy the lowest N_valence neutron levels in the HF basis
            self.occupations[sorted_neutron_indices[:self.N_valence]] = 1.0
        else:
            print(f"Warning: Not enough neutron states for {self.N_valence} valence neutrons")
            raise ValueError

        if self.Z_valence <= len(sorted_proton_indices):
            # Occupy the lowest Z_valence proton levels in HF basis
            self.occupations[sorted_proton_indices[:self.Z_valence]] = 1.0
        else:
            print(f"Warning: Not enough proton states for {self.Z_valence} valence protons")
            raise ValueError

    def get_hf_quantum_numbers(self, state_index: int) -> dict:
        """
        Get quantum numbers for a specific HF basis state.

        Parameters
        ----------
        state_index : int
            Index of the HF basis state

        Returns
        -------
        dict
            Dictionary containing quantum numbers: n, l, two_j, two_m, isospin, parity

        Raises
        ------
        IndexError
            If HF quantum numbers are not available or index is invalid
        """
        if self.hf_quantum_numbers is None or state_index >= len(self.hf_quantum_numbers):
            raise IndexError("HF quantum numbers not available or invalid index")
        return self.hf_quantum_numbers[state_index]

    def get_all_hf_quantum_numbers(self) -> list:
        """
        Get quantum numbers for all HF basis states.

        Returns
        -------
        list
            List of dictionaries, each containing quantum numbers for a HF state
        """
        if self.hf_quantum_numbers is None:
            return []
        return self.hf_quantum_numbers

    def calculate_quadrupole_expectation(self) -> float:
        """
        Calculate the expectation value of the quadrupole operator Q20.

        Computes <Q20> = Tr(rho * Q20) where Q20 is the quadrupole operator
        matrix in the computational basis.

        Returns:
            float: Expectation value <Q20> in units appropriate for the basis

        Notes:
            - Requires that the basis has quadrupole matrix elements computed
              (via build_quadrupole_matrix_elements)
            - Returns 0.0 if quadrupole matrix elements are not available
            - The quadrupole operator is defined as Q20 = r^2 * Y20

        Example:
            >>> q20_exp = sd.calculate_quadrupole_expectation()
            >>> print(f"<Q20> = {q20_exp:.4f}")
            >>> if abs(q20_exp) > 1e-6:
            ...     print(f"Non-zero quadrupole moment detected")
        """

        # Calculate expectation value: <Q20> = Tr(rho * Q20)
        q20_expectation = np.trace(self.rho @ self.basis.Q20)

        return q20_expectation

    def print_hf_basis(self):
        """
        Print single-particle information in the Hartree-Fock basis.

        Prints separate tables for protons and neutrons with quantum numbers and occupations.
        Each table shows state index, energy, parity, magnetic quantum number (2m), and occupation.

        Notes
        -----
        - States are sorted by energy within each isospin group
        - Uses HF single-particle energies (self.hf_sp_energies)
        - Displays quantum numbers from the HF basis
        - Shows occupation numbers for each state

        Example
        -------
        >>> sd.print_hf_basis()
        Proton states
        ================================================================================
        Index   Energy (MeV)  Parity     2m      Occupation
        --------------------------------------------------------------------------------
        0      -12.3456       +1        +1       1.00
        1      -10.2345       -1        -1       1.00
        ...
        
        Neutron states
        ================================================================================
        Index   Energy (MeV)  Parity     2m      Occupation
        --------------------------------------------------------------------------------
        0      -15.6789       +1        +3       1.00
        1      -13.5790       +1        +1       1.00
        ...
        """

        # Use HF energies if available, otherwise use single-particle energies from basis
        eps = self.hf_sp_energies

        # Separate into proton and neutron states
        proton_mask  = self.hf_isospin ==  1
        neutron_mask = self.hf_isospin == -1
        parities     = self.hf_parity
        two_m_values = self.hf_two_m
        occupations  = self.occupations

        #------------------------------------------------------
        # First print all proton information
        #------------------------------------------------------
        # Sort all proton states by energy
        sorted_indices = np.argsort(eps[proton_mask])

        print(f"Proton states")
        print("="*80)
        print(f"{'Index':<6} {'Energy (MeV)':<12} {'Parity':<8} {'2m':<6} {'Occupation':<10}")
        print("-"*80)
        for i, idx in enumerate(sorted_indices):
            energy = eps[proton_mask][idx]
            two_m  = two_m_values[proton_mask][idx]
            occ    = occupations[proton_mask][idx]
            p      = parities[proton_mask][idx]

            print(f"{i:<6} {energy:<+12.4f} {p:<+8d} {two_m:<+6d} {occ:<10.2f}")

        #---------------------------------------------
        # ... and then print all neutron information
        #---------------------------------------------
        # Sort all neutron states by energy
        sorted_indices = np.argsort(eps[neutron_mask])

        print()
        print(f"Neutron states")
        print("="*80)
        print(f"{'Index':<6} {'Energy (MeV)':<12} {'Parity':<8} {'2m':<6} {'Occupation':<10}")
        print("-"*80)
        for i, idx in enumerate(sorted_indices):
            energy = eps[neutron_mask][idx]
            two_m  = two_m_values[neutron_mask][idx]
            occ    = occupations[neutron_mask][idx]
            p      = parities[neutron_mask][idx]

            print(f"{i:<6} {energy:<+12.4f} {p:<+8d} {two_m:<+6d} {occ:<10.2f}")
        print("="*80)

    def __str__(self) -> str:
        """
        String representation of the Slater determinant.
        
        Returns:
            str: Formatted string with basis size and particle counts
        
        Example:
            >>> sd = SlaterDeterminant(basis, energies, 4, 4)
            >>> print(sd)
            SlaterDeterminant:
              Basis size: 20
              Particle number: 8.00
              Neutrons: 4.00
              Protons: 4.00
        """
        info = []
        info.append(f"SlaterDeterminant:")
        info.append(f"  Basis size: {self.basis.size}")
        info.append(f"  Particle number: {self.particle_number():.2f}")
        info.append(f"  Neutrons: {self.neutron_number():.2f}")
        info.append(f"  Protons: {self.proton_number():.2f}")

        return "\n".join(info)

@njit
def _build_gamma_numba_core(tbme_keys, tbme_vals, basis_size, rho):
    """
    Numba-optimized serial implementation of build_gamma.
    
    Computes the Gamma matrix from fundamental TBMEs and density matrix,
    applying antisymmetry relations explicitly.
    
    Parameters:
        tbme_keys (np.ndarray): Array of shape (N,4) containing TBME indices (a,b,c,d)
        tbme_vals (np.ndarray): Array of shape (N,) containing TBME values
        basis_size (int): Size of the single-particle basis
        rho (np.ndarray): Density matrix of shape (basis_size, basis_size)
    
    Returns:
        np.ndarray: Gamma matrix of shape (basis_size, basis_size)
    
    Notes:
        - Applies antisymmetry relations: v_{ikjl} = -v_{jkil} = -v_{iklj} = v_{jkl i}
        - Iterates only over fundamental TBMEs for efficiency
        - Uses serial computation (parallelization could be added)
    
    Example:
        >>> # This function is called internally by build_gamma()
        >>> # Not typically called directly by users
        >>> Gamma = _build_gamma_numba_core(tbme_keys, tbme_vals, basis_size, rho)
    """
    Gamma = np.zeros((basis_size, basis_size))

    # Iterate over all TBME items serially
    for i in range(len(tbme_vals)):
        a = tbme_keys[i, 0]
        b = tbme_keys[i, 1]
        c = tbme_keys[i, 2]
        d = tbme_keys[i, 3]
        fundamental_val = tbme_vals[i]

        # Variant 1: (a,b,c,d) with phase (+1, +1)
        tbme_val = fundamental_val * 1.0 * 1.0
        Gamma[a, c] += tbme_val * rho[d, b]

        # Variant 2: (b,a,c,d) with phase (-1, +1) if a≠b
        if a != b:
            tbme_val = fundamental_val * -1.0 * 1.0
            Gamma[b, c] += tbme_val * rho[d, a]

        # Variant 3: (a,b,d,c) with phase (+1, -1) if c≠d
        if c != d:
            tbme_val = fundamental_val * 1.0 * -1.0
            Gamma[a, d] += tbme_val * rho[c, b]

        # Variant 4: (b,a,d,c) with phase (-1, -1) if a≠b and c≠d
        if a != b and c != d:
            tbme_val = fundamental_val * -1.0 * -1.0
            Gamma[b, d] += tbme_val * rho[c, a]

    return Gamma
