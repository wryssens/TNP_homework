# An introduction to nuclear mean-field calculations

The code in this repository accompanies the numerical homework for the course "Theoretical Nuclear Physics", taught by Dr. W. Ryssens in 2025-2026. Students will build a simple Hartree-Fock solver for use in Configuration Interaction Shell Model (CISM) spaces based on ingredients provided in this repository. They will use the resulting code to answer several questions in the full statement of the assignment. 

## Contents 

- ```README.md ```       : this file with information on the code ingredients provided
- ```environment.yml ``` : environment file to help you satisfy the code's requirements.
- ```assignment.pdf ```  : a copy of the assignment featuring more information on the physics
- ```data/```            : datafiles defining single particle bases and corresponding Hamiltonians.
- ```solver/```          : Python modules with ingredients for building a Hartree-Fock solver.

## Code requirements & environment.yml
  This set of functions depends on the following - rather standard - scientific Python packages. 

  - Numba
  - Scipy
  - Numpy
  - Sympy

  The package contains an `environment.yml` file that can be used to generate a `homework` environment with the right packages with the environment manager conda (https://docs.conda.io/projects/conda/en/latest/index.html).
  If you work in the terminal, you can type 
  
  ```
  conda create --file environment.yml
  ```
  
  into your preferred terminal. For somewhat older versions of conda, this won't work and you need to type instead 

  ```
  conda env create --file environment.yml
  ```
  
  Afterwards, you can type 
  ```
  conda activate homework
  ```
  and you should be good to go. 
  
  If you do not use conda, you should guarantee the availability of these modules yourself. 


## Single particle bases and Hamiltonians: the ```data/``` directory

This directory contains the full definition of four different CISM single particle bases and associated Hamiltonians. More specifically, these are 

- `sd-shell/`: six valence orbitals outside of an $^{16}$O core
- `fp-shell/`: ten valence orbitals outside of an $^{40}$Ca core 
- `rare-earth/`: fourteen valence orbitals outside of an $^{120}$Sn core 
- `actinides/`: fifteen valence orbitals outside of an $^{208}$Pb core

Each subfolder contains 

- `pn.sps`: the specification of the valence orbitals
- `r2.red`: reduced matrix elements of $r^2$y
- `*.int*`: the specification of the Hamiltonian, i.e. its single-particle energies and two-body matrix elements.

## Code ingredients: the ```solver/``` package

The `solver/` directory contains several Python modules; in order of importance to this assignment: 

- `slater_determinant.py`: the implementation of the `SlaterDeterminant` class
- `singleparticlebasis.py`: the implementation of the `SingleParticleBasis` class.
- `hamiltonian.py`: the implementation of the `Hamiltonian` class
- `tbme.py` : the `TBME` class to deal efficiently with two-body matrix elements.
- `angmomcoupling.py`: some utility functions for angular momentum coupling.

The latter two modules -- `tbme.py` and `angmomcoupling.py` -- are mostly technical, understanding them is not crucial to this assignment which is why they are not documented in detail here. Nevertheless the reader might be interested in taking a look at these modules.

This is not true for the other three modules: these contain the ingredients you will need for the assignment. Nevertheless, not ALL of their attributes/methods/details are important to you in practice. Below, I provide documentation for the most important aspects; you can safely assume that the things this files does not cover are of secondary importance to you.

All of the code in `solver/` is accessible as a Python package; i.e.~you can access all methods with import statements such as

```python 
from solver import Hamiltonian, SingleParticleBasis, SlaterDeterminant
```


### 1. `SingleParticleBasis` Class from the `singleparticlebasis.py` file

The `SingleParticleBasis` class represents the computational basis for nuclear shell model calculations and stores all relevant information about all the single particle states. In particular, it has the following key attributes 

**Key Attributes**

- single-particle quantum numbers
    - `n` : the principal quantum number $n$
    - `l` : the orbital angular momentum $\ell$
    - `two_j` : twice the total angular momentum $J$
    - `two_m` : twice the magnetic quantum number $m$)
    - `isospin`: the projection of isospin, $t_z = +1/-1$ for proton/neutron)
    - `parity` : the parity quantum number $P = \pm 1$
- `Q20`: array containing the single-particle matrix elements of $Q_{20}$

Other attributes are more technical and less crucial to finishing this assignment: blocks of single-particle states grouped by quantum numbers and blocks of two-body states grouped by quantum numbers.

The class provides you with several useful methods 

**Key Methods:**

- `size`: return the total number of single-particle states in the single particle basis.
- `build_single_particle_basis(filename, r2_filename)`: returns a `SingleParticleBasis` object based on a `.sps` file and an associated `r2.red` file.
- `__str__` : returns a string representation of the class such that you can call `print(basis)`

**Example Usage:**
```python
basis = build_single_particle_basis("data/sd-shell/pn.sps", "data/sd-shell/r2.red")
print(basis)
```

### 2. `Hamiltonian` Class from the `hamiltonian.py` file

The `Hamiltonian` class stores all information for the evaluation of a generic two-body Hamiltonian $H$ as treated in class. It features the following attributes 

**Key attributes**

- `basis` : the SingleParticleBasis for which the Hamiltonian is defined 
- `SPE`   : the single particle energies, i.e. the one-body part of $H$
- `TBME`  : the two-body matrix elements, i.e. the two-body part of $H$

**Key Methods:**

- `__init__(self, basis, filename)`: initializes a Hamiltonian object for a given SingleParticleBasis and a given `.int` file. 
- `__str__` : returns a string representation of the class such that you can call `print(H)`

**Example Usage:**
```python
basis = build_single_particle_basis("data/sd-shell/pn.sps", "data/sd-shell/r2.red")
H = Hamiltonian(basis, 'data/sd-shell/sd-shell.int')
print(H)
```

### 3. `SlaterDeterminant` Class from the `slater_determinant.py` file

The `SlaterDeterminant` class represents a Slater determinant many-body state in nuclear Hartree-Fock calculations. The methods associated with this object are the ones that are *most important* to this assignment.

 **Key Concepts:**
- A Slater determinant is defined by a single-particle basis and occupation numbers
- The Hartree-Fock basis diagonalizes the single-particle Hamiltonian
- The density matrix ρ represents the occupation of single-particle states
- The mean-field Γ represents the average potential from two-body interactions

**Key Attributes:**
- `basis`: The computational single-particle basis, an instance of SingleParticleBasis
- `rho`: The one-body density matrix $\rho$, expressed in the computational basis
- `Gamma`: The mean-field potential matrix $\Gamma$ expressed in the computational basis
- `h`: The single-particle hamiltonian matrix $h$ 
- `hf_sp_energies`: The Hartree-Fock single-particle energies
- `hf_basis`: Transformation matrix from the computational to the Hartree-Fock basis
- `occupations`: Single-particle occupation factors = the diagonal elements of $\rho$ in the Hartree-Fock basis

**Key Methods:**

1. **Initialization and Setup:**
   - `__init__(basis, hf_energies, N_valence, Z_valence, shape)`: Initialize a Slater Determinant with
       - `N/Z_valence` neutrons/protons 
       - a trivial HF transformation, i.e. the HF basis is equal to the computational basis 
       - the specified set of single-particle energies `hf_energies` 
       - ... but **offset** by a small energy shift based on `shape`. 
         If set to `prolate`, the single-particle energies with high |m| will be (slightly) disfavoured.
         If set to `oblate`, the single-particle energies with low |m| will be (slightly) disfavoured.
       - occupation factors set in such a way that the *lowest-lying** single-particle states are occupied  
   - `set_occupations()`: Set occupation numbers to occupy the *lowest-lying* states as decided by the attribute `hf_sp_energies`

2. **Hartree-Fock Calculations:**
   - `diagonalize()`: Diagonalize the single-particle Hamiltonian with symmetry constraints
   - `build_gamma(H)`: Calculate $\Gamma$ from TBMEs and density matrix
   - `build_single_particle_hamiltonian(H)`: Build $h$, based on precalculated $\Gamma$
   - `build_density()`: Build $\rho$ from the HF basis and occupations

3. **Energy and Particle Number Calculations:**
   - `calculate_energy(H)`: Calculate the total energy - requires passing in a Hamiltonian object
   - `particle_number()`: Calculate total particle number
   - `neutron_number()`: Calculate neutron number
   - `proton_number()`: Calculate proton number

4. **Visualization and Analysis:**
   - `print_hf_basis()`: Print *extensive* HF basis information with quantum numbers
   - `__str__()`: String representation with basis size and particle counts

**Example Usage:**
```python
basis = build_single_particle_basis("data/sd-shell/pn.sps", "data/sd-shell/r2.red")
H     = Hamiltonian(basis, "data/sd-shell/sd-shell.int")
sd    = SlaterDeterminant(basis, H.SPE, 4, 4)
```
