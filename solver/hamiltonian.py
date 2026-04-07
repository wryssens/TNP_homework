#!/usr/bin/env python3
from modelspace import SingleParticleBasis, J_coupling
import numpy as np
from angmomcoupling import clebsch
from tbme import TBME

class Hamiltonian:
    """
    Represents a nuclear shell model Hamiltonian consisting of single-particle energies (SPE)
    and two-body matrix elements (TBME).
    
    The Hamiltonian is defined as:
        H = H_1body + H_2body
        H_1body = sum_{\mu} \epsilon_\mu c_\mu† c_\mu
        H_2body = (1/4) sum_{\mu \nu \kappa \lamba} \bar{v}_{\mu \nu \kappa \lambda} c_\mu† c_\nu† c_\lambda c_\kappa
    
    where
       - \epsilon_\mu are single-particle energies
       - \bar{v}_{\mu \nu \kappa \lambda} are antisymmetrized two-body matrix elements (TBMEs).
    
    Attributes:
        basis (SingleParticleBasis): The single-particle basis defining the model space
        SPE (numpy.ndarray)        : Single-particle energies for each basis state
        TBME (TBME)                : Container for antisymmetrized two-body matrix elements
    
    Methods:
        __init__(basis, filename=None): Initialize Hamiltonian with given basis and read SPEs and TBMEs from file
        read_values(filename)         : Read SPE and TBME values from file
        __str__()                     : Return string representation of Hamiltonian
    """

    def __init__(self,  basis : SingleParticleBasis, filename : str = None):
        Nspwf = basis.size

        self.basis   = basis           # Single-particle basis
        self.SPE     = np.zeros(Nspwf) # One-body part of the Hamiltonian
        self.TBME    = TBME()          # Use TBME class for proper antisymmetrization

        # Automatically read values if filename is provided
        if filename is not None:
            self.read_values(filename)

    def __str__(self):
        """
        Return a string representation of the Hamiltonian.
        
        Returns:
            str: String containing basis information, SPE count, and TBME count
        
        Example:
            >>> hamiltonian = Hamiltonian(basis, "hamiltonian_file.txt")
            >>> print(hamiltonian)
            Hamiltonian(basis=SingleParticleBasis(size=10), spe_count=10, tbme_count=45)
        """
        return (f"Hamiltonian(basis={self.basis}, "
                f"spe_count={len(self.SPE)}, "
                f"tbme_count={len(self.TBME)})")

    def read_values(self, filename : str):
        """
        Read the single-particle energies (SPEs) and two-body matrix elements (TBMEs)
        from a file.
        
        The file format is:
        - First two lines: SPE values (first line starts with TBME count, then SPEs)
        - Remaining lines: TBME data in format "a b c d J v" where:
          * a,b,c,d: orbit indices (1-based)
          * J: coupled angular momentum
          * v: TBME value
        
        Input:
         - filename : str
                      Name of the file containing the Hamiltonian data
        
        Note:
         - SPE values are given per orbit but expanded to all m-states
         - TBMEs are read in reduced form and then expanded using Clebsch-Gordan coefficients
         - Proper antisymmetrization is applied based on isospin components
         - Hermitian symmetry is enforced but not really leveraged
        """
        with open(filename, 'r') as f:
            # Read first two lines which contain the SPEs
            line1 = f.readline()
            line2 = f.readline()

            # Combine and parse all SPE values, skipping the first number (TBME count)
            all_spe_values = []
            for line in [line1, line2]:
                parts = line.split()
                # Skip first number in first line (it's the TBME count, not an SPE)
                if line == line1 and len(parts) > 0:
                    parts = parts[1:]  # Skip first element
                all_spe_values.extend([float(x) for x in parts])

            # We now have the SPEs per ORBIT, but we need them for each m-state
            for k in range(self.basis.size):
                idx         = self.basis.orbit_map[k]
                self.SPE[k] = all_spe_values[idx - 1]

            # ... and now we start reading the TBME's in reduced form
            v_orbit = {}
            for line in f:
                parts = line.split()
                # All lines have the following format
                #  orbit_a, orbit_b, orbit_c, orbit_d, angular momentum, TBME value
                a, b, c, d, J, v  = parts
                # ensure correct types
                a, b, c, d = map(int, (a, b, c, d))
                J = int(J)
                v = float(v)

                v_orbit[(a,b,c,d,J)] = v
                exp_ab = (self.basis.orbit_twoj[a-1] + self.basis.orbit_twoj[b-1])//2 + J + 1
                exp_cd = (self.basis.orbit_twoj[c-1] + self.basis.orbit_twoj[d-1])//2 + J + 1
                pab    = (-1)**(exp_ab)
                pcd    = (-1)**(exp_cd)

                # same isospin components - fully antisymmetrized
                if( self.basis.orbit_isospin[a-1] == self.basis.orbit_isospin[b-1] ):
                 v_orbit[(b,a,c,d,J)] = v * pab
                 v_orbit[(a,b,d,c,J)] = v * pcd
                 v_orbit[(b,a,d,c,J)] = v * pab * pcd
                 v_orbit[(c,d,a,b,J)] = v
                 v_orbit[(c,d,b,a,J)] = v * pab
                 v_orbit[(d,c,a,b,J)] = v * pcd
                 v_orbit[(d,c,b,a,J)] = v * pab * pcd
                else :
                 # opposite isospin TBME - not fully antisymmetrized
                 v_orbit[(c,d,a,b,J)] = v
                 v_orbit[(b,a,d,c,J)] = v * pab * pcd
                 v_orbit[(d,c,b,a,J)] = v * pab * pcd

            # ... and now we decouple into matrix elements for the m-states
            #   \bar{v}_{1234} = \sum_{JM} [N_{ab}(J) N_{cd}(J)]^{-1}
            #                     < j1 m1 j2 m2 | J M > < j3 m3 j4 m4 | J M >
            #                     < ab ; J | V | cd ; J >
            #
            #  where
            #   - a,b,c,d are the orbits of the individual
            #                     single-particle states 1,2,3 and 4.
            #   - ja ma jb mb | J M > are Clebsch-Gordan coefficients
            #   - N_{ab}(J) = (1 + delta_{ab} (-1)^J)/(1+\delta_{ab}).
            #

            # Note: we don't do naive loops over all posible single-particle
            #       states - that would take too long!

            # Set up the memoization of the calculation of the Clebsch-Gordan coefficients
            #clebsch_memo = Memoize(clebsch)

            for block, indices in self.basis.TB_blocks.items():
                (M,P,T) = block
                for i,ab in enumerate(indices):
                     (a,b) = self.basis.TB_pair_index[ab]

                     aa = self.basis.orbit_map[a]
                     bb = self.basis.orbit_map[b]

                     two_ja = self.basis.two_j[a]
                     two_jb = self.basis.two_j[b]
                     two_ma = self.basis.two_m[a]
                     two_mb = self.basis.two_m[b]

                     assert two_ma + two_mb == 2*M
                     assert abs(two_ma) <= two_ja

                     # Now the inner loop is NOT over all possible pairs; we leverage that the Hamiltonian is hermitian
                     # and the TBMEs are real, such that V(ab,cd) = V(cd,ab)
                     for cd in indices[i:]:
                         (c,d) = self.basis.TB_pair_index[cd]

                         cc = self.basis.orbit_map[c]
                         dd = self.basis.orbit_map[d]

                         two_jc = self.basis.two_j[c]
                         two_jd = self.basis.two_j[d]
                         two_mc = self.basis.two_m[c]
                         two_md = self.basis.two_m[d]

                         minJ, maxJ = J_coupling(two_ja,two_jb,two_jc,two_jd)

                         # Check if this is a canonical ordering considering both antisymmetry and Hermitian symmetry
                         # Canonical: (a <= b, c <= d) AND (ab_pair <= cd_pair lexicographically)
                         is_canonical_ab = (a <= b)
                         is_canonical_cd = (c <= d)
                         
                         # For Hermitian symmetry, we only calculate when (ab) <= (cd) in lex order
                         ab_pair = (min(a, b), max(a, b))
                         cd_pair = (min(c, d), max(c, d))
                         is_hermitian_canonical = (ab_pair <= cd_pair)
                         
                         is_canonical = is_canonical_ab and is_canonical_cd and is_hermitian_canonical
                         
                         if is_canonical:
                             # Calculate the TBME value for canonical ordering
                             tbme_value = 0.0
                             for J in range(minJ, maxJ+1):
                                 if((aa,bb,cc,dd,J) not in v_orbit.keys()):
                                     continue

                                 cg1 = clebsch(two_ja, two_ma, two_jb, two_mb, J, M)
                                 cg2 = clebsch(two_jc, two_mc, two_jd, two_md, J, M)

                                 tbme_value = tbme_value + cg1 * cg2 * v_orbit[(aa,bb,cc,dd,J)]

                             # Set the fundamental TBME value (let TBME class handle all symmetries)
                             self.TBME[(a,b,c,d)] = tbme_value
                             # ... and manually enforce the Hamiltonian to be hermitian
                             self.TBME[(c,d,a,b)] = tbme_value
                         else:
                             # For non-canonical orderings, the TBME will be automatically
                             # generated by the symmetry properties of the TBME class
                             # when accessed later. No need to calculate it now.
                             pass

            # Apply normalization factors for diagonal elements
            # This replaces the manual normalization that was done in the loop
            self.TBME.apply_normalization(self.basis)
