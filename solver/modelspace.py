import numpy as np
from .angmomcoupling import wigner

class SingleParticleBasis:
    """
    Container for single-particle quantum states in nuclear shell model calculations.
    
    Stores quantum numbers for each single-particle state and provides methods
    for building symmetry-restricted blocks and pairs used in Hamiltonian construction.
    
    Each state is characterized by quantum numbers:
        - n: principal quantum number
        - l: orbital angular momentum
        - two_j: twice the total angular momentum (2j)
        - two_m: twice the magnetic quantum number (2m)
        - isospin: isospin projection (±1 for proton/neutron)
        - parity: parity (±1)
        - orbit_map: mapping to original orbital index
    
    Attributes:
        n, l, two_j, two_m, isospin, parity (numpy.ndarray): Quantum numbers for each state
        orbit_twoj, orbit_isospin, orbit_l (numpy.ndarray): Quantum numbers for each orbital
        orbit_map (numpy.ndarray): Maps each state to its parent orbital
        one_body_blocks (dict): States grouped by (m, parity, isospin) quantum numbers
        TB_pair_index (dict): Maps pair indices to (state_i, state_j) tuples
        TB_blocks (dict): Two-body pairs grouped by (M, parity, total_isospin) quantum numbers
        n_pairs (int): Total number of two-body pairs
        r2_red (dict): Reduced matrix elements of r^2 operator
        qred (numpy.ndarray): Reduced matrix elements of quadrupole operator Q2
        Q20 (numpy.ndarray): Single-particle matrix elements of Q20 operator
    
    The class provides methods for:
        - Adding multiple states efficiently (vectorized operations)
        - Building one-body blocks for symmetry-restricted calculations
        - Building two-body pairs for TBME calculations
        - Filtering and selecting subsets of states based on quantum numbers
        - Reading reduced matrix elements and building quadrupole operators
    
    Methods:
        __init__(): Initialize empty basis
        add_states(): Add multiple states to basis
        build_one_body_blocks(): Group states by conserved quantum numbers
        build_pairs(): Build all two-body pairs
        mask(): Filter states by quantum number criteria
        read_reduced_matrix_elements_r2(): Read reduced matrix elements of r^2
        build_quadrupole_matrix_elements(): Build quadrupole operator matrix elements
    """
    def __init__(self):
        # Single-particle quantum numbers
        self.n         = np.empty(0, dtype=np.int32)
        self.l         = np.empty(0, dtype=np.int32)
        self.two_j     = np.empty(0, dtype=np.int32)
        self.two_m     = np.empty(0, dtype=np.int32)
        self.isospin   = np.empty(0, dtype=np.int8)
        self.parity    = np.empty(0, dtype=np.int8)

        self.orbit_twoj    = np.empty(0, dtype=np.int8)
        self.orbit_l       = np.empty(0, dtype=np.int8)
        self.orbit_isospin = np.empty(0, dtype=np.int8)

        # Reduced matrix elements of r2
        self.r2_red        = {}
        # Reduced matrix elements of Q2 (will be numpy array after building)
        self.qred          = np.empty(0)
        # Single-particle matrix elements of Q20
        self.Q20           = np.empty(0)
        # Labelling the single-particle states w.r.t. to the original orbital
        self.orbit_map = np.empty(0, dtype=np.int8)

        # Constructing the symmetry-restriced one-body blocks
        self.one_body_blocks = {}
        # Indexing all possible two-body states
        self.TB_pair_index = {}
        self.n_pairs       = 0
        # Indexing the possible two-body states by quantum numbers ( M = m1+m2, P = p1*p2, isospin)
        self.TB_blocks     = {}
        
    @property
    def size(self) -> int:
        return self.n.size

    @property
    def size_orbit(self) -> int:
        return self.orbit_twoj.size

    def __str__(self) -> str:
        """
        Return a string representation of the single-particle basis.
        
        Returns:
            str: Formatted string containing total states, proton states, and neutron states
        
        Example:
            >>> basis = SingleParticleBasis()
            >>> # ... add states to basis ...
            >>> print(basis)
            SingleParticleBasis(total_states=10, proton_states=5, neutron_states=5)
        """
        total_states = self.size
        proton_states = np.sum(self.isospin == 1)    # isospin = +1 for protons
        neutron_states = np.sum(self.isospin == -1)  # isospin = -1 for neutrons
        
        return (
            f"SingleParticleBasis("
            f"total_states={total_states}, "
            f"proton_states={proton_states}, "
            f"neutron_states={neutron_states}"
            f")"
        )

    def add_states(
        self,
        n: np.ndarray,
        l: np.ndarray,
        two_j: np.ndarray,
        two_m: np.ndarray,
        isospin: np.ndarray,
        parity: np.ndarray,
        orbit_map : np.ndarray
    ):
        """
        Add multiple single-particle states to the basis (vectorized operation).
        
        Parameters:
            n (np.ndarray): Principal quantum numbers
            l (np.ndarray): Orbital angular momentum quantum numbers
            two_j (np.ndarray): Twice the total angular momentum (2j)
            two_m (np.ndarray): Twice the magnetic quantum numbers (2m)
            isospin (np.ndarray): Isospin projections (±1)
            parity (np.ndarray): Parity values (±1)
            orbit_map (np.ndarray): Mapping to original orbital indices
        
        Raises:
            ValueError: If quantum numbers are inconsistent (|m| > j, invalid parity/isospin)
        
        Note:
            - All input arrays must have the same length and will be concatenated
              to the existing basis states.
            - This method performs vectorized consistency checks for efficiency.
            - The basis maintains separate tracking of orbital vs state-level quantum numbers.
        
        Example:
            >>> n = np.array([1, 1])
            >>> l = np.array([1, 1])
            >>> two_j = np.array([3, 3])  # p3/2 orbital
            >>> two_m = np.array([3, 1])   # m = 3/2, 1/2
            >>> isospin = np.array([1, 1]) # protons
            >>> parity = np.array([-1, -1]) # negative parity
            >>> orbit_map = np.array([1, 1]) # both from orbit 1
            >>> basis.add_states(n, l, two_j, two_m, isospin, parity, orbit_map)
        """

        n       = np.asarray(n      , dtype=np.int32)
        l       = np.asarray(l      , dtype=np.int32)
        two_j   = np.asarray(two_j  , dtype=np.int32)
        two_m   = np.asarray(two_m  , dtype=np.int32)
        isospin = np.asarray(isospin, dtype=np.int8)
        parity  = np.asarray(parity , dtype=np.int8)
        orbit_map = np.asarray(orbit_map, dtype=np.int8)

        # consistency checks (vectorized)
        if not np.all(np.abs(two_m) <= two_j):
            raise ValueError("|m| > j detected")
        if not np.all(np.isin(parity, (-1, +1))):
            raise ValueError("Parity must be ±1")
        if not np.all(np.isin(isospin, (-1, +1))):
            raise ValueError("Isospin must be ±1")

        self.n         = np.concatenate((self.n, n))
        self.l         = np.concatenate((self.l, l))
        self.two_j     = np.concatenate((self.two_j, two_j))
        self.two_m     = np.concatenate((self.two_m, two_m))
        self.isospin   = np.concatenate((self.isospin, isospin))
        self.parity    = np.concatenate((self.parity, parity))
        self.orbit_map = np.concatenate((self.orbit_map, orbit_map))

    def build_one_body_blocks(self):
        """
        Build one-body blocks grouped by conserved quantum numbers.
        
        Creates a dictionary mapping (two_m, parity, isospin) tuples to lists
        of state indices that share those quantum numbers. This grouping enables
        efficient symmetry-restricted calculations.
        
        The one-body blocks are stored in self.one_body_blocks as:
            {(two_m, P, T): [state_indices]}
        
        Where:
            - two_m: twice the magnetic quantum number
            - P: parity (±1)
            - T: isospin projection (±1)
        
        Note:
            - This method must be called after all states have been added.
            - The blocks are used for efficient Hamiltonian construction and
              other symmetry-restricted operations.
        
        Example:
            >>> basis.build_one_body_blocks()
            >>> # Access states with m=1/2, positive parity, proton isospin
            >>> block_key = (1, 1, 1)
            >>> if block_key in basis.one_body_blocks:
            ...     states = basis.one_body_blocks[block_key]
            ...     print(f"Found {len(states)} states in this block")
        """

        N = self.size
        self.one_body_blocks = {}

        for i in range(N):
            two_m = self.two_m[i]
            P     = self.parity[i]
            T     = self.isospin[i]

            key = (two_m, P, T)
            if key not in self.one_body_blocks.keys():
               self.one_body_blocks[key] = []
            self.one_body_blocks[key].append(i)

    def build_pairs(self):
        """
        Build all possible two-body pairs and group them by conserved quantum numbers.
        
        Creates two data structures:
        1. self.TB_pair_index: Maps pair indices to (state_i, state_j) tuples
        2. self.TB_blocks: Groups pairs by (M, parity, total_isospin) quantum numbers
        
        For each pair (i,j) where i ≠ j:
            - M = m_i + m_j (total magnetic quantum number)
            - P = p_i * p_j (total parity)
            - T = t_i + t_j (total isospin)
        
        The pairs are stored as:
            self.TB_pair_index: {pair_index: (state_i, state_j)}
            self.TB_blocks: {(M, P, T): [pair_indices]}
        
        Note:
            - This method excludes pairs where i == j (no self-interactions).
            - Must be called after all states have been added.
            - The pair structure is essential for efficient TBME calculations.
        
        Example:
            >>> basis.build_pairs()
            >>> print(f"Total pairs: {basis.n_pairs}")
            >>> # Access pairs with specific quantum numbers
            >>> key = (0, 1, 0)  # M=0, positive parity, total isospin=0
            >>> if key in basis.TB_blocks:
            ...     pair_indices = basis.TB_blocks[key]
            ...     print(f"Found {len(pair_indices)} pairs in this block")
        """

        idx = 0
        N = self.size

        self.TB_pair_index = {}
        self.TB_blocks     = {}
        for i in range(N):
            for j in range(N):
                if( i == j ):
                    continue
                self.TB_pair_index[idx] = (i,j)
                two_M = self.two_m[i] + self.two_m[j]
                P     = self.parity[i] * self.parity[j]
                T     = self.isospin[i] + self.isospin[j]

                key = (two_M//2, P, T)
                if key not in self.TB_blocks.keys():
                    self.TB_blocks[key] = []
                self.TB_blocks[key].append(idx)
                idx = idx + 1
        self.n_pairs = idx

    def read_reduced_matrix_elements_r2(self, filename):
        """
        Read reduced matrix elements of r^2 from file and store them.
        
        Parameters:
            filename (str): Path to file containing reduced matrix elements
        
        The file can contain one or more square matrices of reduced matrix elements
        for different orbital subsets (protons and neutrons). Each matrix should be in row-major order.
        
        Stores the reduced matrix elements in self.r2_red as a
        dictionary: {(orbit_i, orbit_j): value}
        
        File format:
            - Each matrix is stored in row-major order
            - Multiple matrices can be concatenated (e.g., protons then neutrons)
            - Values are space-separated, one row per line
        
        Note:
            - Orbit indices are 1-based and determined by the order of states in the file.
            - This method should be called before build_quadrupole_matrix_elements().
        
        Example:
            >>> basis.read_reduced_matrix_elements_r2("r2_matrix_elements.txt")
            >>> # Access specific reduced matrix element
            >>> if (1, 2) in basis.r2_red:
            ...     value = basis.r2_red[(1, 2)]
            ...     print(f"<1||r^2||2> = {value}")
        """
        # Read all data from file
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse the file line by line to handle variable matrix sizes
        self.r2_red = {}
        current_row = 0
        current_block_start = 1  # Starting orbit index for current block (1-based)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse the values in this line
            values = list(map(float, line.split()))
            n_cols = len(values)
            
            # Determine matrix size from first line of each block
            if current_row == 0:
                matrix_size = n_cols
            
            # Store the values with correct orbit indexing
            for col, value in enumerate(values):
                # Convert to 1-based indexing for orbits within the current block
                orbit_i = current_block_start + current_row
                orbit_j = current_block_start + col
                self.r2_red[(orbit_i, orbit_j)] = value
            
            current_row += 1
            
            # Reset if we've completed a matrix, and update block start
            if current_row >= matrix_size:
                current_block_start += matrix_size
                current_row = 0

    def build_quadrupole_matrix_elements(self, quadrupole_filename):
        """
        Build the full quadrupole matrix elements from the reduced matrix elements of r^2.
        
        Computes the matrix elements of the quadrupole operator Q20 using
        the Wigner-Eckart theorem:
            <j1 m1| Q20 |j2 m2> = <j2 2 m2 0| j1 m1> * <j1|| Q2 ||j2> / sqrt(2j1+1)
        
        where <j1|| Q2 ||j2> are the reduced matrix elements.
        
        The quadrupole matrix elements are stored in self.Q20
        as a 2D numpy array of shape (N_sp, N_sp) where N_sp is the number
        of single-particle states.
        
        Parameters:
            quadrupole_filename (str): Path to file containing reduced matrix elements of r^2.
        
        Note:
            - This method transforms reduced matrix elements of r^2 to those of Q2 using
              Wigner 3j symbols and angular momentum coupling.
            - The resulting Q20 matrix is used for quadrupole moment calculations.
            - Requires that reduced matrix elements have been read first.
        
        Example:
            >>> basis.build_quadrupole_matrix_elements("r2_elements.txt")
            >>> # Access specific matrix element
            >>> q20_value = basis.Q20[0, 1]  # <state0| Q20 |state1>
            >>> print(f"Q20 matrix element: {q20_value}")
        """
        if quadrupole_filename is not None:
            self.read_reduced_matrix_elements_r2(quadrupole_filename)
        else:
            raise ValueError("Reduced matrix elements must be read first using read_reduced_matrix_elements_r2() or provide quadrupole_filename")

        #-------------------------------------------------------------
        # Transform the reduced matrix elements of r^2 to those of Q_2
        #
        #  < a || Q_{\ell} || b > = P sqrt(2*\ell+1)sqrt(2*j_a+1) sqrt(2*j_b+1)
        #                              ( j_a \ell   j_c ) <a || R^2 || b >
        #                              ( 0.5   0   -0.5 )
        # with P = 1/2 * (-1)^(j_b + \ell - 0.5) (1 + (-1)^(l_a + l_b + \ell)).
        
        # Create numpy matrix for reduced matrix elements
        n_orbits = self.size_orbit
        self.qred = np.zeros((n_orbits, n_orbits))
        
        for (orbit_a, orbit_b), ME in self.r2_red.items():
            # Quantum number verification
            la = self.orbit_l[orbit_a-1]
            lb = self.orbit_l[orbit_b-1]

            if( (-1)**la != (-1)**lb):
                continue

            ta = self.orbit_isospin[orbit_a-1]
            tb = self.orbit_isospin[orbit_b-1]

            if(ta != tb):
                continue

            two_ja = self.orbit_twoj[orbit_a-1]
            two_jb = self.orbit_twoj[orbit_b-1]
            # Wigner 3j symbol
            fac = wigner(two_ja, 2, two_jb, +1, 0, -1)
            fac = fac * 2 * np.sqrt(float(two_ja+1)) * np.sqrt(float(two_jb+1))
            fac = fac * (-1)**(two_ja/2 + 0.5)
            self.qred[orbit_a-1, orbit_b-1] = fac * ME
        
        #-------------------------------------------------------------
        # ... and now perform angular momentum decoupling to move to
        # the single-particle matrix elements of Q20
        #
        #  < a | T^{ell}_m | b > = phase *  (  ja  ell  jb ) * < ja || Q_2 ||jb >
        #                                   ( -ma   q   mb )
        #  with
        #      phase = (-1)^(ja - ma).
        N_sp     = self.size
        self.Q20 = np.zeros((N_sp, N_sp))
        
        # Loop over all single-particle states
        for i in range(N_sp):
            for j in range(N_sp):
                # Get quantum numbers for states i and j
                two_ji = self.two_j[i]
                two_mi = self.two_m[i]
                two_jj = self.two_j[j]
                two_mj = self.two_m[j]
                
                orbit_i = self.orbit_map[i]
                orbit_j = self.orbit_map[j]
                
                # Check if these orbits have reduced matrix elements
                if self.qred[orbit_i-1, orbit_j-1] == 0:
                    continue

                wigner_coeff  = wigner(two_ji, 2, two_jj, -two_mi, 0, two_mj)
                self.Q20[i,j] = (-1)**((two_ji - two_mi)//2) *  wigner_coeff * self.qred[orbit_i-1, orbit_j-1]
                

def generate_m_substates(n, l, two_j, isospin, orbit, timereversal=False):
    """
    Generate all magnetic substates for a given orbital.
    
    Creates the complete set of (2j+1) magnetic substates for an orbital
    characterized by quantum numbers (n, l, j, isospin).
    
    Parameters:
        n (int): Principal quantum number
        l (int): Orbital angular momentum
        two_j (int): Twice the total angular momentum (2j)
        isospin (int): Isospin projection (±1)
        orbit (int): Orbital index (1-based)
        timereversal (bool): If True, generate only time-reversal symmetric states
    
    Returns:
        tuple: (n_arr, l_arr, j_arr, two_m, iso_arr, parity_arr, orbit_map)
        where each array has length (2j+1) [or (j+1) if timereversal=True]
    
    The magnetic quantum numbers are ordered to facilitate time-reversal symmetry:
        [j, j-2, j-4, ..., -j+2, -j] for normal ordering
        [j, j-2, ..., middle] for time-reversal ordering
    
    Note:
        - For timereversal=True, only (j+1) states are generated (positive m values)
        - The ordering is designed to make time-reversal symmetry easier to implement
        - All returned arrays have the same length
    
    Example:
        >>> # Generate all substates for p3/2 orbital (j=3/2)
        >>> n_arr, l_arr, j_arr, two_m, iso_arr, parity_arr, orbit_map = \
        ...     generate_m_substates(1, 1, 3, 1, 1, timereversal=False)
        >>> print(f"Generated {len(n_arr)} states: m values = {two_m}")
        Generated 4 states: m values = [3 1 -1 -3]
    """

    if(timereversal):
        two_m = np.zeros((two_j+1)//2,dtype=np.int32)
        for i in range((two_j+1)//2):
            two_m[i]   = 2*(0.5 + (i)) * (-1)**(i+2)    #- qj(i) + (j-1)
    else:
        two_m = np.zeros((two_j+1),dtype=np.int32)
        for i in range(0,two_j+1,2):
            # This ordering is intended to make time-reversal symmetry
            # easier to get working!
            two_m[i]   =   (1 + (i))
            two_m[i+1] = - (1 + (i))
        #two_m      = np.arange(-two_j, two_j + 1, 2, dtype=np.int32)

    n_arr      = np.full_like(two_m, n)
    l_arr      = np.full_like(two_m, l)
    j_arr      = np.full_like(two_m, two_j)
    iso_arr    = np.full_like(two_m, isospin, dtype=np.int8)
    parity_arr = np.full_like(two_m, (-1)**l, dtype=np.int8)
    orbit_map  = np.full_like(two_m, orbit, dtype = np.int8)

    return n_arr, l_arr, j_arr, two_m, iso_arr, parity_arr, orbit_map

def build_model_space(filename, r2_filename, timereversal=False ):
     """
     Build a complete single-particle basis from a model space definition file.
     
     Reads orbital definitions from a file and generates all magnetic substates
     to create a complete single-particle basis for nuclear shell model calculations.
     
     Parameters:
         filename (str)     : Path to file containing orbital definitions
         timereversal (bool): If True, use time-reversal symmetry to reduce basis size
         r2_filename (str)  : Path to file containing the reduced matrix elements of r^2
     Returns:
         SingleParticleBasis: Fully constructed basis with all states and blocks
     
     File format: Each line should contain (n, l, j, isospin) where:
         - Column 1: Orbital index (ignored, but must be present)
         - Column 2: Principal quantum number n
         - Column 3: Orbital angular momentum l
         - Column 4: Total angular momentum j (will be doubled internally)
         - Column 5: Isospin (will be doubled internally)
     
     Example file line: 1 1 1 1.5 0.5  # p1/2 orbital
     
     The function automatically:
         1. Reads orbital definitions
         2. Generates all magnetic substates for each orbital
         3. Builds one-body blocks
         4. Builds two-body pairs
         5. Reads reduced matrix elements and builds quadrupole matrix
     
     Note:
         - This is the main entry point for creating a complete basis.
         - The basis is ready for Hamiltonian construction after this function.
         - If r2_filename is None, quadrupole matrix elements are not built.
     
     Example:
         >>> basis = build_model_space("modelspace_def.txt", "r2_elements.txt", timereversal=False)
         >>> print(f"Created basis with {basis.size} states")
         >>> print(f"Protons: {np.sum(basis.isospin == 1)}, Neutrons: {np.sum(basis.isospin == -1)}")
     """

     basis = SingleParticleBasis()
     sps_info = np.loadtxt(filename)
     n_orbits = sps_info.shape[0]

     basis.orbit_twoj    = np.zeros( n_orbits , dtype=np.int8 )
     basis.orbit_isospin = np.zeros( n_orbits , dtype=np.int8 )
     basis.orbit_l       = np.zeros( n_orbits , dtype=np.int8 )

     for i in range(n_orbits):
         # quantum numbers of this orbits
         n       = sps_info[i,1]  # principal quantum number
         l       = sps_info[i,2]  # orbital quantum number
         two_j   = int(2*sps_info[i,3])  # total angular momentem
         isospin = int(2*sps_info[i,4])  # isospin
         orbit   = i + 1  # Note the +1 offset!

         # update orbit information
         basis.orbit_twoj[i]    = two_j
         basis.orbit_isospin[i] = isospin
         basis.orbit_l[i]       = l

         data = generate_m_substates(n,l,two_j,isospin, orbit, timereversal)
         basis.add_states(*data)

     basis.build_one_body_blocks()
     basis.build_pairs()

     if r2_filename is not None:
         basis.build_quadrupole_matrix_elements(r2_filename)

     return basis

def J_coupling(Ja,Jb,Jc,Jd):
    """
    Determine the range of allowed total angular momentum J for two-body coupling.
    
    Calculates the minimum and maximum possible J values for coupling
    two pairs of single-particle states with angular momenta (Ja, Jb) and (Jc, Jd).
    
    Parameters:
        Ja, Jb, Jc, Jd (int): TWICE the angular momenta of four single-particle states
                              (i.e., 2*j for each state)
    
    Returns:
        tuple: (min_J, max_J) where:
            min_J (int): Minimum allowed total angular momentum
            max_J (int): Maximum allowed total angular momentum
    
    The coupling follows angular momentum addition rules:
        |Ja - Jb|/2 ≤ J ≤ (Ja + Jb)/2
        |Jc - Jd|/2 ≤ J ≤ (Jc + Jd)/2
        
    The function returns the intersection of these ranges.
    
    Example:
        >>> # For p1/2 orbitals (j=1/2, so 2j=1)
        >>> min_J, max_J = J_coupling(1, 1, 1, 1)
        >>> print(f"Allowed J range: {min_J} to {max_J}")
        Allowed J range: 0 to 1
        
        >>> # For p3/2 orbitals (j=3/2, so 2j=3)
        >>> min_J, max_J = J_coupling(3, 3, 3, 3)
        >>> print(f"Allowed J range: {min_J} to {max_J}")
        Allowed J range: 0 to 3
    
    Note:
        - All input values should be even integers (since they represent 2j).
        - The returned J values are integers representing the actual angular momentum.
        - This function is used in TBME calculations to determine valid J couplings.
    """

    min_twoJab = abs(Ja - Jb)
    min_twoJcd = abs(Jc - Jd)
    min_J   = max(min_twoJab,min_twoJcd) // 2

    max_twoJab = Ja + Jb
    max_twoJcd = Jc + Jd
    max_J   = min(max_twoJab,max_twoJcd) // 2

    return min_J, max_J
