#!/usr/bin/env python3
"""
TBME (Two-Body Matrix Element) class that properly handles antisymmetrization.

This class provides a clean interface for storing and retrieving TBMEs while
automatically applying the correct antisymmetry properties for fermionic systems.
"""

import numpy as np

class TBME:
    """
    A class to store and manage two-body matrix elements with proper antisymmetrization.
    
    This class handles the antisymmetry properties of fermionic TBMEs:
    V(ab,cd) = -V(ba,cd) = -V(ab,dc) = V(ba,dc)
    
    For diagonal elements where a==b or c==d, additional factors apply.

    NOTE: this class does NOT leverage by itself the hermeticity of
          the Hamiltonian, i.e. it does not enforce that

             V(ab,cd) = V(cd,ab)^*
    """
    
    def __init__(self):
        """Initialize an empty TBME storage."""
        # Use a dictionary to store the fundamental TBMEs
        self._data = {}
        # Cache for frequently accessed TBMEs
        self._cache = {}
        
    def __getitem__(self, key):
        """Get a TBME with proper antisymmetrization."""
        a, b, c, d = key
        
        # Check cache first
        if key in self._cache:
            return self._cache[key]
        
        # Create canonical key (sorted pairs)
        ab_pair = tuple(sorted((a, b)))
        cd_pair = tuple(sorted((c, d)))
        canonical_key = (ab_pair[0], ab_pair[1], cd_pair[0], cd_pair[1])
        
        # Get the fundamental value
        fundamental_value = self._data.get(canonical_key, 0.0)
        
        # Apply antisymmetry phase factors
        phase_factor = 1.0
        
        # Phase from (a,b) ordering
        if a != b:
            if (a, b) != ab_pair:
                phase_factor *= -1.0
        
        # Phase from (c,d) ordering  
        if c != d:
            if (c, d) != cd_pair:
                phase_factor *= -1.0
        
        result = fundamental_value * phase_factor
        
        # Cache the result
        self._cache[key] = result
        return result
    
    def __setitem__(self, key, value):
        """Set a TBME value, storing it in canonical form."""
        a, b, c, d = key
        
        # Create canonical key (sorted pairs)
        ab_pair = tuple(sorted((a, b)))
        cd_pair = tuple(sorted((c, d)))
        canonical_key = (ab_pair[0], ab_pair[1], cd_pair[0], cd_pair[1])
        
        # Apply inverse phase factor to store fundamental value
        inverse_phase = 1.0
        if a != b and (a, b) != ab_pair:
            inverse_phase *= -1.0
        if c != d and (c, d) != cd_pair:
            inverse_phase *= -1.0
        
        fundamental_value = value * inverse_phase
        self._data[canonical_key] = fundamental_value
        
        # Clear cache for all related keys
        self._clear_related_cache(ab_pair, cd_pair)
    
    def _clear_related_cache(self, ab_pair, cd_pair):
        """Clear cache entries related to these orbital pairs."""
        # Generate all possible orderings that would be affected
        a, b = ab_pair
        c, d = cd_pair
        
        # All possible (a,b) orderings
        ab_variants = [(a, b), (b, a)] if a != b else [(a, b)]
        # All possible (c,d) orderings  
        cd_variants = [(c, d), (d, c)] if c != d else [(c, d)]
        
        # Clear all combinations
        for ab_var in ab_variants:
            for cd_var in cd_variants:
                key = (ab_var[0], ab_var[1], cd_var[0], cd_var[1])
                if key in self._cache:
                    del self._cache[key]
    
    def get_fundamental(self, ab_pair, cd_pair):
        """Get the fundamental TBME value for sorted orbital pairs."""
        a, b = sorted(ab_pair)
        c, d = sorted(cd_pair)
        return self._data.get((a, b, c, d), 0.0)
    
    def set_fundamental(self, ab_pair, cd_pair, value):
        """Set the fundamental TBME value for sorted orbital pairs."""
        a, b = sorted(ab_pair)
        c, d = sorted(cd_pair)
        self._data[(a, b, c, d)] = value
        self._clear_related_cache((a, b), (c, d))
    
    def __contains__(self, key):
        """Check if a TBME exists (considering antisymmetry)."""
        a, b, c, d = key
        ab_pair = tuple(sorted((a, b)))
        cd_pair = tuple(sorted((c, d)))
        canonical_key = (ab_pair[0], ab_pair[1], cd_pair[0], cd_pair[1])
        return canonical_key in self._data
    
    def __len__(self):
        """Return the number of fundamental TBMEs stored."""
        return len(self._data)
    
    def keys(self):
        """Return all fundamental TBME keys."""
        return self._data.keys()
    
    def items(self):
        """Return all fundamental TBME items."""
        return self._data.items()
    
    def clear(self):
        """Clear all TBME data and cache."""
        self._data.clear()
        self._cache.clear()
    
    def apply_normalization(self, basis):
        """Apply normalization factors for diagonal elements."""
        new_data = {}
        for (a, b, c, d), value in self._data.items():
            # Get the orbital indices for these states
            aa = basis.orbit_map[a]
            bb = basis.orbit_map[b]
            cc = basis.orbit_map[c]
            dd = basis.orbit_map[d]
            
            # Apply normalization based on orbital indices (matching original logic)
            normalized_value = value
            if aa == bb:
                normalized_value *= np.sqrt(2.0)
            if cc == dd:
                normalized_value *= np.sqrt(2.0)
            new_data[(a, b, c, d)] = normalized_value
        
        self._data = new_data
        self._cache.clear()
