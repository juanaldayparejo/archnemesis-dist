from __future__ import annotations #  for 3.9 compatability

from typing import Self

import numpy as np

import archnemesis.helpers.maths_helper as maths_helper
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor


class GasMixture:
    """
    Class for storing a gas mixture and getting properties of each gas within the mixture.
    
    The fractions of a gas in the mixture always add to one, they are normalised by the
    `normalise()` method, and after every `set(...)` method.
    """
    def __init__(self, gas_frac_pairs : tuple[tuple[RadtranGasDescriptor, float],...]):
        self._gf = dict(gas_frac_pairs)
        self.normalise()
    
    @property
    def gasses(self) -> tuple[RadtranGasDescriptor]:
        return tuple(self._gf.keys())
    
    @property
    def fractions(self) -> tuple[float]:
        return tuple(self._gf.values())
    
    def __contains__(self, k) -> bool:
        return k in self._gf
    
    def __getitem__(self, k) -> float:
        return self._gf[k]
    
    def normalise(self):
        if len(self._gf) == 0:
            return
        
        a = np.sum([f for g,f in self._gf.items()])
        self._gf = dict([(g,f/a) for g,f in self._gf.items()])
    
    def is_normalised(self) -> bool: # ideally this should always be True when using the public interface of this class
        if len(self._gf) == 0:
            return True
        
        a = np.sum([f for g,f in self._gf.items()])
        return (a - 1) < 1E-30
    
    def set(self, gas_frac_pairs : tuple[tuple[RadtranGasDescriptor, float],...]) -> Self:
        for g, f in gas_frac_pairs:
            self._gf[g] = f
        
        self.normalise()
        return self
    
    def get_number_densities(self, press : float | np.ndarray, temp : float | np.ndarray) -> dict[RadtranGasDescriptor, float | np.ndarray]:
        n = maths_helper.ideal_gas_number_density(press, temp)
        
        ns = dict()
        for g, f in self._gf.items():
            ns[g] = n * f
        return ns
    
    
    