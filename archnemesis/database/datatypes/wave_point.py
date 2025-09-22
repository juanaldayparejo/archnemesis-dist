

from typing import Self, NamedTuple

import numpy as np

import archnemesis as ans
import archnemesis.enums

# NOTE: There must be a better way of doing unit conversions like this
#       maybe look into astropy.unit to either use or take inspiration from?

class WavePoint(NamedTuple):
    value : float | np.ndarray
    unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm
    
    def to_unit(self, new_unit : ans.enums.WaveUnit) -> Self:
        if self.unit == ans.enums.WaveUnit.Wavenumber_cm:
            if new_unit == ans.enums.WaveUnit.Wavenumber_cm:
                    return self
            elif new_unit ==  ans.enums.WaveUnit.Wavelength_um:
                return WavePoint((1.0/self.value)*1E4, new_unit)
            else:
                raise ValueError(f'No conversion from {self.unit} to {new_unit} was found.')
        elif self.unit ==  ans.enums.WaveUnit.Wavelength_um:
            if new_unit == ans.enums.WaveUnit.Wavenumber_cm:
                return WavePoint(1.0/(self.value*1E-4), new_unit)
            elif new_unit == ans.enums.WaveUnit.Wavelength_um:
                return self
            else:
                raise ValueError(f'No conversion from {self.unit} to {new_unit} was found.')
        else:
            raise ValueError(f'No conversion from {self.unit} to anything else was found.')
    
    def as_unit(self, new_unit : ans.enums.WaveUnit) -> Self:
        return self.to_unit(new_unit)
    
    def __getattr__(self, name):
        """
        Pass through to underlying value
        """
        return getattr(self.value, name)
    
    def __setattr__(self, name, x):
        """
        Pass through to underlying value
        """
        return setattr(self.value, name, x)
    
    def __getitem__(self, i):
        """
        Pass through to underlying value
        """
        if hasattr(self.value, '__getitem__'):
            return self.value[i]
        else:
            raise AttributeError(f'Wrapped type {type(self.value)} does not have attribute "__getitem__"')
    
    def __setitem__(self, i, x):
        """
        Pass through to underlying value
        """
        if hasattr(self.value, '__setitem__'):
            self.value[i] = x
        else:
            raise AttributeError(f'Wrapped type {type(self.value)} does not have attribute "__setitem__"')
