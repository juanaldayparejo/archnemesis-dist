

from typing import Self, NamedTuple

import numpy as np

import archnemesis as ans
import archnemesis.enum

class WaveRange(NamedTuple):
    
    min : float = 0
    max : float = 0
    unit : ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm
    
    def to_unit(self, new_unit : ans.enum.WaveUnitEnum) -> Self:
        if self.unit == ans.enum.WaveUnitEnum.Wavenumber_cm:
            if new_unit == ans.enum.WaveUnitEnum.Wavenumber_cm:
                    return self
            elif new_unit ==  ans.enum.WaveUnitEnum.Wavelength_um:
                x1 = (1.0 / self.min) * 1E4
                x2 = (1.0 / self.max) * 1E4
                return WaveRange(*sorted((x1, x2)), new_unit)
            else:
                raise ValueError(f'No conversion from {self.unit} to {new_unit} was found.')
        elif self.unit ==  ans.enum.WaveUnitEnum.Wavelength_um:
            if new_unit == ans.enum.WaveUnitEnum.Wavenumber_cm:
                x1 = 1.0 / (self.min * 1E-4)
                x2 = 1.0 / (self.max * 1E-4)
                return WaveRange(*sorted((x1, x2)), new_unit)
            elif new_unit == ans.enum.WaveUnitEnum.Wavelength_um:
                return self
            else:
                raise ValueError(f'No conversion from {self.unit} to {new_unit} was found.')
        else:
            raise ValueError(f'No conversion from {self.unit} to anything else was found.')
    
    def as_unit(self, new_unit : ans.enum.WaveUnitEnum) -> Self:
        return self.to_unit(new_unit)
    
    @property
    def size(self) -> float:
        return self.max - self.min
    
    def values(self) -> tuple[float,float]:
        return self.min, self.max
    
    def contains(self, other) -> bool:
        other = other.as_unit(self.unit)
        return (self.min - other.min) <= 1E-30 and (other.max - self.max) <= 1E-30
    
    def union(self, *others) -> Self:
        vmin, vmax = self.values()
        for another in others:
            other = another.as_unit(self.unit)
            if other.min < vmin:
                vmin = other.min
            if vmax < other.max:
                vmax = other.max
        return WaveRange(vmin, vmax, self.unit)
    
    def linspace(self, n : float = 50, endpoint=True):
        return np.linspace(self.min, self.max, n, endpoint=True)
    
    def arange(self, step : float):
        return np.arange(self.min, self.max, step)

