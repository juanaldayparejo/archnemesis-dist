from __future__ import annotations #  for 3.9 compatability

import dataclasses as dc
from typing import Self

import archnemesis as ans
import archnemesis.enums

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

@dc.dataclass(slots=True)
class WaveRange:
    
    min : float = 0
    max : float = 0
    unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm
    
    def to_unit(self, new_unit : ans.enums.WaveUnit) -> Self:
        if self.unit == ans.enums.WaveUnit.Wavenumber_cm:
            if new_unit == ans.enums.WaveUnit.Wavenumber_cm:
                    return self
            elif new_unit ==  ans.enums.WaveUnit.Wavelength_um:
                x1 = (1.0 / self.min) * 1E4
                x2 = (1.0 / self.max) * 1E4
                if x1 < x2:
                    self.min = x1
                    self.max = x2
                else:
                    self.min = x2
                    self.max = x1
                return self
            else:
                raise ValueError(f'No conversion from {self.unit} to {new_unit} was found.')
        elif self.unit ==  ans.enums.WaveUnit.Wavelength_um:
            if new_unit == ans.enums.WaveUnit.Wavenumber_cm:
                x1 = 1.0 / (self.min * 1E-4)
                x2 = 1.0 / (self.max * 1E-4)
                if x1 < x2:
                    self.min = x1
                    self.max = x2
                else:
                    self.min = x2
                    self.max = x1
                return self
            elif new_unit == ans.enums.WaveUnit.Wavelength_um:
                return self
            else:
                raise ValueError(f'No conversion from {self.unit} to {new_unit} was found.')
        else:
            raise ValueError(f'No conversion from {self.unit} to anything else was found.')
    
    def as_unit(self, new_unit : ans.enums.WaveUnit) -> Self:
        instance = WaveRange(self.min, self.max, self.unit)
        return instance.to_unit(new_unit)
    
    def values(self) -> tuple[float,float]:
        return self.min, self.max
    
    def contains(self, other) -> bool:
        other = other.as_unit(self.unit)
        return self.min <= other.min and other.max <= self.max
    
    def union(self, *others) -> Self:
        result = WaveRange(self.min, self.max, self.unit)
        for another in others:
            other = another.as_unit(self.unit)
            if other.min < result.min:
                result.min = other.min
            if result.max < other.max:
                result.max = other.max
        return result


