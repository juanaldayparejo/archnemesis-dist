

from typing import Callable

# Import lineshape functions
import archnemesis.lineshape as lineshape

# Import ENUM that chooses lineshape function
from archnemesis.enum import SpectroscopicLineProfileEnum


def SpectroscopicLineProfileEnum_to_lineshape_fn(
        enum : SpectroscopicLineProfileEnum,
) -> Callable:
    """
    Performs mapping between `SpectroscopicLineProfile` and lineshape functions
    """
    if enum == SpectroscopicLineProfileEnum.VOIGT:
        lineshape_fn = lineshape.voigt
    
    elif enum == SpectroscopicLineProfileEnum.SUBLORENTZ_CO2_BROADENING:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfileEnum(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfileEnum.VANVLECK_WEISSKOPF:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfileEnum(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfileEnum.ROSENKRANTZ_BENREUVEN_FARIR:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfileEnum(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfileEnum.LORENTZ:
        lineshape_fn = lineshape.lorentz
    
    elif enum == SpectroscopicLineProfileEnum.LEVY1994:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfileEnum(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfileEnum.ROSENKRANTZ_BENREUVEN:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfileEnum(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfileEnum.SUBLORENTZ_CO2_BROADENING_VENUS:
        lineshape_fn = lineshape.tonkov96_sublorentz_CO2_venus
    
    elif enum == SpectroscopicLineProfileEnum.DOPPLER:
        lineshape_fn = lineshape.gaussian
    
    else:
        raise ValueError(f'Cannot find lineshape function for unknown SpectroscopicLineProfile value {enum}')
    
    return lineshape_fn