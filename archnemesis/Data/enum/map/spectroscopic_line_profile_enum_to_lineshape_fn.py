

from typing import Callable

# Import lineshape functions
import archnemesis.Data.lineshape as lineshape

# Import ENUM that chooses lineshape function
from archnemesis.enums import SpectroscopicLineProfile


def SpectroscopicLineProfileEnum_to_lineshape_fn(
        enum : SpectroscopicLineProfile,
) -> Callable:
    """
    Performs mapping between `SpectroscopicLineProfile` and lineshape functions
    """
    if enum == SpectroscopicLineProfile.VOIGT:
        lineshape_fn = lineshape.voigt
    
    elif enum == SpectroscopicLineProfile.SUBLORENTZ_CO2_BROADENING:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfile(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfile.VANVLECK_WEISSKOPF:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfile(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfile.ROSENKRANTZ_BENREUVEN_FARIR:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfile(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfile.LORENTZ:
        lineshape_fn = lineshape.lorentz
    
    elif enum == SpectroscopicLineProfile.LEVY1994:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfile(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfile.ROSENKRANTZ_BENREUVEN:
        raise NotImplementedError(f'Lineshape function for {SpectroscopicLineProfile(enum)} is not implemented yet')
    
    elif enum == SpectroscopicLineProfile.SUBLORENTZ_CO2_BROADENING_VENUS:
        lineshape_fn = lineshape.tonkov96_sublorentz_CO2_venus
    
    elif enum == SpectroscopicLineProfile.DOPPLER:
        lineshape_fn = lineshape.gaussian
    
    else:
        raise ValueError(f'Cannot find lineshape function for unknown SpectroscopicLineProfile value {enum}')
    
    return lineshape_fn