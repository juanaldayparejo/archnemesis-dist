from enum import IntEnum

class RayleighScatteringModeEnum(IntEnum):
    """
    Defines values for 'IRAY'
    
    Used in 'Scatter_0.py'
    """
    NOT_INCLUDED = 0 # Rayleigh scattering optical depth not included
    GAS_GIANT_ATM = 1 # Rayleigh scattering for gas giant atmospheres
    C02_DOMINATED_ATM = 2 # Rayleigh scattering for CO2 dominated atmospheres
    N2_O2_DOMINATED_ATM = 3 # Rayleigh scattering for N2-O2 dominated atmospheres
    JOVIAN_AIR = 4 # Rayleigh scattering for Jovian air (adaptive from Larry Sromovsky)