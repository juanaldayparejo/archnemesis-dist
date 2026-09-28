from enum import IntEnum

class AerosolPhaseFunctionCalculationModeEnum(IntEnum):
    """
    Defines values for 'IMIE'
    
    Used in 'Scatter_0.py'
    """
    HENYEY_GREENSTEIN = 0 # Henyey-Greenstein parameters
    MIE_THEORY = 1 # Explicitly calculated from Mie theory
    LEGENDRE_POLYNOMIALS = 2 # Legendre polynomials