from enum import IntEnum

class LowerBoundaryConditionEnum(IntEnum):
    """
    Defines values for 'LOWBC'
    
    Used in 'Surface_0.py'
    """
    THERMAL = 0 # Thermal emission only (i.e. no reflection)
    LAMBERTIAN = 1 # Lambertian surface
    HAPKE = 2 # Hapke surface
    OREN_NAYAR = 3 # Oren-Nayar surface