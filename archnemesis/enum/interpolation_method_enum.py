from enum import IntEnum

class InterpolationMethodEnum(IntEnum):
    """
    Defines interpolation method used by SCIPY routines.
    
    Used in 'Layer_0.py'
    """
    LINEAR = 0
    QUADRATIC_SPLINE = 1
    CUBIC_SPLINE = 2