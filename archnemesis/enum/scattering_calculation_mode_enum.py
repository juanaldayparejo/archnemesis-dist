from enum import IntEnum

class ScatteringCalculationModeEnum(IntEnum):
    """
    Defines values for 'ISCAT'
    
    Used in 'Scatter_0.py'
    """
    THERMAL_EMISSION = 0 # Thermal emission only (i.e. no scattering)
    MULTIPLE_SCATTERING = 1 # Multiple scattering
    INTERNAL_RADIATION_FIELD = 2 # Internal radiation field
    SINGLE_SCATTERING_PLANE_PARALLEL = 3 # Single scattering in a plane parallel atmosphere
    SINGLE_SCATTERING_SPHERICAL = 4 # Single scattering in a spherical atmosphere
    INTERNAL_NET_FJLUX = 5 # Internal net flux
    DOWNWARD_BOTTOM_FLUX = 6 # Downward flux at the bottom of the atmosphere