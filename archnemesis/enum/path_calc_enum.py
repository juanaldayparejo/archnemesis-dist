from enum import IntFlag, auto

class PathCalcEnum(IntFlag):
    """
    Defines path calculation type used when calculating radiative transfer.
    
    Used as elements of 'IMOD' in 'AtmCalc_0.py', 'Path_0.py', 'ForwardModel_0.py'
    """
    WEIGHTING_FUNCTION = auto() # Weighting function
    NET_FLUX = auto() # Net flux calculation
    UPWARD_FLUX = auto() # Internal upward flux calculation
    OUTWARD_FLUX = auto() # Upward flux at top of topmost layer
    DOWNWARD_FLUX = auto() # Downward flux at bottom of lowest layer
    CURTIS_GODSON = auto() # Curtis Godson
    THERMAL_EMISSION = auto() # Thermal emission
    HEMISPHERE = auto() # Integrate emission into hemisphere
    MULTIPLE_SCATTERING = auto() # Full scattering calculation
    NEAR_LIMB = auto() # Near-limb scattering calculation
    SINGLE_SCATTERING_PLANE_PARALLEL = auto() # Single scattering calculation (plane parallel)
    SINGLE_SCATTERING_SPHERICAL = auto() # Single scattering calculation (spherical atm.)
    ABSORBTION = auto() # calculate absorption not transmission
    PLANCK_FUNCTION_AT_BIN_CENTRE = auto() # use planck function at bin centre in genlbl (also denoted as BINBB in old code)
    BROADENING = auto() # calculate emission outside of genlbl