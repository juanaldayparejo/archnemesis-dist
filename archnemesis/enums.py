from __future__ import annotations #  for 3.9 compatability
from enum import IntEnum, IntFlag, auto

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

class PlanetEnum(IntEnum):
    """
    Define values for 'IPLANET', the planet being observed.
    """
    UNDEFINED = -1
    Mercury = 1
    Venus = 2
    Earth = 3
    Mars = 4
    Jupiter = 5
    Saturn = 6
    Uranus = 7
    Neptune = 8
    Pluto = 9
    Sun = 10
    Titan = 11
    NGTS_10b = 85
    WASP_43b = 87
    

class AtmosphericProfileFormatEnum(IntEnum):
    """
    Defines values for 'AMFORM', the atmospheric profile format.
    """
    MOLECULAR_WEIGHT_DEFINED = 0
    CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE = 1
    CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR = 2

class ParaH2Ratio(IntEnum):
    """
    Defines values for 'INORMAL' and elements of 'INORMALT', the para-hydrogen ratio in the atmosphere.
    """
    EQUILIBRIUM = 0 # 1:1 ratio
    NORMAL = 1 # 3:1 ratio

class Gas(IntEnum):
    """
    Enum to define the gases used in the retrievals.
    
    Used in 'IPAIRG1', 'IPAIRG2'
    """
    H2O = 1
    CO2 = 2
    O3 = 3
    N2O = 4
    CO = 5
    CH4 = 6
    O2 = 7
    NO = 8
    SO2 = 9
    NO2 = 10
    NH3 = 11
    HNO3 = 12
    OH = 13
    HF = 14
    HCl = 15
    HBr = 16
    HI = 17
    ClO = 18
    OCS = 19
    H2CO = 20
    HOCl = 21
    N2 = 22
    HCN = 23
    CH3Cl = 24
    H2O2 = 25
    C2H2 = 26
    C2H6 = 27
    PH3 = 28
    C2N2 = 29
    C4H2 = 30
    HC3N = 31
    C2H4 = 32
    GeH4 = 33
    C3H8 = 34
    HCOOH = 35
    H2S = 36
    COF2 = 37
    SF6 = 38
    H2 = 39
    He = 40
    AsH3 = 41
    C3H4 = 42
    ClONO2 = 43
    HO2 = 44
    O = 45
    NO_PLUS = 46
    CH3OH = 47
    H = 48
    C6H6 = 49
    CH3CN = 50
    CH2NH = 51
    C2H3CN = 52
    HCP = 53
    CS = 54
    HC5N = 55
    HC7N = 56
    C2H5CN = 57
    CH3NH2 = 58
    HNC = 59
    Na = 60
    K = 61
    TiO = 62
    VO = 63
    CH2CCH2 = 64
    C4N2 = 65
    C5H5N = 66
    C5H4N2 = 67
    C7H8 = 68
    C8H6 = 69
    C5H5CN = 70
    HOBr = 71
    CH3Br = 72
    CF4 = 73
    SO3 = 74
    Ne = 75
    Ar = 76
    COCl2 = 77
    SO = 78
    H2SO4 = 79
    E_MINUS = 80
    H3_PLUS = 81
    FeH = 82
    AlO = 83
    AlCl = 84
    AlF = 85
    AlH = 86
    BeH = 87
    C2 = 88
    CaF = 89
    CaH = 90
    H_MINUS = 91
    CaO = 92
    CH = 93
    CH3 = 94
    CH3F = 95
    CN = 96
    CP = 97
    CrH = 98
    HD_PLUS = 99
    HeH_PLUS = 100
    KCl = 101
    KF = 102
    LiCl = 103
    LiF = 104
    LiH = 105
    LiH_PLUS = 106
    MgF = 107
    MgH = 108
    MgO = 109
    NaCl = 110
    NaF = 111
    NaH = 112
    NH = 113
    NS = 114
    OH_PLUS = 115
    CIS_P2H2 = 116
    TRANS_P2H2 = 117
    PH = 118
    PN = 119
    PO = 120
    PS = 121
    ScH = 122
    SH = 123
    SiH = 124
    SiH2 = 125
    SiH4 = 126
    SiO = 127
    SiS = 129
    TiH = 130
    Cl2 = 131
    ClO2 = 132
    BrO = 133
    IO = 134
    NO3 = 135
    BrO2 = 136
    IO2 = 137

class InstrumentLineshape(IntEnum):
    """
    Defines values for 'ISHAPE'
    """
    Square = 0
    Triangular = 1
    Gaussian = 2
    Hamming = 3
    Hanning = 4

class WaveUnit(IntEnum):
    """
    Defines values for 'ISPACE'
    """
    Wavenumber_cm = 0
    Wavelength_um = 1

class SpectraUnit(IntEnum):
    """
    Defines values for 'IFORM'
    """
    Radiance = 0 # W cm-2 sr-1 (cm-1)-1 if ISPACE=0 ---- W cm-2 sr-1 μm-1 if ISPACE=1
    FluxRatio = 1 # F_planet/F_star - Dimensionless
    A_Ratio = 2 # A_planet/A_star - 100.0 * A_planet/A_star (dimensionless)
    Integrated_spectral_power = 3 # Integrated spectral power of planet - W (cm-1)-1 if ISPACE=0 ---- W um-1 if ISPACE=1
    Atmospheric_transmission = 4 # Atmospheric transmission multiplied by solar flux
    Normalised_radiance = 5 # Normalised radiance to a given wavelength (VNORM)

class SpectralCalculationMode(IntEnum):
    """
    Defines values for 'ILBL'
    """
    K_TABLES = 0 # use pre-tabulated correlated k-tables
    LINE_BY_LINE_RUNTIME = 1 # calculate line-by-line during runtime
    LINE_BY_LINE_TABLES = 2 # use pre-tabulated line-by-line tables

class LowerBoundaryCondition(IntEnum):
    """
    Defines values for 'LOWBC'
    
    Used in 'Surface_0.py'
    """
    THERMAL = 0 # Thermal emission only (i.e. no reflection)
    LAMBERTIAN = 1 # Lambertian surface
    HAPKE = 2 # Hapke surface
    OREN_NAYAR = 3 # Oren-Nayar surface

class RetrievalStrategy(IntEnum):
    """
    Defines values for 'retrieval_method'
    """
    Optimal_Estimation = 0
    Nested_Sampling = 1

class ScatteringCalculationMode(IntEnum):
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

class RayleighScatteringMode(IntEnum):
    """
    Defines values for 'IRAY'
    
    Used in 'Scatter_0.py'
    """
    NOT_INCLUDED = 0 # Rayleigh scattering optical depth not included
    GAS_GIANT_ATM = 1 # Rayleigh scattering for gas giant atmospheres
    C02_DOMINATED_ATM = 2 # Rayleigh scattering for CO2 dominated atmospheres
    N2_O2_DOMINATED_ATM = 3 # Rayleigh scattering for N2-O2 dominated atmospheres
    JOVIAN_AIR = 4 # Rayleigh scattering for Jovian air (adaptive from Larry Sromovsky)

class AerosolPhaseFunctionCalculationMode(IntEnum):
    """
    Defines values for 'IMIE'
    
    Used in 'Scatter_0.py'
    """
    HENYEY_GREENSTEIN = 0 # Henyey-Greenstein parameters
    MIE_THEORY = 1 # Explicitly calculated from Mie theory
    LEGENDRE_POLYNOMIALS = 2 # Legendre polynomials

class ZenithAngleOrigin(IntEnum):
    """
    Defines values for 'IPZEN'

    Used in 'AtmCalc_0.py'
    """
    BOTTOM = 0 # Zenith angle is defined at the bottom of the bottom layer
    ALTITUDE_ZERO = 1 # Zenith angle is defined at 0km atltitude
    TOP = 2 # Zenith angle is defined at the top of the top layer


class PathObserverPointing(IntEnum):
    """
    Defines location of the PathObserver, used when calculating radiative transfer.

    Used in 'AtmCalc_0.py'
    """
    LIMB = 0 # Limb path, path observer is looking at the limb of the planet
    NADIR = 1 # Nadir path, path observer is on the planet looking upwards
    DISK = 2 # Disk path, path observer is looking at the disk of the planet


class PathCalc(IntFlag):
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


class LayerType(IntEnum):
    """
    Defines layer type used when calculating radiative transfer.
    
    Used as 'LAYTYP' in 'Layer_0.py'
    """
    EQUAL_PRESSURE = 0
    EQUAL_LOG_PRESSURE = 1
    EQUAL_HEIGHT = 2
    EQUAL_PATH_LENGTH = 3
    BASE_PRESSURE = 4
    BASE_HEIGHT = 5

class LayerIntegrationScheme(IntEnum):
    """
    Defines layer integration scheme used when calculating radiative transfer.
    
    Used as 'LAYINT' in 'Layer_0.py'
    """
    MID_PATH = 0
    ABSORBER_WEIGHTED_AVERAGE = 1

class InterpolationMethod(IntEnum):
    """
    Defines interpolation method used by SCIPY routines.
    
    Used in 'Layer_0.py'
    """
    LINEAR = 0
    QUADRATIC_SPLINE = 1
    CUBIC_SPLINE = 2


class AtmosphericProfileType(IntEnum):
    """
    Defines the atmospheric profile type that a model parameterises
    
    Used as 'ipar' in 'Models.py', 'ForwardModel_0.py'
    """
    NOT_PRESENT = -1
    GAS_VOLUME_MIXING_RATIO = 0
    TEMPERATURE = 1
    AEROSOL_DENSITY = 2
    PARA_H2_FRACTION = 3
    FRACTIONAL_CLOUD_COVERAGE = 4
