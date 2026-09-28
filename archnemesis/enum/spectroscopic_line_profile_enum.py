from enum import IntEnum

class SpectroscopicLineProfileEnum(IntEnum):
    """
    Defines the line profile to be used in the spectroscopic line shape calculations
    """
    VOIGT = 0
    SUBLORENTZ_CO2_BROADENING = 1
    VANVLECK_WEISSKOPF = 2
    ROSENKRANTZ_BENREUVEN_FARIR = 3
    LORENTZ = 4
    LEVY1994 = 5
    ROSENKRANTZ_BENREUVEN = 6
    SUBLORENTZ_CO2_BROADENING_VENUS = 7
    DOPPLER = 12