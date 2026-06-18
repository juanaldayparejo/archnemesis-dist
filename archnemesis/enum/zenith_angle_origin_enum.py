from enum import IntEnum

class ZenithAngleOriginEnum(IntEnum):
    """
    Defines values for 'IPZEN'

    Used in 'AtmCalc_0.py'
    """
    BOTTOM = 0 # Zenith angle is defined at the bottom of the bottom layer
    ALTITUDE_ZERO = 1 # Zenith angle is defined at 0km atltitude
    TOP = 2 # Zenith angle is defined at the top of the top layer