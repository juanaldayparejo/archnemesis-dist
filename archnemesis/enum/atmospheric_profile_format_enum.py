

from enum import IntEnum

class AtmosphericProfileFormatEnum(IntEnum):
    """
    Defines values for 'AMFORM', the atmospheric profile format.
    """
    MOLECULAR_WEIGHT_DEFINED = 0
    CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE = 1
    CALC_MOLECULAR_WEIGHT_DO_NOT_SCALE_VMR = 2