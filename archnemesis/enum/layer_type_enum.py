from enum import IntEnum

class LayerTypeEnum(IntEnum):
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