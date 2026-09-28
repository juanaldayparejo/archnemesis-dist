from enum import IntEnum

class EmissionTypeEnum(IntEnum):
    """
    Defines the type of atmospheric emission included
    """
    FLUORESCENCE=0
    CHEMICAL=1
    PHOTOLYSIS=2