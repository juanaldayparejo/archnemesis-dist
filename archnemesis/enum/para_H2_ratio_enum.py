from enum import IntEnum

class ParaH2Ratio(IntEnum):
    """
    Defines values for 'INORMAL' and elements of 'INORMALT', the para-hydrogen ratio in the atmosphere.
    """
    EQUILIBRIUM = 0 # 1:1 ratio
    NORMAL = 1 # 3:1 ratio