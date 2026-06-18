from enum import IntEnum

class InstrumentLineshapeEnum(IntEnum):
    """
    Defines values for 'ISHAPE'
    """
    Square = 0
    Triangular = 1
    Gaussian = 2
    Hamming = 3
    Hanning = 4