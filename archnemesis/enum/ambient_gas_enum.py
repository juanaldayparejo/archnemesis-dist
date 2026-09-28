from enum import IntEnum

class AmbientGasEnum(IntEnum):
    """
    Defines the ambient gas used in the line spectroscopy calculations

    Used as ambient_gas in 'LineData_0.py'
    """
    AIR = 0
    CO2 = 1
    H2 = 2