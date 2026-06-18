from enum import IntEnum

class LayerIntegrationSchemeEnum(IntEnum):
    """
    Defines layer integration scheme used when calculating radiative transfer.
    
    Used as 'LAYINT' in 'Layer_0.py'
    """
    MID_PATH = 0
    ABSORBER_WEIGHTED_AVERAGE = 1