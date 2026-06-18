from enum import IntEnum

class AtmosphericProfileTypeEnum(IntEnum):
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