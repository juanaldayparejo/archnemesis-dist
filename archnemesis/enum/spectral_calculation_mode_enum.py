from enum import IntEnum

class SpectralCalculationModeEnum(IntEnum):
    """
    Defines values for 'ILBL'
    """
    K_TABLES = 0 # use pre-tabulated correlated k-tables
    LINE_BY_LINE_RUNTIME = 1 # calculate line-by-line during runtime
    LINE_BY_LINE_TABLES = 2 # use pre-tabulated line-by-line tables