from enum import IntEnum

class PathObserverPointingEnum(IntEnum):
    """
    Defines location of the PathObserver, used when calculating radiative transfer.

    Used in 'AtmCalc_0.py'
    """
    LIMB = 0 # Limb path, path observer is looking at the limb of the planet
    NADIR = 1 # Nadir path, path observer is on the planet looking upwards
    DISK = 2 # Disk path, path observer is looking at the disk of the planet