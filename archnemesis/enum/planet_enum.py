from enum import IntEnum

class PlanetEnum(IntEnum):
    """
    Define values for 'IPLANET', the planet being observed.
    """
    CUSTOM = -1
    Mercury = 1
    Venus = 2
    Earth = 3
    Mars = 4
    Jupiter = 5
    Saturn = 6
    Uranus = 7
    Neptune = 8
    Pluto = 9
    Sun = 10
    Titan = 11
    NGTS_10b = 85
    WASP_43b = 87