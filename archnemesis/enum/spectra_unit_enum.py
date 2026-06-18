from enum import IntEnum

class SpectraUnitEnum(IntEnum):
    """
    Defines values for 'IFORM'
    """
    Radiance = 0 # W cm-2 sr-1 (cm-1)-1 if ISPACE=0 ---- W cm-2 sr-1 μm-1 if ISPACE=1
    FluxRatio = 1 # F_planet/F_star - Dimensionless
    TransitDepth = 2 # Transit depth - 100.0 * Area_planet/Area_star (dimensionless)
    Integrated_spectral_power = 3 # Integrated spectral power of planet - W (cm-1)-1 if ISPACE=0 ---- W um-1 if ISPACE=1
    Atmospheric_transmission = 4 # Atmospheric transmission multiplied by solar flux
    Normalised_radiance = 5 # Normalised radiance to a given wavelength (VNORM)
    Integrated_radiance = 6 # Integrated radiance over filter function - W cm-2 sr-1 if ISPACE=0 ---- W cm-2 sr-1 if ISPACE=1