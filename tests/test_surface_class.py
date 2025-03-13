import pytest
import archnemesis as ans
import numpy as np

def create_example_surface(NEM, NLOCATIONS):
    surface = ans.Surface_0(
        GASGIANT=True, 
        ISPACE=1, # Spectral units (0 = wavenumber, 1 = um)
        LOWBC=1, # Lower boundary condition
        GALB=-1.0, # Ground albedo
        NEM=NEM, # Number of points in the EMissivity spectrum of the surface
        NLOCATIONS=NLOCATIONS # Number of points "on the surface" for which we define a spectrum/temperature etc.
    )
    
    # Make a grid of LAT-LON values
    nx = int(np.floor(np.sqrt(surface.NLOCATIONS)))
    ny = surface.NLOCATIONS // nx
    nz = surface.NLOCATIONS - nx*ny
    surface.LATITUDE = np.repeat(np.linspace(-90,90,nx), ny)
    surface.LATITUDE = np.concatenate((surface.LATITUDE, np.zeros((nz,))))

    surface.LONGITUDE = np.repeat(np.linspace(-90,90,ny), nx).reshape((ny,nx)).T.flatten()
    surface.LONGITUDE = np.concatenate((surface.LONGITUDE, np.zeros((nz,))))

    surface.TSURF=np.ones((surface.NLOCATIONS,))*300
    
    # Wavelengths where surface parameters are defined (ISPACE = 1, so is in um)
    surface.VEM = np.geomspace(1E-9,100,surface.NEM) 
    
    return surface


@pytest.mark.parametrize(
    'NEM, NLOCATIONS', [
        (201,100), 
        (201,1), 
        (100,201), 
        (1,1),
        (1,2)
    ]
)
def test_surface_radiance_calculation(NEM, NLOCATIONS):
    surface = create_example_surface(NEM, NLOCATIONS)
    
    # inside Surface_0.calc_radground(...), bbsurf has shape (self.NEM, self.NLOCATIONS)
    # so surface.EMISSIVITY must have that shape as well.
    surface.EMISSIVITY = np.ones((surface.NEM, surface.NLOCATIONS))
    
    # To test the 1D case, squeeze the 2nd dimension if we can
    if (surface.EMISSIVITY.shape[1] == 1):
        surface.EMISSIVITY = np.squeeze(surface.EMISSIVITY, axis=1)
    
    # This should not throw an error
    surface_radiance = surface.calc_radground(surface.ISPACE)