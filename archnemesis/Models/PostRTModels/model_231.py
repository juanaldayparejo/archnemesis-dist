

from typing import TYPE_CHECKING, IO, Self

import numpy as np

from ._base import PostRTModelBase
from ..ModelParameter import ModelParameter

from ..log import _lgr  # noqa # Ignore if _lgr is not used

if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0

    nx = 'number of elements in state vector'
    m = 'an undetermined number, but probably less than "nx"'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'
    nparam = 'the number of parameters a model has'
    NCONV = 'number of spectral bins'
    NGEOM = 'number of geometries'
    NX = 'number of elements in state vector'
    NDEGREE = 'number of degrees in a polynomial'
    NWINDOWS = 'number of spectral windows'

class Model231(PostRTModelBase):
    """
        Scaling of spectrum using a varying scaling factor (following a polynomial of degree N)
        
        The computed spectra is multiplied by `R = R0 * POL`, where the polynomial function POL depends on the wavelength and is given by:
        
            POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
    """
    id : int = 231
    
    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            n_geom : int,
            #   The number of geometries that this model applies to, applies from the first to the last geometry.
            #   Geometries with index >= n_geom will not be affected by this model.
            
            n_degree : int,
            #   The degree of the polynomial
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model.
        # NOTE: It is best to define these in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        self.parameters = (
            ModelParameter('coeff', slice(None), 'coefficients for the polynomial', 'NUMBER'),
        )
        
        self.n_geom = n_geom
        self.n_degree = n_degree
        
        return
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            #   Modelled spectrum
            
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            #   Gradient of modelled spectrum
            
            WAVE : np.ndarray[['NCONV'], float],
            #   Wavelengths/wavenumbers the spectrum values are defined at
            
            COEFF : np.ndarray[['NDEGREE+1'],float],
            #   Coefficients of the polynomial
            
            state_vector_slice : slice,
            #   A slice that chooses parts of the state vector corresponding to the
            #   parameters used by this model
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        WAVE0 = WAVE[0]
        spec = np.zeros(WAVE.size)
        spec[:] = SPECMOD[:WAVE.size]
        POL = np.zeros_like(spec)
        
        dW = WAVE-WAVE0
        for j in range(COEFF.shape[0]):
            POL[:] = POL[:] + COEFF[j] * dW**j
        
        SPECMOD[:WAVE.size] *= POL
        dSPECMOD[:WAVE.size,:] *= POL[:,None]
        
        dspecmod_part = dSPECMOD[:WAVE.size, state_vector_slice]
        for j in range(COEFF.shape[0]):
            dspecmod_part[:WAVE.size,j] = spec * dW**j

        return SPECMOD, dSPECMOD
    
    
    @classmethod
    def from_apr_to_state_vector(
            cls,
            variables : "Variables_0",
            f : IO,
            varident : np.ndarray[[3],int],
            varparam : np.ndarray[["mparam"],float],
            ix : int,
            lx : np.ndarray[["mx"],int],
            x0 : np.ndarray[["mx"],float],
            sx : np.ndarray[["mx","mx"],float],
            inum : np.ndarray[["mx"],int],
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='int')
        ngeom = int(tmp[0])
        ndegree = int(tmp[1])
        varparam[0] = ngeom
        varparam[1] = ndegree
        for ilevel in range(ngeom):
            tmp = f1.readline().split()
            for ic in range(ndegree+1):
                r0 = float(tmp[2*ic])
                err0 = float(tmp[2*ic+1])
                x0[ix] = r0
                sx[ix,ix] = (err0)**2.
                inum[ix] = 0
                ix = ix + 1
        return cls(ix_0, ix-ix_0, ngeom, ndegree)
        
    
    @classmethod
    def from_bookmark(
            cls,
            variables : "Variables_0",
            varident : np.ndarray[[3],int],
            varparam : np.ndarray[["mparam"],float],
            ix : int,
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
        ) -> Self:
        ix_0 = ix
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
        ngeom = int(varparam[0])
        ndegree = int(varparam[1])
        for ilevel in range(ngeom):
            for ic in range(ndegree+1):
                ix = ix + 1
        return cls(ix_0, ix-ix_0, ngeom, ndegree)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:

        #coeff_shape = (self.n_degree+1, self.n_geom)
        coeff_shape = (self.n_geom, self.n_degree+1)
        coeff = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)[0].reshape(coeff_shape)

        for i_geom in range(self.n_geom):
            _lgr.debug(f'coefficients for geometry {i_geom}: {coeff[i_geom,:]}')

        ixx = ix
        for i_geom in range(self.n_geom):
            SPECMOD[:,i_geom], dSPECMOD[:,i_geom,:] = self.calculate(
                SPECMOD[:,i_geom],
                dSPECMOD[:,i_geom,:],
                forward_model.Measurement.VCONV[:forward_model.Measurement.NCONV[i_geom], i_geom],
                coeff[i_geom,:],
                slice(ixx, ixx + self.n_degree + 1)
            )
            ixx += self.n_degree + 1
        
        return


