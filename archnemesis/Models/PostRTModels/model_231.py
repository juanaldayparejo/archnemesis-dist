
from typing import TYPE_CHECKING, Self, IO, ClassVar, Any
import dataclasses as dc

import numpy as np

from ..param import (
    StateParam, 
    #ConstParam, 
    VarParam,
)

from ._base import PostRTModelBase

from archnemesis.enum import ArchNemesisFileTypeEnum

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


@dc.dataclass
class Model231(PostRTModelBase):
    """
        Scaling of spectrum using a varying scaling factor (following a polynomial of degree N)
        
        The computed spectra is multiplied by `R = R0 * POL`, where the polynomial function POL depends on the wavelength and is given by:
        
            POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
    """
    id : ClassVar[int] = 231
    
    coeff : StateParam.using(slice(None), 'coefficients of the polynomial', 'NUMBER') # noqa: F722 F821
    
    ngeom : VarParam.using(int, 'Number of geometires that this model applies to, geometries with index >= ngeom will not be affected by this model') # noqa: F722 F821
    ndegree : VarParam.using(int, 'The degree of the polynomial') # noqa: F722 F821
    
    
    
    @classmethod
    def read_polyfile(
            cls,
            fpath : str
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any]]:
        """
        first line contains `ngeom` `ndegree`
        next `ngeom` lines contain `ndegree` (coefficient, error) pairs
        """
        with open(fpath, 'r') as f:
            ngeom, ndegree = cls.read_apr_entries(f, (int, int))
            n = ngeom * (ndegree+1)
            xvals = np.zeros((n,))
            xerrs = np.zeros((n,))
            for i in range(ngeom):
                pairs = f.readline().split()
                xvals[i*ngeom:(i+1)*ngeom] = map(float, pairs[::2])
                xerrs[i*ngeom:(i+1)*ngeom] = map(float, pairs[1::2])
            
            return xvals, xerrs, (ngeom, ndegree)
    
    
    
    @classmethod
    def from_apr_file(
            cls,
            f : IO,
            varident : np.ndarray[[3],int],
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
            runname : str,
            sxminfac : float,
            input_file_type : ArchNemesisFileTypeEnum,
    ) -> Self:
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
        
        xvals, xerrs, (ngeom, ndegree) = cls.read_polyfile(f.readline().split()[0])
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            ngeom,
            ndegree
        )
        
        instance.coeff.log = False
        
        return instance
        
    
    
    def calculate(
            self,
            SPECMOD : np.ndarray[['NCONV'],float],
            #   Modelled spectrum
            
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            #   Gradient of modelled spectrum
            
            WAVE : np.ndarray[['NCONV'], float],
            #   Wavelengths/wavenumbers the spectrum values are defined at
            
            geom_slice : slice,
            #   Slice for this geometry
    ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        COEFF = self.coeff.v
        
        WAVE0 = WAVE[0]
        spec = np.zeros(WAVE.size)
        spec[:] = SPECMOD[:WAVE.size]
        POL = np.zeros_like(spec)
        
        dW = WAVE-WAVE0
        for j in range(COEFF.shape[0]):
            POL[:] = POL[:] + COEFF[j] * dW**j
        
        SPECMOD[:WAVE.size] *= POL
        dSPECMOD[:WAVE.size,:] *= POL[:,None]
        
        dspecmod_part = dSPECMOD[:WAVE.size, geom_slice]
        for j in range(COEFF.shape[0]):
            dspecmod_part[:WAVE.size,j] = spec * dW**j

        return SPECMOD, dSPECMOD
    
    
    
        
    
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
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
        ngeom = int(varparam[0])
        ndegree = int(varparam[1])
        return cls(
            (
                np.zeros((ngeom*(ndegree+1),)), 
                np.zeros((ngeom*(ndegree+1),))
            ), 
            ngeom, 
            ndegree
        )
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        
        for i_geom in range(self.ngeom.v):
            _lgr.debug(f'coefficients for geometry {i_geom}: {self.coeff.v[i_geom*self.ndegree.v:(i_geom+1)*self.ndegree.v]}')

        self.pull_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)

        ixx = ix
        for i_geom in range(self.ngeom.v):
            SPECMOD[:,i_geom], dSPECMOD[:,i_geom,:] = self.calculate(
                i_geom,
                SPECMOD[:,i_geom],
                dSPECMOD[:,i_geom,:],
                forward_model.Measurement.VCONV[:forward_model.Measurement.NCONV[i_geom], i_geom],
                slice(ixx, ixx+self.ndegree.v + 1),
            )
            ixx += self.ndegree + 1
        
        return


