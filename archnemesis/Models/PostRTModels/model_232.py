

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
    from archnemesis.Spectroscopy_0 import Spectroscopy_0

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
class Model232(PostRTModelBase):
    """
        Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
    """
    id : ClassVar[int] = 232
    
    opacity_coeff_pairs : StateParam.using(slice(None), 'Opacity and angstrom coefficient pairs level at `wavenorm`') # noqa: F722 F821

    n_levels : VarParam[int].using('Number of levels') # noqa: F722 F821
    wavenorm : VarParam[float].using('Wavenumber of normalisation') # noqa: F722 F821
    
    @classmethod
    def read_taufile(
            cls,
            fpath : str
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any,...]]:
        with open(fpath, 'r') as f:
            n_levels = cls.read_apr_entries(f, (int,))
            xvals = np.zeros((2*n_levels,))
            xerrs = np.zeros((2*n_levels,))
            
            opacity, opacity_err, ang_coeff, ang_coeff_err = cls.read_apr_entries(f, (float,float,float,float), n_levels)
            xvals[::2] = opacity
            xvals[1::2] = ang_coeff
            xerrs[::2] = opacity_err
            xerrs[1::2] = ang_coeff_err
            
        return xvals, xerrs, (n_levels,)
                
    
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
        """
        Continuum addition to transmission spectra using the Angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """
        
        wavenorm = cls.read_apr_entries(f, (float,))
        taufile_path = cls.read_apr_entries(f, (str,))
        
        xvals, xerrs, (n_levels,) = cls.read_taufile(taufile_path)
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            n_levels,
            wavenorm
        )
        
        instance.opacity_coeff_pairs.log = False
        
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            igeom_slice : slice,
            Spectroscopy : "Spectroscopy_0",
            TAU0 : float,
            ALPHA : float,
            WAVE0 : float,
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        spec = np.array(SPECMOD)
        factor = np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) )

        #Changing the state vector based on this parameterisation
        SPECMOD *= factor

        #Changing the rest of the gradients based on the impact of this parameterisation
        dSPECMOD *= factor[:,None]

        #Defining the analytical gradients for this parameterisation
        dspecmod_part = SPECMOD[:,igeom_slice]
        dspecmod_part[:,0] = spec[:] * ( -((Spectroscopy.WAVE/WAVE0)**(-ALPHA)) * np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) ) )
        dspecmod_part[:,1] = spec[:] * TAU0 * np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) ) * np.log(Spectroscopy.WAVE/WAVE0) * (Spectroscopy.WAVE/WAVE0)**(-ALPHA)

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
        """
        Continuum addition to transmission spectra using the Angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """
        nlevel = int(varparam[0])
        wavenorm = float(varparam[1])
        return cls(
            (np.zeros(nlevel*2), np.zeros(nlevel*2)),
            nlevel,
            wavenorm
        )
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 232. Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """

        #The effect of this model takes place after the computation of the spectra in CIRSrad!
        if int(forward_model.Variables.NXVAR[ivar]/2)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

        self.pull_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(2), ix+(igeom+1)*(2)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))

        if NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):
                TAU0 = self.opacity_coeff_pairs.v[2*i]
                ALPHA = self.opacity_coeff_pairs.v[2*i+1]
                WAVE0 = self.wavenorm.v
                
                SPECMOD[:,i], dSPECMOD[:,i] = self.calculate(
                    SPECMOD[:,i], 
                    dSPECMOD[:,i], 
                    igeom_slices[i], 
                    forward_model.SpectroscopyX,
                    TAU0,
                    ALPHA,
                    WAVE0
                )

        else:
            TAU0 = self.opacity_coeff_pairs.v[0]
            ALPHA = self.opacity_coeff_pairs.v[1]
            WAVE0 = self.wavenorm.v
            
            SPECMOD[:], dSPECMOD[:] = self.calculate(
                SPECMOD[:], 
                dSPECMOD[:], 
                igeom_slices[0], 
                forward_model.SpectroscopyX,
                TAU0,
                ALPHA,
                WAVE0
            )

