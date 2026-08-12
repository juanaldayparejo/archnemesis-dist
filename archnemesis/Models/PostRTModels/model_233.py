
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
class Model233(PostRTModelBase):
    """
        Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

            np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.
    """
    id : ClassVar[int] = 233
    
    coefficient_triplets : StateParam.using(slice(None), 'a0, a1, a2 coefficients') # noqa: F722 F821
    
    nlevels : VarParam[int].using('Number of coefficient triplets') # noqa: F722 F821
    
    
    @classmethod
    def read_datafile(
            cls,
            fpath : str
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any,...]]:
        
        with open(fpath, 'r') as f:
            nlevel = cls.read_apr_entries(f, (int,))
            a0,e0,a1,e1,a2,e2 = cls.read_apr_entries(f, (float,float,float,float,float,float), nlevel)
            
            xvals = np.zeros((3*nlevel,))
            xerrs = np.zeros((3*nlevel,))
            
            xvals[::3] = a0
            xvals[1::3] = a1
            xvals[2::3] = a2
            
            xerrs[::3] = e0
            xerrs[1::3] = e1
            xerrs[2::3] = e2
        
        return xvals, xerrs, nlevel
    
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
        Aerosol opacity modelled with a variable angstrom coefficient. Applicable to transmission spectra.

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.                  
        """

        datafile_path = cls.read_apr_entries(f, (str,))
        xvals, xerrs, (nlevel) = cls.read_datafile(datafile_path)
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            nlevel
        )
        
        instance.coefficient_triplets.log = False
        
        return instance
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            igeom_slice : slice,
            Spectroscopy : "Spectroscopy_0",
            A0 : float,
            A1 : float,
            A2 : float,
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        spec = np.array(SPECMOD)

        #Calculating the aerosol opacity at each wavelength
        TAU = np.exp(A0 + A1 * np.log(Spectroscopy.WAVE) + A2 * np.log(Spectroscopy.WAVE)**2.)

        #Changing the state vector based on this parameterisation
        SPECMOD *= np.exp ( -TAU )

        #Changing the rest of the gradients based on the impact of this parameterisation
        dSPECMOD *= np.exp ( -TAU )
        
        #Defining the analytical gradients for this parameterisation
        dspecmod_part = SPECMOD[:,igeom_slice]
        dspecmod_part[:,0] = spec[:] * (-TAU) * np.exp(-TAU)
        dspecmod_part[:,1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Spectroscopy.WAVE)
        dspecmod_part[:,2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Spectroscopy.WAVE)**2.

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
        Aerosol opacity modelled with a variable angstrom coefficient. Applicable to transmission spectra.

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.                  
        """
        nlevel = int(varparam[0])
        for ilevel in range(nlevel):             
            ix = ix + 3
        return cls(
            (np.zeros(nlevel*3), np.zeros(nlevel*3)),
            nlevel,
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
        Model 232. Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)
        ***************************************************************

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.

        The effect of this model takes place after the computation of the spectra in CIRSrad!
        """
        
        if int(forward_model.Variables.NXVAR[ivar]/3)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')
    
        self.pull_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(3), ix+(igeom+1)*(3)) for igeom in range(NGEOM))


        if forward_model.MeasurementX.NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):

                A0 = self.coefficient_triplets.v[3*i]
                A1 = self.coefficient_triplets.v[3*i+1]
                A2 = self.coefficient_triplets.v[3*i+2]
                
                SPECMOD[:,i], dSPECMOD[:,i] = self.calculate(
                    SPECMOD[:,i], 
                    dSPECMOD[:,i],
                    igeom_slices[i],
                    forward_model.SpectroscopyX,
                    A0,
                    A1,
                    A2
                )

        else:
            A0 = self.coefficient_triplets.v[0]
            A1 = self.coefficient_triplets.v[1]
            A2 = self.coefficient_triplets.v[2]

            SPECMOD[:], dSPECMOD[:] = self.calculate(
                SPECMOD[:], 
                dSPECMOD[:],
                slice(ix,ix+3),
                forward_model.SpectroscopyX,
                A0,
                A1,
                A2
            )


