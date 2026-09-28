

from typing import TYPE_CHECKING, IO, Self

import numpy as np

from ._base import PostRTModelBase

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

class Model233(PostRTModelBase):
    """
        Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

            np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.
    """
    id : int = 233
    
    
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
        """
        Aerosol opacity modelled with a variable angstrom coefficient. Applicable to transmission spectra.

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.                  
        """

        #Reading the file where the a priori parameters are stored
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
        nlevel = int(tmp[0])
        varparam[0] = nlevel
        for ilevel in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=6,dtype='float')
            a0 = float(tmp[0])   #A0
            err0 = float(tmp[1])
            a1 = float(tmp[2])   #A1
            err1 = float(tmp[3])
            a2 = float(tmp[4])   #A2
            err2 = float(tmp[5])
            x0[ix] = a0
            sx[ix,ix] = (err0)**2.
            x0[ix+1] = a1
            sx[ix+1,ix+1] = err1**2.
            x0[ix+2] = a2
            sx[ix+2,ix+2] = err2**2.
            inum[ix] = 0
            inum[ix+1] = 0    
            inum[ix+2] = 0                  
            ix = ix + 3
        return cls(ix_0, ix-ix_0)
        
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
        return cls(ix_0, ix-ix_0)
    
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

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(3), ix+(igeom+1)*(3)) for igeom in range(NGEOM))


        if forward_model.MeasurementX.NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):

                A0 = forward_model.Variables.XN[ix]
                A1 = forward_model.Variables.XN[ix+1]
                A2 = forward_model.Variables.XN[ix+2]
                
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
            A0 = forward_model.Variables.XN[ix]
            A1 = forward_model.Variables.XN[ix+1]
            A2 = forward_model.Variables.XN[ix+2]

            SPECMOD[:], dSPECMOD[:] = self.calculate(
                SPECMOD[:], 
                dSPECMOD[:],
                slice(ix,ix+3),
                forward_model.SpectroscopyX,
                A0,
                A1,
                A2
            )


