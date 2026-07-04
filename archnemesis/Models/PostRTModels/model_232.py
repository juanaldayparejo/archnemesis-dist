

from typing import TYPE_CHECKING, IO, Self

import numpy as np

from ._base import PostRTModelBase

import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.DEBUG)
_lgr.setLevel(logging.INFO)

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

class Model232(PostRTModelBase):
    """
        Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
    """
    id : int = 232
    
    
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
        Continuum addition to transmission spectra using the Angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """
        s = f.readline().split()
        wavenorm = float(s[0])                    

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
        nlevel = int(tmp[0])
        varparam[0] = nlevel
        varparam[1] = wavenorm
        for ilevel in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=4,dtype='float')
            r0 = float(tmp[0])   #Opacity level at wavenorm
            err0 = float(tmp[1])
            r1 = float(tmp[2])   #Angstrom coefficient
            err1 = float(tmp[3])
            x0[ix] = r0
            sx[ix,ix] = (err0)**2.
            x0[ix+1] = r1
            sx[ix+1,ix+1] = err1**2.
            inum[ix] = 0
            inum[ix+1] = 0                        
            ix = ix + 2
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
        Continuum addition to transmission spectra using the Angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """
        nlevel = int(varparam[0])
        wavenorm = float(varparam[1])
        for ilevel in range(nlevel):                
            ix = ix + 2
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
        Model 232. Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """

        #The effect of this model takes place after the computation of the spectra in CIRSrad!
        if int(forward_model.Variables.NXVAR[ivar]/2)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(2), ix+(igeom+1)*(2)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))

        if NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):
                TAU0 = forward_model.Variables.XN[ix]
                ALPHA = forward_model.Variables.XN[ix+1]
                WAVE0 = forward_model.Variables.VARPARAM[ivar,1]
                
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
            TAU0 = forward_model.Variables.XN[ix]
            ALPHA = forward_model.Variables.XN[ix+1]
            WAVE0 = forward_model.Variables.VARPARAM[ivar,1]
            
            _lgr.warning(f'It looks like there is no calculation for NGEOM=1 for model id = {self.id}')


