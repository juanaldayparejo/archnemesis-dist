
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Scatter_0 import Scatter_0

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

class Model887(PreRTModelBase):
    """
        In this model, the cross-section spectrum of IDUST is changed given the parameters in 
        the state vector
    """
    id : int = 887


    @classmethod
    def calculate(cls, Scatter,xsc,idust,MakePlot=False):

        """
            FUNCTION NAME : model887()

            DESCRIPTION :

                Function defining the model parameterisation 887 in NEMESIS.
                In this model, the cross-section spectrum of IDUST is changed given the parameters in 
                the state vector

            INPUTS :

                Scatter :: Python class defining the spectral properties of aerosols in the atmosphere
                xsc :: New cross-section spectrum of aerosol IDUST
                idust :: Index of the aerosol to be changed (from 0 to NDUST-1)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model887(Scatter,xsc,idust)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        if len(xsc)!=Scatter.NWAVE:
            raise ValueError('error in model 887 :: Cross-section array must be defined at the same wavelengths as in .xsc')
        else:
            kext = np.zeros([Scatter.NWAVE,Scatter.DUST])
            kext[:,:] = Scatter.KEXT
            kext[:,idust] = xsc[:]
            Scatter.KEXT = kext

        if MakePlot==True:
            fig,ax1=plt.subplots(1,1,figsize=(10,3))
            ax1.semilogy(Scatter.WAVE,Scatter.KEXT[:,idust])
            ax1.grid()
            if Scatter.ISPACE==1:
                ax1.set_xlabel(r'Wavelength ($\mu$m)')
            else:
                ax1.set_xlabel(r'Wavenumber (cm$^{-1}$')
            plt.tight_layout()
            plt.show()


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
        #******** Cloud x-section spectrum

        #Read in number of points, cloud id, and correlation between elements.
        s = f.readline().split()
        nwv = int(s[0]) #number of spectral points (must be the same as in .xsc)
        icloud = int(s[1])  #aerosol ID
        clen = float(s[2])  #Correlation length (in wavelengths/wavenumbers)

        varparam[0] = nwv
        varparam[1] = icloud

        #Read the wavelengths and the extinction cross-section value and error
        wv = np.zeros(nwv)
        xsc = np.zeros(nwv)
        err = np.zeros(nwv)
        for iw in range(nwv):
            s = f.readline().split()
            wv[iw] = float(s[0])
            xsc[iw] = float(s[1])
            err[iw] = float(s[2])
            if xsc[iw]<=0.0:
                raise ValueError('error in read_apr :: Cross-section in model 887 must be greater than 0')

        #It is important to check that the wavelengths in .apr and in .xsc are the same
        Aero0 = Scatter_0()
        Aero0.read_xsc(runname)
        for iw in range(Aero0.NWAVE):
            if (wv[iw]-Aero0.WAVE[iw])>0.01:
                raise ValueError('error in read_apr :: Number of wavelengths in model 887 must be the same as in .xsc')

        #Including the parameters in state vector and covariance matrix
        for j in range(nwv):
            x0[ix+j] = np.log(xsc[j])
            lx[ix+j] = 1
            inum[ix+j] = 1
            sx[ix+j,ix+j] = (err[j]/xsc[j])**2.

        for j in range(nwv):
            for k in range(nwv):
                delv = wv[j] - wv[k]
                arg = abs(delv/clen)
                xfac = np.exp(-arg)
                if xfac>0.001:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        #jxsc = ix

        ix = ix + nwv

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
        #******** Cloud x-section spectrum
        nwv = varparam[0]
        #icloud = varparam[1]
        ix = ix + nwv

        return cls(ix_0, ix-ix_0)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError


