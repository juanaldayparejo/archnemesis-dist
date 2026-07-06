
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase

from archnemesis.enum import WaveUnitEnum

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

class Model500(PreRTModelBase):
    """
        This allows the retrieval of CIA opacity with a gaussian basis.
        Assumes a constant P/T dependence.
    """
    id : int = 500


    @classmethod
    def calculate(cls, k_cia, waven, icia, vlo, vhi, nbasis, amplitudes):
        """
            FUNCTION NAME : model500()

            DESCRIPTION :

                Function defining the model parameterisation 500.
                This allows the retrieval of CIA opacity with a gaussian basis.
                Assumes a constant P/T dependence.

            INPUTS :

                cia :: CIA class

                icia :: CIA pair to be modelled

                vlo :: Lower wavenumber bound

                vhi :: Upper wavenumber bound

                nbasis :: Number of gaussians in the basis

                amplitudes :: Amplitudes of each gaussian


            OUTPUTS :

                cia :: Updated CIA class
                xmap :: Gradient (not implemented)

            CALLING SEQUENCE:

                cia,xmap = model500(cia, icia, nbasis, amplitudes)

            MODIFICATION HISTORY : Joe Penn (14/01/25)

        """

        ilo = np.argmin(np.abs(waven-vlo))
        ihi = np.argmin(np.abs(waven-vhi))
        width = (ihi - ilo)/nbasis          # Width of the Gaussian functions
        centers = np.linspace(ilo, ihi, int(nbasis))

        def gaussian_basis(x, centers, width):
            return np.exp(-((x[:, None] - centers[None, :])**2) / (2 * width**2))

        x = np.arange(ilo,ihi+1)

        G = gaussian_basis(x, centers, width)
        gaussian_cia = G @ amplitudes

        k_cia = k_cia * 0

        k_cia[icia,:,:,ilo:ihi+1] = gaussian_cia

        xmap = np.zeros(1)
        return k_cia,xmap


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

        s = f.readline().split()
        amp_f = open(s[0],'r')

        tmp = np.fromfile(amp_f,sep=' ',count=2,dtype='float')

        nbasis = int(tmp[0])
        clen = float(tmp[1])

        amp = np.zeros([nbasis])
        eamp = np.zeros([nbasis])

        for j in range(nbasis):
            tmp = np.fromfile(amp_f,sep=' ',count=2,dtype='float')
            amp[j] = float(tmp[0])
            eamp[j] = float(tmp[1])

            lx[ix+j] = 1
            x0[ix+j] = np.log(amp[j])
            sx[ix+j,ix+j] = ( eamp[j]/amp[j]  )**2.

        for j in range(nbasis):
            for k in range(nbasis):

                deli = j-k
                arg = abs(deli/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        varparam[0] = nbasis
        ix = ix + nbasis

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
        nbasis = int(varparam[0])
        ix = ix + nbasis

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:

        icia = forward_model.Variables.VARIDENT[ivar,1]

        if forward_model.Measurement.ISPACE == WaveUnitEnum.Wavelength_um:
            vlo = 1e4/(forward_model.SpectroscopyX.WAVE.max())
            vhi = 1e4/(forward_model.SpectroscopyX.WAVE.min())
        else:
            vlo = forward_model.SpectroscopyX.WAVE.min()
            vhi = forward_model.SpectroscopyX.WAVE.max()

        nbasis = forward_model.Variables.VARPARAM[ivar,0]
        amplitudes = np.exp(forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]])*1e-40

        new_k_cia, xmap1 = self.calculate(forward_model.CIA.K_CIA.copy(), forward_model.CIA.WAVEN, icia, vlo, vhi, nbasis, amplitudes)

        forward_model.CIA.K_CIA = new_k_cia
        forward_model.CIAX.K_CIA = new_k_cia

        ix = ix + forward_model.Variables.NXVAR[ivar]


