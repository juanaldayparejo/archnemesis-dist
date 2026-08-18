
from typing import TYPE_CHECKING, Self, IO, ClassVar, Any

import numpy as np

from ..param import (
    StateParam, 
    ConstParam, 
    VarParam,
)
from ._base import PreRTModelBase

from archnemesis.enum import WaveUnitEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.CIA_0 import CIA_0
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
    id : ClassVar[int] = 500


    amplitude : StateParam.using(slice(0,None), 'Amplitudes of each gaussian') # noqa: F722 F821
    amplitude_file : ConstParam.using(str, 'File that contains the amplitude data for this model') # noqa: F722 F821
    nbasis : ConstParam.using(int, 'Number of basis gaussians') # noqa: F722 F821
    icia : ConstParam.using(int, 'CIA pair to be modelled') # noqa: F722 F82
    correlation_length : VarParam.using(float, 'Correlation length of the gaussians, note: "distance" between gaussians is their index.') # noqa: F722 F821

    @classmethod
    def read_amplitude_file(
        cls,
        fpath : str,
    ) -> tuple[np.ndarray, np.ndarray, tuple[Any,...]]:
        with open(fpath, 'r') as f:
            nbasis, clen = cls.read_apr_entries(f, (int,float))
            
            xvals, xerrs = cls.read_apr_value_error_pairs(f, nbasis)
        
        return xvals, xerrs, (clen, nbasis)

    @classmethod
    def from_apr_file(
            cls,
            f: IO,
            varident: np.ndarray[[3], int],
            npro: int,
            ngas: int,
            ndust: int,
            nlocations: int,
            runname: str,
            sxminfac: float,
            input_file_type: ArchNemesisFileTypeEnum,
    ) -> Self:
        icia = varident[1]
        xvals, xerrs, (clen, nbasis) = cls.read_amplitude_file(f.readline().split(maxsplit=1)[0])
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            clen,
            icia,
            nbasis,
        )
        
        return instance

    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["mx","mx"],float],
            sxminfac : float,
    ):
        """
        Model 500 requires a custom covariance matrix calculation
        """
        self.push_to_covariance_matrix_with_correlation_length(
            sx,
            sxminfac,
            correlation_lengths = (0, self.correlation_length.v),
            distances = np.arange(self.nbasis.v, dtype=int),
        )

    @staticmethod
    def gaussian_basis(x, centers, width):
        """
        Computes gaussian at `x` for each `centers` and `widths`
        """
        return np.exp(-((x[:, None] - centers[None, :])**2) / (2 * width**2))

    def calculate(
            self, 
            k_cia : "CIA_0", 
            waven : np.ndarray, 
            vlo : float, 
            vhi : float,
    ) -> tuple["CIA_0", np.ndarray]:
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
        icia = self.icia.v
        nbasis = self.nbasis.v
        amplitudes = self.amplitudes.v * 1e-40 # Usure exactly why the `1e-40` factor is here.

        ilo = np.argmin(np.abs(waven-vlo))
        ihi = np.argmin(np.abs(waven-vhi))
        width = (ihi - ilo)/nbasis          # Width of the Gaussian functions
        centers = np.linspace(ilo, ihi, int(nbasis))

        x = np.arange(ilo,ihi+1)

        G = self.gaussian_basis(x, centers, width)
        gaussian_cia = G @ amplitudes

        k_cia = k_cia * 0

        k_cia[icia,:,:,ilo:ihi+1] = gaussian_cia

        xmap = np.zeros(1)
        return k_cia,xmap

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
        raise NotImplementedError


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
    ) -> None:

        if forward_model.Measurement.ISPACE == WaveUnitEnum.Wavelength_um:
            vlo = 1e4/(forward_model.SpectroscopyX.WAVE.max())
            vhi = 1e4/(forward_model.SpectroscopyX.WAVE.min())
        else:
            vlo = forward_model.SpectroscopyX.WAVE.min()
            vhi = forward_model.SpectroscopyX.WAVE.max()

        new_k_cia, _ = self.calculate(forward_model.CIA.K_CIA.copy(), forward_model.CIA.WAVEN, vlo, vhi)

        forward_model.CIA.K_CIA = new_k_cia
        forward_model.CIAX.K_CIA = new_k_cia


