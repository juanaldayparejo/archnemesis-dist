
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase

from archnemesis.enum import AtmosphericProfileTypeEnum
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

class Model500(PreRTModelBase):
    """
        This allows the retrieval of CIA opacity with a gaussian basis.
        Assumes a constant P/T dependence.

        .apr format expected (two lines for this variable block):
            <amplitudes_filename>
            <vlo> <vhi>
    """
    id: int = 500

    def __init__(
        self,
        state_vector_start: int,
        n_state_vector_entries: int,
        icia: int,
        nbasis: int,
        vlo: float,
        vhi: float,
        atm_profile_type: AtmosphericProfileTypeEnum = AtmosphericProfileTypeEnum.NOT_PRESENT,
    ):
        """
        Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)

        self.icia = icia
        self.nbasis = nbasis
        self.vlo = vlo
        self.vhi = vhi

        # Define the layout of the state vector for this model
        self.parameters = (
            ModelParameter(
                'amplitudes',
                slice(0, self.nbasis),
                'Amplitudes of each gaussian in the basis (stored in log-space).',
                'cm-1/amagat^2'
            ),
        )

    @classmethod
    def calculate(cls, k_cia: np.ndarray, waven: np.ndarray, icia: int, vlo: float, vhi: float, nbasis: int, amplitudes: np.ndarray) -> np.ndarray:
        """
        Calculates the new CIA profile based on Gaussian basis functions.
        """
        # Find the indices in the wavenumber array closest to the bounds.
        ilo = np.argmin(np.abs(waven - vlo))
        ihi = np.argmin(np.abs(waven - vhi))

        # --- Validation Block ---
        if ihi <= ilo:
            _lgr.error(
                f"Model 500: Invalid wavenumber range for CIA pair {icia}. "
                f"Resulted in an empty slice (ilo={ilo}, ihi={ihi}).\n"
                f"  - Provided Range: vlo={vlo:.2f}, vhi={vhi:.2f}\n"
                f"  - CIA Data Range: waven.min()={waven.min():.2f}, waven.max()={waven.max():.2f}\n"
                "  - Check your .apr file to ensure vlo < vhi and the range overlaps the CIA data."
            )
            raise ValueError("Model 500 failed due to invalid wavenumber range.")

        if icia >= k_cia.shape[0]:
            raise IndexError(
                f"Model 500: CIA pair index {icia} is out of bounds. "
                f"The CIA data has only {k_cia.shape[0]} pairs (indexed 0 to {k_cia.shape[0]-1})."
            )
        # --- End Validation ---

        width = (ihi - ilo) / nbasis
        centers = np.linspace(ilo, ihi, int(nbasis))

        def gaussian_basis(x, centers, width):
            return np.exp(-((x[:, None] - centers[None, :])**2) / (2 * width**2))

        x = np.arange(ilo, ihi + 1)
        G = gaussian_basis(x, centers, width)
        gaussian_cia = G @ amplitudes

        new_k_cia = k_cia.copy()
        new_k_cia[icia, :, :, ilo:ihi+1] = gaussian_cia

        return new_k_cia

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

        # The CIA pair index is the second element of the varident array.
        icia = int(varident[1])

        # Line 1: amplitudes filename
        s = f.readline().split()
        with open(s[0], 'r') as amp_f:
            # Line 2: CIA bounds (vlo vhi)
            bounds = f.readline().split()
            if len(bounds) < 2:
                raise ValueError("Model500 .apr expects a second line with 'vlo vhi' after the filename.")
            vlo = float(bounds[0])
            vhi = float(bounds[1])

            # amplitudes file: first line -> nbasis, correlation length
            tmp = np.fromfile(amp_f, sep=' ', count=2, dtype='float')
            nbasis = int(tmp[0])
            clen = float(tmp[1])

            # Read a priori values for basis function amplitudes
            for j in range(nbasis):
                tmp_amp = np.fromfile(amp_f, sep=' ', count=2, dtype='float')
                amp_val, amp_err = float(tmp_amp[0]), float(tmp_amp[1])

                x0[ix + j] = np.log(amp_val)
                lx[ix + j] = 1
                sx[ix + j, ix + j] = (amp_err / amp_val)**2.
                inum[ix + j] = 1

            # Calculate covariance between basis function amplitudes
            for j in range(nbasis):
                for k in range(nbasis):
                    deli = j - k
                    arg = abs(deli / clen)
                    xfac = np.exp(-arg)
                    if xfac >= sxminfac:
                        sx[ix + j, ix + k] = np.sqrt(sx[ix + j, ix + j] * sx[ix + k, ix + k]) * xfac
                        sx[ix + k, ix + j] = sx[ix + j, ix + k]

        ix_end = ix + nbasis

        # Persist the definition needed by .pre/.raw readers, which only
        # retain VARIDENT, VARPARAM, and the state vector, not the .apr files.
        varparam[0:3] = (nbasis, vlo, vhi)

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification"

        return cls(
            state_vector_start=ix_0,
            n_state_vector_entries=ix_end - ix_0,
            icia=icia,
            nbasis=nbasis,
            vlo=vlo,
            vhi=vhi,
            atm_profile_type=model_classification[1]
        )

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
        vlo, vhi = float(varparam[1]), float(varparam[2])
        if nbasis < 1 or not np.isfinite([vlo, vhi]).all() or vhi <= vlo:
            raise ValueError(
                'Model500 bookmark is missing a valid basis count or spectral bounds; '
                'recreate it from the .apr file.'
            )
        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        return cls(ix_0, nbasis, int(varident[1]), nbasis, vlo, vhi, model_classification[1])


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:

        # Get the amplitudes from the state vector. This handles log-space values.
        (amplitudes,) = self.get_parameter_values_from_state_vector(
            forward_model.Variables.XN, forward_model.Variables.LX
        )

        # A very small scaling factor was present in the original code.
        # This might be better handled in the a priori data itself.
        amplitudes = np.atleast_1d(amplitudes) * 1e-40

        # Call the calculation method using the stored instance variables
        new_k_cia = self.calculate(
            k_cia=forward_model.CIA.K_CIA,
            waven=forward_model.CIA.WAVEN,
            icia=self.icia,
            vlo=self.vlo,
            vhi=self.vhi,
            nbasis=self.nbasis,
            amplitudes=amplitudes
        )

        # Update the forward model's CIA data
        forward_model.CIA.K_CIA = new_k_cia

        # Update the working copy of the CIA data as well
        if hasattr(forward_model, 'CIAX') and forward_model.CIAX is not None:
            forward_model.CIAX.K_CIA = new_k_cia

        # Gradients (xmap) are not calculated by this model.
        return
