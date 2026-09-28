from typing import TYPE_CHECKING, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter
from ..log import _lgr

if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Scatter_0 import Scatter_0

    # Dimension labels used by the array annotations, as in the other models.
    nparam = 'the number of parameters a model has'
    mparam = 'the number of parameters a model has'
    mx = 'number of elements in state vector'

class Model501(PreRTModelBase):
    """
    Multiplier on the imaginary refractive index spectrum of an aerosol species.
    Useful for brightening/darkening a pre-fitted nimag spectrum with model 444.

    Design:
      - State vector holds ln(darkening_parameter), i.e. positive multiplier via log-flag.
      - During subprofretg, writes exp(param) into `Scatter.DARKENING_PARAMETER[idust]`.

    Parameters
    ----------
    darkening_parameter : dimensionless (stored ln in state vector)
        Multiplier applied to the imaginary refractive index spectrum for the
        specified aerosol species.
    """

    id: int = 501

    def __init__(
        self,
        state_vector_start: int,
        n_state_vector_entries: int,
        aerosol_species_index: int,
    ):
        super().__init__(state_vector_start, n_state_vector_entries)
        self.parameters = (
            ModelParameter(
                "darkening_parameter",
                slice(0, 1),
                "Multiplier applied to the imaginary refractive index spectrum "
                "for the specified aerosol species (stored in log-space).",
            ),
        )
        self.aerosol_species_idx = aerosol_species_index

    @classmethod
    def calculate(
        cls,
        Scatter: "Scatter_0",
        idust: int,
        xparam: np.ndarray[["nparam"], float],
    ) -> "Scatter_0":
        """
        Set the darkening multiplier used by downstream scattering models.

        Inputs
        ------
        Scatter : Scatter_0
        idust   : int
            Aerosol species index to be affected.
        xparam  : array-like
            [ ln(darkening_parameter) ]
        """
        darkening = float(np.exp(xparam[0]))

        # Ensure the attribute exists and is long enough; fill with ones by default.
        if not hasattr(Scatter, "DARKENING_PARAMETER") or Scatter.DARKENING_PARAMETER is None:
            # Try to infer NDUST from existing arrays; fall back to at least idust+1
            ndust = None
            if hasattr(Scatter, "KEXT") and Scatter.KEXT is not None and hasattr(Scatter.KEXT, "shape"):
                if len(Scatter.KEXT.shape) >= 2:
                    ndust = int(Scatter.KEXT.shape[1])
            if ndust is None:
                ndust = int(idust + 1)
            Scatter.DARKENING_PARAMETER = np.ones(ndust, dtype=float)

        # Extend if the array is too short
        if idust >= len(Scatter.DARKENING_PARAMETER):
            extra = idust + 1 - len(Scatter.DARKENING_PARAMETER)
            Scatter.DARKENING_PARAMETER = np.concatenate(
                [Scatter.DARKENING_PARAMETER, np.ones(extra, dtype=float)]
            )

        Scatter.DARKENING_PARAMETER[idust] = darkening
        _lgr.debug(f"Model501: set DARKENING_PARAMETER[{idust}] = {darkening:.6g}")

        return Scatter

    @classmethod
    def from_bookmark(
        cls, variables, varident, varparam, ix, npro, ngas, ndust, nlocations,
    ) -> "Model501":
        return cls(ix, 1, int(varident[1]) - 1)

    @classmethod
    def from_apr_to_state_vector(
        cls,
        variables: "Variables_0",
        f: IO,
        varident: np.ndarray[[3], int],
        varparam: np.ndarray[["mparam"], float],
        ix: int,
        lx: np.ndarray[["mx"], int],
        x0: np.ndarray[["mx"], float],
        sx: np.ndarray[["mx", "mx"], float],
        inum: np.ndarray[["mx"], int],
        npro: int,
        ngas: int,
        ndust: int,
        nlocations: int,
        runname: str,
        sxminfac: float,
    ) -> "Model501":
        """
        Read apriori for model 501 and pack state vector / covariance.
        Format in .apr file (one line):
            scaling_factor  uncertainty

        Behaviour:
          - scaling_factor must be > 0
          - store ln(scaling_factor) in x0[ix], set lx[ix]=1, variance=(unc/scale)^2
        """
        ix_0 = ix

        vals = np.fromfile(f, sep=" ", count=2, dtype=float)
        if vals.size != 2:
            raise ValueError("Model 501 expects two floats: <scaling_factor> <uncertainty>")

        xfac = float(vals[0])
        err = float(vals[1])
        if not (xfac > 0.0):
            raise ValueError("Model 501: scaling factor must be > 0")

        x0[ix] = np.log(xfac)
        lx[ix] = 1
        sx[ix, ix] = (err / xfac) ** 2.0
        ix += 1

        aerosol_species_idx = int(varident[1]) - 1

        return cls(ix_0, ix - ix_0, aerosol_species_idx)

    def calculate_from_subprofretg(
        self,
        forward_model: "ForwardModel_0",
        ix: int,
        ipar: int,
        ivar: int,
        xmap: np.ndarray,
    ) -> None:
        """
        Push the darkening multiplier onto Scatter for the appropriate species.
        """
        forward_model.ScatterX = self.calculate(
            forward_model.ScatterX,
            self.aerosol_species_idx,
            self.get_state_vector_slice(forward_model.Variables.XN),
        )
