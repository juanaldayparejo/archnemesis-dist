
from typing import TYPE_CHECKING, Self, IO, Any

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

from archnemesis.Scatter_0 import kk_new_sub

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

class Model444(PreRTModelBase):
    """
        Allows for retrieval of the particle size distribution and imaginary refractive index.
    """
    
    id: int = 444

    def __init__(
        self,
        state_vector_start: int,
        n_state_vector_entries: int,
        haze_params: dict[str, Any],
        aerosol_species_index: int,
        scattering_type_id: int,
    ):
        """
        Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)

        # State-vector layout: [ ln(a), ln(b), ln(k_im[0]), ln(k_im[1]), ... ]
        self.parameters = (
            ModelParameter(
                "particle_size_distribution_params",
                slice(0, 2),
                "Values that define the particle size distribution (stored ln-space).",
            ),
            ModelParameter(
                "imaginary_ref_idx",
                slice(2, None),
                "Imaginary refractive index samples across the haze file grid (stored ln-space).",
            ),
        )

        self.haze_params = haze_params
        self.aerosol_species_idx = aerosol_species_index
        self.scattering_type_id = scattering_type_id

    @staticmethod
    def _get_darkening_for_species(Scatter: "Scatter_0", idust: int) -> float:
        """
        Robustly read the darkening factor for species idust; default to 1.0.
        Does not modify REFIND_IM; only returns the factor.
        """
        try:
            val = getattr(Scatter, "DARKENING_PARAMETER", None)
            if val is None:
                return 1.0
            # Accept list/np.ndarray; guard for short arrays
            if np.ndim(val) == 0:
                return float(val)
            if idust < len(val):
                return float(val[idust])
            return 1.0
        except Exception:
            return 1.0

    @classmethod
    def calculate(
        cls,
        Scatter: "Scatter_0",
        idust: int,
        iscat: int,
        xprof: np.ndarray[["nparam"], float],
        haze_params: dict[str, Any],
    ) -> "Scatter_0":
        """
        FUNCTION NAME : model444()

        DESCRIPTION :
            NEMESIS model 444. Retrieves particle size distribution parameters and
            an imaginary refractive index spectrum. Applies a darkening multiplier
            from Scatter.DARKENING_PARAMETER[idust] (default 1.0).

        INPUTS :
            Scatter    :: Scattering container
            idust      :: Index of the aerosol distribution (0 .. NDUST-1)
            iscat      :: Flag indicating the particle size distribution
            xprof      :: [ ln(a), ln(b), ln(k_im[0..]) ]
            haze_params:: Haze constants read from 444 file (WAVE, NREAL, WAVE_REF, WAVE_NORM)

        OUTPUTS :
            Scatter :: Updated Scatter class
        """
        _lgr.debug(f"{idust=} {iscat=} {xprof=} {type(xprof)=}")
        for key in ("WAVE", "NREAL", "WAVE_REF", "WAVE_NORM"):
            _lgr.debug(f"haze_params[{key}] : {type(haze_params[key])} = {haze_params[key]}")

        # Particle size distribution parameters (stored in ln-space)
        a = float(np.exp(xprof[0]))
        b = float(np.exp(xprof[1]))
        if iscat == 1:
            pars = (a, b, (1 - 3 * b) / b)
        elif iscat == 2:
            pars = (a, b, 0.0)
        elif iscat == 4:
            pars = (a, 0.0, 0.0)
        else:
            _lgr.warning(f"ISCAT = {iscat} not implemented for model 444 yet! Defaulting to iscat = 1.")
            pars = (a, b, (1 - 3 * b) / b)

        # Set wavelength grid and base (unscaled) imaginary refractive index
        Scatter.WAVER = haze_params["WAVE"]
        k_im_base = np.exp(xprof[2:])  # may be length-1 scalar or spectrum samples

        # Obtain darkening multiplier for this species (defaults to 1.0 if absent)
        darkening = cls._get_darkening_for_species(Scatter, idust)

        # Apply darkening to the imaginary refractive index definition
        if np.size(k_im_base) == 1:
            # Broadcast to the haze wavelength grid after scaling
            k_im = float(darkening) * float(k_im_base) * np.ones_like(Scatter.WAVER, dtype=float)
        else:
            k_im = float(darkening) * np.array(k_im_base, dtype=float)

        Scatter.REFIND_IM = k_im

        # Real refractive index via KK transform anchored at (WAVE_REF, NREAL)
        reference_nreal = float(haze_params["NREAL"])
        reference_wave = float(haze_params["WAVE_REF"])
        normalising_wave = float(haze_params["WAVE_NORM"])

        Scatter.REFIND_REAL = kk_new_sub(
            np.array(Scatter.WAVER, dtype=float),
            np.array(Scatter.REFIND_IM, dtype=float),
            reference_wave,
            reference_nreal,
        )

        # Build phase function / optical properties for this species
        Scatter.makephase(idust, iscat, pars)

        # Normalise extinction and scattering at the specified normalising wavelength
        xextnorm = np.interp(normalising_wave, Scatter.WAVE, Scatter.KEXT[:, idust])
        Scatter.KEXT[:, idust] = Scatter.KEXT[:, idust] / xextnorm
        Scatter.KSCA[:, idust] = Scatter.KSCA[:, idust] / xextnorm

        _lgr.debug(
            f"Model444: applied darkening={darkening:.6g}; normalised at {normalising_wave}"
        )
        return Scatter

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
    ) -> "Model444":
        """
        Read the 444 haze file and pack state vector with ln-PSD params and ln(k_im) samples.
        Preserves legacy covariance-building behaviour including optional correlation length.
        """
        ix_0 = ix

        # Read haze file path
        s = f.readline().split()
        haze_f = open(s[0], "r")

        haze_waves = []

        # First two lines: ln(a), ln(b), with uncertainties
        for _ in range(2):
            line = haze_f.readline().split()
            xai, xa_erri = line[:2]
            x0[ix] = np.log(float(xai))
            lx[ix] = 1
            sx[ix, ix] = (float(xa_erri) / float(xai)) ** 2.0
            ix += 1

        # Meta and grid
        nwave, clen = haze_f.readline().split("!")[0].split()
        vref, nreal_ref = haze_f.readline().split("!")[0].split()
        v_od_norm = haze_f.readline().split("!")[0]

        # k_im samples (ln) with uncertainties
        for _ in range(int(nwave)):
            line = haze_f.readline().split()
            v, xai, xa_erri = line[:3]
            x0[ix] = np.log(float(xai))
            lx[ix] = 1
            sx[ix, ix] = (float(xa_erri) / float(xai)) ** 2.0
            ix += 1
            haze_waves.append(float(v))
            if float(clen) < 0:
                break

        aerosol_species_idx = int(varident[1]) - 1

        haze_params = dict()
        haze_params["NX"] = 2 + len(haze_waves)
        haze_params["WAVE"] = haze_waves
        haze_params["NREAL"] = float(nreal_ref)
        haze_params["WAVE_REF"] = float(vref)
        haze_params["WAVE_NORM"] = float(v_od_norm)

        varparam[0] = 2 + len(haze_waves)
        varparam[1] = float(clen)
        varparam[2] = float(vref)
        varparam[3] = float(nreal_ref)
        varparam[4] = float(v_od_norm)

        # Optional spectral correlation for ln(k_im)
        if float(clen) > 0:
            # Start of the ln(k_im) block within this model's parameters
            start = ix - int(nwave)
            for j in range(int(nwave)):
                for k in range(int(nwave)):
                    delv = haze_waves[k] - haze_waves[j]
                    arg = abs(delv / float(clen))
                    xfac = np.exp(-arg)
                    if xfac >= sxminfac:
                        sx[start + j, start + k] = np.sqrt(sx[start + j, start + j] * sx[start + k, start + k]) * xfac
                        sx[start + k, start + j] = sx[start + j, start + k]

        scattering_type_id = 1  # Future: allow this to be set from input

        return cls(ix_0, ix - ix_0, haze_params, aerosol_species_idx, scattering_type_id)

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
        #******** model for retrieving an aerosol particle size distribution and imaginary refractive index spectrum
        _lgr.warn(f"{cls.__name__}.from_bookmark(...) only sets model parameters that have been stored in `varident`, `varparam`. Therefore it cannot set `haze_params['WAVE']` at the moment as those values are in an external file whose name is not stored in those locations. Use with caution.")
        
        #haze_waves = []
        for j in range(2):
            ix = ix + 1

        nwave = varparam[0] - 2
        #clen = varparam[1]
        vref = varparam[2]
        nreal_ref = varparam[3]
        v_od_norm = varparam[4]
        
        haze_params = dict()
        haze_params['NX'] = nwave
        #haze_params['WAVE'] = haze_waves    !This needs to be fixed!
        haze_params['NREAL'] = float(nreal_ref)
        haze_params['WAVE_REF'] = float(vref)
        haze_params['WAVE_NORM'] = float(v_od_norm)

        for j in range(int(nwave)):
            ix = ix + 1

        aerosol_species_idx = varident[1]-1
        scattering_type_id = 1 # Should add a way to alter this value from the input files.

        return cls(ix_0, ix-ix_0, haze_params, aerosol_species_idx, scattering_type_id)


    def calculate_from_subprofretg(
        self,
        forward_model: "ForwardModel_0",
        ix: int,
        ipar: int,
        ivar: int,
        xmap: np.ndarray,
    ) -> None:
        """
        Apply this model's effect to Scatter, consuming any darkening already placed
        by Model501 (or defaulting to 1.0 if absent).
        """
        forward_model.ScatterX = self.calculate(
            forward_model.ScatterX,
            self.aerosol_species_idx,
            self.scattering_type_id,
            self.get_state_vector_slice(forward_model.Variables.XN),
            self.haze_params,
        )
