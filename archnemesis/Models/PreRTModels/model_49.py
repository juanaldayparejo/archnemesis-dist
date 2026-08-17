from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from ._base import PreRTModelBase
from ..param import (
    StateParam, 
    ConstParam, 
    #VarParam,
)

from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Atmosphere_0 import Atmosphere_0
    mparam = 'the number of parameters a model has'


@dc.dataclass(slots=True)
class Model49(PreRTModelBase):
    """
        Profile scaling model (single multiplicative factor)
    """
    id: ClassVar[int] = 49

    scale: StateParam.using(slice(0,1), 'Factor to scale reference profile by', 'NUMBER') # noqa: F722 F821
    atm_profile_type: ConstParam.using(AtmosphericProfileTypeEnum, 'Atmospheric profile type this model applies to') # noqa: F722 F821
    ref_gas: ConstParam.using(int, 'Reference gas') # noqa: F722 F821
    ref_iso: ConstParam.using(int, 'Reference isotope') # noqa: F722 F821

    def calculate(self, atm: "Atmosphere_0", ipar: int, ref_gas: int, ref_iso: int) -> tuple["Atmosphere_0", np.ndarray]:
        npar = atm.NVMR + 2 + atm.NDUST
        iref_vmr = np.where((atm.ID == ref_gas) & (atm.ISO == ref_iso))[0][0]
        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        xref[:] = atm.VMR[:, iref_vmr]
        scf = self.scale.v
        x1[:] = xref * scf
        atm.VMR[:, ipar] = x1
        xmap = np.zeros([1, npar, atm.NP])
        xmap[0, ipar, :] = xref[:]
        return atm, xmap

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
        # profgas profiso on first line, then scale & error
        s = f.readline().rsplit('!', 1)[0].split()
        profgas = int(s[0])
        profiso = int(s[1])
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 1)
        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            profgas,
            profiso,
        )
        instance.scale.log = True
        return instance

    @classmethod
    def from_bookmark(
            cls,
            variables: "Variables_0",
            varident: np.ndarray[[3], int],
            varparam: np.ndarray[["mparam"], float],
            ix: int,
            npro: int,
            ngas: int,
            ndust: int,
            nlocations: int,
        ) -> Self:
        xvals = np.zeros(1)
        xerrs = np.zeros(1)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            varparam[1],
            varparam[2],
        )

    def calculate_from_subprofretg(self, forward_model: "ForwardModel_0", ix: int, ipar: int, ivar: int, xmap: np.ndarray) -> None:
        scale_gas, scale_iso = forward_model.Variables.VARPARAM[ivar, 1:3]
        # Pull state parameters into self
        self.pull_from_state_vector(forward_model.Variables.XN)
        forward_model.AtmosphereX, xmap1 = self.calculate(forward_model.AtmosphereX, ipar, scale_gas, scale_iso)
        xmap[self.state_vector_slice, ipar, 0:forward_model.AtmosphereX.NP] = xmap1
