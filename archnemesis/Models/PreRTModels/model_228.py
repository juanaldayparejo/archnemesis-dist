from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from ._base import PreRTModelBase
from ..param import (
    StateParam, 
    #ConstParam, 
    #VarParam,
)

from archnemesis.helpers.maths_helper import ngauss
from archnemesis.enum import ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    mparam = 'the number of parameters a model has'


@dc.dataclass(slots=True)
class Model228(PreRTModelBase):
    """
        Wavelength calibration + double-Gaussian ILS model for ACS MIR
    """
    id: ClassVar[int] = 228

    V0: StateParam.using(slice(0,1), 'Wavelength/Wavenumber of the first data point') # noqa: F722 F821
    C0: StateParam.using(slice(1,2), 'Coefficient C0 for step size') # noqa: F722 F821
    C1: StateParam.using(slice(2,3), 'Coefficient C1 for step size') # noqa: F722 F821
    C2: StateParam.using(slice(3,4), 'Coefficient C2 for step size') # noqa: F722 F821
    P0: StateParam.using(slice(4,5), 'Offset of the second gaussian with respect to the first one') # noqa: F722 F821
    P1: StateParam.using(slice(5,6), 'FWHM of the main gaussian') # noqa: F722 F821
    P2: StateParam.using(slice(6,7), 'Relative amplitude at lowest wavenumber') # noqa: F722 F821
    P3: StateParam.using(slice(7,8), 'Relative amplitude at highest wavenumber') # noqa: F722 F821

    def calculate(self, Measurement, Spectroscopy, MakePlot=False):
        nconv = Measurement.NCONV[0]
        V0 = self.V0.v
        C0 = self.C0.v
        C1 = self.C1.v
        C2 = self.C2.v
        P0 = self.P0.v
        P1 = self.P1.v
        P2 = self.P2.v
        P3 = self.P3.v

        vconv1 = np.zeros(nconv)
        vconv1[0] = V0
        xx = np.linspace(0, nconv - 2, nconv - 1)
        dV = C0 + C1 * xx + C2 * (xx)**2.0
        for i in range(nconv - 1):
            vconv1[i + 1] = vconv1[i] + dV[i]
        for i in range(Measurement.NGEOM):
            Measurement.VCONV[0:Measurement.NCONV[i], i] = vconv1[:]

        ng = 2
        offset = np.zeros([nconv, ng])
        offset[:, 0] = 0.0
        offset[:, 1] = P0
        fwhm = np.zeros([nconv, ng])
        fwhml = P1 / vconv1[0]**2.0
        for i in range(nconv):
            fwhm[i, 0] = fwhml * (vconv1[i])**2.0
            fwhm[i, 1] = fwhm[i, 0]
        amp = np.zeros([nconv, ng])
        ampgrad = (P3 - P2) / (vconv1[nconv - 1] - vconv1[0])
        for i in range(nconv):
            amp[i, 0] = 1.0
            amp[i, 1] = (vconv1[i] - vconv1[0]) * ampgrad + P2

        nfil = np.zeros(nconv, dtype='int32')
        mfil1 = 200
        vfil1 = np.zeros([mfil1, nconv])
        afil1 = np.zeros([mfil1, nconv])
        for i in range(nconv):
            xlim = 0.0
            xdist = 5.0
            for j in range(ng):
                xcen = offset[i, j]
                xmin = abs(xcen - xdist * fwhm[i, j] / 2.0)
                if xmin > xlim:
                    xlim = xmin
                xmax = abs(xcen + xdist * fwhm[i, j] / 2.0)
                if xmax > xlim:
                    xlim = xmax
            xsamp = 7.0
            xhwhm = 10000.0
            for j in range(ng):
                xhwhmx = fwhm[i, j] / 2.0
                if xhwhmx < xhwhm:
                    xhwhm = xhwhmx
            deltawave = xhwhm / xsamp
            np1 = 2.0 * xlim / deltawave
            npx = int(np1) + 1
            iamp = np.zeros([ng])
            imean = np.zeros([ng])
            ifwhm = np.zeros([ng])
            fun = np.zeros([npx])
            xwave = np.linspace(vconv1[i] - deltawave * (npx - 1) / 2.0, vconv1[i] + deltawave * (npx - 1) / 2.0, npx)
            for j in range(ng):
                iamp[j] = amp[i, j]
                imean[j] = offset[i, j] + vconv1[i]
                ifwhm[j] = fwhm[i, j]
            fun = ngauss(npx, xwave, ng, iamp, imean, ifwhm)
            nfil[i] = npx
            vfil1[0:nfil[i], i] = xwave[:]
            afil1[0:nfil[i], i] = fun[:]

        mfil = nfil.max()
        vfil = np.zeros([mfil, nconv])
        afil = np.zeros([mfil, nconv])
        for i in range(nconv):
            vfil[0:nfil[i], i] = vfil1[0:nfil[i], i]
            afil[0:nfil[i], i] = afil1[0:nfil[i], i]

        Measurement.NFIL = nfil
        Measurement.VFIL = vfil
        Measurement.AFIL = afil

        return Measurement, Spectroscopy

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
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 8)
        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
        )

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
        xvals = np.zeros(8)
        xerrs = np.zeros(8)
        return cls.from_arrays(
            xvals,
            xerrs,
        )

    def calculate_from_subprofretg(
            self,
            forward_model: "ForwardModel_0",
            ix: int,
            ipar: int,
            ivar: int,
            xmap: np.ndarray,
        ) -> None:
        par_vals = self.pull_from_state_vector(forward_model.Variables.XN)
        V0 = par_vals[0]
        C0 = par_vals[1]
        C1 = par_vals[2]
        C2 = par_vals[3]
        P0 = par_vals[4]
        P1 = par_vals[5]
        P2 = par_vals[6]
        P3 = par_vals[7]

        forward_model.MeasurementX, forward_model.SpectroscopyX = self.calculate(
            forward_model.MeasurementX, forward_model.SpectroscopyX, V0, C0, C1, C2, P0, P1, P2, P3
        )

        ix = ix + int(forward_model.Variables.NXVAR[ivar])
