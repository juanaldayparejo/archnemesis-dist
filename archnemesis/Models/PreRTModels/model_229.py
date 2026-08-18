from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

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
class Model229(PreRTModelBase):
    """
        Model for representing the double-Gaussian parameterisation of the instrument lineshape
        for retrievals from the Atmospheric Chemistry Suite aboard the ExoMars Trace Gas Orbiter
    """
    id: ClassVar[int] = 229

    A0: StateParam.using(slice(0,1), 'Wavenumber offset of main at lowest wavenumber','cm-1') # noqa: F722 F821
    A1: StateParam.using(slice(1,2), 'Wavenumber offset of main at wavenumber in the middle','cm-1') # noqa: F722 F821
    A2: StateParam.using(slice(2,3), 'Wavenumber offset of main at highest wavenumber','cm-1') # noqa: F722 F821
    DELDG: StateParam.using(slice(3,4), 'Offset of the second gaussian with respect to the first one (assumed spectrally constant)','cm-1') # noqa: F722 F821
    FWHM: StateParam.using(slice(4,5), 'FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units)','cm-1') # noqa: F722 F821
    AMP1: StateParam.using(slice(5,6), 'Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber') # noqa: F722 F821
    AMP2: StateParam.using(slice(6,7), 'Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var)') # noqa: F722 F821

    def calculate(self, Measurement, MakePlot=False):
        nconv = Measurement.NCONV[0]
        vconv1 = Measurement.VCONV[0:nconv, 0]
        ng = 2

        par1 = self.A0.v
        par2 = self.A1.v
        par3 = self.A2.v
        par4 = self.DELDG.v
        par5 = self.FWHM.v
        par6 = self.AMP1.v
        par7 = self.AMP2.v

        iconvmid = int(nconv / 2.0)
        wavemax = vconv1[nconv - 1]
        wavemin = vconv1[0]
        wavemid = vconv1[iconvmid]
        offgrad1 = (par2 - par1) / (wavemid - wavemin)
        offgrad2 = (par2 - par3) / (wavemid - wavemax)
        offset = np.zeros([nconv, ng])
        for i in range(iconvmid):
            offset[i, 0] = (vconv1[i] - wavemin) * offgrad1 + par1
            offset[i, 1] = offset[i, 0] + par4
        for i in range(nconv - iconvmid):
            offset[i + iconvmid, 0] = (vconv1[i + iconvmid] - wavemax) * offgrad2 + par3
            offset[i + iconvmid, 1] = offset[i + iconvmid, 0] + par4

        fwhm = np.zeros([nconv, ng])
        fwhml = par5 / wavemin**2.0
        for i in range(nconv):
            fwhm[i, 0] = fwhml * (vconv1[i])**2.0
            fwhm[i, 1] = fwhm[i, 0]

        amp = np.zeros([nconv, ng])
        ampgrad = (par7 - par6) / (wavemax - wavemin)
        for i in range(nconv):
            amp[i, 0] = 1.0
            amp[i, 1] = (vconv1[i] - wavemin) * ampgrad + par6

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

        if MakePlot:
            fig, ([ax1, ax2, ax3]) = plt.subplots(1, 3, figsize=(12, 4))
            ix = 0
            ax1.plot(vfil[0:nfil[ix], ix], afil[0:nfil[ix], ix], linewidth=2.0)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax1.set_ylabel(r'f($\nu$)')
            ax1.set_xlim([vfil[0:nfil[ix], ix].min(), vfil[0:nfil[ix], ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()
            ix = int(nconv / 2) - 1
            ax2.plot(vfil[0:nfil[ix], ix], afil[0:nfil[ix], ix], linewidth=2.0)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax2.set_ylabel(r'f($\nu$)')
            ax2.set_xlim([vfil[0:nfil[ix], ix].min(), vfil[0:nfil[ix], ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()
            ix = nconv - 1
            ax3.plot(vfil[0:nfil[ix], ix], afil[0:nfil[ix], ix], linewidth=2.0)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax3.set_ylabel(r'f($\nu$)')
            ax3.set_xlim([vfil[0:nfil[ix], ix].min(), vfil[0:nfil[ix], ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()
            plt.tight_layout()
            plt.show()

        return Measurement


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
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 7)
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
        xvals = np.zeros(7)
        xerrs = np.zeros(7)
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
        par1 = par_vals[0]
        par2 = par_vals[1]
        par3 = par_vals[2]
        par4 = par_vals[3]
        par5 = par_vals[4]
        par6 = par_vals[5]
        par7 = par_vals[6]

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX, par1, par2, par3, par4, par5, par6, par7)

        ix = ix + int(forward_model.Variables.NXVAR[ivar])
