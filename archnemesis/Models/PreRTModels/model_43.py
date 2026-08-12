from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from ._base import PreRTModelBase
from ..param import StateParam, ConstParam

from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.Atmosphere_0 import Atmosphere_0
    mparam = 'the number of parameters a model has'


@dc.dataclass(slots=True)
class Model43(PreRTModelBase):
    """
        Temperature profile from double grey analytic formulation (Parmentier and Guillot (2014) and Line et al. (2013))
    """
    id: ClassVar[int] = 43

    alpha: StateParam.using(slice(0,1), 'Parameter alpha. Weighting between two streams') # noqa: F722 F821
    beta: StateParam.using(slice(1,2), 'Parameter beta. Albedo/emissivity weight.') # noqa: F722 F821
    k_ir: StateParam.using(slice(2,3), 'Parameter k_ir. Thermal IR opacity parameter. (units: cm2/g)') # noqa: F722 F821
    gammav1: StateParam.using(slice(3,4), 'Ratio of visible stream 1 opacity to thermal opacity') # noqa: F722 F821
    gammav2: StateParam.using(slice(4,5), 'Ratio of visible stream 2 opacity to thermal opacity') # noqa: F722 F821

    atm_profile_type: ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to') # noqa: F722 F821
    T_star: ConstParam.using('Star temperature (K)') # noqa: F722 F821
    R_star: ConstParam.using('Star radius (km)') # noqa: F722 F821
    sdist: ConstParam.using('Planet-star distance (km)') # noqa: F722 F821
    T_int: ConstParam.using('Internal temperature (K)') # noqa: F722 F821

    def calculate(
            self,
            atm: "Atmosphere_0",
            MakePlot=False,
        ) -> tuple["Atmosphere_0", np.ndarray]:

        def e2(xin):
            n = 100
            yl = np.array([0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,-2.58860e-08,
                        -2.58860e-08, -2.58860e-08, -2.58860e-08, -5.17719e-08,-5.17719e-08, -7.76579e-08, -7.76579e-08, -1.03544e-07,
                        -1.55316e-07, -2.07088e-07, -2.58860e-07, -3.36518e-07,-4.40062e-07, -5.43606e-07, -7.24808e-07, -9.31896e-07,
                        -1.21664e-06, -1.55316e-06, -2.01911e-06, -2.61449e-06,-3.39108e-06, -4.40064e-06, -5.66906e-06, -7.32579e-06,
                        -9.47437e-06, -1.22442e-05, -1.58166e-05, -2.04245e-05,-2.63527e-05, -3.39637e-05, -4.37754e-05, -5.64092e-05,
                        -7.26162e-05, -9.34584e-05, -0.000120179, -0.000154463,-0.000198384, -0.000254637, -0.000326597, -0.000418571,
                        -0.000536015, -0.000685872, -0.000876814, -0.00111995,-0.00142903, -0.00182161, -0.00231951, -0.00295015,
                        -0.00374779, -0.00475518, -0.00602551, -0.00762473,-0.00963471, -0.0121566, -0.0153151, -0.0192639,
                        -0.0241918, -0.0303303, -0.0379631, -0.0474377, -0.0591792,-0.0737073, -0.0916590, -0.113815, -0.141135, -0.174802,
                        -0.216279, -0.267385, -0.330392, -0.408156, -0.504287,-0.623379, -0.771314, -0.955672, -1.18628, -1.47592, -1.84134,
                        -2.30451, -2.89434, -3.64897, -4.61874, -5.87009, -7.49083,-9.59684, -12.3411, -15.9257, -20.6167, -26.7657, -34.8358])

            z = np.zeros(len(yl))
            x = np.zeros(len(yl))
            for i in range(100):
                z[i] = -10. + 12. * i / 100.
                x[i] = 10.**z[i]

            z1 = np.log10(xin)
            if z1 < -10.:
                y = 1.0
                grad = -20.0
            elif z1 > 1.89:
                y = 0.0
                grad = 0.0
            else:
                i = int((z1 + 10.0) / 0.12)
                if i == n:
                    i = n - 2

                x1 = x[i]
                x2 = x[i + 1]
                fx = (xin - x1) / (x2 - x1)
                g1 = (yl[i + 1] - yl[i]) / (x2 - x1)

                ylint = (1.0 - fx) * yl[i] + fx * yl[i + 1]
                y = 10.0**ylint
                grad = y * np.log(10.0) * g1

                if xin < 1.0e-8:
                    grad = -20.0

            return y, grad

        def calc_zeta(gamma, tau):
            c0 = 2.0 / 3.0
            c1 = c0 / gamma
            c2 = c1 / gamma
            sarg = gamma * tau
            x = c0 + c1 * (1.0 + (0.5 * sarg - 1.0) * np.exp(-sarg))
            y, grad = e2(sarg)
            zeta = x + c0 * gamma * (1.0 - 0.5 * tau**2.0) * y
            dzeta_dtau = c1 * (-(0.5 * sarg - 1.0) * gamma * np.exp(-sarg))
            dzeta_dtau += c1 * 0.5 * gamma * np.exp(-sarg)
            dzeta_dtau += c0 * gamma * (1.0 - 0.5 * tau**2.0) * grad * gamma
            dzeta_dtau -= c0 * gamma * y * tau
            dzeta_dgamma = c1 * (-(0.5 * sarg - 1.0) * np.exp(-sarg) * tau + 0.5 * tau * np.exp(-sarg))
            dzeta_dgamma -= c2 * (1.0 + (0.5 * sarg - 1.0) * np.exp(-sarg))
            c3 = c0 * (1.0 - 0.5 * tau**2.0)
            dzeta_dgamma += c3 * (gamma * grad * tau + y)
            return zeta, dzeta_dtau, dzeta_dgamma

        alpha = self.alpha.v
        beta = self.beta.v
        k_ir = self.k_ir.v
        gammav1 = self.gammav1.v
        gammav2 = self.gammav2.v
        T_star = self.T_star.v
        R_star = self.R_star.v
        sdist = self.sdist.v
        T_int = self.T_int.v

        T_eq = T_star * np.sqrt(0.5 * R_star / sdist)
        T_irr = beta * T_eq
        c1 = 3.0 / 4.0 * (T_int**4.0)
        cx = 3.0 / 4.0 * (T_irr**4.0)
        dcx_dbeta = 3.0 * (T_irr**3.0) * T_eq
        atm.calc_grav()
        G0 = atm.GRAV[0]
        T_out = np.zeros(atm.NP)
        xmap = np.zeros((5, atm.NP))
        for i in range(atm.NP):
            tau = k_ir * atm.P[i] / G0 / 10.0
            x = c1 * (2.0 / 3.0 + tau)
            zeta1, dz1_dtau, dz1_dgamma = calc_zeta(gammav1, tau)
            zeta2, dz2_dtau, dz2_dgamma = calc_zeta(gammav2, tau)
            x += cx * ((1.0 - alpha) * zeta1 + alpha * zeta2)
            T_out[i] = x**0.25
            g1 = 0.25 * x**(-0.75)
            dx_dalpha = -cx * zeta1 + cx * zeta2
            dx_dbeta = ((1.0 - alpha) * zeta1 + alpha * zeta2) * dcx_dbeta
            dx_dtau = cx * ((1.0 - alpha) * dz1_dtau + alpha * dz2_dtau) + c1
            dx_dkir = dx_dtau * tau / k_ir
            dx_dg1 = cx * (1.0 - alpha) * dz1_dgamma
            dx_dg2 = cx * alpha * dz2_dgamma
            xmap[0, i] = dx_dalpha * g1 * alpha
            xmap[1, i] = dx_dbeta * g1 * beta
            xmap[2, i] = dx_dkir * g1 * k_ir
            xmap[3, i] = dx_dg1 * g1 * gammav1
            xmap[4, i] = dx_dg2 * g1 * gammav2
        atm.edit_T(T_out)
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
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 5)
        s = f.readline().split()
        T_star = float(s[0])
        R_star = float(s[1])
        sdist = float(s[2])
        T_int = float(s[3])

        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            T_star,
            R_star,
            sdist,
            T_int,
        )

        instance.alpha.log = True
        instance.beta.log = True
        instance.k_ir.log = True
        instance.gammav1.log = True
        instance.gammav2.log = True

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
        xvals = np.zeros(5)
        xerrs = np.zeros(5)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            varparam[0],
            varparam[1],
            varparam[2],
            varparam[3],
        )

    # Use PreRTModelBase.calculate_from_subprofretg
