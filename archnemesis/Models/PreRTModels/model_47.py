
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..param import (
    StateParam, 
    ConstParam, 
    #VarParam,
)

import archnemesis.Data.constants as const

from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Atmosphere_0 import Atmosphere_0

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




@dc.dataclass(slots=True)
class Model47(PreRTModelBase):
    """
        Profile is represented by a Gaussian with a specified optical thickness centred
        at a variable pressure level plus a variable FWHM (log press).
    """
    id : ClassVar[int] = 47

    
    tau   : StateParam.using(slice(0,1), 'Integrated optical thickness', 'ln(RATIO)') # noqa: F722 F821
    p_ref : StateParam.using(slice(1,2), 'Mean pressure of the cloud', 'atm') # noqa: F722 F821
    fwhm  : StateParam.using(slice(2,3), 'Full-width-half-maximum of the log-Gaussian', 'atm') # noqa: F722 F821
    
    atm_profile_type : ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to') # noqa: F722 F821
    
    @classmethod
    def from_apr_file(
            cls,
            f : IO,
            varident : np.ndarray[[3],int],
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
            runname : str,
            sxminfac : float,
            input_file_type : ArchNemesisFileTypeEnum,
    ) -> Self:
    
        xvals, xerrs = cls.read_apr_value_error_pairs(f, len(cls._stateparam_names))
    
        instance = cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust)
        )
        
        if varident[0] == 0:
            instance.tau.log = False
        
        assert instance.atm_profile_type.v == AtmosphericProfileTypeEnum.AEROSOL_DENSITY, \
            f"{cls.__name__}[id={instance.id}] is only valid for aerosol profiles"
        
        return instance
        

    def calculate(
            self, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            MakePlot=False
    ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model47()

            DESCRIPTION :

                Profile is represented by a Gaussian with a specified optical thickness centred
                at a variable pressure level plus a variable FWHM (log press).

            INPUTS :

                atm :: Python class defining the atmosphere
                
                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                tau :: Integrated optical thickness.

                pref :: Mean pressure (atm) of the cloud.

                fwhm :: FWHM of the log-Gaussian.

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(mparam,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model47(atm, atm_profile_type, atm_profile_idx, tau, pref, fwhm)

            MODIFICATION HISTORY : Joe Penn (08/10/2024)
        """
        
        tau = self.tau.v
        pref = self.p_ref.v
        fwhm = self.fwhm.v
        _lgr.debug(f'{atm_profile_type=} {atm_profile_idx=} {tau=} {pref=} {fwhm=}')
        
        
        # Calculate atmospheric properties
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        # Convert pressures to atm
        P = atm.P / 101325.0  # Pressure in atm

        # Compute Y0 = np.log(pref)
        Y0 = np.log(pref)

        # Compute XWID, the standard deviation of the Gaussian
        # [JD] this calculation is not correct
        XWID = fwhm

        # Initialize arrays
        Q = np.zeros(atm.NP)
        ND = np.zeros(atm.NP)
        OD = np.zeros(atm.NP)
        X1 = np.zeros(atm.NP)

        XOD = 0.0

        for j in range(atm.NP):
            Y = np.log(P[j])

            # Compute Q[j]
            Q[j] = 1.0 / (XWID * np.sqrt(np.pi)) * np.exp(-((Y - Y0) / XWID) ** 2)  #Q is specific density in particles per gram of atm

            # Compute ND[j]
            ND[j] = Q[j] * (rho[j] / 1.0e3) #ND is m-3

            # Compute OD[j]
            OD[j] = ND[j] * scale[j] * 1e5  # The factor 1e5 converts m to cm

            # Check for NaN or small values
            if np.isnan(OD[j]) or OD[j] < 1e-36:
                OD[j] = 1e-36
            if np.isnan(Q[j]) or Q[j] < 1e-36:
                Q[j] = 1e-36

            XOD += OD[j]

            X1[j] = Q[j]

        # Empirical correction to XOD
        XOD = XOD * 0.25

        # Rescale Q[j]
        for j in range(atm.NP):
            X1[j] = Q[j] * tau / XOD  # XDEEP is tau

            # Check for NaN or small values
            if np.isnan(X1[j]) or X1[j] < 1e-36:
                X1[j] = 1e-36

        # Now compute the Jacobian matrix xmap, note that we only need one entry for the 'ipar' entry
        # so we don't have to use 'ipar' in here, we pass the calcuated slice out and is is the
        # the containing scope's job to put our returned xmap part into the 'real xmap' at the
        # correct location.
        xmap = np.zeros((3, atm.NP))

        for j in range(atm.NP):
            Y = np.log(P[j])

            # First parameter derivative: xmap[0, ipar, j] = X1[j] / tau
            xmap[0, j] = X1[j] / tau  # XDEEP is tau

            # Derivative of X1[j] with respect to Y0 (pref)
            xmap[1, j] = 2.0 * (Y - Y0) / XWID ** 2 * X1[j]

            # Derivative of X1[j] with respect to XWID (fwhm)
            xmap[2, j] = (2.0 * ((Y - Y0) ** 2) / XWID ** 3 - 1.0 / XWID) * X1[j]

        # Update the atmosphere class with the new profile
        atm.DUST[:, atm_profile_idx] = X1[:]
        _lgr.debug(f'{X1=}')

        if MakePlot:
            fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))
            ax1.plot(atm.DUST[:, atm_profile_idx], atm.P / 101325.0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max() / 101325.0, atm.P.min() / 101325.0)
            ax1.set_xlabel('Cloud density (particles/kg)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()
            plt.show()

        atm.DUST_RENORMALISATION[atm_profile_idx] = tau   #Adding flag to ensure that the dust optical depth is tau

        return atm, xmap



