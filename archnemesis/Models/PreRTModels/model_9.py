

from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
from scipy.integrate import simpson

from ._base import PreRTModelBase
from ..param import StateParam, ConstParam

import archnemesis.Data.constants as const

from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
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
class Model9(PreRTModelBase):
    """
    In this model, the profile (cloud profile) is represented by a value
    at a certain height, plus a fractional scale height. Below the reference height 
    the profile is set to zero, while above it the profile decays exponentially with
    altitude given by the fractional scale height. In addition, this model scales
    the profile to give the requested integrated cloud optical depth.
    """

    id: ClassVar[int] = 9

    tau: StateParam.using(slice(0,1), 'Total integrated column density of the cloud (aerosol)', r'$m^{-2}$') # noqa: F722 F821
    frac_scale_height: StateParam.using(slice(1,2), 'Fractional scale height (decays above `h_ref` zero below)', 'km') # noqa: F722 F821
    h_ref: StateParam.using(slice(2,3), 'Base height of cloud profile', 'km') # noqa: F722 F821
    atm_profile_type: ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to') # noqa: F722 F821

    def calculate(
        self,
        atm: "Atmosphere_0",
        #   Instance of Atmosphere_0 class we are operating upon

        atm_profile_type: AtmosphericProfileTypeEnum,
        #   ENUM of atmospheric profile type we are altering.

        atm_profile_idx: int | None,
        #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

        MakePlot=False,
    ):

        """
            FUNCTION NAME : model9()

            DESCRIPTION :

                Function defining the model parameterisation 9 in NEMESIS.
                In this model, the profile (cloud profile) is represented by a value
                at a certain height, plus a fractional scale height. Below the reference height 
                the profile is set to zero, while above it the profile decays exponentially with
                altitude given by the fractional scale height. In addition, this model scales
                the profile to give the requested integrated cloud optical depth.

            INPUTS :

                atm :: Python class defining the atmosphere

                tau :: Total integrated column density of the cloud (m-2)

                fsh :: Fractional scale height (km)

                href :: Base height of cloud profile (km)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(3,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model9(atm,atm_profile_type,atm_profile_idx,href,fsh,tau)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        


        tau = self.tau.v
        fsh = self.frac_scale_height.v
        href = self.h_ref.v

        if atm_profile_type != AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            _msg = f'Model id={self.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)
        
        
        #Calculating the actual atmospheric scale height in each level
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        xmap = np.zeros([3,atm.NP])
        for itest in range(4):

            xdeep = tau
            xfsh = fsh
            hknee = href

            if itest==0:
                _ = 1
            elif itest==1: #For calculating the gradient wrt tau
                dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xdeep = np.exp( np.log(tau) + dx )
            elif itest==2: #For calculating the gradient wrt fsh
                dx = 0.05 * np.log(fsh)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xfsh = np.exp( np.log(fsh) + dx )
            elif itest==3: #For calculating the gradient wrt href
                dx = 0.05 * href
                if dx==0.0:
                    dx = 0.1
                hknee = href + dx

            #Initialising some arrays
            ND = np.zeros(atm.NP)   #Dust density (m-3)

            #Calculating the density in each level
            jfsh = -1
            if atm.H[0]/1.0e3>=hknee:
                jfsh = 1
                ND[0] = 1.

            for jx in range(atm.NP-1):
                j = jx + 1
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfsh

                if atm.H[j]/1.0e3>=hknee:

                    if jfsh<0:
                        ND[j]=1.0
                        jfsh = 1
                    else:
                        ND[j]=ND[j-1]*np.exp(-delh/xfac)


            for j in range(atm.NP):
                if(atm.H[j]/1.0e3<hknee):
                    if(atm.H[j+1]/1.0e3>=hknee):
                        ND[j] = ND[j] * (1.0 - (hknee*1.0e3-atm.H[j])/(atm.H[j+1]-atm.H[j]))
                    else:
                        ND[j] = 0.0

            #Calculating column density (m-2) by integrating the number density (m-3) over column (m)
            #Note that when doing the layering, the total column density in the atmosphere might not be
            #exactly the same as in xdeep due to misalignments at the boundaries of the cloud
            totcol = simpson(ND, x=atm.H)
            ND = ND / totcol * xdeep

            if itest==0:
                xprof[:] = ND[:]
            else:
                xmap[itest-1,:] = (ND[:]-xprof[:])/dx

        atm.DUST[0:atm.NP,atm_profile_idx] = xprof

        

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
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 3)
        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )

        instance.frac_scale_height.log = True
        if varident[0] != 0:
            instance.tau.log = True

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
        xvals = np.zeros(3)
        xerrs = np.zeros(3)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )


    


