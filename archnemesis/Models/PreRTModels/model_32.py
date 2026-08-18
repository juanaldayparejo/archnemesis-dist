
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
# legacy ModelParameter removed; using new param API

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
class Model32(PreRTModelBase):
    """
        In this model, the profile (cloud profile) is represented by a value
        at a certain pressure level, plus a fractional scale height which defines an exponential
        drop of the cloud at higher altitudes. Below the pressure level, the cloud is set 
        to exponentially decrease with a scale height of 1 km. 
    """
    id : ClassVar[int] = 32

    
    tau                : StateParam.using(slice(0,1), 'Integrated dust column density', r'$m^{-2}$') # noqa: F722 F821
    frac_scale_height  : StateParam.using(slice(1,2), 'Fractional scale height', 'km') # noqa: F722 F821
    p_ref              : StateParam.using(slice(2,3), 'Reference pressure', 'atm') # noqa: F722 F821
    
    atm_profile_type : ConstParam.using(AtmosphericProfileTypeEnum, 'Atmospheric profile type this model applies to') # noqa: F722 F821
    
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
            f"{cls.__name__}[id={instance.id}] is only valid for Aerosol Density profiles but was defined with {instance.atm_profile_type.v.name}"
        
        return instance
        

    def calculate(
            self, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            MakePlot : bool = False
    ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model32()

            DESCRIPTION :

                Function defining the model parameterisation 32 in NEMESIS.
                In this model, the profile (cloud profile) is represented by a value
                at a certain pressure level, plus a fractional scale height which defines an exponential
                drop of the cloud at higher altitudes. Below the pressure level, the cloud is set 
                to exponentially decrease with a scale height of 1 km. 


            INPUTS :

                atm :: Python class defining the atmosphere
                
                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
                    
                p_ref :: Base pressure of cloud profile (atm)
                
                frac_scale_height :: Fractional scale height (km)
                
                tau :: Total integrated column density of the cloud (m-2) or cloud optical depth (if kext is normalised)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(3,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model32(atm,atm_profile_type,atm_profile_idx,p_ref,frac_scale_height,tau)

            MODIFICATION HISTORY : Juan Alday (29/05/2024)

        """
        tau = self.tau.v
        frac_scale_height = self.frac_scale_height.v
        p_ref = self.p_ref.v
        
        _lgr.debug(f'{atm_profile_type=}')
        _lgr.debug(f'{atm_profile_idx=}')
        _lgr.debug(f'{p_ref=}')
        _lgr.debug(f'{frac_scale_height=}')
        _lgr.debug(f'{tau=}')

        #Calculating the actual atmospheric scale height in each level
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        #npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((3,atm.NP))
        for itest in range(4):

            xdeep = tau
            xfrac_scale_height = frac_scale_height
            pknee = p_ref
            if itest==0:
                pass
            elif itest==1: #For calculating the gradient wrt tau
                dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xdeep = np.exp( np.log(tau) + dx )
            elif itest==2: #For calculating the gradient wrt frac_scale_height
                dx = 0.05 * np.log(frac_scale_height)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xfrac_scale_height = np.exp( np.log(frac_scale_height) + dx )
            elif itest==3: #For calculating the gradient wrt p_ref
                dx = 0.05 * np.log(p_ref) #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                pknee = np.exp( np.log(p_ref) + dx )

            #Getting the altitude level based on the pressure/height relation
            isort = np.argsort(atm.P)
            hknee = np.interp(pknee,atm.P[isort]/101325.,atm.H[isort])  #metres

            #Initialising some arrays
            ND = np.zeros(atm.NP)   #Dust density (m-3)
            OD = np.zeros(atm.NP)   #Column density (m-2)
            Q = np.zeros(atm.NP)    #Specific density (particles/gram of atmosphere)


            #Finding the levels in the atmosphere that span pknee
            jknee = -1
            for j in range(atm.NP-1):
                if((atm.P[j]/101325. >= pknee) & (atm.P[j+1]/101325.< pknee)):
                    jknee = j


            if jknee < 0:
                jknee = 0

            #Calculating cloud density at the first level occupied by the cloud
            delh = atm.H[jknee+1] - hknee   #metres
            xfac = 0.5 * (scale[jknee]+scale[jknee+1]) * xfrac_scale_height  #metres
            ND[jknee+1] = np.exp(-delh/xfac)


            delh = hknee - atm.H[jknee]  #metres
            xf = 1000.  #The cloud below is set to decrease with a scale height of 1 km
            ND[jknee] = np.exp(-delh/xf)

            #Calculating the cloud density above this level
            for j in range(jknee+2,atm.NP):
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfrac_scale_height
                ND[j] = ND[j-1] * np.exp(-delh/xfac)

            #Calculating the cloud density below this level
            for j in range(0,jknee):
                delh = atm.H[jknee] - atm.H[j]
                xf = 1000.    #The cloud below is set to decrease with a scale height of 1 km
                ND[j] = np.exp(-delh/xf)

            #Now that we have the initial cloud number density (m-3) we can just divide by the mass density to get specific density
            Q[:] = ND[:] / rho[:] / 1.0e3 #particles per gram of atm

            #Now we integrate the optical thickness (calculate column density essentially)
            OD[atm.NP-1] = ND[atm.NP-1] * (scale[atm.NP-1] * xfrac_scale_height * 1.0e2)  #the factor 1.0e2 is for converting from m to cm
            #jfrac_scale_height = -1
            for j in range(atm.NP-2,-1,-1):
                if j>jknee:
                    delh = atm.H[j+1] - atm.H[j]   #m
                    xfac = scale[j] * xfrac_scale_height
                    OD[j] = OD[j+1] + (ND[j] - ND[j+1]) * xfac * 1.0e2
                elif j==jknee:
                    delh = atm.H[j+1] - hknee
                    xfac = 0.5 * (scale[j]+scale[j+1])*xfrac_scale_height
                    OD[j] = OD[j+1] + (1. - ND[j+1]) * xfac * 1.0e2
                    xfac = 1000.
                    OD[j] = OD[j] + (1.0 - ND[j]) * xfac * 1.0e2
                else:
                    delh = atm.H[j+1] - atm.H[j]
                    xfac = 1000.
                    OD[j] = OD[j+1] + (ND[j+1]-ND[j]) * xfac * 1.0e2

            ODX = OD[0]

            #Now we normalise the specific density profile
            #This should be done later to make this totally secure
            for j in range(atm.NP):
                OD[j] = OD[j] * xdeep / ODX
                ND[j] = ND[j] * xdeep / ODX
                Q[j] = Q[j] * xdeep / ODX
                if Q[j]>1.0e10:
                    Q[j] = 1.0e10
                if Q[j]<1.0e-36:
                    Q[j] = 1.0e-36

                #if ND[j]>1.0e10:
                #    ND[j] = 1.0e10
                #if ND[j]<1.0e-36:
                #    ND[j] = 1.0e-36

            if itest==0:  #First iteration, using the values in the state vector
                xprof[:] = Q[:]
            else:  #Next iterations used to calculate the derivatives
                xmap[itest-1,:] = (Q[:] - xprof[:])/dx

        #Now updating the atmosphere class with the new profile
        atm.DUST[:,atm_profile_idx] = xprof[:]
        _lgr.debug(f'{xprof=}')

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(atm.DUST[:,atm_profile_idx],atm.P/101325.)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max()/101325.,atm.P.min()/101325.)
            ax1.set_xlabel('Cloud density (m$^{-3}$)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()

        # [JD] Question: What is this actually doing? The `tau` variable is associated with a deep abundance
        #                not a total optical depth as far as I can tell.
        atm.DUST_RENORMALISATION[atm_profile_idx] = tau  #Adding flag to ensure that the dust optical depth is tau

        return atm,xmap
