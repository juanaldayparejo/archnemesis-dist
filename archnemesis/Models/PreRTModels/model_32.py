
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

import archnemesis.Data.constants as const

from archnemesis.enum import AtmosphericProfileTypeEnum

import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.DEBUG)
_lgr.setLevel(logging.INFO)


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
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

class Model32(PreRTModelBase):
    """
        In this model, the profile (cloud profile) is represented by a value
        at a certain pressure level, plus a fractional scale height which defines an exponential
        drop of the cloud at higher altitudes. Below the pressure level, the cloud is set 
        to exponentially decrease with a scale height of 1 km. 
    """
    
    id : int = 32


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM that tells us what kind of atmospheric profile this model instance represents
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model.
        # NOTE: It is best to define these in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        self.parameters = (
            ModelParameter('tau', slice(0,1), 'Integrated dust column density', r'$m^{-2}$'),
            ModelParameter('frac_scale_height', slice(1,2), 'Fractional scale height', 'km'),
            ModelParameter('p_ref', slice(2,3), 'Reference pressure', 'atm'),
        )
        
        return


    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            tau : float,
            #   Integrated dust column-density (m-2) or opacity
            
            frac_scale_height : float,
            #   Fractional scale height
            
            p_ref : float,
            #   reference pressure (atm)
            
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
        _lgr.debug(f'{atm_profile_type=}')
        _lgr.debug(f'{atm_profile_idx=}')
        _lgr.debug(f'{p_ref=}')
        _lgr.debug(f'{frac_scale_height=}')
        _lgr.debug(f'{tau=}')

        
        if atm_profile_type != AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)

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
        #******** cloud profile is represented by a value at a 
        #******** variable pressure level and fractional scale height.
        #******** Below the knee pressure the profile is set to drop exponentially.

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfsh = tmp[0]
        efsh = tmp[1]

        #optical depth
        if varident[0]==0:
            #temperature - leave alone
            x0[ix] = xdeep
            err = edeep
        else:
            if xdeep>0.0:
                x0[ix] = np.log(xdeep)
                lx[ix] = 1
                err = edeep/xdeep
                #inum[ix] = 1
            else:
                raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        sx[ix,ix] = err**2.

        ix = ix + 1

        #cloud fractional scale height
        if xfsh>0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

        err = efsh/xfsh
        sx[ix,ix] = err**2.

        ix = ix + 1

        #cloud pressure level
        if pknee>0.0:
            x0[ix] = np.log(pknee)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter pknee (cloud pressure level) must be positive')

        err = eknee/pknee
        sx[ix,ix] = err**2.

        ix = ix + 1

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])

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
        #******** cloud profile is represented by a value at a 
        #******** variable pressure level and fractional scale height.
        #******** Below the knee pressure the profile is set to drop exponentially.
        ix = ix + 3

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 32. Cloud profile is represented by a value at a variable
            #pressure level and fractional scale height.
            #Below the knee pressure the profile is set to drop exponentially.
        #***************************************************************
        #tau = np.exp(forward_model.Variables.XN[ix])   #Base pressure (atm)
        #fsh = np.exp(forward_model.Variables.XN[ix+1])  #Integrated dust column-density (m-2) or opacity
        #pref = np.exp(forward_model.Variables.XN[ix+2])  #Fractional scale height
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        if atm_profile_type != AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            _msg = f'Model id={self.id} is only defined for {AtmosphericProfileTypeEnum.AEROSOL_DENSITY}.'
            _lgr.error(_msg)
            raise ValueError(_msg)
            
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


