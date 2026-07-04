
from typing import TYPE_CHECKING, Self, IO

import numpy as np
from scipy.integrate import simpson

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

class Model9(PreRTModelBase):
    """
    In this model, the profile (cloud profile) is represented by a value
    at a certain height, plus a fractional scale height. Below the reference height 
    the profile is set to zero, while above it the profile decays exponentially with
    altitude given by the fractional scale height. In addition, this model scales
    the profile to give the requested integrated cloud optical depth.
    """
    
    id : int = 9

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
            ModelParameter('tau', slice(2,3), 'Total integrated column density of the cloud (aerosol)', r'$m^{-2}$'),
            ModelParameter('frac_scale_height', slice(1,2), 'Fractional scale height (decays above `h_ref` zero below)', 'km'),
            ModelParameter('h_ref', slice(0,1), 'Base height of cloud profile', 'km'),
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
            
            tau,
            #   Total integrated column density of the cloud (m-2)
            
            fsh,
            #   Fractional scale height (km)
            
            href,
            #   Base height of cloud profile (km)
            
            MakePlot=False
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

        


        if atm_profile_type != AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
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
            totcol = simpson(ND,x=atm.H)
            ND = ND / totcol * xdeep

            if itest==0:
                xprof[:] = ND[:]
            else:
                xmap[itest-1,:] = (ND[:]-xprof[:])/dx

        atm.DUST[0:atm.NP,atm_profile_idx] = xprof

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
        #******** cloud profile held as total optical depth plus
        #******** base height and fractional scale height. Below the knee
        #******** pressure the profile is set to zero - a simple
        #******** cloud in other words!
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        hknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfsh = tmp[0]
        efsh = tmp[1]

        if xdeep>0.0:
            x0[ix] = np.log(xdeep)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        err = edeep/xdeep
        sx[ix,ix] = err**2.

        ix = ix + 1

        if xfsh>0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

        err = efsh/xfsh
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = hknee
        #inum[ix] = 1
        sx[ix,ix] = eknee**2.

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
        #******** cloud profile held as total optical depth plus
        #******** base height and fractional scale height. Below the knee
        #******** pressure the profile is set to zero - a simple
        #******** cloud in other words!
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
        #Model 9. Simple cloud represented by base height, fractional scale height
            #and the total integrated cloud density
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


