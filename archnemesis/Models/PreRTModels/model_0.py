
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


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

class Model0(PreRTModelBase):
    """
    In this model, the atmospheric parameters are modelled as continuous profiles
    in which each element of the state vector corresponds to the atmospheric profile 
    at each altitude level
    """
    
    id : int = 0


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
            ModelParameter('full_profile', slice(None), 'Every value for each level of the profile', 'PROFILE_TYPE'),
        )
        
        _lgr.debug(f'Constructed {self.__class__.__name__} with {self.state_vector_start=} {self.n_state_vector_entries=} {self.parameters=}')
        
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
            
            xprof : np.ndarray[['mparam'],float],
            #   Full profile, this model defines every value for each profile level. Has been unlogged as required
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 0 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                in which each element of the state vector corresponds to the atmospheric profile 
                at each altitude level

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                xprof(npro) :: Atmospheric profile

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                model parameters.

            CALLING SEQUENCE:

                atm,xmap = model0(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        _lgr.debug(f'Calculating {cls.__name__} {atm=} {atm_profile_type=} {atm_profile_idx=} {xprof.shape=}')
        _lgr.debug(f'{xprof[:10]=}')

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 0 :: Number of levels in atmosphere does not match the passed profile')
        
        xmap = np.diag(np.ones_like(xprof)) # This is always true. NOTE: Logged profiles are handled in `calculate_from_subprofretg`
        
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            temp = np.array(atm.VMR)
            temp[:,atm_profile_idx] = xprof
            atm.edit_VMR(temp)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            atm.edit_T(xprof)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            temp = np.array(atm.DUST)
            temp[:,atm_profile_idx] = xprof
            atm.edit_DUST(temp)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            atm.PARAH2(xprof)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            atm.FRAC(xprof)
        
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')
        
        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

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
        _lgr.debug(f'Reading model {cls.__name__} setup from "{runname}.apr" file')
        ix_0 = ix
        
        #********* continuous profile ************************
        s = f.readline().split()
        
        with open(s[0],'r') as f1:
            tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
            
            nlevel = int(tmp[0])
            if nlevel != npro:
                raise ValueError('profiles must be listed on same grid as .prf')
            
            clen = float(tmp[1])
            pref = np.zeros([nlevel])
            ref = np.zeros([nlevel])
            eref = np.zeros([nlevel])
            for j in range(nlevel):
                tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                pref[j] = float(tmp[0])
                ref[j] = float(tmp[1])
                eref[j] = float(tmp[2])

        if varident[0] == 0:  # *** temperature, leave alone ****
            x0[ix:ix+nlevel] = ref[:]
            for j in range(nlevel):
                sx[ix+j,ix+j] = eref[j]**2.
                if varident[1] == -1: #Gradients computed numerically
                    inum[ix+j] = 1

        else:                   #**** vmr, cloud, para-H2 , fcloud, take logs ***
            for j in range(nlevel):
                lx[ix+j] = 1
                x0[ix+j] = np.log(ref[j])
                sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel

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
        _lgr.debug(f'Initialising model {cls.__name__} setup from bookmark')
        ix_0 = ix
        #********* continuous profile ************************
        if varident[2] != cls.id:
            raise ValueError('error in Model0.from_bookmark() :: wrong model id')
        
        ix = ix + npro

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
        _lgr.debug(f'Calculating {self.__class__.__name__} from subprofretg {forward_model=} {ix=} {ipar=} {ivar=} {xmap.shape=}')
        
        #Model 0. Continuous profile
        #***************************************************************
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        xn_params = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        xn_params_1d = np.concatenate([np.atleast_1d(v) for v in xn_params])

        xprof = xn_params[0]  #The profile is the first (and only) slice of the state vector entries

        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            xprof,
        )
        
        #Calculating derivatives of the atmospheric profile with respect to the state vector
        for i in range(self.n_state_vector_entries):
            if forward_model.Variables.LX[self.state_vector_start+i] == 1: #If carried in log space, then the derivative is multiplied by the value of the profile at that level
                xmap1[i,:] = xmap1[i,:] * xn_params_1d[i]
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


