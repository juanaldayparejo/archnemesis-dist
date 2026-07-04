
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter


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

class Model45(PreRTModelBase):
    """
        Variable deep tropospheric and stratospheric abundances, along with tropospheric humidity.
    """
    
    id : int = 45

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
            ModelParameter('deep_vmr', slice(0,1), 'deep (topospheric) gas volume mixing ratio', 'RATIO'),
            ModelParameter('humidity', slice(1,2), 'relative humidity of gas', 'RATIO'),
            ModelParameter('strato_vmr', slice(2,3), 'high (stratospheric) gas volume mixing ratio', 'RATIO'),
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
            
            tropo, 
            humid, 
            strato, 
            MakePlot=True
        ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model45()

            DESCRIPTION :

                Irwin CH4 model. Variable deep tropospheric and stratospheric abundances,
                along with tropospheric humidity.

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                tropo :: Deep methane VMR

                humid :: Relative methane humidity in the troposphere

                strato :: Stratospheric methane VMR

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model45(atm, atm_profile_type, atm_profile_idx, tropo, humid, strato)

            MODIFICATION HISTORY : Joe Penn (09/10/2024)

        """

        _lgr.debug(f'{atm_profile_type=} {atm_profile_idx=} {tropo=} {humid=} {strato=}')

        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            _msg = f'Model id={cls.id} is only defined for gas VMR profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)
            
        SCH40 = 10.6815
        SCH41 = -1163.83
        # psvp is in bar
        NP = atm.NP

        xnew = np.zeros(NP)
        xnewgrad = np.zeros(NP)
        pch4 = np.zeros(NP)
        pbar = np.zeros(NP)
        psvp = np.zeros(NP)

        for i in range(NP):
            pbar[i] = atm.P[i] /100000#* 1.013
            tmp = SCH40 + SCH41 / atm.T[i]
            psvp[i] = 1e-30 if tmp < -69.0 else np.exp(tmp)

            pch4[i] = tropo * pbar[i]
            if pch4[i] / psvp[i] > 1.0:
                pch4[i] = psvp[i] * humid

            if pbar[i] < 0.1 and pch4[i] / pbar[i] > strato:
                pch4[i] = pbar[i] * strato

            if pbar[i] > 0.5 and pch4[i] / pbar[i] > tropo:
                pch4[i] = pbar[i] * tropo
                xnewgrad[i] = 1.0

            xnew[i] = pch4[i] / pbar[i]

        _lgr.debug(f'{xnew=}')
        atm.VMR[:, atm_profile_idx] = xnew

        return atm, xnewgrad


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
        #******** Irwin CH4 model. Represented by tropospheric and stratospheric methane 
        #******** abundances, along with methane humidity. 
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        tropo = tmp[0]
        etropo = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        humid = tmp[0]
        ehumid = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        strato = tmp[0]
        estrato = tmp[1]



        x0[ix] = np.log(tropo)
        lx[ix] = 1
        err = etropo/tropo
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = np.log(humid)
        lx[ix] = 1
        err = ehumid/humid
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = np.log(strato)
        lx[ix] = 1
        err = estrato/strato
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
        #******** Irwin CH4 model. Represented by tropospheric and stratospheric methane 
        #******** abundances, along with methane humidity. 
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
        #Model 45. Irwin CH4 model. Variable deep tropospheric and stratospheric abundances,
            #along with tropospheric humidity.
        #***************************************************************
        #tropo = np.exp(forward_model.Variables.XN[ix])   # Deep tropospheric abundance
        #humid = np.exp(forward_model.Variables.XN[ix+1])  # Humidity
        #strato = np.exp(forward_model.Variables.XN[ix+2])  # Stratospheric abundance
        
        #forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX, ipar, tropo, humid, strato)
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm, 
            atm_profile_type, 
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[ix, ipar, :] = xmap1

        #ix = ix + forward_model.Variables.NXVAR[ivar]
        return


