
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from ._base import PreRTModelBase
from ..param import StateParam, ConstParam, VarParam
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




@dc.dataclass(slots=True)
class Model45(PreRTModelBase):
    """
        Variable deep tropospheric and stratospheric abundances, along with tropospheric humidity.
    """
    id : ClassVar[int] = 45

    
    deep_vmr   : StateParam.using(slice(0,1), 'deep (topospheric) gas volume mixing ratio', 'RATIO')
    humidity   : StateParam.using(slice(1,2), 'relative humidity of gas', 'RATIO')
    strato_vmr : StateParam.using(slice(2,3), 'high (stratospheric) gas volume mixing ratio', 'RATIO')
    
    atm_profile_type : ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to')
    
    
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
    ) -> Self:
        xvals, xerrs = cls.read_apr_value_error_pairs(f, len(cls._stateparam_names))
    
        instance = cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust)
        )
        
        assert instance.atm_profile_type.v == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO, \
            f"{cls.__name__}[id={instance.id}] is only valid for Gas VMR profiles"
        
        return instance
        

    def calculate(
            self, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
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
        tropo = self.deep_vmr.v
        humid = self.humidity.v
        strato = self.strato_vmr.v
        
        _lgr.debug(f'{atm_profile_type=} {atm_profile_idx=} {tropo=} {humid=} {strato=}')

        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            _msg = f'Model id={self.id} is only defined for gas VMR profiles.'
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

