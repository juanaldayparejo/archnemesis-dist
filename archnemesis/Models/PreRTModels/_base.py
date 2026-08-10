
from typing import TYPE_CHECKING
import abc

import numpy as np

from ..ModelBase import ModelBase
from ..param import (
    #StateParam, 
    ConstParam, 
    #VarParam,
)
from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.ForwardModel_0 import ForwardModel_0
    
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


class PreRTModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact 
    with Components before the radiative transfer calculation is performed.
    """
    
    
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
    ) -> bool:
        return varident[2]==cls.id
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
    ) -> None:
        assert isinstance(atm_profile_type := getattr(self, "atm_profile_type", None), ConstParam) and isinstance(atm_profile_type.v, AtmosphericProfileTypeEnum), \
            'Default method "calculate_from_subprofretg(...)" requires class to have an attribute like `atm_profile_type : ConstParam[AtmosphericProfileTypeEnum]` otherwise the "calculate_from_subprofretg(...)" method must be provided.'
    
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        assert atm_profile_type == self.atm_profile_type.v, \
            f"Model[id={self.id}] was defined with {self.atm_profile_type.v}, but is being used with {atm_profile_type}"
        
        self.pull_from_state_vector(forward_model.Variables.XN)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
