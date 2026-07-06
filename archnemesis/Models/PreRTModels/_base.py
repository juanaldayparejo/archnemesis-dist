
from typing import TYPE_CHECKING
import abc

import numpy as np

from ..ModelBase import ModelBase
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
    
    def __init__(
            self,
            state_vector_start : int, 
            n_state_vector_entries : int,
            atm_profile_type : AtmosphericProfileTypeEnum = AtmosphericProfileTypeEnum.NOT_PRESENT,
            #   ENUM that tells us what kind of atmospheric profile this model instance represents
    ):
        super().__init__(state_vector_start, n_state_vector_entries)
        self.target = atm_profile_type
        return
    
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
    ) -> bool:
        return varident[2]==cls.id
    
    
    def patch_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
    ) -> None:
        """
        Patches values of components based upon values of model parameters in the state vector. Called from ForwardModel_0::subprofretg.
        """
        _lgr.debug(f'Model id {self.id} method "patch_from_subprofretg" does nothing...')
    
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
    ) -> None:
        """
        Updated spectra based upon values of model parameters in the state vector. Called from ForwardModel_0::subspecret.
        """
        _lgr.debug(f'Model id {self.id} method "calculate_from_subspecret" does nothing...')
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    @abc.abstractmethod
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
    ) -> None:
        raise NotImplementedError('calculate_from_subprofretg must be implemented for all Atmospheric models')

