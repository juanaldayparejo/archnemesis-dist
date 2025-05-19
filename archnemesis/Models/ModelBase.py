from __future__ import annotations #  for 3.9 compatability

import numpy as np
import numpy.ma
from typing import TYPE_CHECKING, IO, Any, Self
from collections import namedtuple

import matplotlib.pyplot as plt # used in some of the models, I should really remove the plotting code or at least separate it from the calculation code

from archnemesis.Scatter_0 import kk_new_sub
from archnemesis.helpers.maths_helper import ngauss
from archnemesis.enums import WaveUnit

from .ModelParameterEntry import ModelParameterEntry


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Scatter_0 import Scatter_0
    from archnemesis.Spectroscopy_0 import Spectroscopy_0
    from archnemesis.Measurement_0 import Measurement_0
    
    nx = 'number of elements in state vector'
    m = 'an undetermined number, but probably less than "nx"'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'
    NCONV = 'number of spectral bins'
    NGEOM = 'number of geometries'
    NX = 'number of elements in state vector'
    NDEGREE = 'number of degrees in a polynomial'
    NWINDOWS = 'number of spectral windows'

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)




class ModelBase:
    """
    Abstract base class of all parameterised models used by ArchNemesis. This class should be subclassed further for models of a particular component.
    """
    id : int = None # All "*ModelBase" classes that are not meant to be used should have an id of 'None'
    name : str = 'name should be overwritten in subclass'
    
    def __init__(
            self, 
            i_state_vector_start : int, 
            n_state_vector_entries : int
        ):
        """
        Initialise an instance of the model.
        
        ARGUMENTS
            i_state_vector_start : int
                The index of the first entry of the model parameters in the state vector
            n_state_vector_entries : int
                The number of model parameters that are stored in the state vector
        
        RETURNS
            An initialised instance of this object
        """
        # Store where the model parameters are positioned within the state vector
        self.state_vector_start = i_state_vector_start
        self.n_state_vector_entries = n_state_vector_entries
        self.state_vector_slice = slice(i_state_vector_start, i_state_vector_start+n_state_vector_entries)
        _lgr.debug(f'{self.id=} {self.state_vector_start=} {self.n_state_vector_entries=}')
        
    
    @property
    def parameter_slices(self) -> dict[str, slice]:
        """
        A dictionary that maps parameter names to the sub-slices of the state vector that holds the parameter values.
        This should be overloaded in subclasses to make it easy to get the apriori and posterior values of the
        parameters for this model. If not overwritten only one parameter called "all_parameters" is defined, which
        contains all the parameters for the model.
        
        QUESTION: Should this be a more general 'parameter_definitions' property that includes more things that just
        state vector slice information (e.g. descriptions, units, short name, long name, etc.)?
        """
        return {
            'all_parameters' : slice(None)
        }
    
    def get_state_vector_slice(self, state_vector : np.ndarray[['nx'], float]):
        """
        Gets the slice of a `state_vector` that holds only the parameters for the model
        """
        return state_vector[self.state_vector_slice]
    
    def get_value_from_state_vector(
            self,
            state_vector : np.ndarray[['nx'],float],
            state_vector_log : np.ndarray[['nx'],int],
            sub_slice : slice = slice(None),
        ) -> np.ndarray[['m'],float]:
        """
        Returns the value of elements of a (sub-slice of a) state vector associated with the model.
        
        ARGUMENTS
            state_vector : np.ndarray[['nx'],float]
                Array that we want to pull from. Will normally be the apriori or posterior state vector.
            state_vector_log : np.ndarray[['nx'],int]
                Array of boolean flags indicating if the value stored in `state_vector` is the exponential logarithm
                of the 'real' value.
            sub_slice : slice = slice(None)
                A sub-slice that is applied after the state vector is sliced the first time to only contain elements
                associated with the model. For example this can be set to 'slice(0,1)' to only get the first element
                of the state vector associated with the model, useful when splitting up the "whole model" state vector
                to get each individual parameter of the model.
        
        RETURNS
            value : np.ndarray[['m'],float]
                Array of 'real' (i.e. unlogged where applicable) values of (a sub-slice of) the parameters of the model.
        """
        
        a_val = state_vector[self.state_vector_slice][sub_slice]
        a_exp = np.exp(a_val)
        a_log_flag = state_vector_log[self.state_vector_slice][sub_slice] != 0
        
        return np.where(a_log_flag, a_exp, a_val)
    
    def set_value_to_state_vector(
            self,
            value : np.ndarray[['m'],float],
            state_vector : np.ndarray[['nx'],float],
            state_vector_log : np.ndarray[['nx'],int],
            sub_slice : slice = slice(None),
        ):
        """
        Sets the value of elements of a (sub-slice of a) state vector associated with the model.
        
        ARGUMENTS
            value : np.ndarray[['m'],float]
                Array of 'real' (i.e. unlogged where applicable) values of (a sub-slice of) the parameters of the model
                that we want to store in the state vector.
            state_vector : np.ndarray[['nx'],float]
                Array that we want to push to. Will normally be the apriori or posterior state vector.
            state_vector_log : np.ndarray[['nx'],int]
                Array of boolean flags indicating if the value stored in `state_vector` is the exponential logarithm
                of the 'real' value.
            sub_slice : slice = slice(None)
                A sub-slice that is applied after the state vector is sliced the first time to only contain elements
                associated with the model. For example this can be set to 'slice(0,1)' to only set the first element
                of the state vector associated with the model, useful when setting a single parameter of the model.
        
        RETURNS:
            None
        """
        
        value_log = np.log(value)
        log_flag = state_vector_log[self.state_vector_slice][sub_slice] != 0
        
        state_vector[self.state_vector_slice][sub_slice] = np.where(log_flag, value_log, value)
    
    def get_parameters_from_state_vector(
            self,
            apriori_state_vector : np.ndarray[['nx'],float],
            posterior_state_vector : np.ndarray[['nx'],float],
            state_vector_log : np.ndarray[['nx'],int],
            state_vector_fix : np.ndarray[['nx'],int],
        ) -> dict[str, ModelParameterEntry]:
        """
        Retrieve parameters from state vector as a dictionary of name : value pairs
        
        ARGUMENTS
            
            apriori_state_vector : np.ndarray[['nx'],float]
                The complete apriori state vector with 'nx' entries
            
            posterior_state_vector : np.ndarray[['nx'],float]
                The complete posterior state vector with 'nx' entries
            
            state_vector_log : np.ndarray[['nx'],int]
                Array of 'log' flags for each entry in either state vector (they share these flags)
                if the flag is non-zero the value stored in both state vectors is the exponential
                logarithm of the 'real' value.
            
            state_vector_fix : np.ndarray[['nx'],int]
                Array of the 'fix' flags for each entry in either state vector (they share these flags)
                if the flag is non-zero, the value stored in both state vectors is not retrieved, and
                therefore should be the same in each of the apriori/posterior state vectors.
        
        RETURNS
            
            parameters : dict[str, ModelParameterEntry]
                A dictionary that maps parameter names to the "ModelParameterEntry" associated with that parameter.
        """
        parameters = dict()
        
        assert self.state_vector_slice.step is None or self.state_vector_slice.step == 1, "A step larger than 1 is not supported when slicing a state vector"
        
        for name, pslice in self.parameter_slices.items():
            assert pslice.step is None or pslice.step == 1, "A step larger than 1 is not supported when sub-slicing a state vector"
            
            apriori_value = self.get_value_from_state_vector(
                apriori_state_vector, 
                state_vector_log, 
                pslice
            )
            posterior_value = self.get_value_from_state_vector(
                posterior_state_vector, 
                state_vector_log, 
                pslice
            )
            
            fix_flag = self.get_state_vector_slice(state_vector_fix)[pslice] != 0
            
            p_start, p_stop, p_step = pslice.indices(self.n_state_vector_entries)
            parameters[name] = ModelParameterEntry(
                self.id,
                name,
                slice(self.state_vector_slice.start + p_start, self.state_vector_slice.start + p_stop),
                fix_flag,
                apriori_value,
                posterior_value
            )
        return parameters
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int], # 3 integers that specify the identity (and some parameters) of the model
        ) -> bool:
        ...
    
    
    @classmethod
    def from_apr_to_state_vector(
            cls,
            variables : "Variables_0", # An instance of the archnemesis.Variables_0.Variables_0 class that is reading the *.apr file
            f : IO, # The open file descriptor of the *.apr file
            varident : np.ndarray[[3],int], # Should be the correct slice of the original (which should be a reference to the sub-array)
            varparam : np.ndarray[["mparam"],float], # Should be the correct slice of the original (which should be a reference to the sub-array)
            ix : int, # The next free entry in the state vector
            lx : np.ndarray[["mx"],int], # state vector flags denoting if the value in the state vector is a logarithm of the 'real' value. Should be a reference to the original
            x0 : np.ndarray[["mx"],float], # state vector, holds values to be retrieved. Should be a reference to the original
            sx : np.ndarray[["mx","mx"],float], # Covariance matrix for the state vector. Should be a reference to the original
            inum : np.ndarray[["mx"],int], # state vector flags denoting if the gradient is to be numerically calulated (1) or analytically calculated (0) for the state vector entry. Should be a reference to the original
            npro : int, # Number of altitude levels defined for the atmosphere component
            nlocations : int, # Number of locations defined for the atmosphere component
            runname : str, # Name of the *.apr file, without extension.
            sxminfac : float, # Minimum factor to bother calculating covariance matrix entries between current model's parameters and other model's parameters.
        ) -> Self:
        """
        Constructs a model from its entry in a *.apr file.
        """
        ...
    
    
    @classmethod
    def calculate(cls, *args, **kwargs) -> Any:
        """
        This class method should perform the lowest-level calculation for the model. Note that it is a class
        method (so we can easily call it from other models if need be) so you must pass any instance attributes
        as arguments.
        
        Models are so varied in here that I cannot make any specific interface at this level of abstraction.
        """
        ...
    
    
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        """
        Updated values of components based upon values of model parameters in the state vector. Called from ForwardModel_0::subprofretg.
        """
        ...

    
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
        ...
    
    
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
        ...