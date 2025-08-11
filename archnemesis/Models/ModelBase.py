from __future__ import annotations #  for 3.9 compatability
import sys, os
import abc
from typing import TYPE_CHECKING, IO, Any, Self
from collections import namedtuple
import textwrap
import inspect


import matplotlib.pyplot as plt # used in some of the models, I should really remove the plotting code or at least separate it from the calculation code

import numpy.ma
import numpy as np
from archnemesis.Scatter_0 import kk_new_sub
from archnemesis.helpers.maths_helper import ngauss
from archnemesis.enums import WaveUnit
from archnemesis.helpers.io_helper import OutWidth

from .ModelParameterEntry import ModelParameterEntry
from .ModelParameter import ModelParameter



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
#_lgr.setLevel(logging.DEBUG)
_lgr.setLevel(logging.INFO)




class ModelBase(abc.ABC):
    """
        Abstract base class of all parameterised models used by ArchNemesis. This class should be subclassed further for models of a particular component.
    """
    
    id : int = None # All "*ModelBase" classes that are not meant to be used should have an id of 'None'
    
    @classmethod
    def to_string(cls):
        desc_wrapper_first = textwrap.TextWrapper(
            width = OutWidth.get(),
            expand_tabs=True,
            tabsize=2,
            replace_whitespace=True,
            initial_indent='- description: ',
            subsequent_indent=' '*15
        )
        desc_wrapper_rest = textwrap.TextWrapper(
            width = OutWidth.get(),
            expand_tabs=True,
            tabsize=2,
            replace_whitespace=True,
            initial_indent=' '*15,
            subsequent_indent=' '*15
        )
        
        ds : list[str,...] = textwrap.dedent(cls.__doc__.strip('\n') if cls.__doc__ is not None else 'DESCRIPTION NOT FOUND').split('\n\n')
        
        docstr = textwrap.indent('\n'.join((
            desc_wrapper_first.fill(x) if i==0 else desc_wrapper_rest.fill(x) for i,x in enumerate(ds)
        )), '|', lambda x: True)
        return '\n'.join((
            f'{cls.__name__}:',
            f'|- id : {cls.id}',
            f'|- parent classes: {", ".join((x.__name__ for x in cls.__bases__))}',
            docstr,
        ))
    
    def __init__(
            self, 
            state_vector_start : int, 
            #   The index of the first entry of the model parameters in the state vector
            
            n_state_vector_entries : int,
            #   The number of model parameters that are stored in the state vector
        ):
        """
            Initialise an instance of the model.
            
            ## ARGUMENTS ##
                
                state_vector_start : int
                    The index of the first entry of the model parameters in the state vector
                
                n_state_vector_entries : int
                    The number of model parameters that are stored in the state vector
            
            ## RETURNS ##
                
                None
        """
        
        # Store where the model parameters are positioned within the state vector
        self.state_vector_start = state_vector_start
        self.n_state_vector_entries = n_state_vector_entries
        self.state_vector_slice = slice(state_vector_start, state_vector_start+n_state_vector_entries)
        
        # default parameters, the default will make a fake "all_parameters" parameter
        # that contains everything.
        self.parameters = (
            ModelParameter('all_parameters', slice(None), f'default parameter that is all of the parameters for model id {self.id}', 'UNDEFINED'),
        )
        
        _lgr.debug(f'{self.id=} {self.state_vector_start=} {self.n_state_vector_entries=}')
        return
    
    
    def info(self, log_flag_state_vector = None, apriori_state_vector=None, posterior_state_vector=None)->str:
        param_apriori_values = None
        param_posterior_values = None
        if log_flag_state_vector is not None and apriori_state_vector is not None:
            param_apriori_values = self.get_parameter_values_from_state_vector(apriori_state_vector, log_flag_state_vector)
            _lgr.debug(f'{param_apriori_values=}')
        if log_flag_state_vector is not None and posterior_state_vector is not None:
            param_posterior_values = self.get_parameter_values_from_state_vector(posterior_state_vector, log_flag_state_vector)
        
        return self.__str__(param_apriori_values, param_posterior_values)
    
    
    def __str__(self, param_apriori_values = None, param_posterior_values = None) -> str:
        s = self.to_string()
        #attrs = tuple(filter(lambda x: not x.startswith('_') and x != "parameters", dir(self)))
        attrs = inspect.getmembers(self, lambda x: not inspect.ismethod(x))
        attrs = tuple(filter(lambda x: not x[0].startswith('_') and x[0] not in ("parameters", "id"), attrs))
        attrs_str = '\n'.join(
            (
                f'|- {name} : {attr}' for name, attr in attrs
            )
        )
        param_desc_wrapper_first = textwrap.TextWrapper(
            width = OutWidth.get(),
            expand_tabs=True,
            tabsize=2,
            replace_whitespace=False,
            initial_indent='- description: ',
            subsequent_indent=' '*15
        )
        param_desc_wrapper_rest = textwrap.TextWrapper(
            width = OutWidth.get(),
            expand_tabs=True,
            tabsize=2,
            replace_whitespace=False,
            initial_indent=' '*15,
            subsequent_indent=' '*15
        )
        
        
        
        params_strs = [
            '\n'.join((
                f'|  |- {p.name} :',
                f'|  |  |- slice : {p.slice}',
                f'|  |  |- unit : {p.unit}',
                textwrap.indent(
                    '\n'.join((
                        param_desc_wrapper_first.fill(x) if i==0 else param_desc_wrapper_rest(x) for i,x in enumerate(p.description.strip().split('\n'))
                    )),
                    '|  |  |',
                    lambda x: True
                ),
            )) for p in self.parameters
        ]
        
        for i in range(len(params_strs)):
            if param_apriori_values is not None:
                params_strs[i] += f'\n|  |  |- apriori value : {param_apriori_values[i]}'
            if param_posterior_values is not None:
                params_strs[i] += f'\n|  |  |- posterior value : {param_posterior_values[i]}'
        
        
        return '\n'.join((
            s,
            attrs_str,
            '|- Parameters:',
            '\n'.join(params_strs),
        ))
        
    
    def get_state_vector_slice(
            self, 
            state_vector : np.ndarray[['nx'], float|int],
            #   Array that we want to pull from. Will normally be the apriori or posterior state vector.
            
        ) -> np.ndarray[['mparam'],float|int]:
        """
            Gets the slice of a `state_vector`-like object (i.e. same shape and ordering as state vector)
            that holds only the parameters for the model.
        """
        return state_vector[self.state_vector_slice]
    
    
    def get_parameter_entries_from_state_vector(
            self,
            state_vector : np.ndarray[['nx'],float],
            #   Array that we want to pull from. Will normally be the apriori or posterior state vector.
            
        ) -> tuple[float | np.ndarray[[int],float],...]:
        """
            Returns a tuple of copies of the entries (raw entries, not unlogged etc.) of parameters (defined via `self.parameters`).
            Parameters with a single entry will be a 'float', parameters that consist of multiple entries will be a numpy array.
            
            ## RETURNS ##
            
                entries : tuple[float | np.ndarray[[int],float],...]
                    Entries retrieved from the state vector, no processing is done to the entries.
        """
        sv_slice = self.get_state_vector_slice(state_vector)
        return tuple(np.array(sv_slice[p.slice]) if len(sv_slice[p.slice]) > 1 else sv_slice[p.slice][0] for p in self.parameters)
    
    
    def get_parameter_values_from_state_vector(
            self,
            state_vector : np.ndarray[['nx'],float],
            #   Array that we want to pull from. Will normally be the apriori or posterior state vector.
            
            state_vector_log : np.ndarray[['nx'],int],
            #   Array of boolean flags indicating if the value stored in `state_vector` is the exponential logarithm
            #   of the 'real' value.
            
        ) -> tuple[float | np.ndarray[[int],float],...]:
        """
            Returns a tuple of copies of the values (entry will be unlogged etc. if required) of parameters (defined via `self.parameters`).
            Parameters with a single entry will be a 'float', parameters that consist of multiple entries will be a numpy array.
            
            ## RETURNS ##
            
                values : tuple[float | np.ndarray[[int],float],...]
                    Values retrieved from the state vector, unlogged as required depending upon the `state_vector_log` array entries.
        """
        sv_slice = self.get_state_vector_slice(state_vector)
        sv_slice_exp = np.exp(sv_slice)
        log_slice = (self.get_state_vector_slice(state_vector_log) > 0)
        values = np.where(log_slice, sv_slice_exp, sv_slice)
        return tuple(np.array(values[p.slice]) if len(values[p.slice]) > 1 else values[p.slice][0] for p in self.parameters)
    
    
    def get_value_from_state_vector(
            self,
            state_vector : np.ndarray[['nx'],float],
            #   Array that we want to pull from. Will normally be the apriori or posterior state vector.
            
            state_vector_log : np.ndarray[['nx'],int],
            #   Array of boolean flags indicating if the value stored in `state_vector` is the exponential logarithm
            #   of the 'real' value.
            
            sub_slice : slice = slice(None),
            #   A sub-slice that is applied after the state vector is sliced the first time to only contain elements
            #   associated with the model. For example this can be set to 'slice(0,1)' to only get the first element
            #   of the state vector associated with the model, useful when splitting up the "whole model" state vector
            #   to get each individual parameter of the model.
            
        ) -> np.ndarray[['m'],float]:
        """
            Returns the value (entry will be unlogged if required) of elements of a (sub-slice of a) state vector associated with the model.
            
            ## RETURNS ##
                
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
            #   Array of values we want to put into the `state_vector` array
            
            state_vector : np.ndarray[['nx'],float],
            #   Array that we want to push to. Will normally be the apriori or posterior state vector.
            
            state_vector_log : np.ndarray[['nx'],int],
            #   Array of boolean flags indicating if the value stored in `state_vector` is the exponential logarithm
            #   of the 'real' value.
            
            sub_slice : slice = slice(None),
            #   A sub-slice that is applied after the state vector is sliced the first time to only contain elements
            #   associated with the model. For example this can be set to 'slice(0,1)' to only get the first element
            #   of the state vector associated with the model, useful when splitting up the "whole model" state vector
            #   to get each individual parameter of the model.
            
        ) -> None:
        """
            Sets the value (will be logged if required) of elements of a (sub-slice of a) state vector associated with the model.
            
            ## RETURNS ##
                
                None
        """
        
        value_log = np.log(value)
        log_flag = state_vector_log[self.state_vector_slice][sub_slice] != 0
        
        state_vector[self.state_vector_slice][sub_slice] = np.where(log_flag, value_log, value)
        return
    
    
    def get_parameters_from_state_vector(
            self,
            apriori_state_vector : np.ndarray[['nx'],float],
            #   The complete apriori state vector with 'nx' entries
            
            posterior_state_vector : np.ndarray[['nx'],float],
            #   The complete posterior state vector with 'nx' entries
            
            state_vector_log : np.ndarray[['nx'],int],
            #   Array of 'log' flags for each entry in either state vector (they share these flags)
            #   if the flag is non-zero the value stored in both state vectors is the exponential
            #   logarithm of the 'real' value.
                
            state_vector_fix : np.ndarray[['nx'],int],
            #   Array of the 'fix' flags for each entry in either state vector (they share these flags)
            #   if the flag is non-zero, the value stored in both state vectors is not retrieved, and
            #   therefore should be the same in each of the apriori/posterior state vectors.
            
        ) -> dict[str, ModelParameterEntry]:
        """
            Retrieve parameters from state vector as a dictionary of name : value pairs
            
            ## ARGUMENTS ##
                
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
            
            ## RETURNS ##
                
                parameters : dict[str, ModelParameterEntry]
                    A dictionary that maps parameter names to the "ModelParameterEntry" associated with that parameter.
        """
        parameters = dict()
        
        assert self.state_vector_slice.step is None or self.state_vector_slice.step == 1, "A step larger than 1 is not supported when slicing a state vector"
        
        for p in self.parameters:
            
            assert p.slice.step is None or p.slice.step == 1, "A step larger than 1 is not supported when sub-slicing a state vector"
            
            apriori_value = self.get_value_from_state_vector(
                apriori_state_vector, 
                state_vector_log, 
                p.slice
            )
            posterior_value = self.get_value_from_state_vector(
                posterior_state_vector, 
                state_vector_log, 
                p.slice
            )
            
            fix_flag = self.get_state_vector_slice(state_vector_fix)[p.slice] != 0
            
            p_start, p_stop, p_step = p.slice.indices(self.n_state_vector_entries)
            parameters[p.name] = ModelParameterEntry(
                self.id,
                p.name,
                slice(self.state_vector_slice.start + p_start, self.state_vector_slice.start + p_stop),
                fix_flag,
                apriori_value,
                posterior_value
            )
        return parameters
    
    
    
    def set_parameters_to_state_vector(
            self,
            parameter_entries : dict[str,ModelParameterEntry],
            #   A dictionary that maps parameter names to the "ModelParameterEntry" associated with that parameter.
            
            state_vector_log : np.ndarray[['nx'],int],
            #   Array of 'log' flags for each entry in either state vector (they share these flags)
            #   if the flag is non-zero the value stored in both state vectors is the exponential
            #   logarithm of the 'real' value.
            
            apriori_state_vector : np.ndarray[['nx'],float],
            #   The complete apriori state vector with 'nx' entries
            
            posterior_state_vector : np.ndarray[['nx'],float] | None = None,
            #   The complete posterior state vector with 'nx' entries
            
            
            
        ) -> dict[str, ModelParameterEntry]:
        """
            Retrieve parameters from state vector as a dictionary of name : value pairs
            
            ## ARGUMENTS ##
                
                model_parameter_entries : dict[str, ModelParameterEntry]
                    A dictionary that maps parameter names to the "ModelParameterEntry" associated with that parameter.
                    
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
            
            ## RETURNS ##
        """
        
        assert self.state_vector_slice.step is None or self.state_vector_slice.step == 1, "A step larger than 1 is not supported when slicing a state vector"
        
        for p in self.parameters:
            
            self.set_value_to_state_vector(
                parameter_entries[p.name].apriori_value,
                apriori_state_vector,
                state_vector_log,
                p.slice
            )
            
            if posterior_state_vector is not None:
                self.set_value_to_state_vector(
                    parameter_entries[p.name].posterior_value,
                    posterior_state_vector,
                    state_vector_log,
                    p.slice
                )
        return
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    @classmethod
    @abc.abstractmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
            #   "Variable Identifier" from a *.apr file. Consists of 3 integers. Exact interpretation depends on the model
            #   subclass.
            
        ) -> bool:
        """
            Accepts a varident from a *.apr file, returns True if the varident is compatible with the model, False otherwise.
            Should be overwritten by a subclass
            
            ## ARGUMENTS ##
            
                varident : np.ndarra[[3],int]
                    "Variable Identifier" from a *.apr file. Consists of 3 integers. Exact interpretation depends on the model
                    subclass.
            
            ## RETURNS ##
            
                flag : bool
                    True if varident is compatible with the model, False otherwise.
        """
        ...
    
    
    @classmethod
    @abc.abstractmethod
    def from_apr_to_state_vector(
            cls,
            variables : "Variables_0", 
            #   An instance of the archnemesis.Variables_0.Variables_0 class that is reading the *.apr file
            
            f : IO, 
            #   The open file descriptor of the *.apr file
            
            varident : np.ndarray[[3],int], 
            #   Should be the correct slice of the original (which should be a reference to the sub-array)
            
            varparam : np.ndarray[["mparam"],float], 
            #   Should be the correct slice of the original (which should be a reference to the sub-array)
            
            ix : int, 
            #   The next free entry in the state vector
            
            lx : np.ndarray[["mx"],int], 
            #   state vector flags denoting if the value in the state vector is a logarithm of the 'real' value. 
            #   Should be a reference to the original
            
            x0 : np.ndarray[["mx"],float], 
            #   state vector, holds values to be retrieved. Should be a reference to the original
            
            sx : np.ndarray[["mx","mx"],float], 
            #   Covariance matrix for the state vector. Should be a reference to the original
            
            inum : np.ndarray[["mx"],int], 
            #   state vector flags denoting if the gradient is to be numerically calculated (1) 
            #   or analytically calculated (0) for the state vector entry. Should be a reference to the original
            
            npro : int, 
            #   Number of altitude levels defined for the atmosphere component
            
            ngas : int,
            #   Number of gas volume mixing ratio profiles defined for the reference atmosphere
            
            ndust : int,
            #   Number of aerosol species density profiles define for the reference atmosphere

            nlocations : int, 
            #   Number of locations defined for the atmosphere component
            
            runname : str, 
            #   Name of the *.apr file, without extension.
            
            sxminfac : float, 
            #   Minimum factor to bother calculating covariance matrix entries between current model's 
            #   parameters and other model's parameters.
            
        ) -> Self:
        """
            Constructs a model from its entry in a *.apr file. Should be overwritten by a subclass
            
            ## ARGUMENTS ##
            
                variables : Variables_0
                    The "Variables_0" instance that is reading the *.apr file
                
                f : IO
                    An open file descriptor for the *.apr file.
                
                varident : np.ndarray[[3],int]
                    "Variable Identifier" from a *.apr file. Consists of 3 integers. Exact interpretation depends on the model
                    subclass.
                
                varparam : np.ndarray[["mparam"], float]
                    "Variable Parameters" from a *.apr file. Holds "extra parameters" for the model. Exact interpretation depends on the model
                    subclass. NOTE: this is a holdover from the FORTRAN code, the better way to give extra data to the model is to store it on the
                    model instance itself.
                
                ix : int
                    The index of the next free entry in the state vector
                
                lx : np.ndarray[["mx"],int]
                    State vector flags denoting if the value in the state vector is a logarithm of the 'real' value. 
                    Should be a reference to the original
                
                x0 : np.ndarray[["mx"],float]
                    The actual state vector, holds values to be retrieved. Should be a reference to the original
                
                sx : np.ndarray[["mx","mx"],float]
                    Covariance matrix for the state vector. Should be a reference to the original
                
                inum : np.ndarray[["mx"],int]
                    state vector flags denoting if the gradient is to be numerically calulated (1) 
                    or analytically calculated (0) for the state vector entry. Should be a reference to the original
                
                npro : int
                    Number of altitude levels defined for the atmosphere component of the retrieval setup.
                
                ngas : int,
                    Number of gas volume mixing ratio profiles defined for the reference atmosphere
                
                ndust : int,
                    Number of aerosol species density profiles define for the reference atmosphere
                
                n_locations : int
                    Number of locations defined for the atmosphere component of the retrieval setup.
                
                runname : str
                    Name of the *.apr file, without extension. For example '/path/to/neptune.apr' has 'neptune'
                    as `runname`
                
                sxminfac : float
                    Minimum factor to bother calculating covariance matrix entries between current 
                    model's parameters and another model's parameters.
            
            
            ## RETURNS ##
            
                instance : Self
                    A constructed instance of the model class that has parameters set from information in the *.apr file
        """
        
        ...
    
    
    @classmethod
    @abc.abstractmethod
    def calculate(
            cls, 
            *args, 
            **kwargs
        ) -> Any:
        """
            This class method should perform the lowest-level calculation for the model. Note that it is a class
            method (so we can easily call it from other models if need be) so you must pass any instance attributes
            as arguments.
            
            NOTE: Models are so varied in here that I cannot make any specific interface at this level of abstraction.
        """
        ...
    
    @abc.abstractmethod
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            #   The ForwardModel_0 instance that is calling this function. We need this so we can alter components of the forward model
            #   inside this function.
            
            ix : int,
            #   The index of the state vector that corresponds to the start of the model's parameters
            
            ipar : int,
            #   An integer that encodes which part of the atmospheric component of the forward model this model should alter. Only
            #   used for some Atmospheric models.
            
            ivar : int,
            #   The model index, the order in which the models were instantiated. NOTE: this is a vestige from the
            #   FORTRAN version of the code, we don't really need to know this as we should be: 1) storing any model-specific
            #   values on the model instance itself; 2) passing any model-specific data from the outside directly
            #   instead of having the model instance look it up from a big array. However, the code for each model
            #   was recently ported from a more FORTRAN-like implementation so this is still required by some of them
            #   for now.
            
            xmap : np.ndarray,
            #   Functional derivatives of the state vector w.r.t Atmospheric profiles at each Atmosphere location.
            #   The array is sized as [nx,NVMR+2+NDUST,NP,NLOCATIONS]:
            #    
            #       nx - number of state vector entries.
            #    
            #       NVMR - number of gas volume mixing ratio profiles in the Atmosphere component of the forward model.
            #    
            #       NDUST - number of aerosol profiles in the Atmosphere component of the forward model.
            #    
            #       NP - number of points in an atmospheric profile, all profiles in an Atmosphere component of the forward model 
            #               should have the same number of points.
            #    
            #       NLOCATIONS - number of locations defined in the Atmosphere component of the forward model.
            #
            #   The size of the 1st dimension (NVMR+2+NDUST) is like that because it packs in 4 different atmospheric profile
            #   types: gas volume mixing ratios (NVMR), aerosol densities (NDUST), fractional cloud cover (1), para H2 fraction (1).
            
        ) -> None:
        """
            Updated values of components based upon values of model parameters in the state vector. Called from ForwardModel_0::subprofretg.
            
            ## ARGUMENTS ##
            
                forward_model : ForwardModel_0
                    The ForwardModel_0 instance that is calling this function. We need this so we can alter components of the forward model
                    inside this function.
                
                ix : int
                    The index of the state vector that corresponds to the start of the model's parameters
                    
                ipar : int
                    An integer that encodes which part of the atmospheric component of the forward model this model should alter. Only
                    used for some Atmospheric models.
                
                ivar : int
                    The model index, the order in which the models were instantiated. NOTE: this is a vestige from the
                    FORTRAN version of the code, we don't really need to know this as we should be: 1) storing any model-specific
                    values on the model instance itself; 2) passing any model-specific data from the outside directly
                    instead of having the model instance look it up from a big array. However, the code for each model
                    was recently ported from a more FORTRAN-like implementation so this is still required by some of them
                    for now.
                
                xmap : np.ndarray[[nx,NVMR+2+NDUST,NP,NLOCATIONS],float]
                    Functional derivatives of the state vector w.r.t Atmospheric profiles at each Atmosphere location.
                    The array is sized as:
                        
                        nx - number of state vector entries.
                        
                        NVMR - number of gas volume mixing ratio profiles in the Atmosphere component of the forward model.
                        
                        NDUST - number of aerosol profiles in the Atmosphere component of the forward model.
                        
                        NP - number of points in an atmospheric profile, all profiles in an Atmosphere component of the forward model 
                             should have the same number of points.
                        
                        NLOCATIONS - number of locations defined in the Atmosphere component of the forward model.
                    
                    The size of the 1st dimension (NVMR+2+NDUST) is like that because it packs in 4 different atmospheric profile
                    types: gas volume mixing ratios (NVMR), aerosol densities (NDUST), fractional cloud cover (1), para H2 fraction (1).
                    
                
            ## RETURNS ##
            
                None
        """
        ...

    @abc.abstractmethod
    def patch_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            #   The ForwardModel_0 instance that is calling this function. We need this so we can alter components of the forward model
            #   inside this function.
            
            ix : int,
            #   The index of the state vector that corresponds to the start of the model's parameters
            
            ipar : int,
            #   An integer that encodes which part of the atmospheric component of the forward model this model should alter. Only
            #   used for some Atmospheric models.
            
            ivar : int,
            #   The model index, the order in which the models were instantiated. NOTE: this is a vestige from the
            #   FORTRAN version of the code, we don't really need to know this as we should be: 1) storing any model-specific
            #   values on the model instance itself; 2) passing any model-specific data from the outside directly
            #   instead of having the model instance look it up from a big array. However, the code for each model
            #   was recently ported from a more FORTRAN-like implementation so this is still required by some of them
            #   for now.
            
            xmap : np.ndarray,
            #   Functional derivatives of the state vector w.r.t Atmospheric profiles at each Atmosphere location.
            #   The array is sized as:
            #    
            #       nx - number of state vector entries.
            #    
            #       NVMR - number of gas volume mixing ratio profiles in the Atmosphere component of the forward model.
            #    
            #       NDUST - number of aerosol profiles in the Atmosphere component of the forward model.
            #    
            #       NP - number of points in an atmospheric profile, all profiles in an Atmosphere component of the forward model 
            #               should have the same number of points.
            #    
            #       NLOCATIONS - number of locations defined in the Atmosphere component of the forward model.
            #
            #   The size of the 1st dimension (NVMR+2+NDUST) is like that because it packs in 4 different atmospheric profile
            #   types: gas volume mixing ratios (NVMR), aerosol densities (NDUST), fractional cloud cover (1), para H2 fraction (1).
        ) -> None:
        """
            Patches values of components based upon values of model parameters in the state vector. Called from ForwardModel_0::subprofretg.
            
            ## ARGUMENTS ##
            
                forward_model : ForwardModel_0
                    The ForwardModel_0 instance that is calling this function. We need this so we can alter components of the forward model
                    inside this function.
                
                ix : int
                    The index of the state vector that corresponds to the start of the model's parameters
                    
                ipar : int
                    An integer that encodes which part of the atmospheric component of the forward model this model should alter. Only
                    used for some Atmospheric models.
                
                ivar : int
                    The model index, the order in which the models were instantiated. NOTE: this is a vestige from the
                    FORTRAN version of the code, we don't really need to know this as we should be: 1) storing any model-specific
                    values on the model instance itself; 2) passing any model-specific data from the outside directly
                    instead of having the model instance look it up from a big array. However, the code for each model
                    was recently ported from a more FORTRAN-like implementation so this is still required by some of them
                    for now.
                
                xmap : np.ndarray[[nx,NVMR+2+NDUST,NP,NLOCATIONS],float]
                    Functional derivatives of the state vector w.r.t Atmospheric profiles at each Atmosphere location.
                    The array is sized as:
                        
                        nx - number of state vector entries.
                        
                        NVMR - number of gas volume mixing ratio profiles in the Atmosphere component of the forward model.
                        
                        NDUST - number of aerosol profiles in the Atmosphere component of the forward model.
                        
                        NP - number of points in an atmospheric profile, all profiles in an Atmosphere component of the forward model 
                             should have the same number of points.
                        
                        NLOCATIONS - number of locations defined in the Atmosphere component of the forward model.
                    
                    The size of the 1st dimension (NVMR+2+NDUST) is like that because it packs in 4 different atmospheric profile
                    types: gas volume mixing ratios (NVMR), aerosol densities (NDUST), fractional cloud cover (1), para H2 fraction (1).
                    
                
            ## RETURNS ##
            
                None
        """
        ...
    
    @abc.abstractmethod
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            #   The ForwardModel_0 instance that is calling this function. We need this so we can alter components of the forward model
            #   inside this function.
            
            ix : int,
            #   The index of the state vector that corresponds to the start of the model's parameters
            
            ivar : int,
            #   The model index, the order in which the models were instantiated. NOTE: this is a vestige from the
            #   FORTRAN version of the code, we don't really need to know this as we should be: 1) storing any model-specific
            #   values on the model instance itself; 2) passing any model-specific data from the outside directly
            #   instead of having the model instance look it up from a big array. However, the code for each model
            #   was recently ported from a more FORTRAN-like implementation so this is still required by some of them
            #   for now.
            
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            #   Modelled spectrum that we want to alter with this model. NOTE: do not assign directly to this, always
            #   assign to slices of it. If you assign directly, then only the **reference** will change and the value
            #   outside the function will not be altered.
            #
            #   The shape is defined as:
            #
            #       NCONV - number of "convolution points" (i.e. wavelengths/wavenumbers) in the modelled spectrum.
            #
            #       NGEOM - number of "geometries" (i.e. different observation setups) in the modelled spectrum.
            
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
            #   Gradients of the spectrum w.r.t each entry of the state vector.NOTE: do not assign directly to this, always
            #   assign to slices of it. If you assign directly, then only the **reference** will change and the value
            #   outside the function will not be altered.
            #
            #   The shape is defined as:
            #
            #       NCONV - number of "convolution points" (i.e. wavelengths/wavenumbers) in the modelled spectrum.
            #
            #       NGEOM - number of "geometries" (i.e. different observation setups) in the modelled spectrum.
            #
            #       NX - Number of entries in the state vector.
            
            
            
        ) -> None:
        """
            Updated spectra based upon values of model parameters in the state vector. Called from ForwardModel_0::subspecret.
        """
        ...