
import abc
from typing import TYPE_CHECKING, IO, Any, Self, Type

import numpy.ma
import numpy as np
from archnemesis.enum import AtmosphericProfileTypeEnum, GasEnum, ArchNemesisFileTypeEnum

from .state_vector_modifier import StateVectorModifier
from .model_tree_printer import ModelTreePrinter
from .param import ParamMixin

#from .log import _lgr

if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    
    nx = 'number of elements in state vector'
    m = 'an undetermined number, but probably less than "nx"'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'
    NCONV = 'number of spectral bins'
    NGEOM = 'number of geometries'
    NX = 'number of elements in state vector'
    NDEGREE = 'number of degrees in a polynomial'
    NWINDOWS = 'number of spectral windows'





class ModelBase(StateVectorModifier, ModelTreePrinter, ParamMixin, abc.ABC):
    """
        Abstract base class of all parameterised models used by ArchNemesis. This class should be subclassed further for models of a particular component.
        
        These models work a bit differently to "normal" classes in Python. Normally, a python class holds onto it's state,
        however in this case the "state vector" `Variables.XN` holds the state of the model's parameters (or `Variables.XA` for apriori 
        values of the parameters) and `Variables.VARPARAM` holds the model's constants.
        
        This means that `__init__` actually defines the names of the model's parameters (in `self.parameters`), and
        stores the number and index of the instance's parameters on the state vector.
        
        In other words, what this class actually stores is a "pointer" (or an index) to the model's state. Plus a set of
        methods that let us do the following: 
            1) Read/write the model state stored in the state vector.
            2) Perform calculations based upon the model state.
            3) Put the results from calculations into the correct part of a ForwardModel instance.
    """
    
    id : int = None # All "*ModelBase" classes that are not meant to be used should have an id of 'None'
    
    @staticmethod
    def get_model_profile_type_enum_from_varident(
            varident : np.ndarray[[3],int],
            ngas : int,
            ndust : int
        ) -> None | AtmosphericProfileTypeEnum:
        """
        Works out the type of model (and subtype if applicable) identified by a VARIDENT triplet.
        
        ## ARGUMENTS ##
            
            varident : np.ndarray[[3],int]
                Three integers that identify a model
                
            ngas : int
                The number of gases present in the reference atmosphere
            
            ndust : int
                The number of aerosol species present in the reference atmosphere
            
        ## RETURNS ##
        
            None | AtmosphericProfileTypeEnum
                The retrieval component that the model parameterises (and therefore
                alters). This is 'None' when unknown, or an ENUM corresponding to an attribute
                of the retrieval component that the model parameterises.
        """
        if varident[0] == 0:
            return AtmosphericProfileTypeEnum.TEMPERATURE
        elif (varident[0] > 0) and int(varident[0]) in iter(GasEnum):
            return AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO
        elif (varident[0] < 0) and (-varident[0]) <= ndust:
            return AtmosphericProfileTypeEnum.AEROSOL_DENSITY
        elif (varident[0] < 0) and (-varident[0]) == ndust + 1:
            return AtmosphericProfileTypeEnum.PARA_H2_FRACTION
        elif (varident[0] < 0) and (-varident[0]) == ndust + 2:
            return AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE
        
        return None # Not an AtmosphericProfileType
    
    
    @staticmethod
    def read_apr_value_error_pairs(
        f : IO, # File object of *.apr file.
        n_pairs : int, # Number of pairs to read from the file
    ) -> tuple[np.ndarray,np.ndarray]:
        xvals = np.zeros((n_pairs,), dtype=float)
        xerrs = np.zeros((n_pairs,), dtype=float)
        for i in range(xvals.size):
            xvals[i], xerrs[i] = map(float, f.readline().rsplit('!',maxsplit=1)[0].strip().split(maxsplit=1))
        return xvals, xerrs
    
    @staticmethod
    def read_apr_entries(
        f : IO, # File object of *.apr file.
        types : tuple[Type,...] = (float, float), # Types to read per line
        n_lines : int = 1, # Number of pairs to read from the file
        return_values_if_single_line : bool = True,
        return_values_if_single_type : bool = True,
    ) -> list[np.ndarray]:
        entries = []
        n_entries_per_line = len(types)
        for j in range(n_entries_per_line):
            if types[j] is str:
                dtype = np.dtype('T')
            else:
                dtype = types[j]
            
            entries.append(
                np.zeros((n_lines,), dtype=dtype)
            )
        
        #print(f'TESTING: {n_lines=}')
        #print(f'TESTING: {entries=}')
        for i in range(n_lines):
            #print(f'TESTING: {i=} {types=}')
            for j, x in enumerate(f.readline().rsplit('!',maxsplit=1)[0].strip().split(maxsplit=(n_entries_per_line-1))):
                #print(f'TESTING: {j=} {x=}')
                #print(f'TESTING: {entries[j]=}')
                entries[j][i] = types[j](x)
        
        if return_values_if_single_line and n_lines == 1:
            if return_values_if_single_type and n_entries_per_line == 1:
                return entries[0][0]
            else:
                return [x[0] for x in entries]
        else:
            if return_values_if_single_type and n_entries_per_line == 1:
                return entries[0]
            else:
                return entries
    
    
    @classmethod
    def constparam_read(cls, constparam_bytes : bytes) -> tuple[Any]:
        if len(cls._constparam_names) >0:
            raise NotImplementedError()
        else:
            return tuple()
    
    def constparam_write(self) -> bytes:
        if len(self._constparam_names) >0:
            raise NotImplementedError()
    
    @classmethod
    def varparam_read(cls, varparam : np.ndarray) ->tuple[float,...]:
        x = []
        for i, vp in enumerate(cls.iter_varparam_objs()):
            x.append(
                vp.__specialised_types__[0](varparam[i])
            )
        return tuple(x)
    
    def varparam_write(self) -> np.ndarray:
        if self._n_varparams > 0:
            return np.array([x for x in self.iter_varparam_values()], dtype=float)
        return np.zeros((0,), dtype=float)
    
    
    @classmethod
    def from_arrays(
            cls, 
            xvals : np.ndarray, 
            xerrs : np.ndarray, 
            *args : tuple[Any,...,],
            **kwargs : dict[str, Any],
    ) -> Self:
        """
        Create a model instance from an array of values, an array of errors, and arguments of values for ConstParam and VarParam attributes.
        """
        stateparam_parts = []
        for stateparam_name, stateparam_type in cls.iter_stateparam_types():
            assert stateparam_name not in kwargs, f'{cls.__name__}.from_arrays(...) cannot have stateparams set via arguments or keyword arguments. StateParams are set from input `xvals` and `xerrs` vectors.'
            stateparam_parts.append(
                (
                    xvals[stateparam_type.slice],
                    xerrs[stateparam_type.slice],
                )
            )
        
        return cls(
            *stateparam_parts,
            *args,
            **kwargs
        )
    
    
    def get_n_stateparam_entries(self):
        """
        Get the number of entries in the state vector required for all stateparams of this model.
        """
        n = 0
        for v in self.iter_stateparam_values():
            n += v.size
        return n
    
    
    @classmethod
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
            
            input_file_type : ArchNemesisFileTypeEnum,
            #   Type of input files that we are reading the model from.
    ) -> Self:
        """
            Constructs a model from its entry in a *.apr file. 
            A default implementation is provided, may be overwritten by subclass if different logic is required.
            
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
        
        instance = cls.from_apr_file(
            f,
            varident,
            npro,
            ngas,
            ndust,
            nlocations,
            runname,
            sxminfac,
            input_file_type,
        )
        
        instance.set_state_vector_region(ix, instance.get_n_stateparam_entries())
        instance.push_to_state_vector(x0, lx)
        instance.push_to_covariance_matrix(sx, sxminfac)
        instance.push_to_numerical_differentiation_vector(inum)
        varparam[:cls._n_varparams] = instance.varparam_write()
        
        return instance
    
    
    @classmethod
    def from_bookmark(
            cls,
            variables : "Variables_0", 
            #   An instance of the archnemesis.Variables_0 class
            
            varident : np.ndarray[[3],int], 
            #   Should be the correct slice of the original (which should be a reference to the sub-array)
            
            varparam : np.ndarray[["mparam"],float], 
            #   Should be the correct slice of the original (which should be a reference to the sub-array)
            
            ix : int, 
            #   The next free entry in the state vector
            
            npro : int, 
            #   Number of altitude levels defined for the atmosphere component
            
            ngas : int,
            #   Number of gas volume mixing ratio profiles defined for the reference atmosphere
            
            ndust : int,
            #   Number of aerosol species density profiles define for the reference atmosphere

            nlocations : int, 
            #   Number of locations defined for the atmosphere component
    ) -> Self:
        """
            Constructs the model when it is loaded from a bookmark. 
            
            The state vector, `varident`, `varparms`, etc. should all have been loaded from the bookmark by 
            this point, therefore this does not need to set any state vector information.
            
            What this method *should* do is construct and return a model instance similarly to 
            `self.from_apr_to_state_vector(...)`, but *not* set anything on the state vector as the state vector
            is populated elsewhere in this case.
            
            ## ARGUMENTS ##
            
                variables : Variables_0
                    The "Variables_0" instance that enables acccess to `variables.classify_model_type_from_varident`
                
                varident : np.ndarray[[3],int]
                    "Variable Identifier" from a *.apr file. Consists of 3 integers. Exact interpretation depends on the model
                    subclass.
                
                varparam : np.ndarray[["mparam"], float]
                    "Variable Parameters" from a *.apr file. Holds "extra parameters" for the model. Exact interpretation depends on the model
                    subclass. NOTE: this is a holdover from the FORTRAN code, the better way to give extra data to the model is to store it on the
                    model instance itself.
                
                ix : int
                    The index of the next free entry in the state vector
                
                npro : int
                    Number of altitude levels defined for the atmosphere component of the retrieval setup.
                
                ngas : int,
                    Number of gas volume mixing ratio profiles defined for the reference atmosphere
                
                ndust : int,
                    Number of aerosol species density profiles define for the reference atmosphere
                
                n_locations : int
                    Number of locations defined for the atmosphere component of the retrieval setup.
            
            ## RETURNS ##
            
                instance : Self
                    A constructed instance of the model class that has parameters set from information the bookmark provides.
        """
        assert cls.is_varident_valid(varident)
        
        constparam_vals = cls.constparam_read()
        varparam_vals = cls.varparam_read(varparam)
        
        sp_placeholders = []
        n_stateparam_entries = 0
        for sp_name, sp_type in cls.iter_stateparam_types():
            if sp_type.slice.stop is None:
                raise RuntimeError('Cannot use automatically generated "from_bookmark(...)" method as state parameters are not a constant size.')
            varparam_vals += (sp_type.slice.stop - sp_type.slice.start)
            
            sp_placeholders.append(
                (None,None)
            )
        
        instance = cls(
            *sp_placeholders,
            *constparam_vals,
            *varparam_vals
        )
        
        instance.set_state_vector_region(ix, n_stateparam_entries)
        
        instance.pull_from_state_vector(variables.XN)
        instance.pull_from_covariance_matrix(variables.SX)
    
        return instance
    
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
    def from_apr_file(
            cls,
            f : IO,                         # The open file descriptor of the *.apr file
            varident : np.ndarray[[3],int], # The three "varident" integers that were last read from the *.apr file
            npro : int,                     # Number of altitude levels defined for the Atmosphere component
            ngas : int,                     # Number of gas volume mixing ratio profiles defined for the reference atmosphere
            ndust : int,                    # Number of aerosol species density profiles define for the reference atmosphere
            nlocations : int,               # Number of locations defined for the atmosphere component
            runname : str,                  # Name of the *.apr file, without extension.
            sxminfac : float,               # Minimum factor to bother calculating covariance matrix entries, below this value entries will be zero
            input_file_type : ArchNemesisFileTypeEnum, # Type of input files that we are reading the model from.
    ) -> Self: # Instance of model class
        """
            Instance of model from values in *.apr file. Should be overwritten by subclass.
            NOTE: The region of the state vector the model instance interacts with is set later in `cls.from_apr_to_state_vector`.
            
            ## ARGUMENTS ##
            
                f : IO
                    The open file descriptor of the *.apr file
                
                varident : np.ndarray[[3],int]
                    The three "varident" integers that were last read from the *.apr file
                
                npro : int
                    Number of altitude levels defined for the Atmosphere component
                
                ngas : int
                    Number of gas volume mixing ratio profiles defined for the reference atmosphere
                
                ndust : int
                    Number of aerosol species density profiles define for the reference atmosphere
                
                nlocations : int
                    Number of locations defined for the atmosphere component
                
                runname : str
                    Name of the *.apr file, without extension.
                
                sxminfac : float
                    Minimum factor to bother calculating covariance matrix entries, below this value entries will be zero
            
            
            ## RETURNS ##
            
                instance : Self
                    Instance of model class with values assigned from contents of *.apr file.
        """
        ...
    

    
    
    @abc.abstractmethod
    def calculate(
            self, 
            *args, 
            **kwargs
        ) -> Any:
        """
            This class method should perform the lowest-level calculation for the model. Should be overwritten by each model.
            
            This is generally called from the `self.calculate_from_*` methods which are also specific to each model.
            
            NOTE: This is an instance method so all StateParam, ConstParam, VarParam attributes are available (remember to get their value via `self.<param>.v`).
            
            NOTE: Models are so varied in here that I cannot make any specific interface at this level of abstraction.
        """
        ...