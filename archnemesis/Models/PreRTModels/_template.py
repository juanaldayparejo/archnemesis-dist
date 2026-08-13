
# flake8: noqa # Do not check this file as it is just a template

from typing import TYPE_CHECKING, Self, IO, ClassVar

import numpy as np


from ..param import StateParam, ConstParam
from ._base import PreRTModelBase
import dataclasses as dc


from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum


if TYPE_CHECKING:
    from ..type_checking import ForwardModel_0, Atmosphere_0, Variables_0, mparam



"""
Below is a template model that has some example parameters and more comments than a normal model would.

All model class must inherit from `ModelBase` at some point, usually either through `PreRTModelBase` or `PostRTModelBase`.

For ease of use, model classes should be [*dataclasses* ](https://docs.python.org/3/library/dataclasses.html). This allows us
to define instance attributes in the class body instead of having to make an `__init__()` method. 

Having instance attributes in the class body means we can easily see exactly what attributes the model has and what they are for,
instance attributes can be one of three types:

* StateParam - parameters that are retrieved by ArchNemesis, these are packed into the state vector.

* ConstParam - parameters that are not retrieved by ArchNemesis, they are stored on the model instance.

* VarParam - parameters that are not retrieved by ArchNemesis, they are stored on the model instance and 
             in the `varparams` array. These are used when porting over old FORTRAN-NEMESIS code as 
             `varparams` is used by FORTRAN-NEMESIS to make various files (e.g. *.mre files and bookmarks).
             However, only floating-point (or easily converted) values can be saved here. Therefore, for new
             models using ConstParam is recommended.


Values are assigned to `StateParam`, `ConstParam`, and `VarParam` attributes when the model class is constructed,
the helper class method `from_arrays(...)` is available to assist.



## StateParam ##

To define an instance attribute as a `StateParam`, set the type-hint as below:

    `StateParam.using(<slice>, <description>, <unit (optional)>)`

where:

    <slice> - A python `slice` object that denotes where in the model's region of the state vector the StateParam's
              value is stored. NOTE: This determines where in the state vector a StateParam's value is stored.
    
    <description> - A string that describes the intended use of the attribute. Will appear in various diagnostic information.
    
    <unit (optional)> - A string representation of the unit, or "UNKNOWN" if not set. Will appear in various diagnostic information.


A `StateParam` object has the following members:

    stateparam.slice - Set when defined via `<slice>`
    
    stateparam.description - Set when defined via `<description>`
    
    stateparam.unit - Set when defined via `<unit>`
    
    stateparam.v - The value held by the `StateParam`, this is what gets packed into the state vector.
                   NOTE: packing into and unpacking from the state vector handles log and 
                   un-log operations, so this always holds the un-logged value.
    
    stateparam.e - The error held by the `StateParam`, this is what gets packed into the covariance matrix.
                   NOTE: packing into and unpacking from the covariance matrix handles log and
                   un-log operations, so this always holds the un-logged value.
    
    stateparam.log - A boolean flag, if `True` will store the logarithm of `stateparam.v` and `stateparam.e`
                     in to the state vector and covariance matrix respecitvely. If `False` stores raw value.
                     Defaults to `True`.
    
    stateparam.num_diff - A boolean flag, if `True` will use numerical differentiation, if `False` will use
                          analytical differentiation (if available). Defaults to `False`



## ConstParam and VarParam ##

To define an instance attribute as a `ConstParam` or `VarParam`, set the type-hint to one of the following:

    `ConstParam[<type>].using(<description>, <units (optional)>)`

    `VarParam[<type>].using(<description>, <units (optional)>)`

where:

    <type> - The type of the variable, NOTE: The type is not checked at any point (python is dynamically typed), so be on guard
             when using `VarParam` as the value may be a `float` unless you have explicity cast it somewhere else.

    <description> - A string that describes the intended use of the attribute. Will appear in various diagnostic information.
    
    <unit (optional)> - A string representation of the unit, or "UNKNOWN" if not set. Will appear in various diagnostic information.


A `ConstParam` object and a `VarParam` object have the same members:

    xparam.description - Set when defined via `<description>`
    
    xparam.unit - Set when defined via `<unit>`
    
    xparam.v - The value stored in the `ConstParam` or `VarParam` object. NOTE: this **should** be of the same
               type as when defined via `<type>`, however no checks are made.

"""


@dc.dataclass # Has to be a dataclass so we can define what attributes are `StateParam`, `ConstParam` and `VarParam`
class TemplatePreRTModel(PreRTModelBase):
    """
        This docstring acts as the description for the model, **REPLACE THIS**.
    """
    
    # The ID of the model **MUST BE A UNIQUE INTEGER**, check with the github to see taken and not-taken model ID numbers.
    id : ClassVar[int] = None 
    
    # Any "StateParam" attributes are retrieved by ArchNemesis
    # It is best to define these in the same order they appear in the state-vector, and the 
    # same order they are read in from the *.apr file.
    retrieved_parameter_1 : StateParam.using(slice(0,1), "retrieve a single floating point value", "units (optional)")                 # noqa: F722 F821
    retrieved_parameter_2 : StateParam.using(slice(1,3), "retrieve two floating point values", "units (optional)")                     # noqa: F722 F821
    retrieved_parameter_3 : StateParam.using(slice(3,None), "retrieve a variable amount of floating point values", "units (optional)") # noqa: F722 F821
    
    # Any "ConstParam" attributes are not retrieved, they store auxiliary information and can hold any values.
    constant_parameter_1 : ConstParam[int].using('A constant integer, e.g. for saving the number of entries in a retrieved parameter of variable length') # noqa: F722 F821
    constant_parameter_2 : ConstParam[str].using('A constant string, e.g. for saving filepaths')                                                          # noqa: F722 F821
    
    # Any "VarParam" attributes are not retrieved, they store auxiliary information like "ConstParam" but they must be 
    # floating point numbers, they are saved and loaded from "varparams" for compatibility with FORTRAN-NEMESIS
    var_param_1 : VarParam[float].using('A constant floating point number, will be saved and loaded from `varparams` for compatibility with FORTRAN-NEMESIS') # noqa: F722 F821
    var_param_2 : VarParam[int].using('A constant integer, will be represented in `varparams` as a float and coerced to an `int` when loaded.')               # noqa: F722 F821
    
    
    
    # NOTE: Pushing the `StateParam` values into the state vector is handled via `cls.from_apr_to_state_vector(...)`
    #       which calls this method `cls.from_apr_file(...)` internally. `cls.from_apr_to_state_vector(...)` is
    #       provided by the `ModelBase` class. You almost certainly do not need to overwrite that method.
    @classmethod
    def from_apr_file(
            cls,
            f : IO, # file object of the *.apr file
            varident : np.ndarray[[3],int], # varident (three integers) for this model
            npro : int, # number of levels in each profile
            ngas : int, # number of gasses
            ndust : int, # number of aerosols
            nlocations : int, # number of locations
            runname : str, # <runname> in "<runname>.apr`
            sxminfac : float, # minimum factor included in covariance matrix
            input_file_type : ArchNemesisFileTypeEnum, # what kind of input files are we using, HDF5 or LEGACY.
    ) -> Self:
        """
        This is a **factory method**, so called because it creates an instance of the model
        class, modifies it to the requirements specified in the *.apr file, and returns it.
        
        
        
        ## RETURNS ##
            instance : Self
                The model instance created by this method.
        """
        raise NotImplementedError('This is a template model and should never be used')
        
        # Helper functions exist to make it easier to interact with *.apr files
        
        n = 3
        xvals, xerrs = cls.read_apr_value_error_pairs(f, n) # reads `n` (value, error) pairs from the *.apr file object `f` and returns them as a pair of arrays (values, errors)
        
        (
            c1, 
            c2 
        ) = cls.read_apr_entries(f, (int,str)) # reads specified types from *.apr file object `f`
        
        # You can also use the file object directly
        v1 = float(f.readline())
        v2 = int(f.readline())
        
        # The `cls.from_arrays(...)` factory method constructs an instance
        # from two arrays that hold the values and errors for the `StateParam`
        # attributes, and other arguments that hold values for the `ConstParam`
        # and `VarParam` attributes.
        # They should be passed in the order they are defined on the class in.
        instance = cls.from_arrays(
            xvals,
            xerrs,
            c1,
            c2,
            v1,
            v2
        )
        
        # Set logarithm and numerical differentiation after the object is created
        instance.retrieved_parameter_1.log = False
        
        instance.retrieved_parameter_3.num_diff = True
        
        # Finally, return the instance we created
        return instance
    
    
    def calculate(
            self, # this is an instance method, so the first argument is called `self`.
            
            # Other arguments are optional, but they are how you get data into this method
            # from `ForwardModel_0` via the `self.calculate_from_subprofretg(...)` method.
            
            atm : "Atmosphere_0", # Instance of Atmosphere_0 class we are operating upon
            atm_profile_type : AtmosphericProfileTypeEnum, # Enum that defines what kind of profile we should operate upon
            atm_profile_idx : int, # Index of the profile we should operate upon
            
    ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            This is a description of the method
            
            ## Returns ##
                atm : Atmosphere_0
                    The `Atmosphere_0` instance we have operated upon
                
                xmap : np.ndarray
                    Matrix relating functional derivatives to elements in the state vector.
                    
        """
        raise NotImplementedError('This is a template model and should never be used')
        
        # Often the instance attributes are unpacked so we can use shorter names in the body of `calculate(...)`
        # NOTE: you need to use the `.v` member of the attribute to get the actual **value** stored in it.
        r1 = self.retrieved_parameter_1.v
        r2 = self.retrieved_parameter_2.v
        r3 = self.retrieved_parameter_3.v
        
        e1 = self.retrieved_parameter_1.e
        e2 = self.retrieved_parameter_2.e
        e3 = self.retrieved_parameter_3.e
        
        c1 = self.constant_parameter_1.v
        c2 = self.constant_parameter_2.v
        c3 = self.var_param_1.v
        c4 = self.var_param_2.v
        
        raise NotImplementedError('This is a template model and should never be used')
        
        atm.do_something(...)
        
        xmap = NotImplemented
        return atm, xmap


    @classmethod
    def from_bookmark(
            cls,
            variables : "Variables_0", # The "Variables_0" instance that enables acccess to `variables.classify_model_type_from_varident`
            varident : np.ndarray[[3],int], #"Variable Identifier" from a *.apr file. Consists of 3 integers. Exact interpretation depends on the model subclass.
            varparam : np.ndarray[["mparam"],float], #"Variable Parameters" from a *.apr file. Holds "extra parameters" for the model. Exact interpretation depends on the model subclass
            ix : int, # The index of the next free entry in the state vector, should not really be used in this method
            npro : int, # Number of altitude levels defined for the atmosphere component of the retrieval setup.
            ngas : int, # Number of gas volume mixing ratio profiles defined for the reference atmosphere
            ndust : int, # Number of aerosol species density profiles define for the reference atmosphere
            nlocations : int, # Number of locations defined for the atmosphere component of the retrieval setup.
    ) -> Self:
        """
            Constructs the model when it is loaded from a bookmark. 
            
            The state vector, `varident`, `varparms`, etc. should all have been loaded from the bookmark by 
            this point, therefore this does not need to set any state vector information.
            
            What this method *should* do is construct and return a model instance similarly to 
            `self.from_apr_to_state_vector(...)`, but *not* set anything on the state vector. The `StateParam`
            attributes should be given **dummy values** as they will be overwritten by the values loaded
            from the state vector and covariance matrix.
            
            ## RETURNS ##
            
                instance : Self
                    A constructed instance of the model class that has parameters set from information the bookmark.
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        v1 = varparam[0]
        v2 = int(varparam[1])
        
        c1, c2 = ... # These should be saved and loaded from somewhere, the exact supported mechanism is in progress.
        
        # Another way to construct a model instance is to give it a
        # (value, error) tuple for each `StateParam`, then
        # values for each `ConstParam` and `VarParam`.
        instance = cls(
            (0,0), # value, error for `retrieved_parameter_1`
            (0,0), # value, error for `retrieved_parameter_2`
            (0,0), # value, error for `retrieved_parameter_3`
            c1,
            c2,
            v1,
            v2
        )
        
        return instance


    # NOTE: The `PreRTModelBase` class provides an implementation of `self.calculate_from_subprofretg(...)` that
    #       is useful for many atmospheric models. However, some models do need a custom one.
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0", # Forward model instance that is calling this function.
            ix : int, # Index of the state vector that corresponds to the start of the model's parameters
            ipar : int, # Integer encoding which parts of the atmosphere this model alters, **not used for all models**.
            ivar : int, # Model index (the order in which the models are instantiated), **vestigal from FORTRAN version**.
            xmap : np.ndarray, # Functional derivatives of the state vector.
    ) -> None:
        """
            This method is called from ForwardModel_0::subprofretg and should:
            
            1) pull values from the state vector
            2) call the `self.calculate(...)` method
            3) put the results of the calculation where they should be
            
            Some example code is placed in this method as the idioms have been found to be useful.
            
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
                    FORTRAN version of the code, we don't really need to know this as we should be: 
                        
                        1) storing any model-specific values on the model instance itself; 
                        
                        2) passing any model-specific data from the outside directly instead of having the model instance 
                        look it up from a big array. 
                    
                    However, the code for each model was recently ported from a more FORTRAN-like implementation so this 
                    is still required by some of them for now.
                
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
                    It is indexed by the `ipar` argument.
                    
                
            ## RETURNS ##
            
                None
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        # Example code for unpacking information from the `ipar` argument
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        # The model instance's `StateParam` values are pulled from the state vector
        self.pull_from_state_vector(
            forward_model.Variables.XN,
            forward_model.Variables.LX,
        )
        # at this point all the `StateParam` attributes have new values set from
        # whatever was in the state vector
        

        # Example code for calling the `self.calculate(...)` class method
        # NOTE: we can call the class method via the `self` instance.
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
        )
        
        # Example code for packing the results of the calculation back into the forward model
        # and the matrix that holds functional derivatives.
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return