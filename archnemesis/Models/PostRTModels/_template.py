
# flake8: noqa # Do not check this file as it is just a template

from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from archnemesis.enum import ArchNemesisFileTypeEnum
from ._base import PreRTModelBase


if TYPE_CHECKING:
    from ..type_checking import ForwardModel_0, Variables_0, NCONV, NX, mparam, NGEOM


@dc.dataclass # Has to be a dataclass so we can define what attributes are `StateParam`, `ConstParam` and `VarParam`
class TemplatePreRTModel(PreRTModelBase):
    
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
            self,
            
            SPECMOD : np.ndarray[['NCONV'],float],
            #   Modelled spectrum
            
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            #   Gradient of modelled spectrum
            
            # Other arguments are optional, but they are how you get data into this method
            # from `ForwardModel_0` via the `self.calculate_from_subprofretg(...)` method.
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        """
            This class method should perform the actual calculation. Ideally it should not know anything
            about the geometries, locations, etc. of the retrieval setup. Try to make it just perform the
            actual calculation and not any "data arranging". 
            
            For example, instead of passing the geometry index, pass sliced arrays, perform the calculation, 
            and put the result back into the "source" arrays.
            
            This makes it easier to use this class method from another source if required.
            
            ## RETURNS ##
                SPECMOD : np.ndarray
                    value of spectra
                
                dSPECMOD : np.ndarray
                    gradient of spectra w.r.t variable parameters
            
        """
        raise NotImplementedError('This is a template model and should never be used')
        
        # Often the instance attributes are unpacked so we can use shorter names in the body of `calculate(...)`
        # NOTE: you need to use the `.v` member of the `*Param` attributes to get the actual **value** stored in it.
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
        
        # Return the results of the calculation
        return SPECMOD, dSPECMOD

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
            Updates the spectra based upon values of model parameters in the state vector. Called from ForwardModel_0::subspecret.
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        # The model instance's `StateParam` values are pulled from the state vector
        self.pull_from_state_vector(
            forward_model.Variables.XN,
            forward_model.Variables.LX,
        )
        # at this point all the `StateParam` attributes have new values set from
        # whatever was in the state vector
        
        # Example code, generally want to loop over geometries here rather than in `self.calculate(...)`
        for i_geom in range(self.n_geom):
        
            # Example code for calling the `self.calculate(...)` class method
            # NOTE: we can call the class method via the `self` instance.
            specmod, dspecmod = self.calculate(
                SPECMOD[:,i_geom],
                dSPECMOD[:,i_geom,:],
            )
        
            # Example code for packing the results of the calculation back into the spectra
            # and the matrix that holds functional derivatives.
            SPECMOD[:,i_geom] = specmod
            dSPECMOD[:,i_geom,:] = dspecmod
        
        return