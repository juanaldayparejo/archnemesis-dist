

from typing import TYPE_CHECKING, Self, IO, Any

import numpy as np


from ..ModelParameter import ModelParameter
from ._base import PreRTModelBase


from archnemesis.enum import AtmosphericProfileTypeEnum


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    #from archnemesis.Scatter_0 import Scatter_0
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

class TemplatePreRTModel(PreRTModelBase):
    """
        This docstring acts as the description for the model, **REPLACE THIS**.
    """
    id : int = None # This is the ID of the model, it **MUST BE A UNIQUE INTEGER**.
    
    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM that tells us what kind of atmospheric profile this model instance represents
            
            # Extra arguments to this method can store constants etc. that the
            # model requires, but we do not want to retrieve. Useful for setting things like dust ID numbers.
            #####################################################################################################
            # Alter (and add, or remove if not needed) the below to set non-retrievable arguments for the model #
            #####################################################################################################
            example_template_argument : Any,
    ):
        """
            Initialise an instance of the model.
        """
        
        # Remove the below line when copying this template and altering it
        raise NotImplementedError('This is a template model and should never be used')
        
        # Initialise the parent class
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)
        
        
        # To store any constants etc. that the model instance needs, pass them
        # as arguments to this method and set them on the instance.
        # These can then be used in any method that is not
        # a class method.
        
        #######################################################################################
        # Alter (and add, or remove if not needed) the below to add arguments the model needs #
        #######################################################################################
        self.example_template_argument : Any = example_template_argument
        
        
        # NOTE: It is best to define the parameters in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        
        ############################################################
        # Alter the below to reflect the parameters for your model #
        ############################################################
        self.parameters : tuple[ModelParameter] = (
            ModelParameter(
                'single_value_parameter_name', 
                slice(0,1), 
                'This parameter takes up a single slot in the state vector', 
                'The unit of this parameter goes here, e.g. "km", "NUMBER", "UNDEFINED"'
            ),
            
            ModelParameter(
                'multi_value_parameter_name', 
                slice(1,11), 
                'This parameter takes up 10 slots in the state vector', 
                'unit_placeholder'
            ),
            
            ModelParameter(
                'variable_length_parameter_name', 
                slice(11,n_state_vector_entries/2), 
                'This parameter takes up a variable number of slots in the state vector dependent on arguments to this __init__(...) method', 
                'unit_placeholder'
            ),
            
            ModelParameter(
                'another_variable_length_parameter_name', 
                slice(n_state_vector_entries/2,None), 
                'The slice set here is bounded by the range of the entire slice of the state vector devoted to this model, so we do not have to specify an end value if we know we want the entire thing.', 
                'unit_placeholder'
            ),
            
        )
        
        
        return
    
    @classmethod
    def calculate(
            # NOTE:
            # This is a CLASS METHOD (as specified by the `@classmethod` decorator.
            # instance attributes (e.g. those set in the __init__(...) method) will
            # not be available, so they should be passed into to this method.
            
            cls, 
            # NOTE:
            # the `cls` argument is the ONLY ARGUMENT THAT MUST BE DEFINED
            # the other arguments here are example ones, but they are commonly used by
            # this kind of model class.
            
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            # NOTE:
            # Extra arguments to this function are usually the values stored in the state vector as 
            # described by the `ModelParameter` instances in the `self.parameters` attribute which is
            # defined in the __init__(...) method.
            #
            # Extra arguments are often of type `float` or `np.ndarray[[int],float]` this is because
            # if the extra arguments are from the state vector, the state vector is an array of `float`s
            # so single values are normally passed here as `float` and multiple values passed as an array
            # of `float`s
            
            ####################################################################################
            # Example extra arguments are below, they correspond to the examples in `__init__` #
            ####################################################################################
            single_value_parameter_name : float,
            multi_value_parameter_name : np.ndarray[[10],float],
            variable_length_parameter_name : np.ndarray[[int],float],
            another_variable_length_parameter_name : np.ndarray[[int],float],
            
            ######################################################################################
            # An example template argument is below, it corresponds to the example in `__init__` #
            ######################################################################################
            example_template_argument : Any,
    ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            This class method should perform the actual calculation. Ideally it should not know anything
            about the geometries, locations, etc. of the retrieval setup. Try to make it just perform the
            actual calculation and not any "data arranging". 
            
            For example, instead of passing the geometry index, pass sliced arrays, perform the calculation, 
            and put the result back into the "source" arrays.
            
            This makes it easier to use this class method from another source if required.
            
            ## Returns ##
                atm : Atmosphere_0
                    The `Atmosphere_0` instance we have operated upon
                
                xmap : np.ndarray
                    Matrix relating functional derivatives to elements in the state vector.
                    
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        xmap = NotImplemented
        return atm, xmap


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
        """
            This class method should read information from the <runname>.apr file, store
            this model class's parameters in the state vector, set appropriate state vector flags,
            pass constants etc. to the class's __init__(...) method, and return the constructed
            model class instance.
            
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
        
        raise NotImplementedError('This is a template model and should never be used')

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        ix_0 = NotImplemented
        
        return cls(ix_0, ix-ix_0, model_classification[1], "example_template_argument_values")


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
                    A constructed instance of the model class that has parameters set from information the bookmark.
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"
        
        ix_0 = NotImplemented
        
        return cls(ix_0, ix-ix_0, model_classification[1], "example_template_argument_values")


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
            2) call the `self.calculate(...)` class method
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
        
        # Example code for unpacking parameters from the state vector
        # NOTE: this takes care of 'unlogging' values when required.
        (
            single_value_parameter_name,
            multi_value_parameter_name,
            variable_length_parameter_name,
            another_variable_length_parameter_name,
        
        ) = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        
        # Example code for calling the `self.calculate(...)` class method
        # NOTE: we can call the class method via the `self` instance.
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            
            # Paramters defined in `__init__`
            single_value_parameter_name,
            multi_value_parameter_name,
            variable_length_parameter_name,
            another_variable_length_parameter_name,
            
            # Template arguments defined in `__init__`. NOTE: These will not be retrieved
            self.example_template_argument,
        )
        
        # Example code for packing the results of the calculation back into the forward model
        # and the matrix that holds functional derivatives.
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return