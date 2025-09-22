#from __future__ import annotations #  for 3.9 compatability

"""
Contains models that alter the replica before radiative transfer is calculated
"""

from typing import TYPE_CHECKING, IO, Self, Any
import abc

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

from .ModelBase import ModelBase
from .ModelParameter import ModelParameter

import archnemesis.Data.constants as const
from archnemesis.helpers import h5py_helper
from archnemesis.helpers.maths_helper import ngauss
from archnemesis.Scatter_0 import kk_new_sub
from archnemesis.enums import AtmosphericProfileType
from archnemesis.enums import WaveUnit

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
    from archnemesis.Scatter_0 import Scatter_0
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

class PreRTModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact 
    with Components before the radiative transfer calculation is performed.
    """
    
    def __init__(
            self,
            state_vector_start : int, 
            n_state_vector_entries : int,
            atm_profile_type : AtmosphericProfileType = AtmosphericProfileType.NOT_PRESENT,
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


class TemplatePreRTModel(PreRTModelBase):
    """
        This docstring acts as the description for the model, REPLACE THIS.
    """
    id : int = None # This is the ID of the model, it **MUST BE A UNIQUE INTEGER**.
    
    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM that tells us what kind of atmospheric profile this model instance represents
            
            # Extra arguments to this method can store constants etc. that the
            # model requires, but we do not want to retrieve. There is a commented
            # out example below.
            #example_template_argument : type_of_argument,
        ):
        """
            Initialise an instance of the model.
        """
        
        # Remove the below line when copying this template and altering it
        raise NotImplementedError('This is a template model and should never be used')
        
        # Initialise the parent class
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)
        
        
        # To store any constants etc. that the model instance needs, pass them
        # as arguments to this method and set them on the instance like the
        # example below. These can then be used in any method that is not
        # a class method.
        """
        self.example_template_argument : type_of_argument = example_template_argument
        """
        
        # NOTE: It is best to define the parameters in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        
        # Uncomment the below and alter it to reflect the parameters for your model
        """
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
        """
        
        return
    
    @classmethod
    def calculate(
            # NOTE:
            # This is a CLASS METHOD (as specified by the `@classmethod` decorator.
            # instance attributes (e.g. those set in the __init__(...) method) will
            # not be available, so they should be passed to this method.
            
            cls, 
            # NOTE:
            # the `cls` argument is the ONLY ARGUMENT THAT MUST BE DEFINED
            # the other arguments here are example ones, but they are commonly used by
            # this kind of model class.
            
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
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
            
            # Example extra arguments are commented out below:
            
            #single_value_parameter_name : float,
            #multi_value_parameter_name : np.ndarray[[10],float],
            #variable_length_parameter_name : np.ndarray[[int],float],
            #another_variable_length_parameter_name : np.ndarray[[int],float],
            
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
        This class method should perform the actual calculation. Ideally it should not know anything
        about the geometries, locations, etc. of the retrieval setup. Try to make it just perform the
        actual calculation and not any "data arranging". 
        
        For example, instead of passing the geometry index, pass sliced arrays, perform the calculation, 
        and put the result back into the "source" arrays.
        
        This makes it easier to use this class method from another source if required.
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
        return cls(ix_0, ix-ix_0, model_classification[1])


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
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
            single_value_parameter_name,
            multi_value_parameter_name,
            variable_length_parameter_name,
            another_variable_length_parameter_name,
        )
        
        # Example code for packing the results of the calculation back into the forward model
        # and the matrix that holds functional derivatives.
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Modelm1(PreRTModelBase):
    """
    In this model, the aerosol profiles is modelled as a continuous profile in units
    of particles per gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
    are in particles per gram of atmosphere
    """
    
    id : int = -1

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('full_profile', slice(None), 'Every value for each level of the profile', 'PROFILE_TYPE'),
        )
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            xprof : np.ndarray[['mparam'],float],
            #   Full profile, this model defines every value for each profile level. Has been unlogged as required
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : modelm1()

            DESCRIPTION :

                Function defining the model parameterisation -1 in NEMESIS.
                In this model, the aerosol profiles is modelled as a continuous profile in units
                of particles perModelm1 gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
                are in particles per gram of atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileType
                        ENUM of atmospheric profile type we are altering.
                    
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                xprof(npro) :: Atmospheric aerosol profile in particles/cm3

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                    elements in state vector

            CALLING SEQUENCE:

                atm,xmap = modelm1(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        
        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model -1 :: Number of levels in atmosphere does not match and profile')
            
        if atm_profile_type == AtmosphericProfileType.AEROSOL_DENSITY:
            temp = np.array(atm.DUST)
            temp[:,atm_profile_idx] = xprof
            atm.edit_DUST(temp)
            xmap = np.diag(xprof)
        
        else:
            raise ValueError(f'error :: Model -1 is only compatible with aerosol profiles, not {atm_profile_type}')
            
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
        #* continuous cloud, but cloud retrieved as particles/cm3 rather than
        #* particles per gram to decouple it from pressure.
        #********* continuous particles/cm3 profile ************************
        ix_0 = ix
        
        if varident[0] >= 0:
            raise ValueError('error in read_apr_nemesis :: model -1 type is only for use with aerosols')

        s = f.readline().split()
        
        with open(s[0], 'r') as f1:
            tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
            
            nlevel = int(tmp[0])
            if nlevel != npro:
                raise ValueError('profiles must be listed on same grid as .prf')
            
            clen = float(tmp[1])
            pref = np.zeros([nlevel])
            ref = np.zeros([nlevel])
            eref = np.zeros([nlevel])
            
            for j in range(nlevel):
                tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                pref[j] = float(tmp[0])
                ref[j] = float(tmp[1])
                eref[j] = float(tmp[2])

                lx[ix+j] = 1
                x0[ix+j] = np.log(ref[j])
                sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]
        ix = ix + nlevel

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
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        if atm_profile_type == AtmosphericProfileType.AEROSOL_DENSITY:
            calculate_fn = lambda *args, **kwargs: Model0.calculate(*args, **kwargs)
        else:
            calculate_fn = lambda *args, **kwargs: self.calculate(*args, **kwargs)
        
        atm, xmap1 = calculate_fn(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


    def patch_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Model0(PreRTModelBase):
    """
    In this model, the atmospheric parameters are modelled as continuous profiles
    in which each element of the state vector corresponds to the atmospheric profile 
    at each altitude level
    """
    
    id : int = 0


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('full_profile', slice(None), 'Every value for each level of the profile', 'PROFILE_TYPE'),
        )
        
        _lgr.debug(f'Constructed {self.__class__.__name__} with {self.state_vector_start=} {self.n_state_vector_entries=} {self.parameters=}')
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            xprof : np.ndarray[['mparam'],float],
            #   Full profile, this model defines every value for each profile level. Has been unlogged as required
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 0 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                in which each element of the state vector corresponds to the atmospheric profile 
                at each altitude level

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileType
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                xprof(npro) :: Atmospheric profile

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model0(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        _lgr.debug(f'Calculating {cls.__name__} {atm=} {atm_profile_type=} {atm_profile_idx=} {xprof.shape=}')
        _lgr.debug(f'{xprof[:10]=}')

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 0 :: Number of levels in atmosphere does not match and profile')
        
        xmap = np.zeros((npro,npro))
        
        if atm_profile_type == AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO:
            temp = np.array(atm.VMR)
            temp[:,atm_profile_idx] = xprof
            atm.edit_VMR(temp)
            xmap[...] = np.diag(xprof)
        
        elif atm_profile_type == AtmosphericProfileType.TEMPERATURE:
            atm.edit_T(xprof)
            xmap[...] = np.diag(np.ones_like(xprof))
        
        elif atm_profile_type == AtmosphericProfileType.AEROSOL_DENSITY:
            temp = np.array(atm.DUST)
            temp[:,atm_profile_idx] = xprof
            atm.edit_DUST(temp)
            xmap[...] = np.diag(xprof)
        
        elif atm_profile_type == AtmosphericProfileType.PARA_H2_FRACTION:
            atm.PARAH2(xprof)
            xmap[...] = np.diag(np.ones_like(xprof))
        
        elif atm_profile_type == AtmosphericProfileType.FRACTIONAL_CLOUD_COVERAGE:
            atm.FRAC(xprof)
            xmap[...] = np.diag(np.ones_like(xprof))
        
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')
        
        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap

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
        _lgr.debug(f'Reading model {cls.__name__} setup from "{runname}.apr" file')
        ix_0 = ix
        
        #********* continuous profile ************************
        s = f.readline().split()
        
        with open(s[0],'r') as f1:
            tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
            
            nlevel = int(tmp[0])
            if nlevel != npro:
                raise ValueError('profiles must be listed on same grid as .prf')
            
            clen = float(tmp[1])
            pref = np.zeros([nlevel])
            ref = np.zeros([nlevel])
            eref = np.zeros([nlevel])
            for j in range(nlevel):
                tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                pref[j] = float(tmp[0])
                ref[j] = float(tmp[1])
                eref[j] = float(tmp[2])

        if varident[0] == 0:  # *** temperature, leave alone ****
            x0[ix:ix+nlevel] = ref[:]
            for j in range(nlevel):
                sx[ix+j,ix+j] = eref[j]**2.
                if varident[1] == -1: #Gradients computed numerically
                    inum[ix+j] = 1

        else:                   #**** vmr, cloud, para-H2 , fcloud, take logs ***
            for j in range(nlevel):
                lx[ix+j] = 1
                x0[ix+j] = np.log(ref[j])
                sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel

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
        _lgr.debug(f'Calculating {self.__class__.__name__} from subprofretg {forward_model=} {ix=} {ipar=} {ivar=} {xmap.shape=}')
        
        #Model 0. Continuous profile
        #***************************************************************
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        if atm_profile_type == AtmosphericProfileType.AEROSOL_DENSITY:
            calculate_fn = lambda *args, **kwargs: Modelm1.calculate(*args, **kwargs)
        else:
            calculate_fn = lambda *args, **kwargs: self.calculate(*args, **kwargs)
        
        _lgr.debug(f'Distributing to callable {calculate_fn}')
        atm, xmap1 = calculate_fn(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        _lgr.debug('Result calculated, setting values...')
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Model2(PreRTModelBase):
    """
        In this model, the atmospheric parameters are scaled using a single factor with 
        respect to the vertical profiles in the reference atmosphere
    """
    id : int = 2

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('scaling_factor', slice(0,1), 'Scaling factor applied to the reference profile', 'PROFILE_TYPE'),
        )
        
        return


    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            scf : float,
            #   Scaling factor to be applied to the reference vertical profile
            
            MakePlot=False
        ):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
                In this model, the atmospheric parameters are scaled using a single factor with 
                respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileType
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                scf :: Scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        xmap = np.zeros((1,atm.NP))
        
        if atm_profile_type == AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO:
            xmap[0,:] = atm.VMR[:, atm_profile_idx]
            atm.VMR[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileType.TEMPERATURE:
            xmap[0,:] = atm.T
            atm.T *= scf
        
        elif atm_profile_type == AtmosphericProfileType.AEROSOL_DENSITY:
            xmap[0,:] = atm.DUST[:, atm_profile_idx]
            atm.DUST[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileType.PARA_H2_FRACTION:
            xmap[0,:] = atm.PARAH2
            atm.PARAH2 *= scf
        
        elif atm_profile_type == AtmosphericProfileType.FRACTIONAL_CLOUD_COVERAGE:
            xmap[0,:] = atm.FRAC
            atm.FRAC *= scf
        
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')
        

        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #**** model 2 - Simple scaling factor of reference profile *******
        #Read in scaling factor

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        x0[ix] = float(tmp[0])
        sx[ix,ix] = (float(tmp[1]))**2.

        ix = ix + 1

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
        #Model 2. Scaling factor
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Model3(PreRTModelBase):
    """
        In this model, the atmospheric parameters are scaled using a single factor 
        in logscale with respect to the vertical profiles in the reference atmosphere
    """
    
    id : int = 3


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('scaling_factor', slice(0,1), 'Scaling factor applied to the reference profile, stored as a log in the state vector', 'PROFILE_TYPE'),
        )
        
        return


    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            scf : float,
            #   scaling factor to be applied to the reference vertical profile
            
            MakePlot=False
        ):

        """
            FUNCTION NAME : model3()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
                In this model, the atmospheric parameters are scaled using a single factor 
                in logscale with respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileType
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                scf :: scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        xmap = np.zeros((1,atm.NP))
        
        if atm_profile_type == AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO:
            xmap[0,:] = atm.VMR[:, atm_profile_idx]
            atm.VMR[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileType.TEMPERATURE:
            xmap[0,:] = atm.T
            atm.T *= scf
        
        elif atm_profile_type == AtmosphericProfileType.AEROSOL_DENSITY:
            xmap[0,:] = atm.DUST[:, atm_profile_idx]
            atm.DUST[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileType.PARA_H2_FRACTION:
            xmap[0,:] = atm.PARAH2
            atm.PARAH2 *= scf
        
        elif atm_profile_type == AtmosphericProfileType.FRACTIONAL_CLOUD_COVERAGE:
            xmap[0,:] = atm.FRAC
            atm.FRAC *= scf
        
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')
        

        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #**** model 3 - Exponential scaling factor of reference profile *******
        #Read in scaling factor

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfac = float(tmp[0])
        err = float(tmp[1])

        if xfac > 0.0:
            x0[ix] = np.log(xfac)
            lx[ix] = 1
            sx[ix,ix] = ( err/xfac ) **2.
        else:
            raise ValueError('Error in read_apr_nemesis().  xfac must be > 0')

        ix = ix + 1

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
        #Model 3. Log scaling factor
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Model9(PreRTModelBase):
    """
    In this model, the profile (cloud profile) is represented by a value
    at a certain height, plus a fractional scale height. Below the reference height 
    the profile is set to zero, while above it the profile decays exponentially with
    altitude given by the fractional scale height. In addition, this model scales
    the profile to give the requested integrated cloud optical depth.
    """
    
    id : int = 9

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('tau', slice(2,3), 'Total integrated column density of the cloud (aerosol)', r'$m^{-2}$'),
            ModelParameter('frac_scale_height', slice(1,2), 'Fractional scale height (decays above `h_ref` zero below)', 'km'),
            ModelParameter('h_ref', slice(0,1), 'Base height of cloud profile', 'km'),
        )
        
        return

    @classmethod
    def calculate(
            cls,
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            tau,
            #   Total integrated column density of the cloud (m-2)
            
            fsh,
            #   Fractional scale height (km)
            
            href,
            #   Base height of cloud profile (km)
            
            MakePlot=False
        ):

        """
            FUNCTION NAME : model9()

            DESCRIPTION :

                Function defining the model parameterisation 9 in NEMESIS.
                In this model, the profile (cloud profile) is represented by a value
                at a certain height, plus a fractional scale height. Below the reference height 
                the profile is set to zero, while above it the profile decays exponentially with
                altitude given by the fractional scale height. In addition, this model scales
                the profile to give the requested integrated cloud optical depth.

            INPUTS :

                atm :: Python class defining the atmosphere

                tau :: Total integrated column density of the cloud (m-2)

                fsh :: Fractional scale height (km)

                href :: Base height of cloud profile (km)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(3,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model9(atm,atm_profile_type,atm_profile_idx,href,fsh,tau)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        


        if atm_profile_type != AtmosphericProfileType.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)
        
        
        #Calculating the actual atmospheric scale height in each level
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        xmap = np.zeros([3,atm.NP])
        for itest in range(4):

            xdeep = tau
            xfsh = fsh
            hknee = href

            if itest==0:
                _ = 1
            elif itest==1: #For calculating the gradient wrt tau
                dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xdeep = np.exp( np.log(tau) + dx )
            elif itest==2: #For calculating the gradient wrt fsh
                dx = 0.05 * np.log(fsh)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xfsh = np.exp( np.log(fsh) + dx )
            elif itest==3: #For calculating the gradient wrt href
                dx = 0.05 * href
                if dx==0.0:
                    dx = 0.1
                hknee = href + dx

            #Initialising some arrays
            ND = np.zeros(atm.NP)   #Dust density (m-3)

            #Calculating the density in each level
            jfsh = -1
            if atm.H[0]/1.0e3>=hknee:
                jfsh = 1
                ND[0] = 1.

            for jx in range(atm.NP-1):
                j = jx + 1
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfsh

                if atm.H[j]/1.0e3>=hknee:

                    if jfsh<0:
                        ND[j]=1.0
                        jfsh = 1
                    else:
                        ND[j]=ND[j-1]*np.exp(-delh/xfac)


            for j in range(atm.NP):
                if(atm.H[j]/1.0e3<hknee):
                    if(atm.H[j+1]/1.0e3>=hknee):
                        ND[j] = ND[j] * (1.0 - (hknee*1.0e3-atm.H[j])/(atm.H[j+1]-atm.H[j]))
                    else:
                        ND[j] = 0.0

            #Calculating column density (m-2) by integrating the number density (m-3) over column (m)
            #Note that when doing the layering, the total column density in the atmosphere might not be
            #exactly the same as in xdeep due to misalignments at the boundaries of the cloud
            totcol = simpson(ND,x=atm.H)
            ND = ND / totcol * xdeep

            if itest==0:
                xprof[:] = ND[:]
            else:
                xmap[itest-1,:] = (ND[:]-xprof[:])/dx

        atm.DUST[0:atm.NP,atm_profile_idx] = xprof

        return atm,xmap


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
        #******** cloud profile held as total optical depth plus
        #******** base height and fractional scale height. Below the knee
        #******** pressure the profile is set to zero - a simple
        #******** cloud in other words!
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        hknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfsh = tmp[0]
        efsh = tmp[1]

        if xdeep>0.0:
            x0[ix] = np.log(xdeep)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        err = edeep/xdeep
        sx[ix,ix] = err**2.

        ix = ix + 1

        if xfsh>0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

        err = efsh/xfsh
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = hknee
        #inum[ix] = 1
        sx[ix,ix] = eknee**2.

        ix = ix + 1

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
        #Model 9. Simple cloud represented by base height, fractional scale height
            #and the total integrated cloud density
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Model32(PreRTModelBase):
    """
        In this model, the profile (cloud profile) is represented by a value
        at a certain pressure level, plus a fractional scale height which defines an exponential
        drop of the cloud at higher altitudes. Below the pressure level, the cloud is set 
        to exponentially decrease with a scale height of 1 km. 
    """
    
    id : int = 32


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('tau', slice(0,1), 'Integrated dust column density', r'$m^{-2}$'),
            ModelParameter('frac_scale_height', slice(1,2), 'Fractional scale height', 'km'),
            ModelParameter('p_ref', slice(2,3), 'Reference pressure', 'atm'),
        )
        
        return


    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            tau : float,
            #   Integrated dust column-density (m-2) or opacity
            
            frac_scale_height : float,
            #   Fractional scale height
            
            p_ref : float,
            #   reference pressure (atm)
            
            MakePlot : bool = False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model32()

            DESCRIPTION :

                Function defining the model parameterisation 32 in NEMESIS.
                In this model, the profile (cloud profile) is represented by a value
                at a certain pressure level, plus a fractional scale height which defines an exponential
                drop of the cloud at higher altitudes. Below the pressure level, the cloud is set 
                to exponentially decrease with a scale height of 1 km. 


            INPUTS :

                atm :: Python class defining the atmosphere
                
                atm_profile_type :: AtmosphericProfileType
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
                    
                p_ref :: Base pressure of cloud profile (atm)
                
                frac_scale_height :: Fractional scale height (km)
                
                tau :: Total integrated column density of the cloud (m-2) or cloud optical depth (if kext is normalised)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(3,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model32(atm,atm_profile_type,atm_profile_idx,p_ref,frac_scale_height,tau)

            MODIFICATION HISTORY : Juan Alday (29/05/2024)

        """
        _lgr.debug(f'{atm_profile_type=}')
        _lgr.debug(f'{atm_profile_idx=}')
        _lgr.debug(f'{p_ref=}')
        _lgr.debug(f'{frac_scale_height=}')
        _lgr.debug(f'{tau=}')

        
        if atm_profile_type != AtmosphericProfileType.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)

        #Calculating the actual atmospheric scale height in each level
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        #npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((3,atm.NP))
        for itest in range(4):

            xdeep = tau
            xfrac_scale_height = frac_scale_height
            pknee = p_ref
            if itest==0:
                pass
            elif itest==1: #For calculating the gradient wrt tau
                dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xdeep = np.exp( np.log(tau) + dx )
            elif itest==2: #For calculating the gradient wrt frac_scale_height
                dx = 0.05 * np.log(frac_scale_height)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xfrac_scale_height = np.exp( np.log(frac_scale_height) + dx )
            elif itest==3: #For calculating the gradient wrt p_ref
                dx = 0.05 * np.log(p_ref) #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                pknee = np.exp( np.log(p_ref) + dx )

            #Getting the altitude level based on the pressure/height relation
            isort = np.argsort(atm.P)
            hknee = np.interp(pknee,atm.P[isort]/101325.,atm.H[isort])  #metres

            #Initialising some arrays
            ND = np.zeros(atm.NP)   #Dust density (m-3)
            OD = np.zeros(atm.NP)   #Column density (m-2)
            Q = np.zeros(atm.NP)    #Specific density (particles/gram of atmosphere)


            #Finding the levels in the atmosphere that span pknee
            jknee = -1
            for j in range(atm.NP-1):
                if((atm.P[j]/101325. >= pknee) & (atm.P[j+1]/101325.< pknee)):
                    jknee = j


            if jknee < 0:
                jknee = 0

            #Calculating cloud density at the first level occupied by the cloud
            delh = atm.H[jknee+1] - hknee   #metres
            xfac = 0.5 * (scale[jknee]+scale[jknee+1]) * xfrac_scale_height  #metres
            ND[jknee+1] = np.exp(-delh/xfac)


            delh = hknee - atm.H[jknee]  #metres
            xf = 1000.  #The cloud below is set to decrease with a scale height of 1 km
            ND[jknee] = np.exp(-delh/xf)

            #Calculating the cloud density above this level
            for j in range(jknee+2,atm.NP):
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfrac_scale_height
                ND[j] = ND[j-1] * np.exp(-delh/xfac)

            #Calculating the cloud density below this level
            for j in range(0,jknee):
                delh = atm.H[jknee] - atm.H[j]
                xf = 1000.    #The cloud below is set to decrease with a scale height of 1 km
                ND[j] = np.exp(-delh/xf)

            #Now that we have the initial cloud number density (m-3) we can just divide by the mass density to get specific density
            Q[:] = ND[:] / rho[:] / 1.0e3 #particles per gram of atm

            #Now we integrate the optical thickness (calculate column density essentially)
            OD[atm.NP-1] = ND[atm.NP-1] * (scale[atm.NP-1] * xfrac_scale_height * 1.0e2)  #the factor 1.0e2 is for converting from m to cm
            #jfrac_scale_height = -1
            for j in range(atm.NP-2,-1,-1):
                if j>jknee:
                    delh = atm.H[j+1] - atm.H[j]   #m
                    xfac = scale[j] * xfrac_scale_height
                    OD[j] = OD[j+1] + (ND[j] - ND[j+1]) * xfac * 1.0e2
                elif j==jknee:
                    delh = atm.H[j+1] - hknee
                    xfac = 0.5 * (scale[j]+scale[j+1])*xfrac_scale_height
                    OD[j] = OD[j+1] + (1. - ND[j+1]) * xfac * 1.0e2
                    xfac = 1000.
                    OD[j] = OD[j] + (1.0 - ND[j]) * xfac * 1.0e2
                else:
                    delh = atm.H[j+1] - atm.H[j]
                    xfac = 1000.
                    OD[j] = OD[j+1] + (ND[j+1]-ND[j]) * xfac * 1.0e2

            ODX = OD[0]

            #Now we normalise the specific density profile
            #This should be done later to make this totally secure
            for j in range(atm.NP):
                OD[j] = OD[j] * xdeep / ODX
                ND[j] = ND[j] * xdeep / ODX
                Q[j] = Q[j] * xdeep / ODX
                if Q[j]>1.0e10:
                    Q[j] = 1.0e10
                if Q[j]<1.0e-36:
                    Q[j] = 1.0e-36

                #if ND[j]>1.0e10:
                #    ND[j] = 1.0e10
                #if ND[j]<1.0e-36:
                #    ND[j] = 1.0e-36

            if itest==0:  #First iteration, using the values in the state vector
                xprof[:] = Q[:]
            else:  #Next iterations used to calculate the derivatives
                xmap[itest-1,:] = (Q[:] - xprof[:])/dx

        #Now updating the atmosphere class with the new profile
        atm.DUST[:,atm_profile_idx] = xprof[:]
        _lgr.debug(f'{xprof=}')

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(atm.DUST[:,atm_profile_idx],atm.P/101325.)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max()/101325.,atm.P.min()/101325.)
            ax1.set_xlabel('Cloud density (m$^{-3}$)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()

        # [JD] Question: What is this actually doing? The `tau` variable is associated with a deep abundance
        #                not a total optical depth as far as I can tell.
        atm.DUST_RENORMALISATION[atm_profile_idx] = tau  #Adding flag to ensure that the dust optical depth is tau

        return atm,xmap


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
        #******** cloud profile is represented by a value at a 
        #******** variable pressure level and fractional scale height.
        #******** Below the knee pressure the profile is set to drop exponentially.

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfsh = tmp[0]
        efsh = tmp[1]

        #optical depth
        if varident[0]==0:
            #temperature - leave alone
            x0[ix] = xdeep
            err = edeep
        else:
            if xdeep>0.0:
                x0[ix] = np.log(xdeep)
                lx[ix] = 1
                err = edeep/xdeep
                #inum[ix] = 1
            else:
                raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        sx[ix,ix] = err**2.

        ix = ix + 1

        #cloud fractional scale height
        if xfsh>0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

        err = efsh/xfsh
        sx[ix,ix] = err**2.

        ix = ix + 1

        #cloud pressure level
        if pknee>0.0:
            x0[ix] = np.log(pknee)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter pknee (cloud pressure level) must be positive')

        err = eknee/pknee
        sx[ix,ix] = err**2.

        ix = ix + 1

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
        #Model 32. Cloud profile is represented by a value at a variable
            #pressure level and fractional scale height.
            #Below the knee pressure the profile is set to drop exponentially.
        #***************************************************************
        #tau = np.exp(forward_model.Variables.XN[ix])   #Base pressure (atm)
        #fsh = np.exp(forward_model.Variables.XN[ix+1])  #Integrated dust column-density (m-2) or opacity
        #pref = np.exp(forward_model.Variables.XN[ix+2])  #Fractional scale height
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        if atm_profile_type != AtmosphericProfileType.AEROSOL_DENSITY:
            _msg = f'Model id={self.id} is only defined for {AtmosphericProfileType.AEROSOL_DENSITY}.'
            _lgr.error(_msg)
            raise ValueError(_msg)
            
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


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
            
            atm_profile_type : AtmosphericProfileType,
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
            
            atm_profile_type : AtmosphericProfileType,
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

                atm_profile_type :: AtmosphericProfileType
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

        if atm_profile_type != AtmosphericProfileType.GAS_VOLUME_MIXING_RATIO:
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


class Model47(PreRTModelBase):
    """
        Profile is represented by a Gaussian with a specified optical thickness centred
        at a variable pressure level plus a variable FWHM (log press).
    """
    id : int = 47


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileType,
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
            ModelParameter('tau', slice(0,1), 'Integrated optical thickness', 'ln(RATIO)'),
            ModelParameter('p_ref', slice(1,2), 'Mean pressure of the cloud', 'atm'),
            ModelParameter('fwhm', slice(2,3), 'FWHM of the log-Gaussian', 'atm?'),
        )
        
        return


    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileType,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            tau : float, 
            #   Integrated optical thickness
            
            pref : float, 
            #   Mean pressure (atm) of the cloud
            
            fwhm : float, 
            #   FWHM of the log-Gaussian
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model47()

            DESCRIPTION :

                Profile is represented by a Gaussian with a specified optical thickness centred
                at a variable pressure level plus a variable FWHM (log press).

            INPUTS :

                atm :: Python class defining the atmosphere
                
                atm_profile_type :: AtmosphericProfileType
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                tau :: Integrated optical thickness.

                pref :: Mean pressure (atm) of the cloud.

                fwhm :: FWHM of the log-Gaussian.

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(mparam,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model47(atm, atm_profile_type, atm_profile_idx, tau, pref, fwhm)

            MODIFICATION HISTORY : Joe Penn (08/10/2024)

        """
        _lgr.debug(f'{atm_profile_type=} {atm_profile_idx=} {tau=} {pref=} {fwhm=}')

        if atm_profile_type != AtmosphericProfileType.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)
        
        
        
        # Calculate atmospheric properties
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        # Convert pressures to atm
        P = atm.P / 101325.0  # Pressure in atm

        # Compute Y0 = np.log(pref)
        Y0 = np.log(pref)

        # Compute XWID, the standard deviation of the Gaussian
        # [JD] this calculation is not correct
        XWID = fwhm

        # Initialize arrays
        Q = np.zeros(atm.NP)
        ND = np.zeros(atm.NP)
        OD = np.zeros(atm.NP)
        X1 = np.zeros(atm.NP)

        XOD = 0.0

        for j in range(atm.NP):
            Y = np.log(P[j])

            # Compute Q[j]
            Q[j] = 1.0 / (XWID * np.sqrt(np.pi)) * np.exp(-((Y - Y0) / XWID) ** 2)  #Q is specific density in particles per gram of atm

            # Compute ND[j]
            ND[j] = Q[j] * (rho[j] / 1.0e3) #ND is m-3

            # Compute OD[j]
            OD[j] = ND[j] * scale[j] * 1e5  # The factor 1e5 converts m to cm

            # Check for NaN or small values
            if np.isnan(OD[j]) or OD[j] < 1e-36:
                OD[j] = 1e-36
            if np.isnan(Q[j]) or Q[j] < 1e-36:
                Q[j] = 1e-36

            XOD += OD[j]

            X1[j] = Q[j]

        # Empirical correction to XOD
        XOD = XOD * 0.25

        # Rescale Q[j]
        for j in range(atm.NP):
            X1[j] = Q[j] * tau / XOD  # XDEEP is tau

            # Check for NaN or small values
            if np.isnan(X1[j]) or X1[j] < 1e-36:
                X1[j] = 1e-36

        # Now compute the Jacobian matrix xmap, note that we only need one entry for the 'ipar' entry
        # so we don't have to use 'ipar' in here, we pass the calcuated slice out and is is the
        # the containing scope's job to put our returned xmap part into the 'real xmap' at the
        # correct location.
        xmap = np.zeros((3, atm.NP))

        for j in range(atm.NP):
            Y = np.log(P[j])

            # First parameter derivative: xmap[0, ipar, j] = X1[j] / tau
            xmap[0, j] = X1[j] / tau  # XDEEP is tau

            # Derivative of X1[j] with respect to Y0 (pref)
            xmap[1, j] = 2.0 * (Y - Y0) / XWID ** 2 * X1[j]

            # Derivative of X1[j] with respect to XWID (fwhm)
            xmap[2, j] = (2.0 * ((Y - Y0) ** 2) / XWID ** 3 - 1.0 / XWID) * X1[j]

        # Update the atmosphere class with the new profile
        atm.DUST[:, atm_profile_idx] = X1[:]
        _lgr.debug(f'{X1=}')

        if MakePlot:
            fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))
            ax1.plot(atm.DUST[:, atm_profile_idx], atm.P / 101325.0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max() / 101325.0, atm.P.min() / 101325.0)
            ax1.set_xlabel('Cloud density (particles/kg)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()
            plt.show()

        atm.DUST_RENORMALISATION[atm_profile_idx] = tau   #Adding flag to ensure that the dust optical depth is tau

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
        ix_0 = ix
        #******** cloud profile is represented by a peak optical depth at a 
        #******** variable pressure level and a Gaussian profile with FWHM (in log pressure)

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xwid = tmp[0]
        ewid = tmp[1]

        #total optical depth
        if varident[0]==0:
            #temperature - leave alone
            x0[ix] = xdeep
            err = edeep
        else:
            if xdeep>0.0:
                x0[ix] = np.log(xdeep)
                lx[ix] = 1
                err = edeep/xdeep
                #inum[ix] = 1
            else:
                raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        sx[ix,ix] = err**2.

        ix = ix + 1

        #pressure level of the cloud
        if pknee>0.0:
            x0[ix] = np.log(pknee)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter pknee (cloud pressure level) must be positive')

        err = eknee/pknee
        sx[ix,ix] = err**2.

        ix = ix + 1

        #fwhm of the gaussian function describing the cloud profile
        if xwid>0.0:
            x0[ix] = np.log(xwid)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xwid (width of the cloud gaussian profile) must be positive')

        err = ewid/xwid
        sx[ix,ix] = err**2.

        ix = ix + 1

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
        #Model 47. Profile is represented by a Gaussian with a specified optical thickness centred
            #at a variable pressure level plus a variable FWHM (log press) in height.
        #***************************************************************
        #tau = np.exp(forward_model.Variables.XN[ix])   #Integrated dust column-density (m-2) or opacity
        #pref = np.exp(forward_model.Variables.XN[ix+1])  #Base pressure (atm)
        #fwhm = np.exp(forward_model.Variables.XN[ix+2])  #FWHM
        #forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX, ipar, tau, pref, fwhm)
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm, 
            atm_profile_type, 
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:forward_model.AtmosphereX.NP] = xmap1
        
        return


class Model49(PreRTModelBase):
    """
        In this model, the atmospheric parameters are modelled as continuous profiles
        in linear space. This parameterisation allows the retrieval of negative VMRs.
    """
    id : int = 49


    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 49 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                in linear space. This parameterisation allows the retrieval of negative VMRs.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                xprof(npro) :: Scaling factor at each altitude level

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model50(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (08/06/2022)

        """

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 49 :: Number of levels in atmosphere and scaling factor profile does not match')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((npro,npar,npro))

        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:] = atm.VMR[:,jvmr]
            x1[:] = xprof
            vmr = np.zeros((atm.NP,atm.NVMR))
            vmr[:,:] = atm.VMR
            vmr[:,jvmr] = x1[:]
            atm.edit_VMR(vmr)
        elif ipar==atm.NVMR: #Temperature
            xref = atm.T
            x1 = xprof
            atm.edit_T(x1)
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST: #Dust in m-3
                xref[:] = atm.DUST[:,jtmp]
                x1[:] = xprof
                dust = np.zeros((atm.NP,atm.NDUST))
                dust[:,:] = atm.DUST
                dust[:,jtmp] = x1
                atm.edit_DUST(dust)
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2
                x1[:] = xprof
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC
                x1[:] = xprof
                atm.FRAC = x1

        for j in range(npro):
            xmap[j,ipar,j] = 1.

        return atm,xmap


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
        #********* continuous profile in linear scale ************************
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref = np.zeros([nlevel])
        ref = np.zeros([nlevel])
        eref = np.zeros([nlevel])
        for j in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
            pref[j] = float(tmp[0])
            ref[j] = float(tmp[1])
            eref[j] = float(tmp[2])
        f1.close()

        #inum[ix:ix+nlevel] = 1
        x0[ix:ix+nlevel] = ref[:]
        for j in range(nlevel):
            sx[ix+j,ix+j] = eref[j]**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel

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
        #Model 50. Continuous profile in linear scale
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model50(PreRTModelBase):
    """
        In this model, the atmospheric parameters are modelled as continuous profiles
        multiplied by a scaling factor in linear space. Each element of the state vector
        corresponds to this scaling factor at each altitude level. This parameterisation
        allows the retrieval of negative VMRs.
    """
    id : int = 50


    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 50 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                multiplied by a scaling factor in linear space. Each element of the state vector
                corresponds to this scaling factor at each altitude level. This parameterisation
                allows the retrieval of negative VMRs.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                xprof(npro) :: Scaling factor at each altitude level

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model50(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (08/06/2022)

        """

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 50 :: Number of levels in atmosphere and scaling factor profile does not match')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((npro,npar,npro))

        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:] = atm.VMR[:,jvmr]
            x1[:] = atm.VMR[:,jvmr] * xprof
            vmr = np.zeros((atm.NP,atm.NVMR))
            vmr[:,:] = atm.VMR
            vmr[:,jvmr] = x1[:]
            atm.edit_VMR(vmr)
        elif ipar==atm.NVMR: #Temperature
            xref = atm.T
            x1 = atm.T * xprof
            atm.edit_T(x1)
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST: #Dust in m-3
                xref[:] = atm.DUST[:,jtmp]
                x1[:] = atm.DUST[:,jtmp] * xprof
                dust = np.zeros((atm.NP,atm.NDUST))
                dust[:,:] = atm.DUST
                dust[:,jtmp] = x1
                atm.edit_DUST(dust)
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2
                x1[:] = atm.PARAH2 * xprof
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC
                x1[:] = atm.FRAC * xprof
                atm.FRAC = x1

        for j in range(npro):
            xmap[j,ipar,j] = xref[j]

        return atm,xmap


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
        #********* continuous profile of a scaling factor ************************
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref = np.zeros([nlevel])
        ref = np.zeros([nlevel])
        eref = np.zeros([nlevel])
        for j in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
            pref[j] = float(tmp[0])
            ref[j] = float(tmp[1])
            eref[j] = float(tmp[2])
        f1.close()

        x0[ix:ix+nlevel] = ref[:]
        for j in range(nlevel):
            sx[ix+j,ix+j] = eref[j]**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel
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
        #Model 50. Continuous profile of scaling factors
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model51(PreRTModelBase):
    """
        In this model, the profile is scaled using a single factor with 
        respect to a reference profile.
    """
    id : int = 51


    @classmethod
    def calculate(cls, atm, ipar, scale, scale_gas, scale_iso):
        """
            FUNCTION NAME : model51()

            DESCRIPTION :

                Function defining the model parameterisation 51 (49 in NEMESIS).
                In this model, the profile is scaled using a single factor with 
                respect to a reference profile.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scale :: Scaling factor
                scale_gas :: Reference gas
                scale_iso :: Reference isotope

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        npar = atm.NVMR+2+atm.NDUST

        iref_vmr = np.where((atm.ID == scale_gas)&(atm.ISO == scale_iso))[0][0]
        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)

        xref[:] = atm.VMR[:,iref_vmr]
        x1[:] = xref * scale
        atm.VMR[:,ipar] = x1

        xmap = np.zeros([1,npar,atm.NP])

        xmap[0,ipar,:] = xref[:]

        return atm,xmap


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
        #********* multiple of different profile ************************
        prof = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='int') # Use "!" as comment character in *.apr files
        profgas = prof[0]
        profiso = prof[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        scale = tmp[0]
        escale = tmp[1]

        varparam[1] = profgas
        varparam[2] = profiso
        x0[ix] = np.log(scale)
        lx[ix] = 1
        err = escale/scale
        sx[ix,ix] = err**2.

        ix = ix + 1

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
        #Model 51. Scaling of a reference profile
        #***************************************************************                
        scale = np.exp(forward_model.Variables.XN[ix])
        scale_gas, scale_iso = forward_model.Variables.VARPARAM[ivar,1:3]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,scale,scale_gas,scale_iso)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model110(PreRTModelBase):
    """
        In this model, the Venus cloud is parameterised using the model of Haus et al. (2016).
        In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes.
        In this parametersiation, we include the Haus cloud model as it is, but we allow the altitude of the cloud
        to vary according to the inputs.

        The units of the aerosol density are in m-3, so the extinction coefficients must not be normalised.
    """
    id : int = 110


    @classmethod
    def calculate(cls, atm, idust0, z_offset):
        """
            FUNCTION NAME : model110()

            DESCRIPTION :

                Function defining the model parameterisation 110.
                In this model, the Venus cloud is parameterised using the model of Haus et al. (2016).
                In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes.
                In this parametersiation, we include the Haus cloud model as it is, but we allow the altitude of the cloud
                to vary according to the inputs.

                The units of the aerosol density are in m-3, so the extinction coefficients must not be normalised.


            INPUTS :

                atm :: Python class defining the atmosphere

                idust0 :: Index of the first aerosol population in the atmosphere class to be changed,
                          but it will indeed affect four aerosol populations.
                          Thus atm.NDUST must be at least 4.

                z_offset :: Offset in altitude (km) of the cloud with respect to the Haus et al. (2016) model.

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        h = atm.H/1.0e3
        nh = len(h)

        if atm.NDUST<idust0+4:
            raise ValueError('error in model 110 :: The cloud model requires at least 4 modes')

        #Cloud mode 1
        ###################################################

        zb1 = 49. + z_offset #Lower base of peak altitude (km)
        zc1 = 16.            #Layer thickness of constant peak particle (km)
        Hup1 = 3.5           #Upper scale height (km)
        Hlo1 = 1.            #Lower scale height (km)
        n01 = 193.5          #Particle number density at zb (cm-3)

        #N1 = 3982.04e5       #Total column particle density (cm-2)
        #tau1 = 3.88          #Total column optical depth at 1 um

        n1 = np.zeros(nh)

        ialt1 = np.where(h<zb1)
        ialt2 = np.where((h<=(zb1+zc1)) & (h>=zb1))
        ialt3 = np.where(h>(zb1+zc1))

        n1[ialt1] = n01 * np.exp( -(zb1-h[ialt1])/Hlo1 )
        n1[ialt2] = n01
        n1[ialt3] = n01 * np.exp( -(h[ialt3]-(zb1+zc1))/Hup1 )

        #Cloud mode 2
        ###################################################

        zb2 = 65. + z_offset  #Lower base of peak altitude (km)
        zc2 = 1.0             #Layer thickness of constant peak particle (km)
        Hup2 = 3.5            #Upper scale height (km)
        Hlo2 = 3.             #Lower scale height (km)
        n02 = 100.            #Particle number density at zb (cm-3)

        #N2 = 748.54e5         #Total column particle density (cm-2)
        #tau2 = 7.62           #Total column optical depth at 1 um

        n2 = np.zeros(nh)

        ialt1 = np.where(h<zb2)
        ialt2 = np.where((h<=(zb2+zc2)) & (h>=zb2))
        ialt3 = np.where(h>(zb2+zc2))

        n2[ialt1] = n02 * np.exp( -(zb2-h[ialt1])/Hlo2 )
        n2[ialt2] = n02
        n2[ialt3] = n02 * np.exp( -(h[ialt3]-(zb2+zc2))/Hup2 )

        #Cloud mode 2'
        ###################################################

        zb2p = 49. + z_offset   #Lower base of peak altitude (km)
        zc2p = 11.              #Layer thickness of constant peak particle (km)
        Hup2p = 1.0             #Upper scale height (km)
        Hlo2p = 0.1             #Lower scale height (km)
        n02p = 50.              #Particle number density at zb (cm-3)

        #N2p = 613.71e5          #Total column particle density (cm-2)
        #tau2p = 9.35            #Total column optical depth at 1 um

        n2p = np.zeros(nh)

        ialt1 = np.where(h<zb2p)
        ialt2 = np.where((h<=(zb2p+zc2p)) & (h>=zb2p))
        ialt3 = np.where(h>(zb2p+zc2p))

        n2p[ialt1] = n02p * np.exp( -(zb2p-h[ialt1])/Hlo2p )
        n2p[ialt2] = n02p
        n2p[ialt3] = n02p * np.exp( -(h[ialt3]-(zb2p+zc2p))/Hup2p )

        #Cloud mode 3
        ###################################################

        zb3 = 49. + z_offset    #Lower base of peak altitude (km)
        zc3 = 8.                #Layer thickness of constant peak particle (km)
        Hup3 = 1.0              #Upper scale height (km)
        Hlo3 = 0.5              #Lower scale height (km)
        n03 = 14.               #Particle number density at zb (cm-3)

        #N3 = 133.86e5           #Total column particle density (cm-2)
        #tau3 = 14.14            #Total column optical depth at 1 um

        n3 = np.zeros(nh)

        ialt1 = np.where(h<zb3)
        ialt2 = np.where((h<=(zb3+zc3)) & (h>=zb3))
        ialt3 = np.where(h>(zb3+zc3))

        n3[ialt1] = n03 * np.exp( -(zb3-h[ialt1])/Hlo3 )
        n3[ialt2] = n03
        n3[ialt3] = n03 * np.exp( -(h[ialt3]-(zb3+zc3))/Hup3 )


        new_dust = np.zeros((atm.NP,atm.NDUST))

        new_dust[:,:] = atm.DUST[:,:]
        new_dust[:,idust0] = n1[:] * 1.0e6 #Converting from cm-3 to m-3
        new_dust[:,idust0+1] = n2[:] * 1.0e6
        new_dust[:,idust0+2] = n2p[:] * 1.0e6
        new_dust[:,idust0+3] = n3[:] * 1.0e6

        atm.edit_DUST(new_dust)

        return atm


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
        #******** model for Venus cloud following Haus et al. (2016) with altitude offset

        if varident[0]>0:
            raise ValueError('error in read_apr model 110 :: VARIDENT[0] must be negative to be associated with the aerosols')

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #z_offset
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

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
        #Model 110. Venus cloud model from Haus et al. (2016) with altitude offset
        #************************************************************************************  

        offset = forward_model.Variables.XN[ix]   #altitude offset in km
        idust0 = np.abs(forward_model.Variables.VARIDENT[ivar,0])-1  #Index of the first cloud mode                
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,idust0,offset)

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model111(PreRTModelBase):
    """
        This is a parametersiation for the Venus cloud following the model of Haus et al. (2016),
        but also includes a parametersiation for the SO2 profiles, whose mixing ratio is tightly linked to the
        altitude of the cloud.

        In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes, and we allow the 
        variation of the cloud altitude. The units of the aerosol density are in m-3, so the extinction coefficients must 
        not be normalised.

        In the case of the SO2 profile, it is tightly linked to the altitude of the cloud, as the mixing ratio
        of these species greatly decreases within the cloud due to condensation and photolysis. This molecule is
        modelled by defining its mixing ratio below and above the cloud, and the mixing ratio is linearly interpolated in
        log-scale within the cloud.
    """
    id : int = 111


    @classmethod
    def calculate(cls, atm, idust0, so2_deep, so2_top, z_offset):
        """
            FUNCTION NAME : model111()

            DESCRIPTION :

                Function defining the model parameterisation 111.

                This is a parametersiation for the Venus cloud following the model of Haus et al. (2016) (same as model 110),
                but also includes a parametersiation for the SO2 profiles, whose mixing ratio is tightly linked to the
                altitude of the cloud.

                In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes, and we allow the 
                variation of the cloud altitude. The units of the aerosol density are in m-3, so the extinction coefficients must 
                not be normalised.

                In the case of the SO2 profile, it is tightly linked to the altitude of the cloud, as the mixing ratio
                of these species greatly decreases within the cloud due to condensation and photolysis. This molecule is
                modelled by defining its mixing ratio below and above the cloud, and the mixing ratio is linearly interpolated in
                log-scale within the cloud.

            INPUTS :

                atm :: Python class defining the atmosphere

                idust0 :: Index of the first aerosol population in the atmosphere class to be changed,
                          but it will indeed affect four aerosol populations.
                          Thus atm.NDUST must be at least 4.

                so2_deep :: SO2 volume mixing ratio below the cloud
                so2_top :: SO2 volume mixing ratio above the cloud

                z_offset :: Offset in altitude (km) of the cloud with respect to the Haus et al. (2016) model.

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model111(atm,idust0,so2_deep,so2_top,z_offset)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        h = atm.H/1.0e3
        nh = len(h)

        if atm.NDUST<idust0+4:
            raise ValueError('error in model 111 :: The cloud model requires at least 4 modes')

        iSO2 = np.where( (atm.ID==9) & (atm.ISO==0) )[0]
        if len(iSO2)==0:
            raise ValueError('error in model 111 :: SO2 must be defined in atmosphere class')
        else:
            iSO2 = iSO2[0]

        #Cloud mode 1
        ###################################################

        zb1 = 49. + z_offset #Lower base of peak altitude (km)
        zc1 = 16.            #Layer thickness of constant peak particle (km)
        Hup1 = 3.5           #Upper scale height (km)
        Hlo1 = 1.            #Lower scale height (km)
        n01 = 193.5          #Particle number density at zb (cm-3)

        #N1 = 3982.04e5       #Total column particle density (cm-2)
        #tau1 = 3.88          #Total column optical depth at 1 um

        n1 = np.zeros(nh)

        ialt1 = np.where(h<zb1)
        ialt2 = np.where((h<=(zb1+zc1)) & (h>=zb1))
        ialt3 = np.where(h>(zb1+zc1))

        n1[ialt1] = n01 * np.exp( -(zb1-h[ialt1])/Hlo1 )
        n1[ialt2] = n01
        n1[ialt3] = n01 * np.exp( -(h[ialt3]-(zb1+zc1))/Hup1 )

        #Cloud mode 2
        ###################################################

        zb2 = 65. + z_offset  #Lower base of peak altitude (km)
        zc2 = 1.0             #Layer thickness of constant peak particle (km)
        Hup2 = 3.5            #Upper scale height (km)
        Hlo2 = 3.             #Lower scale height (km)
        n02 = 100.            #Particle number density at zb (cm-3)

        #N2 = 748.54e5         #Total column particle density (cm-2)
        #tau2 = 7.62           #Total column optical depth at 1 um

        n2 = np.zeros(nh)

        ialt1 = np.where(h<zb2)
        ialt2 = np.where((h<=(zb2+zc2)) & (h>=zb2))
        ialt3 = np.where(h>(zb2+zc2))

        n2[ialt1] = n02 * np.exp( -(zb2-h[ialt1])/Hlo2 )
        n2[ialt2] = n02
        n2[ialt3] = n02 * np.exp( -(h[ialt3]-(zb2+zc2))/Hup2 )

        #Cloud mode 2'
        ###################################################

        zb2p = 49. + z_offset   #Lower base of peak altitude (km)
        zc2p = 11.              #Layer thickness of constant peak particle (km)
        Hup2p = 1.0             #Upper scale height (km)
        Hlo2p = 0.1             #Lower scale height (km)
        n02p = 50.              #Particle number density at zb (cm-3)

        #N2p = 613.71e5          #Total column particle density (cm-2)
        #tau2p = 9.35            #Total column optical depth at 1 um

        n2p = np.zeros(nh)

        ialt1 = np.where(h<zb2p)
        ialt2 = np.where((h<=(zb2p+zc2p)) & (h>=zb2p))
        ialt3 = np.where(h>(zb2p+zc2p))

        n2p[ialt1] = n02p * np.exp( -(zb2p-h[ialt1])/Hlo2p )
        n2p[ialt2] = n02p
        n2p[ialt3] = n02p * np.exp( -(h[ialt3]-(zb2p+zc2p))/Hup2p )

        #Cloud mode 3
        ###################################################

        zb3 = 49. + z_offset    #Lower base of peak altitude (km)
        zc3 = 8.                #Layer thickness of constant peak particle (km)
        Hup3 = 1.0              #Upper scale height (km)
        Hlo3 = 0.5              #Lower scale height (km)
        n03 = 14.               #Particle number density at zb (cm-3)

        #N3 = 133.86e5           #Total column particle density (cm-2)
        #tau3 = 14.14            #Total column optical depth at 1 um

        n3 = np.zeros(nh)

        ialt1 = np.where(h<zb3)
        ialt2 = np.where((h<=(zb3+zc3)) & (h>=zb3))
        ialt3 = np.where(h>(zb3+zc3))

        n3[ialt1] = n03 * np.exp( -(zb3-h[ialt1])/Hlo3 )
        n3[ialt2] = n03
        n3[ialt3] = n03 * np.exp( -(h[ialt3]-(zb3+zc3))/Hup3 )


        new_dust = np.zeros((atm.NP,atm.NDUST))

        new_dust[:,:] = atm.DUST[:,:]
        new_dust[:,idust0] = n1[:] * 1.0e6 #Converting from cm-3 to m-3
        new_dust[:,idust0+1] = n2[:] * 1.0e6
        new_dust[:,idust0+2] = n2p[:] * 1.0e6
        new_dust[:,idust0+3] = n3[:] * 1.0e6

        atm.edit_DUST(new_dust)


        #SO2 vmr profile
        ####################################################

        cloud_bottom = zb1
        cloud_top = zb1 + 20. #Assuming the cloud extends 20 km above the base
        SO2grad = (np.log(so2_top)-np.log(so2_deep))/(cloud_top-cloud_bottom)  #dVMR/dz (km-1)

        #Calculating SO2 profile
        so2 = np.zeros(nh)
        ibelow = np.where(h<cloud_bottom)[0]
        iabove = np.where(h>cloud_top)[0]
        icloud = np.where((h>=cloud_bottom) & (h<=cloud_top))[0]

        so2[ibelow] = so2_deep
        so2[iabove] = so2_top
        so2[icloud] = np.exp(np.log(so2_deep) + SO2grad*(h[icloud]-cloud_bottom))

        #Updating SO2 profile in atmosphere class
        atm.update_gas(9,0,so2)

        return atm


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
        #******** model for Venus cloud and SO2 vmr profile with altitude offset

        if varident[0]>0:
            raise ValueError('error in read_apr model 111 :: VARIDENT[0] must be negative to be associated with the aerosols')

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #z_offset
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #SO2_deep
        so2_deep = float(tmp[0])
        so2_deep_err = float(tmp[1])
        x0[ix] = np.log(so2_deep)
        sx[ix,ix] = (so2_deep_err/so2_deep)**2.
        lx[ix] = 1
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #SO2_top
        so2_top = float(tmp[0])
        so2_top_err = float(tmp[1])
        x0[ix] = np.log(so2_top)
        sx[ix,ix] = (so2_top_err/so2_top)**2.
        lx[ix] = 1
        inum[ix] = 1
        ix = ix + 1

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
        #Model 110. Venus cloud model and SO2 vmr profile with altitude offset
        #************************************************************************************  

        offset = forward_model.Variables.XN[ix]   #altitude offset in km
        so2_deep = np.exp(forward_model.Variables.XN[ix+1])   #SO2 vmr below the cloud
        so2_top = np.exp(forward_model.Variables.XN[ix+2])   #SO2 vmr above the cloud

        idust0 = np.abs(forward_model.Variables.VARIDENT[ivar,0])-1  #Index of the first cloud mode                
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,idust0,so2_deep,so2_top,offset)

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model202(PreRTModelBase):
    """
        In this model, the telluric atmospheric profile is multiplied by a constant 
        scaling factor
    """
    id : int = 202


    @classmethod
    def calculate(cls, telluric,varid1,varid2,scf):

        """
            FUNCTION NAME : model202()

            DESCRIPTION :

                Function defining the model parameterisation 202 in NEMESIS.
                In this model, the telluric atmospheric profile is multiplied by a constant 
                scaling factor

            INPUTS :

                telluric :: Python class defining the telluric atmosphere

                varid1,varid2 :: The first two values of the Variable ID. They follow the 
                                 same convention as in Model parameterisation 2

                scf :: Scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                telluric :: Updated Telluric class

            CALLING SEQUENCE:

                telluric = model52(telluric,varid1,varid2,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2025)

        """

        x1 = np.zeros(telluric.Atmosphere.NP)
        xref = np.zeros(telluric.Atmosphere.NP)

        if(varid1==0): #Temperature

            xref[:] = telluric.Atmosphere.T[:]
            x1[:] = telluric.Atmosphere.T[:] * scf
            telluric.Atmosphere.T[:] = x1[:]

        elif(varid1>0): #Gaseous abundance

            jvmr = -1
            for j in range(telluric.Atmosphere.NVMR):
                if((telluric.Atmosphere.ID[j]==varid1) & (telluric.Atmosphere.ISO[j]==varid2)):
                    jvmr = j
            if jvmr==-1:
                _lgr.info(f'Required ID ::  {(varid1,varid2)}')
                _lgr.info(f'Avaiable ID and ISO ::  {(telluric.Atmosphere.ID,telluric.Atmosphere.ISO)}')
                raise ValueError('error in model 202 :: The required gas is not found in Telluric atmosphere')

            xref[:] = telluric.Atmosphere.VMR[:,jvmr]
            x1[:] = telluric.Atmosphere.VMR[:,jvmr] * scf
            telluric.Atmosphere.VMR[:,jvmr] = x1[:]

        else:
            raise ValueError('error in model 202 :: The retrieved parameter has to be either temperature or a gaseous abundance')

        return telluric


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
        #********* simple scaling of telluric atmospheric profile ************************
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        x0[ix] = float(tmp[0])
        sx[ix,ix] = (float(tmp[1]))**2.
        inum[ix] = 1

        ix = ix + 1

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
        #Model 202. Scaling factor of telluric atmospheric profile
        #***************************************************************

        scafac = forward_model.Variables.XN[ix]
        varid1 = forward_model.Variables.VARIDENT[ivar,0] ; varid2 = forward_model.Variables.VARIDENT[ivar,1]
        if forward_model.TelluricX is not None:
            forward_model.TelluricX = self.calculate(forward_model.TelluricX,varid1,varid2,scafac)

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model1002(PreRTModelBase):
    """ 
        In this model, the atmospheric parameters are scaled using a single factor with 
        respect to the vertical profiles in the reference atmosphere.
        
        The model is applied simultaneously in different planet locations
    """
    id : int = 1002


    @classmethod
    def calculate(cls, atm,ipar,scf,MakePlot=False):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 1002 in NEMESIS.

                This is the same as model 2, but applied simultaneously in different planet locations
                In this model, the atmospheric parameters are scaled using a single factor with 
                respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scf(nlocations) :: Scaling factors at the different locations

            OPTIONAL INPUTS: None

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(nlocations,ngas+2+ncont,npro,nlocations) :: Matrix of relating funtional derivatives to 
                                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model1002(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (19/04/2023)

        """

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))
        #xmap1 = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))

        if len(scf)!=atm.NLOCATIONS:
            raise ValueError('error in model 1002 :: The number of scaling factors must be the same as the number of locations in Atmosphere')

        if atm.NLOCATIONS<=1:
            raise ValueError('error in model 1002 :: This model can be applied only if NLOCATIONS>1')

        x1 = np.zeros((atm.NP,atm.NLOCATIONS))
        xref = np.zeros((atm.NP,atm.NLOCATIONS))
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:,:] = atm.VMR[:,jvmr,:]
            x1[:,:] = atm.VMR[:,jvmr,:] * scf[:]
            atm.VMR[:,jvmr,:] =  x1
        elif ipar==atm.NVMR: #Temperature
            xref[:] = atm.T[:,:]
            x1[:] = np.transpose(np.transpose(atm.T[:,:]) * scf[:])
            atm.T[:,:] = x1 
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST:
                xref[:] = atm.DUST[:,jtmp,:]
                x1[:] = np.transpose(np.transpose(atm.DUST[:,jtmp,:]) * scf[:])
                atm.DUST[:,jtmp,:] = x1
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2[:,:]
                x1[:] = np.transpose(np.transpose(atm.PARAH2[:,:]) * scf)
                atm.PARAH2[:,:] = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC[:,:]
                x1[:] = np.transpose(np.transpose(atm.FRAC[:,:]) * scf)
                atm.FRAC[:,:] = x1


        #This calculation takes a long time for big arrays
        #for j in range(atm.NLOCATIONS):
        #    xmap[j,ipar,:,j] = xref[:,j]


        if MakePlot==True:

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig,ax1 = plt.subplots(1,1,figsize=(6,4))
            im1 = ax1.scatter(atm.LONGITUDE,atm.LATITUDE,c=scf,cmap='jet',vmin=scf.min(),vmax=scf.max())
            ax1.grid()
            ax1.set_xlabel('Longitude / deg')
            ax1.set_ylabel('Latitude / deg')
            ax1.set_xlim(-180.,180.)
            ax1.set_ylim(-90.,90.)
            ax1.set_title('Model 1002')

            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax)
            cbar1.set_label('Scaling factor')

            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #******** scaling of atmospheric profiles at multiple locations (linear scale)

        s = f.readline().split()

        #Reading file with the a priori information
        f1 = open(s[0],'r') 
        s = np.fromfile(f1,sep=' ',count=2,dtype='float')   #nlocations and correlation length
        nlocs = int(s[0])   #number of locations
        clen = int(s[1])    #correlation length (degress)

        if nlocs != nlocations:
            raise ValueError('error in model 1002 :: number of locations must be the same as in Surface and Atmosphere')

        lats = np.zeros(nlocs)
        lons = np.zeros(nlocs)
        sfactor = np.zeros(nlocs)
        efactor = np.zeros(nlocs)
        for iloc in range(nlocs):

            s = np.fromfile(f1,sep=' ',count=4,dtype='float')   
            lats[iloc] = float(s[0])    #latitude of the location
            lons[iloc] = float(s[1])    #longitude of the location
            sfactor[iloc] = float(s[2])   #scaling value
            efactor[iloc] = float(s[3])   #uncertainty in scaling value

        f1.close()

        #Including the parameters in the state vector
        varparam[0] = nlocs
        #iparj = 1
        for iloc in range(nlocs):
            #Including surface temperature in the state vector
            x0[ix+iloc] = sfactor[iloc]
            sx[ix+iloc,ix+iloc] = efactor[iloc]**2.0
            lx[ix+iloc] = 0     #linear scale
            inum[ix+iloc] = 0   #analytical calculation of jacobian


        #Defining the correlation between surface pixels 
        for j in range(nlocs):
            s1 = np.sin(lats[j]/180.*np.pi)
            s2 = np.sin(lats/180.*np.pi)
            c1 = np.cos(lats[j]/180.*np.pi)
            c2 = np.cos(lats/180.*np.pi)
            c3 = np.cos( (lons[j]-lons)/180.*np.pi )
            psi = np.arccos( s1*s2 + c1*c2*c3 ) / np.pi * 180.   #angular distance (degrees)
            arg = abs(psi/clen)
            xfac = np.exp(-arg)
            for k in range(nlocs):
                if xfac[k]>0.001:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac[k]
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        #jsurf = ix

        ix = ix + nlocs

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
        #Model 1002. Scaling factors at multiple locations
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]],MakePlot=False)
        #This calculation takes a long time for big arrays
        #xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP,0:forward_model.AtmosphereX.NLOCATIONS] = xmap1[:,:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model228(PreRTModelBase):
    """
        In this model, the wavelength calibration of a given spectrum is performed, as well as the fit
        of a double Gaussian ILS suitable for ACS MIR solar occultation observations

        The wavelength calibration is performed such that the first wavelength or wavenumber is given by V0. 
        Then the rest of the wavelengths of the next data points are calculated by calculating the wavelength
        step between data points given by dV = C0 + C1*data_number + C2*data_number, where data_number 
        is an array going from 0 to NCONV-1.

        The ILS is fit using the approach of Alday et al. (2019, A&A). In this approach, the parameters to fit
        the ILS are the Offset of the second gaussian with respect to the first one (P0), the FWHM of the main 
        gaussian (P1), Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber (P2)
        , Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (P3), and
        a linear variation of the relative amplitude.
    """
    id : int = 228


    @classmethod
    def calculate(cls, Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3,MakePlot=False):

        """
            FUNCTION NAME : model228()

            DESCRIPTION :

                Function defining the model parameterisation 228 in NEMESIS.

                In this model, the wavelength calibration of a given spectrum is performed, as well as the fit
                of a double Gaussian ILS suitable for ACS MIR solar occultation observations

                The wavelength calibration is performed such that the first wavelength or wavenumber is given by V0. 
                Then the rest of the wavelengths of the next data points are calculated by calculating the wavelength
                step between data points given by dV = C0 + C1*data_number + C2*data_number, where data_number 
                is an array going from 0 to NCONV-1.

                The ILS is fit using the approach of Alday et al. (2019, A&A). In this approach, the parameters to fit
                the ILS are the Offset of the second gaussian with respect to the first one (P0), the FWHM of the main 
                gaussian (P1), Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber (P2)
                , Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (P3), and
                a linear variation of the relative amplitude.

            INPUTS :

                Measurement :: Python class defining the Measurement
                Spectroscopy :: Python class defining the Spectroscopy
                V0 :: Wavelength/Wavenumber of the first data point
                C0,C1,C2 :: Coefficients to calculate the step size in wavelength/wavenumbers between data points
                P0,P1,P2,P3 :: Parameters used to define the double Gaussian ILS of ACS MIR

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated Measurement and Spectroscopy classes

            CALLING SEQUENCE:

                Measurement,Spectroscopy = model228(Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3)

            MODIFICATION HISTORY : Juan Alday (20/12/2021)

        """

        #1.: Defining the new wavelength array
        ##################################################

        nconv = Measurement.NCONV[0]
        vconv1 = np.zeros(nconv)
        vconv1[0] = V0

        xx = np.linspace(0,nconv-2,nconv-1)
        dV = C0 + C1*xx + C2*(xx)**2.

        for i in range(nconv-1):
            vconv1[i+1] = vconv1[i] + dV[i]

        for i in range(Measurement.NGEOM):
            Measurement.VCONV[0:Measurement.NCONV[i],i] = vconv1[:]

        #2.: Calculating the new ILS function based on the new convolution wavelengths
        ###################################################################################

        ng = 2 #Number of gaussians to include

        #Wavenumber offset of the two gaussians
        offset = np.zeros([nconv,ng])
        offset[:,0] = 0.0
        offset[:,1] = P0

        #FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
        fwhm = np.zeros([nconv,ng])
        fwhml = P1 / vconv1[0]**2.0
        for i in range(nconv):
            fwhm[i,0] = fwhml * (vconv1[i])**2.
            fwhm[i,1] = fwhm[i,0]

        #Amplitde of the second gaussian with respect to the main one
        amp = np.zeros([nconv,ng])
        ampgrad = (P3 - P2)/(vconv1[nconv-1]-vconv1[0])
        for i in range(nconv):
            amp[i,0] = 1.0
            amp[i,1] = (vconv1[i] - vconv1[0]) * ampgrad + P2

        #Running for each spectral point
        nfil = np.zeros(nconv,dtype='int32')
        mfil1 = 200
        vfil1 = np.zeros([mfil1,nconv])
        afil1 = np.zeros([mfil1,nconv])
        for i in range(nconv):

            #determining the lowest and highest wavenumbers to calculate
            xlim = 0.0
            xdist = 5.0 
            for j in range(ng):
                xcen = offset[i,j]
                xmin = abs(xcen - xdist*fwhm[i,j]/2.)
                if xmin > xlim:
                    xlim = xmin
                xmax = abs(xcen + xdist*fwhm[i,j]/2.)
                if xmax > xlim:
                    xlim = xmax

            #determining the wavenumber spacing we need to sample properly the gaussians
            xsamp = 7.0   #number of points we require to sample one HWHM 
            xhwhm = 10000.0
            for j in range(ng):
                xhwhmx = fwhm[i,j]/2. 
                if xhwhmx < xhwhm:
                    xhwhm = xhwhmx
            deltawave = xhwhm/xsamp
            np1 = 2.0 * xlim / deltawave
            npx = int(np1) + 1

            #Calculating the ILS in this spectral point
            iamp = np.zeros([ng])
            imean = np.zeros([ng])
            ifwhm = np.zeros([ng])
            fun = np.zeros([npx])
            xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
            for j in range(ng):
                iamp[j] = amp[i,j]
                imean[j] = offset[i,j] + vconv1[i]
                ifwhm[j] = fwhm[i,j]

            fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
            nfil[i] = npx
            vfil1[0:nfil[i],i] = xwave[:]
            afil1[0:nfil[i],i] = fun[:]

        mfil = nfil.max()
        vfil = np.zeros([mfil,nconv])
        afil = np.zeros([mfil,nconv])
        for i in range(nconv):
            vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
            afil[0:nfil[i],i] = afil1[0:nfil[i],i]

        Measurement.NFIL = nfil
        Measurement.VFIL = vfil
        Measurement.AFIL = afil

        #3. Defining new calculations wavelengths and reading again lbl-tables in correct range
        ###########################################################################################

        #Spectroscopy.read_lls(Spectroscopy.RUNNAME)
        #Measurement.wavesetc(Spectroscopy,IGEOM=0)
        #Spectroscopy.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())

        return Measurement,Spectroscopy


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
        #******** model for retrieving the ILS and Wavelength calibration in ACS MIR solar occultation observations

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #V0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #C0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #C1
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #C2
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P0 - Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P1 - FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P2 - Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P3 - Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 228. Retrieval of instrument line shape for ACS-MIR and wavelength calibration
        #**************************************************************************************

        V0 = forward_model.Variables.XN[ix]
        C0 = forward_model.Variables.XN[ix+1]
        C1 = forward_model.Variables.XN[ix+2]
        C2 = forward_model.Variables.XN[ix+3]
        P0 = forward_model.Variables.XN[ix+4]
        P1 = forward_model.Variables.XN[ix+5]
        P2 = forward_model.Variables.XN[ix+6]
        P3 = forward_model.Variables.XN[ix+7]

        forward_model.MeasurementX,forward_model.SpectroscopyX = self.calculate(forward_model.MeasurementX,forward_model.SpectroscopyX,V0,C0,C1,C2,P0,P1,P2,P3)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model229(PreRTModelBase):
    """
        Model for representing the double-Gaussian parameterisation of the instrument lineshape for
        retrievals from the Atmospheric Chemistry Suite aboard the ExoMars Trace Gas Orbiter
    """
    id : int = 229
    
    def __init__(
            self, 
            state_vector_start : int = 0, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int = 7,
            #   Number of parameters for this model stored in the state vector
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('A0', slice(0,1), 'Wavenumber offset of main at lowest wavenumber','cm-1'),
            ModelParameter('A1', slice(1,2), 'Wavenumber offset of main at wavenumber in the middle','cm-1'),
            ModelParameter('A2', slice(2,3), 'Wavenumber offset of main at highest wavenumber','cm-1'),
            ModelParameter('DELDG', slice(3,4), 'Offset of the second gaussian with respect to the first one (assumed spectrally constant)','cm-1'),
            ModelParameter('FWHM', slice(4,5), 'FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units)','cm-1'),
            ModelParameter('AMP1', slice(5,6), 'Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber'),
            ModelParameter('AMP2', slice(6,7), 'Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var)'),
        )


    @classmethod
    def calculate(cls, Measurement,par1,par2,par3,par4,par5,par6,par7,MakePlot=False):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 229 in NEMESIS.
                In this model, the ILS of the measurement is defined from every convolution wavenumber
                using the double-Gaussian parameterisation created for analysing ACS MIR spectra

            INPUTS :

                Measurement :: Python class defining the Measurement
                par1 :: Wavenumber offset of main at lowest wavenumber
                par2 :: Wavenumber offset of main at wavenumber in the middle
                par3 :: Wavenumber offset of main at highest wavenumber 
                par4 :: Offset of the second gaussian with respect to the first one (assumed spectrally constant)
                par5 :: FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units)
                par6 :: Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
                par7 :: Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Updated Measurement class

            CALLING SEQUENCE:

                Measurement = model229(Measurement,par1,par2,par3,par4,par5,par6,par7)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        #Calculating the parameters for each spectral point
        nconv = Measurement.NCONV[0]
        vconv1 = Measurement.VCONV[0:nconv,0]
        ng = 2

        # 1. Wavenumber offset of the two gaussians
        #    We divide it in two sections with linear polynomials     
        iconvmid = int(nconv/2.)
        wavemax = vconv1[nconv-1]
        wavemin = vconv1[0]
        wavemid = vconv1[iconvmid]
        offgrad1 = (par2 - par1)/(wavemid-wavemin)
        offgrad2 = (par2 - par3)/(wavemid-wavemax)
        offset = np.zeros([nconv,ng])
        for i in range(iconvmid):
            offset[i,0] = (vconv1[i] - wavemin) * offgrad1 + par1
            offset[i,1] = offset[i,0] + par4
        for i in range(nconv-iconvmid):
            offset[i+iconvmid,0] = (vconv1[i+iconvmid] - wavemax) * offgrad2 + par3
            offset[i+iconvmid,1] = offset[i+iconvmid,0] + par4

        # 2. FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
        fwhm = np.zeros([nconv,ng])
        fwhml = par5 / wavemin**2.0
        for i in range(nconv):
            fwhm[i,0] = fwhml * (vconv1[i])**2.
            fwhm[i,1] = fwhm[i,0]

        # 3. Amplitde of the second gaussian with respect to the main one
        amp = np.zeros([nconv,ng])
        ampgrad = (par7 - par6)/(wavemax-wavemin)
        for i in range(nconv):
            amp[i,0] = 1.0
            amp[i,1] = (vconv1[i] - wavemin) * ampgrad + par6

        #Running for each spectral point
        nfil = np.zeros(nconv,dtype='int32')
        mfil1 = 200
        vfil1 = np.zeros([mfil1,nconv])
        afil1 = np.zeros([mfil1,nconv])
        for i in range(nconv):

            #determining the lowest and highest wavenumbers to calculate
            xlim = 0.0
            xdist = 5.0 
            for j in range(ng):
                xcen = offset[i,j]
                xmin = abs(xcen - xdist*fwhm[i,j]/2.)
                if xmin > xlim:
                    xlim = xmin
                xmax = abs(xcen + xdist*fwhm[i,j]/2.)
                if xmax > xlim:
                    xlim = xmax

            #determining the wavenumber spacing we need to sample properly the gaussians
            xsamp = 7.0   #number of points we require to sample one HWHM 
            xhwhm = 10000.0
            for j in range(ng):
                xhwhmx = fwhm[i,j]/2. 
                if xhwhmx < xhwhm:
                    xhwhm = xhwhmx
            deltawave = xhwhm/xsamp
            np1 = 2.0 * xlim / deltawave
            npx = int(np1) + 1

            #Calculating the ILS in this spectral point
            iamp = np.zeros([ng])
            imean = np.zeros([ng])
            ifwhm = np.zeros([ng])
            fun = np.zeros([npx])
            xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
            for j in range(ng):
                iamp[j] = amp[i,j]
                imean[j] = offset[i,j] + vconv1[i]
                ifwhm[j] = fwhm[i,j]

            fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
            nfil[i] = npx
            vfil1[0:nfil[i],i] = xwave[:]
            afil1[0:nfil[i],i] = fun[:]

        mfil = nfil.max()
        vfil = np.zeros([mfil,nconv])
        afil = np.zeros([mfil,nconv])
        for i in range(nconv):
            vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
            afil[0:nfil[i],i] = afil1[0:nfil[i],i]

        Measurement.NFIL = nfil
        Measurement.VFIL = vfil
        Measurement.AFIL = afil

        if MakePlot==True:

            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))

            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax1.set_ylabel(r'f($\nu$)')
            ax1.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()

            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax2.set_ylabel(r'f($\nu$)')
            ax2.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()

            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax3.set_ylabel(r'f($\nu$)')
            ax3.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()

            plt.tight_layout()
            plt.show()

        return Measurement


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
        #******** model for retrieving the ILS in ACS MIR solar occultation observations

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #wavenumber offset at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #wavenumber offset at wavenumber in the middle
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #wavenumber offset at highest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
        #***************************************************************

        par1 = forward_model.Variables.XN[ix]
        par2 = forward_model.Variables.XN[ix+1]
        par3 = forward_model.Variables.XN[ix+2]
        par4 = forward_model.Variables.XN[ix+3]
        par5 = forward_model.Variables.XN[ix+4]
        par6 = forward_model.Variables.XN[ix+5]
        par7 = forward_model.Variables.XN[ix+6]

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,par1,par2,par3,par4,par5,par6,par7)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model230(PreRTModelBase):
    """
        In this model, the ILS of the measurement is defined from every convolution wavenumber
        using the double-Gaussian parameterisation created for analysing ACS MIR spectra.
        However, we can define several spectral windows where the ILS is different
    """
    id : int = 230


    @classmethod
    def calculate(cls, Measurement,nwindows,liml,limh,par,MakePlot=False):

        """
            FUNCTION NAME : model230()

            DESCRIPTION :

                Function defining the model parameterisation 230 in NEMESIS.
                In this model, the ILS of the measurement is defined from every convolution wavenumber
                using the double-Gaussian parameterisation created for analysing ACS MIR spectra.
                However, we can define several spectral windows where the ILS is different

            INPUTS :

                Measurement :: Python class defining the Measurement
                nwindows :: Number of spectral windows in which to fit the ILS
                liml(nwindows) :: Low wavenumber limit of each spectral window
                limh(nwindows) :: High wavenumber limit of each spectral window
                par(0,nwindows) :: Wavenumber offset of main at lowest wavenumber for each window
                par(1,nwindows) :: Wavenumber offset of main at wavenumber in the middle for each window
                par(2,nwindows) :: Wavenumber offset of main at highest wavenumber for each window
                par(3,nwindows) :: Offset of the second gaussian with respect to the first one (assumed spectrally constant) for each window
                par(4,nwindows) :: FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units) for each window
                par(5,nwindows) :: Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber for each window
                par(6,nwindows) :: Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var) for each window

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated Measurement class

            CALLING SEQUENCE:

                Measurement = model230(Measurement,nwindows,liml,limh,par)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        #Calculating the parameters for each spectral point
        nconv = Measurement.NCONV[0]
        vconv2 = Measurement.VCONV[0:nconv,0]
        ng = 2


        nfil2 = np.zeros(nconv,dtype='int32')
        mfil2 = 200
        vfil2 = np.zeros([mfil2,nconv])
        afil2 = np.zeros([mfil2,nconv])

        ivtot = 0
        for iwindow in range(nwindows):

            #Calculating the wavenumbers at which each spectral window applies
            ivwin = np.where( (vconv2>=liml[iwindow]) & (vconv2<=limh[iwindow]) )
            ivwin = ivwin[0]

            vconv1 = vconv2[ivwin]
            nconv1 = len(ivwin)


            par1 = par[0,iwindow]
            par2 = par[1,iwindow]
            par3 = par[2,iwindow]
            par4 = par[3,iwindow]
            par5 = par[4,iwindow]
            par6 = par[5,iwindow]
            par7 = par[6,iwindow]

            # 1. Wavenumber offset of the two gaussians
            #    We divide it in two sections with linear polynomials     
            iconvmid = int(nconv1/2.)
            wavemax = vconv1[nconv1-1]
            wavemin = vconv1[0]
            wavemid = vconv1[iconvmid]
            offgrad1 = (par2 - par1)/(wavemid-wavemin)
            offgrad2 = (par2 - par3)/(wavemid-wavemax)
            offset = np.zeros([nconv,ng])
            for i in range(iconvmid):
                offset[i,0] = (vconv1[i] - wavemin) * offgrad1 + par1
                offset[i,1] = offset[i,0] + par4
            for i in range(nconv1-iconvmid):
                offset[i+iconvmid,0] = (vconv1[i+iconvmid] - wavemax) * offgrad2 + par3
                offset[i+iconvmid,1] = offset[i+iconvmid,0] + par4

            # 2. FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
            fwhm = np.zeros([nconv1,ng])
            fwhml = par5 / wavemin**2.0
            for i in range(nconv1):
                fwhm[i,0] = fwhml * (vconv1[i])**2.
                fwhm[i,1] = fwhm[i,0]

            # 3. Amplitde of the second gaussian with respect to the main one
            amp = np.zeros([nconv1,ng])
            ampgrad = (par7 - par6)/(wavemax-wavemin)
            for i in range(nconv1):
                amp[i,0] = 1.0
                amp[i,1] = (vconv1[i] - wavemin) * ampgrad + par6


            #Running for each spectral point
            nfil = np.zeros(nconv1,dtype='int32')
            mfil1 = 200
            vfil1 = np.zeros([mfil1,nconv1])
            afil1 = np.zeros([mfil1,nconv1])
            for i in range(nconv1):

                #determining the lowest and highest wavenumbers to calculate
                xlim = 0.0
                xdist = 5.0 
                for j in range(ng):
                    xcen = offset[i,j]
                    xmin = abs(xcen - xdist*fwhm[i,j]/2.)
                    if xmin > xlim:
                        xlim = xmin
                    xmax = abs(xcen + xdist*fwhm[i,j]/2.)
                    if xmax > xlim:
                        xlim = xmax

                #determining the wavenumber spacing we need to sample properly the gaussians
                xsamp = 7.0   #number of points we require to sample one HWHM 
                xhwhm = 10000.0
                for j in range(ng):
                    xhwhmx = fwhm[i,j]/2. 
                    if xhwhmx < xhwhm:
                        xhwhm = xhwhmx
                deltawave = xhwhm/xsamp
                np1 = 2.0 * xlim / deltawave
                npx = int(np1) + 1

                #Calculating the ILS in this spectral point
                iamp = np.zeros([ng])
                imean = np.zeros([ng])
                ifwhm = np.zeros([ng])
                fun = np.zeros([npx])
                xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
                for j in range(ng):
                    iamp[j] = amp[i,j]
                    imean[j] = offset[i,j] + vconv1[i]
                    ifwhm[j] = fwhm[i,j]

                fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
                nfil[i] = npx
                vfil1[0:nfil[i],i] = xwave[:]
                afil1[0:nfil[i],i] = fun[:]



            nfil2[ivtot:ivtot+nconv1] = nfil[:]
            vfil2[0:mfil1,ivtot:ivtot+nconv1] = vfil1[0:mfil1,:]
            afil2[0:mfil1,ivtot:ivtot+nconv1] = afil1[0:mfil1,:]

            ivtot = ivtot + nconv1

        if ivtot!=nconv:
            raise ValueError('error in model 230 :: The spectral windows must cover the whole measured spectral range')

        mfil = nfil2.max()
        vfil = np.zeros([mfil,nconv])
        afil = np.zeros([mfil,nconv])
        for i in range(nconv):
            vfil[0:nfil2[i],i] = vfil2[0:nfil2[i],i]
            afil[0:nfil2[i],i] = afil2[0:nfil2[i],i]

        Measurement.NFIL = nfil2
        Measurement.VFIL = vfil
        Measurement.AFIL = afil

        if MakePlot==True:

            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))

            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax1.set_ylabel(r'f($\nu$)')
            ax1.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()

            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax2.set_ylabel(r'f($\nu$)')
            ax2.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()

            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax3.set_ylabel(r'f($\nu$)')
            ax3.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()

            plt.tight_layout()
            plt.show()

        return Measurement


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
        #******** model for retrieving multiple ILS (different spectral windows) in ACS MIR solar occultation observations

        s = f.readline().split()
        f1 = open(s[0],'r')
        s = f1.readline().split()
        nwindows = int(s[0])
        varparam[0] = nwindows
        liml = np.zeros(nwindows)
        limh = np.zeros(nwindows)
        for iwin in range(nwindows):
            s = f1.readline().split()
            liml[iwin] = float(s[0])
            limh[iwin] = float(s[1])
            varparam[2*iwin+1] = liml[iwin]
            varparam[2*iwin+2] = limh[iwin]

        par = np.zeros((7,nwindows))
        parerr = np.zeros((7,nwindows))
        for iw in range(nwindows):
            for j in range(7):
                s = f1.readline().split()
                par[j,iw] = float(s[0])
                parerr[j,iw] = float(s[1])
                x0[ix] = par[j,iw]
                sx[ix,ix] = (parerr[j,iw])**2.
                inum[ix] = 0
                ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 230. Retrieval of multiple instrument line shapes for ACS-MIR
        #***************************************************************

        nwindows = int(forward_model.Variables.VARPARAM[ivar,0])
        liml = np.zeros(nwindows)
        limh = np.zeros(nwindows)
        i0 = 1
        for iwin in range(nwindows):
            liml[iwin] = forward_model.Variables.VARPARAM[ivar,i0]
            limh[iwin] = forward_model.Variables.VARPARAM[ivar,i0+1]
            i0 = i0 + 2

        par1 = np.zeros((7,nwindows))
        for iwin in range(nwindows):
            for jwin in range(7):
                par1[jwin,iwin] = forward_model.Variables.XN[ix]
                ix = ix + 1

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,nwindows,liml,limh,par1)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model444(PreRTModelBase):
    """
        Allows for retrieval of the particle size distribution and imaginary refractive index.
    """
    
    id : int = 444


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            haze_params : dict[str,Any],
            #   Optical constants for the aerosol species (haze) this model represents
            
            aerosol_species_index : int,
            #   Index of the aerosol species that this model pertains to
            
            scattering_type_id : int,
            #   The scattering type this model uses
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('particle_size_distribution_params', slice(0,2), 'Values that define the particle size distribution'),
            ModelParameter('imaginary_ref_idx', slice(2,None), 'Imaginary refractive index of the particle size distribution'),
        )
        
        # Store model-specific constants on the model instance for easy access later
        self.haze_params = haze_params
        self.aerosol_species_idx = aerosol_species_index
        self.scattering_type_id = scattering_type_id


    @classmethod
    def calculate(
            cls, 
            Scatter : "Scatter_0",
            #   Scatter_0 instance of the retrieval setup we are calculating this model for
            
            idust : int,
            #   Aerosol species index we are calculating this model for
            
            iscat : int,
            #   scattering type we are using for this mode. NOTE: this is always set to 1 for now
            
            xprof : np.ndarray[["nparam"],float],
            #   The slice of the state vector that parameters of this model are held in.
            
            haze_params : dict[str,Any] ,
            #   A dictionary of constants for the aerosol being represented by this model.
            
        ) -> "Scatter_0":
        """
            FUNCTION NAME : model444()

            DESCRIPTION :

                Function defining the model parameterisation 444 in NEMESIS.

                Allows for retrieval of the particle size distribution and imaginary refractive index.

            INPUTS :

                Scatter :: Python class defining the scattering parameters
                idust :: Index of the aerosol distribution to be modified (from 0 to NDUST-1)
                iscat :: Flag indicating the particle size distribution
                xprof :: Contains the size distribution parameters and imaginary refractive index
                haze_params :: Read from 444 file. Contains relevant constants.

            OPTIONAL INPUTS:


            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model444(Scatter,idust,iscat,xprof,haze_params)

            MODIFICATION HISTORY : Joe Penn (11/9/2024)

        """   
        _lgr.debug(f'{idust=} {iscat=} {xprof=} {type(xprof)=}')
        for item in ('WAVE', 'NREAL', 'WAVE_REF', 'WAVE_NORM'):
            _lgr.debug(f'haze_params[{item}] : {type(haze_params[item])} = {haze_params[item]}')

        a = np.exp(xprof[0])
        b = np.exp(xprof[1])
        if iscat == 1:
            pars = (a,b,(1-3*b)/b)
        elif iscat == 2:
            pars = (a,b,0)
        elif iscat == 4:
            pars = (a,0,0)
        else:
            _lgr.warning(f'ISCAT = {iscat} not implemented for model 444 yet! Defaulting to iscat = 1.')
            pars = (a,b,(1-3*b)/b)

        Scatter.WAVER = haze_params['WAVE']
        Scatter.REFIND_IM = np.exp(xprof[2:])
        reference_nreal = haze_params['NREAL']
        reference_wave = haze_params['WAVE_REF']
        normalising_wave = haze_params['WAVE_NORM']
        if len(Scatter.REFIND_IM) == 1:
            Scatter.REFIND_IM = Scatter.REFIND_IM * np.ones_like(Scatter.WAVER)

        Scatter.REFIND_REAL = kk_new_sub(np.array(Scatter.WAVER), np.array(Scatter.REFIND_IM), reference_wave, reference_nreal)


        Scatter.makephase(idust, iscat, pars)

        xextnorm = np.interp(normalising_wave,Scatter.WAVE,Scatter.KEXT[:,idust])
        Scatter.KEXT[:,idust] = Scatter.KEXT[:,idust]/xextnorm
        Scatter.KSCA[:,idust] = Scatter.KSCA[:,idust]/xextnorm
        return Scatter


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
        #******** model for retrieving an aerosol particle size distribution and imaginary refractive index spectrum
        
        _lgr.debug(f'{ix=}')
        s = f.readline().split()    
        haze_f = open(s[0],'r')
        haze_waves = []
        for j in range(2):
            line = haze_f.readline().split()
            xai, xa_erri = line[:2]

            x0[ix] = np.log(float(xai))
            lx[ix] = 1
            sx[ix,ix] = (float(xa_erri)/float(xai))**2.

            ix = ix + 1
        _lgr.debug(f'{ix=}')

        nwave, clen = haze_f.readline().split('!')[0].split()
        vref, nreal_ref = haze_f.readline().split('!')[0].split()
        v_od_norm = haze_f.readline().split('!')[0]
        _lgr.debug(f'{nwave=} {clen=} {vref=} {nreal_ref=} {v_od_norm=}')

        for j in range(int(nwave)):
            line = haze_f.readline().split()
            v, xai, xa_erri = line[:3]

            x0[ix] = np.log(float(xai))
            lx[ix] = 1
            sx[ix,ix] = (float(xa_erri)/float(xai))**2.

            ix = ix + 1
            haze_waves.append(float(v))

            if float(clen) < 0:
                break
        _lgr.debug(f'{ix=}')

        aerosol_species_idx = varident[1]-1

        haze_params = dict()
        haze_params['NX'] = 2+len(haze_waves)
        haze_params['WAVE'] = haze_waves
        haze_params['NREAL'] = float(nreal_ref)
        haze_params['WAVE_REF'] = float(vref)
        haze_params['WAVE_NORM'] = float(v_od_norm)

        varparam[0] = 2+len(haze_waves)
        varparam[1] = float(clen)
        varparam[2] = float(vref)
        varparam[3] = float(nreal_ref)
        varparam[4] = float(v_od_norm)

        if float(clen) > 0:
            for j in range(int(nwave)):
                for k in range(int(nwave)):

                    delv = haze_waves[k]-haze_waves[j]
                    arg = abs(delv/float(clen))
                    xfac = np.exp(-arg)
                    if xfac >= sxminfac:
                        sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                        sx[ix+k,ix+j] = sx[ix+j,ix+k]
        _lgr.debug(f'{ix=}')
        
        scattering_type_id = 1 # Should add a way to alter this value from the input files.

        return cls(ix_0, ix-ix_0, haze_params, aerosol_species_idx, scattering_type_id)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        
        # NOTE:
        # ix is not required as we have stored that information on the model instance
        # ipar is ignored for this model
        # ivar is ignored for this model
        # xmap is ignored for this model
        forward_model.ScatterX = self.calculate(
            forward_model.ScatterX,
            self.aerosol_species_idx,
            self.scattering_type_id,
            self.get_state_vector_slice(forward_model.Variables.XN),
            self.haze_params
        )


class Model446(PreRTModelBase):
    """
        In this model, we change the extinction coefficient and single scattering albedo 
        of a given aerosol population based on its particle size, and based on the extinction 
        coefficients tabulated in a look-up table
    """
    id : int = 446

    def __init__(
            self, 
            state_vector_start : int, 
            n_state_vector_entries : int,
            lookup_table_fpath : str,
        ):
        """
        Initialise an instance of the model.
        
        ## ARGUMENTS ##
            
            state_vector_start : int
                The index of the first entry of the model parameters in the state vector
            
            n_state_vector_entries : int
                The number of model parameters that are stored in the state vector
            
            lookup_table_fpath: str
                path to the lookup table of extinction coefficient vs particle size
                
        
        ## RETURNS ##
            An initialised instance of this object
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        self.lookup_table_fpath : str = lookup_table_fpath


    @classmethod
    def calculate(cls, Scatter,idust,wavenorm,xwave,rsize,lookupfile,MakePlot=False):

        """
            FUNCTION NAME : model446()

            DESCRIPTION :

                Function defining the model parameterisation 446 in NEMESIS.

                In this model, we change the extinction coefficient and single scattering albedo 
                of a given aerosol population based on its particle size, and based on the extinction 
                coefficients tabulated in a look-up table

            INPUTS :

                Scatter :: Python class defining the scattering parameters
                idust :: Index of the aerosol distribution to be modified (from 0 to NDUST-1)
                wavenorm :: Flag indicating if the extinction coefficient needs to be normalised to a given wavelength (1 if True)
                xwave :: If wavenorm=1, then this indicates the normalisation wavelength/wavenumber
                rsize :: Particle size at which to interpolate the extinction cross section
                lookupfile :: Name of the look-up file storing the extinction cross section data

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model446(Scatter,idust,wavenorm,xwave,rsize,lookupfile)

            MODIFICATION HISTORY : Juan Alday (25/11/2021)

        """

        import h5py
        from scipy.interpolate import interp1d

        #Reading the look-up table file
        with h5py.File(lookupfile,'r') as f:

            #NWAVE = h5py_helper.retrieve_data(f, 'NWAVE', np.int32)
            NSIZE = h5py_helper.retrieve_data(f, 'NSIZE', np.int32)

            WAVE = h5py_helper.retrieve_data(f, 'WAVE', np.array)
            REFF = h5py_helper.retrieve_data(f, 'REFF', np.array)

            KEXT = h5py_helper.retrieve_data(f, 'KEXT', np.array)      #(NWAVE,NSIZE)
            SGLALB = h5py_helper.retrieve_data(f, 'SGLALB', np.array)  #(NWAVE,NSIZE)

        #First we interpolate to the wavelengths in the Scatter class
        sext = interp1d(WAVE,KEXT,axis=0)
        KEXT1 = sext(Scatter.WAVE)
        salb = interp1d(WAVE,SGLALB,axis=0)
        SGLALB1 = salb(Scatter.WAVE)

        #Second we interpolate to the required particle size
        if rsize<REFF.min():
            rsize =REFF.min()
        if rsize>REFF.max():
            rsize=REFF.max()

        sext = interp1d(REFF,KEXT1,axis=1)
        KEXTX = sext(rsize)
        salb = interp1d(REFF,SGLALB1,axis=1)
        SGLALBX = salb(rsize)

        #Now check if we need to normalise the extinction coefficient
        if wavenorm==1:
            snorm = interp1d(Scatter.WAVE,KEXTX)
            vnorm = snorm(xwave)

            KEXTX[:] = KEXTX[:] / vnorm

        KSCAX = SGLALBX * KEXTX

        #Now we update the Scatter class with the required results
        Scatter.KEXT[:,idust] = KEXTX[:]
        Scatter.KSCA[:,idust] = KSCAX[:]
        Scatter.SGLALB[:,idust] = SGLALBX[:]

        f.close()

        if MakePlot==True:

            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)

            for i in range(NSIZE):

                ax1.plot(WAVE,KEXT[:,i])
                ax2.plot(WAVE,SGLALB[:,i])

            ax1.plot(Scatter.WAVE,Scatter.KEXT[:,idust],c='black')
            ax2.plot(Scatter.WAVE,Scatter.SGLALB[:,idust],c='black')

            if Scatter.ISPACE==0:
                label='Wavenumber (cm$^{-1}$)'
            else:
                label=r'Wavelength ($\mu$m)'
            ax2.set_xlabel(label)
            ax1.set_xlabel('Extinction coefficient')
            ax2.set_xlabel('Single scattering albedo')

            ax1.set_facecolor('lightgray')
            ax2.set_facecolor('lightgray')

            plt.tight_layout()

        return Scatter


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
        #******** model for retrieving an aerosol particle size distribution from a tabulated look-up table

        #This model changes the extinction coefficient of a given aerosol population based on 
        #the extinction coefficient look-up table stored in a separate file. 

        #The look-up table specifies the extinction coefficient as a function of particle size, and 
        #the parameter in the state vector is the particle size

        #The look-up table must have the format specified in Models/Models.py (model446)

        s = f.readline().split()
        aerosol_id = int(s[0])    #Aerosol population (from 0 to NDUST-1)
        wavenorm = int(s[1])      #If 1 - then the extinction coefficient will be normalised at a given wavelength

        xwave = 0.0
        if wavenorm==1:
            xwave = float(s[2])   #If 1 - wavelength at which to normalise the extinction coefficient

        varparam[0] = aerosol_id
        varparam[1] = wavenorm
        varparam[2] = xwave

        #Read the name of the look-up table file
        s = f.readline().split()
        fnamex = s[0]

        #Reading the particle size and its a priori error
        s = f.readline().split()
        lx[ix] = 0
        inum[ix] = 1
        x0[ix] = float(s[0])
        sx[ix,ix] = (float(s[1]))**2.

        ix = ix + 1

        return cls(ix_0, ix-ix_0, fnamex)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 446. model for retrieving the particle size distribution based on the data in a look-up table
        #***************************************************************

        #This model fits the particle size distribution based on the optical properties at different sizes
        #tabulated in a pre-computed look-up table. What this model does is to interpolate the optical 
        #properties based on those tabulated.

        idust0 = int(forward_model.Variables.VARPARAM[ivar,0])
        wavenorm = int(forward_model.Variables.VARPARAM[ivar,1])
        xwave = forward_model.Variables.VARPARAM[ivar,2]
        rsize = forward_model.Variables.XN[ix]

        forward_model.ScatterX = self.calculate(forward_model.ScatterX,idust0,wavenorm,xwave,rsize,self.lookup_table_fpath,MakePlot=False)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model447(PreRTModelBase):
    """
        In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
        is common to all geometries, but in the future it will be updated so that each measurement
        can have a different Doppler velocity (in order to retrieve wind speeds).
    """
    id : int = 447


    @classmethod
    def calculate(cls, Measurement,v_doppler):

        """
            FUNCTION NAME : model447()

            DESCRIPTION :

                Function defining the model parameterisation 447 in NEMESIS.
                In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
                is common to all geometries, but in the future it will be updated so that each measurement
                can have a different Doppler velocity (in order to retrieve wind speeds).

            INPUTS :

                Measurement :: Python class defining the measurement
                v_doppler :: Doppler velocity (km/s)

            OPTIONAL INPUTS: none

            OUTPUTS :

                Measurement :: Updated measurement class with the correct Doppler velocity

            CALLING SEQUENCE:

                Measurement = model447(Measurement,v_doppler)

            MODIFICATION HISTORY : Juan Alday (25/07/2023)

        """

        Measurement.V_DOPPLER = v_doppler

        return Measurement


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
        #******** model for retrieving the Doppler shift

        #Read the Doppler velocity and its uncertainty
        s = f.readline().split()
        v_doppler = float(s[0])     #km/s
        v_doppler_err = float(s[1]) #km/s

        #Filling the state vector and a priori covariance matrix with the doppler velocity
        lx[ix] = 0
        x0[ix] = v_doppler
        sx[ix,ix] = (v_doppler_err)**2.
        inum[ix] = 1

        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError


class Model500(PreRTModelBase):
    """
        This allows the retrieval of CIA opacity with a gaussian basis.
        Assumes a constant P/T dependence.
    """
    id : int = 500


    @classmethod
    def calculate(cls, k_cia, waven, icia, vlo, vhi, nbasis, amplitudes):
        """
            FUNCTION NAME : model500()

            DESCRIPTION :

                Function defining the model parameterisation 500.
                This allows the retrieval of CIA opacity with a gaussian basis.
                Assumes a constant P/T dependence.

            INPUTS :

                cia :: CIA class

                icia :: CIA pair to be modelled

                vlo :: Lower wavenumber bound

                vhi :: Upper wavenumber bound

                nbasis :: Number of gaussians in the basis

                amplitudes :: Amplitudes of each gaussian


            OUTPUTS :

                cia :: Updated CIA class
                xmap :: Gradient (not implemented)

            CALLING SEQUENCE:

                cia,xmap = model500(cia, icia, nbasis, amplitudes)

            MODIFICATION HISTORY : Joe Penn (14/01/25)

        """

        ilo = np.argmin(np.abs(waven-vlo))
        ihi = np.argmin(np.abs(waven-vhi))
        width = (ihi - ilo)/nbasis          # Width of the Gaussian functions
        centers = np.linspace(ilo, ihi, int(nbasis))

        def gaussian_basis(x, centers, width):
            return np.exp(-((x[:, None] - centers[None, :])**2) / (2 * width**2))

        x = np.arange(ilo,ihi+1)

        G = gaussian_basis(x, centers, width)
        gaussian_cia = G @ amplitudes

        k_cia = k_cia * 0

        k_cia[icia,:,:,ilo:ihi+1] = gaussian_cia

        xmap = np.zeros(1)
        return k_cia,xmap


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

        s = f.readline().split()
        amp_f = open(s[0],'r')

        tmp = np.fromfile(amp_f,sep=' ',count=2,dtype='float')

        nbasis = int(tmp[0])
        clen = float(tmp[1])

        amp = np.zeros([nbasis])
        eamp = np.zeros([nbasis])

        for j in range(nbasis):
            tmp = np.fromfile(amp_f,sep=' ',count=2,dtype='float')
            amp[j] = float(tmp[0])
            eamp[j] = float(tmp[1])

            lx[ix+j] = 1
            x0[ix+j] = np.log(amp[j])
            sx[ix+j,ix+j] = ( eamp[j]/amp[j]  )**2.

        for j in range(nbasis):
            for k in range(nbasis):

                deli = j-k
                arg = abs(deli/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        varparam[0] = nbasis
        ix = ix + nbasis

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:

        icia = forward_model.Variables.VARIDENT[ivar,1]

        if forward_model.Measurement.ISPACE == WaveUnit.Wavelength_um:
            vlo = 1e4/(forward_model.SpectroscopyX.WAVE.max())
            vhi = 1e4/(forward_model.SpectroscopyX.WAVE.min())
        else:
            vlo = forward_model.SpectroscopyX.WAVE.min()
            vhi = forward_model.SpectroscopyX.WAVE.max()

        nbasis = forward_model.Variables.VARPARAM[ivar,0]
        amplitudes = np.exp(forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]])*1e-40

        new_k_cia, xmap1 = self.calculate(forward_model.CIA.K_CIA.copy(), forward_model.CIA.WAVEN, icia, vlo, vhi, nbasis, amplitudes)

        forward_model.CIA.K_CIA = new_k_cia
        forward_model.CIAX.K_CIA = new_k_cia

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model666(PreRTModelBase):
    """
        In this model, we retrieve the pressure at a given tangent height.
    """
    id : int = 666

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            htan : float,
            #   Tangent height (km) at which the pressure is retrieved
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, htan)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('PTAN', slice(0,1), 'Pressure (at tangent height = '+str(htan)+' km)', 'atm'),
        )
        self.htan = htan

    @classmethod
    def calculate(cls, Atmosphere, htan, ptan, MakePlot=False):

        """
            FUNCTION NAME : model666()

            DESCRIPTION :

                Function defining the model parameterisation 666 in NEMESIS.
                In this model, we retrieve the pressure at a given tangent height.

            INPUTS :

                Atmosphere :: Atmosphere class
                htan :: Tangent height (km)
                ptan :: Pressure at tangent height (atm)

            OPTIONAL INPUTS: None

            OUTPUTS :
                
                Atmosphere :: Updated Atmosphere class with recomputed pressure levels

            CALLING SEQUENCE:

                Atmosphere = model666(Atmosphere,htan,ptan)

            MODIFICATION HISTORY : Juan Alday (15/02/2023)

        """

        hpre = Atmosphere.H
        ppre = Atmosphere.P
    
        _lgr.info(f'Calculating model 666 with htan={htan} km and ptan={ptan} atm')

        Atmosphere.adjust_hydrostatP(htan*1.0e3,ptan*101325.)

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(ppre,hpre/1.0e3,label='Uncorrected')
            ax1.plot(Atmosphere.P,Atmosphere.H/1.0e3,label='Corrected')
            ax1.legend()
            ax1.set_xlabel('Pressure (Pa)')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_xscale('log')
            plt.tight_layout()

        return Atmosphere

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
        #******** pressure at a given tangent height
        s = f.readline().split()
        htan = float(s[0])  #Tangent height (km)
        s = f.readline().split()
        ptan = float(s[0])
        ptanerr = float(s[1])

        varparam[0] = htan

        if ptan>0.0:
            x0[ix] = np.log(ptan)
            lx[ix] = 1
            inum[ix] = 1
        else:
            raise ValueError('error in read_apr_nemesis() :: pressure must be > 0')
    
        sx[ix,ix] = (ptanerr/ptan)**2.
        #jpre = ix
    
        ix = ix + 1

        return cls(ix_0, ix-ix_0, htan)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 666. Retrieval of pressure at a given tangent height
        #***************************************************************

        ptan = np.exp(forward_model.Variables.XN[ix])

        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,self.htan,ptan)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model777(PreRTModelBase):
    """
        In this model, we apply a correction to the tangent heights listed on the 
        Measurement class
    """
    id : int = 777


    @classmethod
    def calculate(cls, Measurement,hcorr,MakePlot=False):

        """
            FUNCTION NAME : model777()

            DESCRIPTION :

                Function defining the model parameterisation 777 in NEMESIS.
                In this model, we apply a correction to the tangent heights listed on the 
                Measurement class

            INPUTS :

                Measurement :: Measurement class
                hcorr :: Correction to the tangent heights (km)

            OPTIONAL INPUTS: None

            OUTPUTS :

                Measurement :: Updated Measurement class with corrected tangent heights

            CALLING SEQUENCE:

                Measurement = model777(Measurement,hcorr)

            MODIFICATION HISTORY : Juan Alday (15/02/2023)

        """

        #Getting the tangent heights
        tanhe = np.zeros(Measurement.NGEOM)
        tanhe[:] = Measurement.TANHE[:,0]

        #Correcting tangent heights
        tanhe_new = tanhe + hcorr

        #Updating Measurement class
        Measurement.TANHE[:,0] = tanhe_new

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.scatter(np.arange(0,Measurement.NGEOM,1),tanhe,label='Uncorrected')
            ax1.scatter(np.arange(0,Measurement.NGEOM,1),Measurement.TANHE[:,0],label='Corrected')
            ax1.set_xlabel('Geometry #')
            ax1.set_ylabel('Tangent height (km)')
            plt.tight_layout()

        return Measurement


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
        #******** tangent height correction
        s = f.readline().split()
        hcorr = float(s[0])
        herr = float(s[1])

        x0[ix] = hcorr
        sx[ix,ix] = herr**2.
        inum[ix] = 1

        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 777. Retrieval of tangent height corrections
        #***************************************************************

        hcorr = forward_model.Variables.XN[ix]

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,hcorr)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model887(PreRTModelBase):
    """
        In this model, the cross-section spectrum of IDUST is changed given the parameters in 
        the state vector
    """
    id : int = 887


    @classmethod
    def calculate(cls, Scatter,xsc,idust,MakePlot=False):

        """
            FUNCTION NAME : model887()

            DESCRIPTION :

                Function defining the model parameterisation 887 in NEMESIS.
                In this model, the cross-section spectrum of IDUST is changed given the parameters in 
                the state vector

            INPUTS :

                Scatter :: Python class defining the spectral properties of aerosols in the atmosphere
                xsc :: New cross-section spectrum of aerosol IDUST
                idust :: Index of the aerosol to be changed (from 0 to NDUST-1)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model887(Scatter,xsc,idust)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        if len(xsc)!=Scatter.NWAVE:
            raise ValueError('error in model 887 :: Cross-section array must be defined at the same wavelengths as in .xsc')
        else:
            kext = np.zeros([Scatter.NWAVE,Scatter.DUST])
            kext[:,:] = Scatter.KEXT
            kext[:,idust] = xsc[:]
            Scatter.KEXT = kext

        if MakePlot==True:
            fig,ax1=plt.subplots(1,1,figsize=(10,3))
            ax1.semilogy(Scatter.WAVE,Scatter.KEXT[:,idust])
            ax1.grid()
            if Scatter.ISPACE==1:
                ax1.set_xlabel(r'Wavelength ($\mu$m)')
            else:
                ax1.set_xlabel(r'Wavenumber (cm$^{-1}$')
            plt.tight_layout()
            plt.show()


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
        #******** Cloud x-section spectrum

        #Read in number of points, cloud id, and correlation between elements.
        s = f.readline().split()
        nwv = int(s[0]) #number of spectral points (must be the same as in .xsc)
        icloud = int(s[1])  #aerosol ID
        clen = float(s[2])  #Correlation length (in wavelengths/wavenumbers)

        varparam[0] = nwv
        varparam[1] = icloud

        #Read the wavelengths and the extinction cross-section value and error
        wv = np.zeros(nwv)
        xsc = np.zeros(nwv)
        err = np.zeros(nwv)
        for iw in range(nwv):
            s = f.readline().split()
            wv[iw] = float(s[0])
            xsc[iw] = float(s[1])
            err[iw] = float(s[2])
            if xsc[iw]<=0.0:
                raise ValueError('error in read_apr :: Cross-section in model 887 must be greater than 0')

        #It is important to check that the wavelengths in .apr and in .xsc are the same
        Aero0 = Scatter_0()
        Aero0.read_xsc(runname)
        for iw in range(Aero0.NWAVE):
            if (wv[iw]-Aero0.WAVE[iw])>0.01:
                raise ValueError('error in read_apr :: Number of wavelengths in model 887 must be the same as in .xsc')

        #Including the parameters in state vector and covariance matrix
        for j in range(nwv):
            x0[ix+j] = np.log(xsc[j])
            lx[ix+j] = 1
            inum[ix+j] = 1
            sx[ix+j,ix+j] = (err[j]/xsc[j])**2.

        for j in range(nwv):
            for k in range(nwv):
                delv = wv[j] - wv[k]
                arg = abs(delv/clen)
                xfac = np.exp(-arg)
                if xfac>0.001:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        #jxsc = ix

        ix = ix + nwv

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError


class Model999(PreRTModelBase):
    """
        In this model, the temperature of the surface is defined.
    """
    id : int = 999 
        
    def __init__(
            self, 
            state_vector_start : int = 0, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int = 1,
            #   Number of parameters for this model stored in the state vector
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('surface temperature', slice(0,1), 'Surface Temperature','K'),
        )
        
        
    @classmethod
    def calculate(cls, Surface, tsurf):

        """
            FUNCTION NAME : model999()

            DESCRIPTION :

                Function defining the model parameterisation 999 in NEMESIS.
                In this model, we fit the surface temperature.

            INPUTS :

                Surface :: Python class defining the surface
                tsurf :: Surface temperature (K)

            OPTIONAL INPUTS: none

            OUTPUTS :

                Surface :: Updated measurement class with the surface temperature

            CALLING SEQUENCE:

                Surface = model999(Surface,tsurf)

            MODIFICATION HISTORY : Juan Alday (25/05/2025)

        """

        Surface.TSURF = tsurf

        return Surface


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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** model for retrieving the Surface temperature

        #Read the surface temperature and its uncertainty
        s = f.readline().split()
        tsurf = float(s[0])     #K
        tsurf_err = float(s[1]) #K

        #Filling the state vector and a priori covariance matrix with the surface temperature
        lx[ix] = 0   #linear scale
        x0[ix] = tsurf
        sx[ix,ix] = (tsurf_err)**2.
        inum[ix] = 0  #analytical gradient

        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 999. Retrieval of surface temperature
        #***************************************************************

        tsurf = forward_model.Variables.XN[ix]

        forward_model.SurfaceX = self.calculate(forward_model.SurfaceX,tsurf)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]











