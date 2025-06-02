from __future__ import annotations #  for 3.9 compatability

from .ModelBase import *

from archnemesis.enums import AtmosphericProfileType

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



class AtmosphericModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the Atmosphere component.
    """
    
    def __init__(
            self,
            state_vector_start : int, 
            n_state_vector_entries : int,
            atm_profile_type : AtmosphericProfileType,
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
        raise NotImplementedError(f'calculate_from_subprofretg must be implemented for all Atmospheric models')


class TemplateAtmosphericModel(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Modelm1(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model0(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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
        
        _lgr.debug(f'Result calculated, setting values...')
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


class Model2(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model3(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model9(AtmosphericModelBase):
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

        from scipy.integrate import simpson
        from archnemesis.Data.gas_data import const


        if atm_profile_type != AtmosphericProfileType.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)
        
        
        #Calculating the actual atmospheric scale height in each level
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        xmap = np.zeros([3,atm.NP])
        for itest in range(4):

            xdeep = tau
            xfsh = fsh
            hknee = href

            if itest==0:
                dummy = 1
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model32(AtmosphericModelBase):
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

        from scipy.integrate import simpson
        from archnemesis.Data.gas_data import const
        
        if atm_profile_type != AtmosphericProfileType.AEROSOL_DENSITY:
            _msg = f'Model id={cls.id} is only defined for aerosol profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)

        #Calculating the actual atmospheric scale height in each level
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((3,atm.NP))
        for itest in range(4):

            xdeep = tau
            xfrac_scale_height = frac_scale_height
            pknee = p_ref
            if itest==0:
                dummy = 1
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
            jfrac_scale_height = -1
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model45(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model47(AtmosphericModelBase):
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
        
        from archnemesis.Data.gas_data import const
        
        
        # Calculate atmospheric properties
        R = const["R"]
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model49(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model50(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model51(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model110(AtmosphericModelBase):
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

        N1 = 3982.04e5       #Total column particle density (cm-2)
        tau1 = 3.88          #Total column optical depth at 1 um

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

        N2 = 748.54e5         #Total column particle density (cm-2)
        tau2 = 7.62           #Total column optical depth at 1 um

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

        N2p = 613.71e5          #Total column particle density (cm-2)
        tau2p = 9.35            #Total column optical depth at 1 um

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

        N3 = 133.86e5           #Total column particle density (cm-2)
        tau3 = 14.14            #Total column optical depth at 1 um

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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model111(AtmosphericModelBase):
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

        N1 = 3982.04e5       #Total column particle density (cm-2)
        tau1 = 3.88          #Total column optical depth at 1 um

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

        N2 = 748.54e5         #Total column particle density (cm-2)
        tau2 = 7.62           #Total column optical depth at 1 um

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

        N2p = 613.71e5          #Total column particle density (cm-2)
        tau2p = 9.35            #Total column optical depth at 1 um

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

        N3 = 133.86e5           #Total column particle density (cm-2)
        tau3 = 14.14            #Total column optical depth at 1 um

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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model202(AtmosphericModelBase):
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
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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


class Model1002(AtmosphericModelBase):
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
        xmap1 = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))

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
        iparj = 1
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

        jsurf = ix

        ix = ix + nlocs

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert model_classification[0] is cls.__bases__[0], "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

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

