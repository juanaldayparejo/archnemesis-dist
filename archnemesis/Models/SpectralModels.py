from __future__ import annotations #  for 3.9 compatability

from .ModelBase import *


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



class SpectralModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the calculated spectrum in the forward model.
    """
    
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        return varident[0]==cls.id
    
    
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
        _lgr.info(f'Model id {self.id} method "patch_from_subprofretg" does nothing...')
    
    
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
        _lgr.info(f'Model id {self.id} method "calculate_from_subprofretg" does nothing...')
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    @abc.abstractmethod
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        raise NotImplementedError(f'calculate_from_subspecret should be implemented for all Spectral models')


class TemplateSpectralModel(SpectralModelBase):
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
        super().__init__(state_vector_start, n_state_vector_entries)
        
        
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
            
            SPECMOD : np.ndarray[['NCONV'],float],
            #   Modelled spectrum
            
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            #   Gradient of modelled spectrum
            
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
            
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        """
        This class method should perform the actual calculation. Ideally it should not know anything
        about the geometries, locations, etc. of the retrieval setup. Try to make it just perform the
        actual calculation and not any "data arranging". 
        
        For example, instead of passing the geometry index, pass sliced arrays, perform the calculation, 
        and put the result back into the "source" arrays.
        
        This makes it easier to use this class method from another source if required.
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        # Return the results of the calculation
        return SPECMOD, dSPECMOD


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

        return cls(ix_0, ix-ix_0)


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
            Updates the spectra based upon values of model parameters in the state vector. Called from ForwardModel_0::subspecret.
        """
        
        raise NotImplementedError('This is a template model and should never be used')
        
        # Example code for unpacking parameters from the state vector
        # NOTE: this takes care of 'unlogging' values when required.
        (
            single_value_parameter_name,
            multi_value_parameter_name,
            variable_length_parameter_name,
            another_variable_length_parameter_name,
        
        ) = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        
        # Example code, generally want to loop over geometries here rather than in `self.calculate(...)`
        for i_geom in range(self.n_geom):
        
            # Example code for calling the `self.calculate(...)` class method
            # NOTE: we can call the class method via the `self` instance.
            specmod, dspecmod = self.calculate(
                SPECMOD[:,i_geom],
                dSPECMOD[:,i_geom,:],
                single_value_parameter_name,
                multi_value_parameter_name,
                variable_length_parameter_name,
                another_variable_length_parameter_name
            )
        
            # Example code for packing the results of the calculation back into the spectra
            # and the matrix that holds functional derivatives.
            SPECMOD[:,i] = specmod
            dSPECMOD[:,i,:] = dspecmod
        
        return


class Model231(SpectralModelBase):
    """
        Scaling of spectrum using a varying scaling factor (following a polynomial of degree N)
        
        The computed spectra is multiplied by `R = R0 * POL`, where the polynomial function POL depends on the wavelength and is given by:
        
            POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
    """
    id : int = 231
    
    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            n_geom : int,
            #   The number of geometries that this model applies to, applies from the first to the last geometry.
            #   Geometries with index >= n_geom will not be affected by this model.
            
            n_degree : int,
            #   The degree of the polynomial
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model.
        # NOTE: It is best to define these in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        self.parameters = (
            ModelParameter('coeff', slice(None), 'coefficients for the polynomial', 'NUMBER'),
        )
        
        self.n_geom = n_geom
        self.n_degree = n_degree
        
        return
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            #   Modelled spectrum
            
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            #   Gradient of modelled spectrum
            
            WAVE : np.ndarray[['NCONV'], float],
            #   Wavelengths/wavenumbers the spectrum values are defined at
            
            COEFF : np.ndarray[['NDEGREE+1'],float],
            #   Coefficients of the polynomial
            
            state_vector_slice : slice,
            #   A slice that chooses parts of the state vector corresponding to the
            #   parameters used by this model
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        W0 = WAVE[0]
        spec = SPECMOD[:WAVE.size]
        POL = np.zeros_like(spec)
        
        dW = WAVE-WAVE0
        for j in range(COEFF.shape[0]):
            POL[:] = POL[:] + COEFF[j] * (dW**j)
        
        SPECMOD[:WAVE.size] *= POL
        
        dSPECMOD[:WAVE.size,:] *= POL[:,None]
        
        dspecmod_part = dSPECMOD[:WAVE.size, state_vector_slice]
        for j in range(COEFF.shape[0]):
            dspecmod_part[:WAVE.size,j] = spec * (dW**j)
        
        return SPECMOD, dSPECMOD
    
    
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
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='int')
        ngeom = int(tmp[0])
        ndegree = int(tmp[1])
        varparam[0] = ngeom
        varparam[1] = ndegree
        for ilevel in range(ngeom):
            tmp = f1.readline().split()
            for ic in range(ndegree+1):
                r0 = float(tmp[2*ic])
                err0 = float(tmp[2*ic+1])
                x0[ix] = r0
                sx[ix,ix] = (err0)**2.
                inum[ix] = 0
                ix = ix + 1
        return cls(ix_0, ix-ix_0, ngeom, ndegree)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        coeff_shape = (self.n_degree+1, self.n_geom)
        coeff = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX).reshape(coeff_shape)
        
        for i_geom in range(self.n_geom):
            SPECMOD[:,i_geom], dSPECMOD[:,i_geom,:] = self.calculate(
                SPECMOD[:,i_geom],
                dSPECMOD[:,i_geom,:],
                forward_model.Measurement.VCONV[:forward_model.Measurement.NCONV[i_geom], i_geom],
                coeff[:,i_geom],
                self.state_vector_slice
            )
        
        return


# [JD] This uses forward_model.SpectroscopyX, whereas Model231 uses forward_model.Measurement, is this correct?
class Model2310(SpectralModelBase):
    """
        Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
        in multiple spectral windows
    """
    id : int = 2310
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
            igeom_slices : tuple[slice,...],
            Spectroscopy : "Spectroscopy_0",
            NGEOM : int, 
            NDEGREE : int, 
            COEFF : np.ndarray[['NDEGREE+1','NGEOM'],float],
            lowin : np.ndarray[['NWINDOWS'],float],
            hiwin : np.ndarray[['NWINDOWS'],float],
        ) -> tuple[np.ndarray[['NCONV','NGEOM'],float], np.ndarray[['NCONV','NGEOM','NX'],float]]:
        
        for IWIN in range(lowin.size):
            ivin = np.where( (Spectroscopy.WAVE>=lowin[IWIN]) & (Spectroscopy.WAVE<hiwin[IWIN]) )[0]
            nvin = len(ivin)
        
            for i in range(NGEOM):
                T = COEFF[i]

                WAVE0 = Spectroscopy.WAVE[ivin].min()
                spec = np.array(SPECMOD[ivin,i])

                #Changing the state vector based on this parameterisation
                POL = np.zeros(nvin)
                for j in range(NDEGREE+1):
                    POL[:] = POL[:] + T[j]*(Spectroscopy.WAVE[ivin]-WAVE0)**j

                SPECMOD[ivin,i] *=  POL[:]

                #Changing the rest of the gradients based on the impact of this parameterisation
                dSPECMOD[ivin,i,:] *= POL[:,None]
                

                #Defining the analytical gradients for this parameterisation
                dspecmod_part = dSPECMOD[ivin,i,igeom_slices[i]]
                for j in range(NDEGREE+1):
                    dspecmod_part[:,j] = spec[:] * (Spectroscopy.WAVE[ivin]-WAVE0)**j

        return SPECMOD, dSPECMOD
    
    
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
        """
        Continuum addition to transmission spectra using a varying scaling factor (following polynomial of degree N)
        in several spectral windows 

        The computed spectra is multiplied by R = R0 * (T0 + POL)
        Where the polynomial function POL depends on the wavelength given by:
        POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
        """
        
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=3,dtype='int')
        nlevel = int(tmp[0])
        ndegree = int(tmp[1])
        nwindows = int(tmp[2])
        varparam[0] = nlevel
        varparam[1] = ndegree
        varparam[2] = nwindows

        i0 = 0
        #Defining the boundaries of the spectral windows
        for iwin in range(nwindows):
            tmp = f1.readline().split()
            varparam[3+i0] = float(tmp[0])
            i0 = i0 + 1
            varparam[3+i0] = float(tmp[1])
            i0 = i0 + 1

        #Reading the coefficients for the polynomial in each geometry and spectral window
        for iwin in range(nwindows):
            for ilevel in range(nlevel):
                tmp = np.fromfile(f1,sep=' ',count=2*(ndegree+1),dtype='float')
                for ic in range(ndegree+1):
                    r0 = float(tmp[2*ic])
                    err0 = float(tmp[2*ic+1])
                    x0[ix] = r0
                    sx[ix,ix] = (err0)**2.
                    inum[ix] = 0
                    ix = ix + 1
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 2310. Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
        in multiple spectral windows
        """

        #NGEOM = int(forward_model.Variables.VARPARAM[ivar,0])
        NGEOM = forward_model.MeasurementX.NGEOM
        NDEGREE = int(forward_model.Variables.VARPARAM[ivar,1])
        NWINDOWS = int(forward_model.Variables.VARPARAM[ivar,2])

        lowin = np.zeros(NWINDOWS)
        hiwin = np.zeros(NWINDOWS)
        i0 = 0
        for IWIN in range(NWINDOWS):
            lowin[IWIN] = float(forward_model.Variables.VARPARAM[ivar,3+i0])
            i0 = i0 + 1
            hiwin[IWIN] = float(forward_model.Variables.VARPARAM[ivar,3+i0])
            i0 = i0 + 1
        
        COEFF = np.array(forward_model.Variables.XN[ix : ix+(NDEGREE+1)*NGEOM]).reshape((NDEGREE+1,NGEOM))
        
        igeom_slices = tuple(slice(ix+(NDEGREE+1)*igeom, ix+(NDEGREE+1)*(igeom+1)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))

        SPECMOD[...], dSPECMOD[...] = self.calculate(
            SPECMOD, 
            dSPECMOD, 
            igeom_slices, 
            forward_model.SpectroscopyX,
            NGEOM, 
            NDEGREE, 
            COEFF,
            lowin,
            hiwin
        )


class Model232(SpectralModelBase):
    """
        Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
    """
    id : int = 232
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            igeom_slice : slice,
            Spectroscopy : "Spectroscopy_0",
            TAU0 : float,
            ALPHA : float,
            WAVE0 : float,
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        spec = np.array(SPECMOD)
        factor = np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) )

        #Changing the state vector based on this parameterisation
        SPECMOD *= factor

        #Changing the rest of the gradients based on the impact of this parameterisation
        dSPECMOD *= factor[:,None]

        #Defining the analytical gradients for this parameterisation
        dspecmod_part = SPECMOD[:,igeom_slice]
        dspecmod_part[:,0] = spec[:] * ( -((Spectroscopy.WAVE/WAVE0)**(-ALPHA)) * np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) ) )
        dspecmod_part[:,1] = spec[:] * TAU0 * np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) ) * np.log(Spectroscopy.WAVE/WAVE0) * (Spectroscopy.WAVE/WAVE0)**(-ALPHA)

        return SPECMOD, dSPECMOD
    
    
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
        """
        Continuum addition to transmission spectra using the Angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """
        s = f.readline().split()
        wavenorm = float(s[0])                    

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
        nlevel = int(tmp[0])
        varparam[0] = nlevel
        varparam[1] = wavenorm
        for ilevel in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=4,dtype='float')
            r0 = float(tmp[0])   #Opacity level at wavenorm
            err0 = float(tmp[1])
            r1 = float(tmp[2])   #Angstrom coefficient
            err1 = float(tmp[3])
            x0[ix] = r0
            sx[ix,ix] = (err0)**2.
            x0[ix+1] = r1
            sx[ix+1,ix+1] = err1**2.
            inum[ix] = 0
            inum[ix+1] = 0                        
            ix = ix + 2
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 232. Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """

        #The effect of this model takes place after the computation of the spectra in CIRSrad!
        if int(forward_model.Variables.NXVAR[ivar]/2)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(2), ix+(igeom+1)*(2)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))

        if NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):
                TAU0 = forward_model.Variables.XN[ix]
                ALPHA = forward_model.Variables.XN[ix+1]
                WAVE0 = forward_model.Variables.VARPARAM[ivar,1]
                
                SPECMOD[:,i], dSPECMOD[:,i] = self.calculate(
                    SPECMOD[:,i], 
                    dSPECMOD[:,i], 
                    igeom_slices[i], 
                    forward_model.SpectroscopyX,
                    TAU0,
                    ALPHA,
                    WAVE0
                )

        else:
            T0 = forward_model.Variables.XN[ix]
            ALPHA = forward_model.Variables.XN[ix+1]
            WAVE0 = forward_model.Variables.VARPARAM[ivar,1]
            
            _lgr.warning(f'It looks like there is no calculation for NGEOM=1 for model id = {self.id}')


class Model233(SpectralModelBase):
    """
        Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

            np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.
    """
    id : int = 233
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            igeom_slice : slice,
            Spectroscopy : "Spectroscopy_0",
            A0 : float,
            A1 : float,
            A2 : float,
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        spec = np.array(SPECMOD)

        #Calculating the aerosol opacity at each wavelength
        TAU = np.exp(A0 + A1 * np.log(Spectroscopy.WAVE) + A2 * np.log(Spectroscopy.WAVE)**2.)

        #Changing the state vector based on this parameterisation
        SPECMOD *= np.exp ( -TAU )

        #Changing the rest of the gradients based on the impact of this parameterisation
        dSPECMOD *= np.exp ( -TAU )
        
        #Defining the analytical gradients for this parameterisation
        dspecmod_part = SPECMOD[:,igeom_slice]
        dspecmod_part[:,0] = spec[:] * (-TAU) * np.exp(-TAU)
        dspecmod_part[:,1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Spectroscopy.WAVE)
        dspecmod_part[:,2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Spectroscopy.WAVE)**2.

        return SPECMOD, dSPECMOD
    
    
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
        """
        Aerosol opacity modelled with a variable angstrom coefficient. Applicable to transmission spectra.

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.                  
        """

        #Reading the file where the a priori parameters are stored
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
        nlevel = int(tmp[0])
        varparam[0] = nlevel
        for ilevel in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=6,dtype='float')
            a0 = float(tmp[0])   #A0
            err0 = float(tmp[1])
            a1 = float(tmp[2])   #A1
            err1 = float(tmp[3])
            a2 = float(tmp[4])   #A2
            err2 = float(tmp[5])
            x0[ix] = a0
            sx[ix,ix] = (err0)**2.
            x0[ix+1] = a1
            sx[ix+1,ix+1] = err1**2.
            x0[ix+2] = a2
            sx[ix+2,ix+2] = err2**2.
            inum[ix] = 0
            inum[ix+1] = 0    
            inum[ix+2] = 0                  
            ix = ix + 3
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 232. Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)
        ***************************************************************

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.

        The effect of this model takes place after the computation of the spectra in CIRSrad!
        """
        
        if int(forward_model.Variables.NXVAR[ivar]/3)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(3), ix+(igeom+1)*(3)) for igeom in range(NGEOM))


        if forward_model.MeasurementX.NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):

                A0 = forward_model.Variables.XN[ix]
                A1 = forward_model.Variables.XN[ix+1]
                A2 = forward_model.Variables.XN[ix+2]
                
                SPECMOD[:,i], dSPECMOD[:,i] = self.calculate(
                    SPECMOD[:,i], 
                    dSPECMOD[:,i],
                    igeom_slices[i],
                    forward_model.SpectroscopyX,
                    A0,
                    A1,
                    A2
                )

        else:
            A0 = forward_model.Variables.XN[ix]
            A1 = forward_model.Variables.XN[ix+1]
            A2 = forward_model.Variables.XN[ix+2]

            SPECMOD[:], dSPECMOD[:] = self.calculate(
                SPECMOD[:], 
                dSPECMOD[:],
                slice(ix,ix+3),
                forward_model.SpectroscopyX,
                A0,
                A1,
                A2
            )


class Model667(SpectralModelBase):
    """
        In this model, the output spectrum is scaled using a dillusion factor to account
        for strong temperature gradients in exoplanets
    """
    id : int = 667


    @classmethod
    def calculate(cls, Spectrum,xfactor,MakePlot=False):

        """
            FUNCTION NAME : model667()

            DESCRIPTION :

                Function defining the model parameterisation 667 in NEMESIS.
                In this model, the output spectrum is scaled using a dillusion factor to account
                for strong temperature gradients in exoplanets

            INPUTS :

                Spectrum :: Modelled spectrum 
                xfactor :: Dillusion factor

            OPTIONAL INPUTS: None

            OUTPUTS :

                Spectrum :: Modelled spectrum scaled by the dillusion factor

            CALLING SEQUENCE:

                Spectrum = model667(Spectrum,xfactor)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        Spectrum = Spectrum * xfactor

        return Spectrum


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
        #******** dilution factor to account for thermal gradients thorughout exoplanet
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfac = float(tmp[0])
        xfacerr = float(tmp[1])
        x0[ix] = xfac
        inum[ix] = 0 
        sx[ix,ix] = xfacerr**2.
        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        #Model 667. Spectrum scaled by dilution factor to account for thermal gradients in planets
        #**********************************************************************************************

        xfactor = forward_model.Variables.XN[ix]
        spec = np.zeros(forward_model.SpectroscopyX.NWAVE)
        spec[:] = SPECMOD
        SPECMOD = self.calculate(SPECMOD,xfactor)
        dSPECMOD = dSPECMOD * xfactor
        dSPECMOD[:,ix] = spec[:]
        ix = ix + 1

