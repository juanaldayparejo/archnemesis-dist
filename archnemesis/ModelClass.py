import numpy as np
import numpy.ma
from typing import IO, Any
from collections import namedtuple

from archnemesis.Scatter_0 import kk_new_sub

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)

StateVectorEntry = namedtuple('StateVectorValue', ['model_id', 'sv_slice', 'is_fixed', 'apriori_value', 'posterior_value'])


class ModelBase:
    """
    Abstract base class of all parameterised models used by ArchNemesis. This class should be subclassed further for models of a particular component.
    """
    id : int = None # All "*ModelBase" classes that are not meant to be used should have an id of 'None'
    name : str = 'name should be overwritten in subclass'
    description : str = 'description should be overwritten in subclass'
    
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
        
    
    @property
    def parameter_slices(self) -> dict[str, slice]:
        """
        A dictionary that maps parameter names to the sub-slices of the state vector that holds the parameter values.
        This should be overloaded in subclasses to make it easy to get the apriori and posterior values of the
        parameters for this model. If not overwritten only one parameter called "unspecified" is defined, which
        contains all the parameters for the model.
        """
        return {
            'unspecified' : slice(None)
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
        ) -> dict[str, StateVectorEntry]:
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
            parameters : dict[str, StateVectorEntry]
                A dictionary that maps parameter names to the "StateVectorEntry" associated with that parameter.
                StateVectorEntry is a named tuple with the following fields:
                    model_id : int
                        The id of the model that the state vector entry is associated with
                    sv_slice : slice
                        Slice of the state vector that the parameter is associated with. sv_slice.end - sv_slice.start = 'm'
                        where 'm' is the number of entries in the state vector associated with the parameter.
                    is_fixed : np.ndarray[['m'],bool]
                        Array of boolean flags for each element of the parameter, if True the element is fixed (i.e. it is
                        not retrieved and should be identical between the apriori/posterior values)
                    apriori_value : np.ndarray[['m'],float]
                        Value of each element of the parameter in the apriori state vector. The value has been 'unlogged'
                        where applicable.
                    posterior_value : np.ndarray[['m'],float]
                        Value of each element of the parameter in the posterior state vector. The value has been 'unlogged'
                        where applicable.
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
            parameters[name] = StateVectorEntry(
                self.id,
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
            varident : np.ndarray[[3],int],
        ) -> bool:
        ...
    
    @classmethod
    def calculate(cls, *args, **kwargs) -> Any:
        """
        Models are so varied in here that I cannot make any specific interface at this level of abstraction
        """
        ...
    
    @classmethod
    def from_apr_to_state_vector(
            cls,
            variables : "Variables_0",
            f : IO,
            varident : np.ndarray[[3],int], # Should be the correct slice of the original (which should be a reference to the sub-array)
            varparam : np.ndarray[["mparam"],float], # Should be the correct slice of the original (which should be a reference to the sub-array)
            ix : int,
            lx : np.ndarray[["mx"],int], # should be a reference to the original
            x0 : np.ndarray[["mx"],float], # should be a reference to the original
            sx : np.ndarray[["mx","mx"],float], # should be a reference to the original
            inum : np.ndarray[["mx"],int], # should be a reference to the original
            varfile : list[str], # should be a reference to the original
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int: # Should return updated value of `ix`
        ...
        
    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        _lgr.info(f'No calculation of model id {cls.id} in subprofretg')
        ix = ix + forward_model.Variables.NXVAR[ivar]
        return ix

    @classmethod
    def patch_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        _lgr.info(f'No patch of model id {cls.id} in subprofretg')
        ix = ix + forward_model.Variables.NXVAR[ivar]
        return ix
    
    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
        _lgr.info(f'No calculation of model id {cls.id} in subspecret')
        ix = ix + forward_model.Variables.NXVAR[ivar]
        return ix
    
    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int: # Should return the number of elements the model has stored in the state vector
        ...

class AtmosphericModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the Atmosphere component.
    """
    name : str = 'name of atmospheric model should be overwritten in subclass'
    description : str = 'description of atmospheric model should be overwritten in subclass'
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        return varident[2]==cls.id
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Atmospheric models')

class NonAtmosphericModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with anything but the Atmosphere component.
    """
    name : str = 'name of non-atmospheric model should be overwritten in subclass'
    description : str = 'description of non-atmospheric model should be overwritten in subclass'
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        return varident[0]==cls.id

class SpectralModelBase(NonAtmosphericModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the calculated spectrum in the forward model.
    """
    name : str = 'name of spectral model should be overwritten in subclass'
    description : str = 'description of spectral model should be overwritten in subclass'
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
        raise NotImplementedError(f'calculate_from_subspecret should be implemented for all Spectral models')

class InstrumentModelBase(NonAtmosphericModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the instrument parameters of the Spectroscopy component.
    """
    name : str = 'name of instrument model should be overwritten in subclass'
    description : str = 'description of instrument model should be overwritten in subclass'
    ## Abstract methods below this line, subclasses must implement all of these methods ##

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Instrument models')

class ScatteringModelBase(NonAtmosphericModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with a Scatter component, i.e. that sets the scattering properties of an aerosol.
    """
    name : str = 'name of scattering model should be overwritten in subclass'
    description : str = 'description of scattering model should be overwritten in subclass'
    ## Abstract methods below this line, subclasses must implement all of these methods ##

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Scattering models')

class DopplerModelBase(NonAtmosphericModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the doppler shift parameters of the Measurement component.
    """
    name : str = 'name of doppler model should be overwritten in subclass'
    description : str = 'description of doppler model should be overwritten in subclass'
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Doppler models')

class CollisionInducedAbsorptionModelBase(NonAtmosphericModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the CIA (CollisionInducedAbsorption) component.
    """
    name : str = 'name of CIA model should be overwritten in subclass'
    description : str = 'description of CIA model should be overwritten in subclass'
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Collision Induced Absorption models')

class TangentHeightCorrectionModelBase(NonAtmosphericModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the tangent height limb-observation parameter of the Measurement component.
    """
    name : str = 'name of tangent height correction model should be overwritten in subclass'
    description : str = 'description of tangent height correction model should be overwritten in subclass'
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int: # Should return updated value of `ix`
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Doppler models')



class Modelm1(AtmosphericModelBase):
    id : int = -1


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : modelm1()

            DESCRIPTION :

                Function defining the model parameterisation -1 in NEMESIS.
                In this model, the aerosol profiles is modelled as a continuous profile in units
                of particles per gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
                are in particles per gram of atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

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

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([npro,npar,npro])

        if ipar<atm.NVMR:  #Gas VMR
            raise ValueError('error :: Model -1 is just compatible with aerosol populations (Gas VMR given)')
        elif ipar==atm.NVMR: #Temperature
            raise ValueError('error :: Model -1 is just compatible with aerosol populations (Temperature given)')
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            x1 = np.exp(xprof)
            if jtmp<atm.NDUST:
                atm.DUST_UNITS_FLAG[jtmp] = -1
                atm.DUST[:,jtmp] = x1 #* 1000. * rho
            elif jtmp==atm.NDUST:
                raise ValueError('error :: Model -1 is just compatible with aerosol populations')
            elif jtmp==atm.NDUST+1:
                raise ValueError('error :: Model -1 is just compatible with aerosol populations')

        for j in range(npro):
            xmap[0:npro,ipar,j] = x1[:] #* 1000. * rho


        if MakePlot==True:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(7,5))

            for i in range(atm.NDUST):
                ax1.semilogx(atm.DUST[:,i]*rho,atm.H/1000.)
                ax2.semilogx(atm.DUST[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax1.set_xlabel('Aerosol density (particles per cm$^{-3}$)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Aerosol density (particles per gram of atm)')
            ax2.set_ylabel('Altitude (km)')
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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #* continuous cloud, but cloud retrieved as particles/cm3 rather than
        #* particles per gram to decouple it from pressure.
        #********* continuous particles/cm3 profile ************************
        if varident[0] >= 0:
            raise ValueError('error in read_apr_nemesis :: model -1 type is only for use with aerosols')

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

            lx[ix+j] = 1
            x0[ix+j] = np.log(ref[j])
            sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        f1.close()

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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        jtmp = ipar - (forward_model.AtmosphereX.NVMR+1)
        if forward_model.Variables.VARPARAM[ivar,0]\
            and ipar > forward_model.AtmosphereX.NVMR\
            and jtmp < forward_model.AtmosphereX.NDUST: # Fortran true so flip aerosol model

            forward_model.AtmosphereX,xmap1 = Model0.calculate(forward_model.AtmosphereX,ipar,xprof)
        else:
            forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,xprof)

        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def patch_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return NPRO


class Model0(AtmosphericModelBase):
    id : int = 0


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 0 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                in which each element of the state vector corresponds to the atmospheric profile 
                at each altitude level

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

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

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 0 :: Number of levels in atmosphere does not match and profile')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([npro,npar,npro])

        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            x1 = np.exp(xprof)
            vmr = np.zeros([atm.NP,atm.NVMR])
            vmr[:,:] = atm.VMR
            vmr[:,jvmr] = x1
            atm.edit_VMR(vmr)
            for j in range(npro):
                xmap[j,ipar,j] = x1[j]
        elif ipar==atm.NVMR: #Temperature
            x1 = xprof
            atm.edit_T(x1)
            for j in range(npro):
                xmap[j,ipar,j] = 1.
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            x1 = np.exp(xprof)
            if jtmp<atm.NDUST: #Dust in m-3
                dust = np.zeros([atm.NP,atm.NDUST])
                dust[:,:] = atm.DUST
                dust[:,jtmp] = x1
                atm.edit_DUST(dust)
                for j in range(npro):
                    xmap[j,ipar,j] = x1[j]
            elif jtmp==atm.NDUST:
                atm.PARAH2 = x1
                for j in range(npro):
                    xmap[j,ipar,j] = x1[j]
            elif jtmp==atm.NDUST+1:
                atm.FRAC = x1
                for j in range(npro):
                    xmap[j,ipar,j] = x1[j]

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #********* continuous profile ************************
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


        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 0. Continuous profile
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        jtmp = ipar - (forward_model.AtmosphereX.NVMR+1)
        if (forward_model.Variables.VARPARAM[ivar,0] != 0
                and ipar > forward_model.AtmosphereX.NVMR
                and jtmp < forward_model.AtmosphereX.NDUST
            ): # Fortran true so flip aerosol model
            forward_model.AtmosphereX,xmap1 = Modelm1.calculate(forward_model.AtmosphereX,ipar,xprof)
        else:
            forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,xprof)

        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return NPRO


class Model2(AtmosphericModelBase):
    id : int = 2


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @classmethod
    def calculate(cls, atm,ipar,scf,MakePlot=False):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
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

                scf :: Scaling factor

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

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([1,npar,atm.NP])

        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:] = atm.VMR[:,jvmr]
            x1[:] = atm.VMR[:,jvmr] * scf
            atm.VMR[:,jvmr] =  x1
        elif ipar==atm.NVMR: #Temperature
            xref[:] = atm.T[:]
            x1[:] = atm.T[:] * scf
            atm.T[:] = x1 
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST:
                xref[:] = atm.DUST[:,jtmp]
                x1[:] = atm.DUST[:,jtmp] * scf
                atm.DUST[:,jtmp] = x1
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2
                x1[:] = atm.PARAH2 * scf
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC
                x1[:] = atm.FRAC * scf
                atm.FRAC = x1

        xmap[0,ipar,:] = xref[:]

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #**** model 2 - Simple scaling factor of reference profile *******
        #Read in scaling factor

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        x0[ix] = float(tmp[0])
        sx[ix,ix] = (float(tmp[1]))**2.

        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 2. Scaling factor
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix])
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model3(AtmosphericModelBase):
    id : int = 3


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def calculate(cls, atm,ipar,scf,MakePlot=False):

        """
            FUNCTION NAME : model3()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
                In this model, the atmospheric parameters are scaled using a single factor 
                in logscale with respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scf :: Log scaling factor

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

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([1,npar,atm.NP])

        x1 = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            x1[:] = atm.VMR[:,jvmr] * np.exp(scf)
            atm.VMR[:,jvmr] =  x1 
        elif ipar==atm.NVMR: #Temperature
            x1[:] = atm.T[:] * np.exp(scf)
            atm.T[:] = x1 
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST:
                x1[:] = atm.DUST[:,jtmp] * np.exp(scf)
                atm.DUST[:,jtmp] = x1
            elif jtmp==atm.NDUST:
                x1[:] = atm.PARAH2 * np.exp(scf)
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                x1[:] = atm.FRAC * np.exp(scf)
                atm.FRAC = x1

        xmap[0,ipar,:] = x1[:]

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #**** model 3 - Exponential scaling factor of reference profile *******
        #Read in scaling factor

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xfac = float(tmp[0])
        err = float(tmp[1])

        if xfac > 0.0:
            x0[ix] = np.log(xfac)
            lx[ix] = 1
            sx[ix,ix] = ( err/xfac ) **2.
        else:
            raise ValueError('Error in read_apr_nemesis().  xfac must be > 0')

        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 3. Log scaling factor
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix])
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model9(AtmosphericModelBase):
    id : int = 9


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def calculate(cls, atm,ipar,href,fsh,tau,MakePlot=False):

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

                href :: Base height of cloud profile (km)

                fsh :: Fractional scale height (km)

                tau :: Total integrated column density of the cloud (m-2)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model9(atm,ipar,href,fsh,tau)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        from scipy.integrate import simpson
        from archnemesis.Data.gas_data import const

        #Checking that profile is for aerosols
        if(ipar<=atm.NVMR):
            raise ValueError('error in model 9 :: This model is defined for aerosol profiles only')

        if(ipar>atm.NVMR+atm.NDUST):
            raise ValueError('error in model 9 :: This model is defined for aerosol profiles only')


        #Calculating the actual atmospheric scale height in each level
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([3,npar,atm.NP])
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
                xmap[itest-1,ipar,:] = (ND[:]-xprof[:])/dx

        icont = ipar - (atm.NVMR+1)
        atm.DUST[0:atm.NP,icont] = xprof

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** cloud profile held as total optical depth plus
        #******** base height and fractional scale height. Below the knee
        #******** pressure the profile is set to zero - a simple
        #******** cloud in other words!
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        hknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 9. Simple cloud represented by base height, fractional scale height
            #and the total integrated cloud density
        #***************************************************************

        tau = np.exp(forward_model.Variables.XN[ix])    #Integrated dust column-density
        fsh = np.exp(forward_model.Variables.XN[ix+1])  #Fractional scale height
        href = forward_model.Variables.XN[ix+2]         #Base height (km)

        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,href,fsh,tau)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 3


class Model32(AtmosphericModelBase):
    id : int = 32


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'tau' : slice(0,1),
            'frac_scale_height' : slice(1,2),
            'p_ref' : slice(2,3),
        }

    @classmethod
    def calculate(cls, atm,ipar,pref,fsh,tau,MakePlot=False):

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
                pref :: Base pressure of cloud profile (atm)
                fsh :: Fractional scale height (km)
                tau :: Total integrated column density of the cloud (m-2) or cloud optical depth (if kext is normalised)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model32(atm,ipar,pref,fsh,tau)

            MODIFICATION HISTORY : Juan Alday (29/05/2024)

        """
        _lgr.debug(f'{ipar=} {pref=} {tau=}')

        from scipy.integrate import simpson
        from archnemesis.Data.gas_data import const

        #Checking that profile is for aerosols
        if(ipar<=atm.NVMR):
            raise ValueError('error in model 32 :: This model is defined for aerosol profiles only')

        if(ipar>atm.NVMR+atm.NDUST):
            raise ValueError('error in model 32 :: This model is defined for aerosol profiles only')

        icont = ipar - (atm.NVMR+1)   #Index of the aerosol population we are modifying

        #Calculating the actual atmospheric scale height in each level
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((3,npar,atm.NP))
        for itest in range(4):

            xdeep = tau
            xfsh = fsh
            pknee = pref
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
            elif itest==3: #For calculating the gradient wrt pref
                dx = 0.05 * np.log(pref) #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                pknee = np.exp( np.log(pref) + dx )

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
            xfac = 0.5 * (scale[jknee]+scale[jknee+1]) * xfsh  #metres
            ND[jknee+1] = np.exp(-delh/xfac)


            delh = hknee - atm.H[jknee]  #metres
            xf = 1000.  #The cloud below is set to decrease with a scale height of 1 km
            ND[jknee] = np.exp(-delh/xf)

            #Calculating the cloud density above this level
            for j in range(jknee+2,atm.NP):
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfsh
                ND[j] = ND[j-1] * np.exp(-delh/xfac)

            #Calculating the cloud density below this level
            for j in range(0,jknee):
                delh = atm.H[jknee] - atm.H[j]
                xf = 1000.    #The cloud below is set to decrease with a scale height of 1 km
                ND[j] = np.exp(-delh/xf)

            #Now that we have the initial cloud number density (m-3) we can just divide by the mass density to get specific density
            Q[:] = ND[:] / rho[:] / 1.0e3 #particles per gram of atm

            #Now we integrate the optical thickness (calculate column density essentially)
            OD[atm.NP-1] = ND[atm.NP-1] * (scale[atm.NP-1] * xfsh * 1.0e2)  #the factor 1.0e2 is for converting from m to cm
            jfsh = -1
            for j in range(atm.NP-2,-1,-1):
                if j>jknee:
                    delh = atm.H[j+1] - atm.H[j]   #m
                    xfac = scale[j] * xfsh
                    OD[j] = OD[j+1] + (ND[j] - ND[j+1]) * xfac * 1.0e2
                elif j==jknee:
                    delh = atm.H[j+1] - hknee
                    xfac = 0.5 * (scale[j]+scale[j+1])*xfsh
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
                xmap[itest-1,ipar,:] = (Q[:] - xprof[:])/dx

        #Now updating the atmosphere class with the new profile
        atm.DUST[:,icont] = xprof[:]
        _lgr.debug(f'{xprof=}')

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(atm.DUST[:,icont],atm.P/101325.)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max()/101325.,atm.P.min()/101325.)
            ax1.set_xlabel('Cloud density (m$^{-3}$)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()

        atm.DUST_RENORMALISATION[icont] = tau  #Adding flag to ensure that the dust optical depth is tau

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** cloud profile is represented by a value at a 
        #******** variable pressure level and fractional scale height.
        #******** Below the knee pressure the profile is set to drop exponentially.

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 32. Cloud profile is represented by a value at a variable
            #pressure level and fractional scale height.
            #Below the knee pressure the profile is set to drop exponentially.
        #***************************************************************
        tau = np.exp(forward_model.Variables.XN[ix])   #Base pressure (atm)
        fsh = np.exp(forward_model.Variables.XN[ix+1])  #Integrated dust column-density (m-2) or opacity
        pref = np.exp(forward_model.Variables.XN[ix+2])  #Fractional scale height
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,pref,fsh,tau)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 3


class Model45(AtmosphericModelBase):
    id : int = 45


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'deep_vmr' : slice(0,1),
            'humidity' : slice(1,2),
            'strato_vmr' : slice(2,3),
        }

    @classmethod
    def calculate(cls, atm, ipar, tropo, humid, strato, MakePlot=True):

        """
            FUNCTION NAME : model45()

            DESCRIPTION :

                Irwin CH4 model. Variable deep tropospheric and stratospheric abundances,
                along with tropospheric humidity.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                tropo :: Deep methane VMR

                humid :: Relative methane humidity in the troposphere

                strato :: Stratospheric methane VMR

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model45(atm, ipar, tropo, humid, strato)

            MODIFICATION HISTORY : Joe Penn (09/10/2024)

        """

        _lgr.debug(f'{ipar=} {tropo=} {humid=} {strato=}')

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
        atm.VMR[:, ipar] = xnew


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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** Irwin CH4 model. Represented by tropospheric and stratospheric methane 
        #******** abundances, along with methane humidity. 
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        tropo = tmp[0]
        etropo = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        humid = tmp[0]
        ehumid = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 45. Irwin CH4 model. Variable deep tropospheric and stratospheric abundances,
            #along with tropospheric humidity.
        #***************************************************************
        tropo = np.exp(forward_model.Variables.XN[ix])   # Deep tropospheric abundance
        humid = np.exp(forward_model.Variables.XN[ix+1])  # Humidity
        strato = np.exp(forward_model.Variables.XN[ix+2])  # Stratospheric abundance
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX, ipar, tropo, humid, strato)
        xmap[ix] = xmap1

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 3


class Model47(AtmosphericModelBase):
    id : int = 47


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'tau' : slice(0,1),
            'p_ref' : slice(1,2),
            'fwhm' : slice(2,3),
        }
    
    @classmethod
    def calculate(cls, atm, ipar, tau, pref, fwhm, MakePlot=False):

        """
            FUNCTION NAME : model47()

            DESCRIPTION :

                Profile is represented by a Gaussian with a specified optical thickness centred
                at a variable pressure level plus a variable FWHM (log press).

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                tau :: Integrated optical thickness.

                pref :: Mean pressure (atm) of the cloud.

                fwhm :: FWHM of the log-Gaussian.

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model47(atm, ipar, tau, pref, fwhm)

            MODIFICATION HISTORY : Joe Penn (08/10/2024)

        """
        _lgr.debug(f'{ipar=} {tau=} {pref=} {fwhm=}')

        from archnemesis.Data.gas_data import const

        # First, check that the profile is for aerosols
        if ipar <= atm.NVMR:
            raise ValueError('Error in model47: This model is defined for aerosol profiles only')

        if ipar > atm.NVMR + atm.NDUST:
            raise ValueError('Error in model47: This model is defined for aerosol profiles only')

        icont = ipar - (atm.NVMR + 1)   # Index of the aerosol population we are modifying

        # Calculate atmospheric properties
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        # Convert pressures to atm
        P = atm.P / 101325.0  # Pressure in atm

        # Compute Y0 = np.log(pref)
        Y0 = np.log(pref)

        # Compute XWID, the standard deviation of the Gaussian
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

        # Now compute the Jacobian matrix xmap
        npar = atm.NVMR + 2 + atm.NDUST  # Assuming this is the total number of parameters
        xmap = np.zeros((3, npar, atm.NP))

        for j in range(atm.NP):
            Y = np.log(P[j])

            # First parameter derivative: xmap[0, ipar, j] = X1[j] / tau
            xmap[0, ipar, j] = X1[j] / tau  # XDEEP is tau

            # Derivative of X1[j] with respect to Y0 (pref)
            xmap[1, ipar, j] = 2.0 * (Y - Y0) / XWID ** 2 * X1[j]

            # Derivative of X1[j] with respect to XWID (fwhm)
            xmap[2, ipar, j] = (2.0 * ((Y - Y0) ** 2) / XWID ** 3 - 1.0 / XWID) * X1[j]

        # Update the atmosphere class with the new profile
        atm.DUST[:, icont] = X1[:]
        _lgr.debug(f'{X1=}')

        if MakePlot:
            fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))
            ax1.plot(atm.DUST[:, icont], atm.P / 101325.0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max() / 101325.0, atm.P.min() / 101325.0)
            ax1.set_xlabel('Cloud density (particles/kg)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()
            plt.show()

        atm.DUST_RENORMALISATION[icont] = tau   #Adding flag to ensure that the dust optical depth is tau

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** cloud profile is represented by a peak optical depth at a 
        #******** variable pressure level and a Gaussian profile with FWHM (in log pressure)

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 47. Profile is represented by a Gaussian with a specified optical thickness centred
            #at a variable pressure level plus a variable FWHM (log press) in height.
        #***************************************************************
        tau = np.exp(forward_model.Variables.XN[ix])   #Integrated dust column-density (m-2) or opacity
        pref = np.exp(forward_model.Variables.XN[ix+1])  #Base pressure (atm)
        fwhm = np.exp(forward_model.Variables.XN[ix+2])  #FWHM
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX, ipar, tau, pref, fwhm)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]



        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 3


class Model49(AtmosphericModelBase):
    id : int = 49


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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



        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 50. Continuous profile in linear scale
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return NPRO


class Model50(AtmosphericModelBase):
    id : int = 50


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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


        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 50. Continuous profile of scaling factors
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]



        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return NPRO


class Model51(AtmosphericModelBase):
    id : int = 51


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #********* multiple of different profile ************************
        prof = np.fromfile(f,sep=' ',count=2,dtype='int')
        profgas = prof[0]
        profiso = prof[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        scale = tmp[0]
        escale = tmp[1]

        varparam[1] = profgas
        varparam[2] = profiso
        x0[ix] = np.log(scale)
        lx[ix] = 1
        err = escale/scale
        sx[ix,ix] = err**2.

        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 51. Scaling of a reference profile
        #***************************************************************                
        scale = np.exp(forward_model.Variables.XN[ix])
        scale_gas, scale_iso = forward_model.Variables.VARPARAM[ivar,1:3]
        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,scale,scale_gas,scale_iso)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]



        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model110(AtmosphericModelBase):
    id : int = 110


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** model for Venus cloud following Haus et al. (2016) with altitude offset

        if varident[0]>0:
            raise ValueError('error in read_apr model 110 :: VARIDENT[0] must be negative to be associated with the aerosols')

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #z_offset
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 110. Venus cloud model from Haus et al. (2016) with altitude offset
        #************************************************************************************  

        offset = forward_model.Variables.XN[ix]   #altitude offset in km
        idust0 = np.abs(forward_model.Variables.VARIDENT[ivar,0])-1  #Index of the first cloud mode                
        forward_model.AtmosphereX = cls.calculate(forward_model.AtmosphereX,idust0,offset)

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model111(AtmosphericModelBase):
    id : int = 111


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** model for Venus cloud and SO2 vmr profile with altitude offset

        if varident[0]>0:
            raise ValueError('error in read_apr model 111 :: VARIDENT[0] must be negative to be associated with the aerosols')

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #z_offset
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #SO2_deep
        so2_deep = float(tmp[0])
        so2_deep_err = float(tmp[1])
        x0[ix] = np.log(so2_deep)
        sx[ix,ix] = (so2_deep_err/so2_deep)**2.
        lx[ix] = 1
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #SO2_top
        so2_top = float(tmp[0])
        so2_top_err = float(tmp[1])
        x0[ix] = np.log(so2_top)
        sx[ix,ix] = (so2_top_err/so2_top)**2.
        lx[ix] = 1
        inum[ix] = 1
        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 110. Venus cloud model and SO2 vmr profile with altitude offset
        #************************************************************************************  

        offset = forward_model.Variables.XN[ix]   #altitude offset in km
        so2_deep = np.exp(forward_model.Variables.XN[ix+1])   #SO2 vmr below the cloud
        so2_top = np.exp(forward_model.Variables.XN[ix+2])   #SO2 vmr above the cloud

        idust0 = np.abs(forward_model.Variables.VARIDENT[ivar,0])-1  #Index of the first cloud mode                
        forward_model.AtmosphereX = cls.calculate(forward_model.AtmosphereX,idust0,so2_deep,so2_top,offset)

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 3


class Model202(AtmosphericModelBase):
    id : int = 202


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
                print('Required ID :: ',varid1,varid2)
                print('Avaiable ID and ISO :: ',telluric.Atmosphere.ID,telluric.Atmosphere.ISO)
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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #********* simple scaling of telluric atmospheric profile ************************
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        x0[ix] = float(tmp[0])
        sx[ix,ix] = (float(tmp[1]))**2.
        inum[ix] = 1

        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 202. Scaling factor of telluric atmospheric profile
        #***************************************************************

        scafac = forward_model.Variables.XN[ix]
        varid1 = forward_model.Variables.VARIDENT[ivar,0] ; varid2 = forward_model.Variables.VARIDENT[ivar,1]
        if forward_model.TelluricX is not None:
            forward_model.TelluricX = cls.calculate(forward_model.TelluricX,varid1,varid2,scafac)

        ix = ix + forward_model.Variables.NXVAR[ivar]



        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model228(InstrumentModelBase):
    id : int = 228


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** model for retrieving the ILS and Wavelength calibration in ACS MIR solar occultation observations

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #V0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C1
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C2
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P0 - Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P1 - FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P2 - Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P3 - Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
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

        forward_model.MeasurementX,forward_model.SpectroscopyX = cls.calculate(forward_model.MeasurementX,forward_model.SpectroscopyX,V0,C0,C1,C2,P0,P1,P2,P3)

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 8


class Model229(InstrumentModelBase):
    id : int = 229


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** model for retrieving the ILS in ACS MIR solar occultation observations

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at wavenumber in the middle
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at highest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1


        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
        #***************************************************************

        par1 = forward_model.Variables.XN[ix]
        par2 = forward_model.Variables.XN[ix+1]
        par3 = forward_model.Variables.XN[ix+2]
        par4 = forward_model.Variables.XN[ix+3]
        par5 = forward_model.Variables.XN[ix+4]
        par6 = forward_model.Variables.XN[ix+5]
        par7 = forward_model.Variables.XN[ix+6]

        forward_model.MeasurementX = cls.calculate(forward_model.MeasurementX,par1,par2,par3,par4,par5,par6,par7)

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 7


class Model230(InstrumentModelBase):
    id : int = 230


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
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

        forward_model.MeasurementX = cls.calculate(forward_model.MeasurementX,nwindows,liml,limh,par1)

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 7*int(varident[0])


class Model444(ScatteringModelBase):
    id : int = 444


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'particle_size_distribution_params' : slice(0,2),
            'imaginary_ref_idx' : slice(2,None),
        }

    @classmethod
    def calculate(cls, Scatter,idust,iscat,xprof,haze_params):
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
            _lgr.debug(f'haze_params[{item},{idust}] = {type(haze_params[item,idust])} {haze_params[item,idust]}')

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

        Scatter.WAVER = haze_params['WAVE',idust]
        Scatter.REFIND_IM = np.exp(xprof[2:])
        reference_nreal = haze_params['NREAL',idust]
        reference_wave = haze_params['WAVE_REF',idust]
        normalising_wave = haze_params['WAVE_NORM',idust]
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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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

        idust = varident[1]-1

        variables.HAZE_PARAMS['NX',idust] = 2+len(haze_waves)
        variables.HAZE_PARAMS['WAVE',idust] = haze_waves
        variables.HAZE_PARAMS['NREAL',idust] = float(nreal_ref)
        variables.HAZE_PARAMS['WAVE_REF',idust] = float(vref)
        variables.HAZE_PARAMS['WAVE_NORM',idust] = float(v_od_norm)

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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        idust = int(forward_model.Variables.VARIDENT[ivar,1]) - 1
        iscat = 1 # Should add an option for this
        xprof = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.ScatterX = cls.calculate(forward_model.ScatterX,idust,iscat,xprof,forward_model.Variables.HAZE_PARAMS)
        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        try:
            idust = varident[1]-1
            return variables.HAZE_PARAMS['NX',idust]
        except: # happens when reading .mre
            return varparam[0]


class Model446(ScatteringModelBase):
    id : int = 446


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
        f = h5py.File(lookupfile,'r')

        NWAVE = np.int32(f.get('NWAVE'))
        NSIZE = np.int32(f.get('NSIZE'))

        WAVE = np.array(f.get('WAVE'))
        REFF = np.array(f.get('REFF'))

        KEXT = np.array(f.get('KEXT'))      #(NWAVE,NSIZE)
        SGLALB = np.array(f.get('SGLALB'))  #(NWAVE,NSIZE)

        f.close()

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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
        varfile[i] = fnamex

        #Reading the particle size and its a priori error
        s = f.readline().split()
        lx[ix] = 0
        inum[ix] = 1
        x0[ix] = float(s[0])
        sx[ix,ix] = (float(s[1]))**2.

        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 446. model for retrieving the particle size distribution based on the data in a look-up table
        #***************************************************************

        #This model fits the particle size distribution based on the optical properties at different sizes
        #tabulated in a pre-computed look-up table. What this model does is to interpolate the optical 
        #properties based on those tabulated.

        idust0 = int(forward_model.Variables.VARPARAM[ivar,0])
        wavenorm = int(forward_model.Variables.VARPARAM[ivar,1])
        xwave = forward_model.Variables.VARPARAM[ivar,2]
        lookupfile = forward_model.Variables.VARFILE[ivar]
        rsize = forward_model.Variables.XN[ix]

        forward_model.ScatterX = cls.calculate(forward_model.ScatterX,idust0,wavenorm,xwave,rsize,lookupfile,MakePlot=False)

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]



        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model447(DopplerModelBase):
    id : int = 447


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        _lgr.warning(f'Model id = {cls.id} is not implemented yet, so it will never be chosen as a valid model')
        return False

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        raise NotImplementedError
        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model500(CollisionInducedAbsorptionModelBase):
    id : int = 500


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:

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


        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:

        icia = forward_model.Variables.VARIDENT[ivar,1]

        if forward_model.Measurement.ISPACE == WaveUnit.Wavelength_um:
            vlo = 1e4/(forward_model.SpectroscopyX.WAVE.max())
            vhi = 1e4/(forward_model.SpectroscopyX.WAVE.min())
        else:
            vlo = forward_model.SpectroscopyX.WAVE.min()
            vhi = forward_model.SpectroscopyX.WAVE.max()

        nbasis = forward_model.Variables.VARPARAM[ivar,0]
        amplitudes = np.exp(forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]])*1e-40

        new_k_cia, xmap1 = cls.calculate(forward_model.CIA.K_CIA.copy(), forward_model.CIA.WAVEN, icia, vlo, vhi, nbasis, amplitudes)

        forward_model.CIA.K_CIA = new_k_cia
        forward_model.CIAX.K_CIA = new_k_cia


        ix = ix + forward_model.Variables.NXVAR[ivar]

        return ix


    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return int(varparam[0])


class Model777(TangentHeightCorrectionModelBase):
    id : int = 777


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        return varident[0]==777

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** tangent height correction
        s = f.readline().split()
        hcorr = float(s[0])
        herr = float(s[1])

        x0[ix] = hcorr
        sx[ix,ix] = herr**2.
        inum[ix] = 1

        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 777. Retrieval of tangent height corrections
        #***************************************************************

        hcorr = forward_model.Variables.XN[ix]

        forward_model.MeasurementX = cls.calculate(forward_model.MeasurementX,hcorr)

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1


class Model887(ScatteringModelBase):
    id : int = 887


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        _lgr.warning(f"Model with id {cls.id} is not implemented")
        return False

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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

        jxsc = ix

        ix = ix + nwv

        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        raise NotImplementedError
        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return int(varparam[0])


class Model1002(AtmosphericModelBase):
    id : int = 1002


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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

            #Including lats and lons in varparam
            #varparam[iparj]  = lats[iloc]
            #iparj = iparj + 1
            #varparam[iparj] = lons[iloc]
            #iparj = iparj + 1

            #if iparj==mparam:
            #    raise ValueError('error in reading .apr :: Need to increase the mparam')

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


        return ix

    @classmethod
    def calculate_from_subprofretg(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> int:
        #Model 1002. Scaling factors at multiple locations
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = cls.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]],MakePlot=False)
        #This calculation takes a long time for big arrays
        #xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP,0:forward_model.AtmosphereX.NLOCATIONS] = xmap1[:,:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1 * nlocations   


class Model231(SpectralModelBase):
    id : int = 231
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
            igeom_slices : tuple[slice,...],
            meas : "Measurement_0",
            NGEOM : int, 
            NDEGREE : int, 
            COEFF : np.ndarray[['NDEGREE+1','NGEOM'],float]
        ) -> tuple[np.ndarray[['NCONV','NGEOM'],float], np.ndarray[['NCONV','NGEOM','NX'],float]]:
        
        for i in range(meas.NGEOM):
            T = COEFF[i]

            WAVE0 = meas.VCONV[0,i]
            spec[:] = np.array(SPECMOD[0:meas.NCONV[i],i])

            #Changing the state vector based on this parameterisation
            POL = np.zeros(meas.NCONV[i])
            for j in range(NDEGREE+1):
                POL[:] = POL[:] + T[j]*(meas.VCONV[0:meas.NCONV[i],i]-WAVE0)**j

            SPECMOD[0:meas.NCONV[i],i] *= POL[:]

            #Changing the rest of the gradients based on the impact of this parameterisation
            dSPECMOD[0:meas.NCONV[i],i,:] *= POL[:,None]

            #Defining the analytical gradients for this parameterisation
            dspecmod_part = dSPECMOD[0:meas.NCONV[i],i,igeom_slices[i]]
            for j in range(NDEGREE+1):
                dspecmod_part[:,j] = spec * (meas.VCONV[0:meas.NCONV[i],i]-WAVE0)**j

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='int')
        nlevel = int(tmp[0])
        ndegree = int(tmp[1])
        varparam[0] = nlevel
        varparam[1] = ndegree
        for ilevel in range(nlevel):
            tmp = f1.readline().split()
            for ic in range(ndegree+1):
                r0 = float(tmp[2*ic])
                err0 = float(tmp[2*ic+1])
                x0[ix] = r0
                sx[ix,ix] = (err0)**2.
                inum[ix] = 0
                ix = ix + 1
        return ix
        
    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
        NGEOM = int(forward_model.Variables.VARPARAM[ivar,0])
        NDEGREE = int(forward_model.Variables.VARPARAM[ivar,1])
        COEFF = np.array(forward_model.Variables.XN[ix : ix+(NDEGREE+1)*forward_model.Measurement.NGEOM]).reshape((NDEGREE+1,forward_model.Measurement.NGEOM))
        
        igeom_slices = tuple(slice(ix+(NDEGREE+1)*igeom, ix+(NDEGREE+1)*(igeom+1)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))
        
        SPECMOD[...], dSPECMOD[...] = cls.calculate(SPECMOD, dSPECMOD, igeom_slices, forward_model.Measurement, NGEOM, NDEGREE, COEFF)
        
        return ix + COEFF.size

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return varparam[0]*(varparam[1]+1)

# [JD] This uses forward_model.SpectroscopyX, whereas Model231 uses forward_model.Measurement, is this correct?
class Model2310(SpectralModelBase):
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
                spec[:] = np.array(SPECMOD[ivin,i])

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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
        return ix
        
    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
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

        SPECMOD[...], dSPECMOD[...] = cls.calculate(
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

        return ix + (COEFF.size * NWINDOWS)

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return varparam[0]*(varparam[1]+1)*varparam[2]


class Model232(SpectralModelBase):
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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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
        varparam[i,0] = nlevel
        varparam[i,1] = wavenorm
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
        return ix
        
    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
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
                
                SPECMOD[:,i], dSPECMOD[:,i] = cls.calculate(
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
            
            _lgr.warning(f'It looks like there is no calculation for NGEOM=1 for model id = {cls.id}')

        return ix + 2*NGEOM

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident : np.ndarray[[3],int],
            varparam : np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 2*varparam[0]


class Model233(SpectralModelBase):
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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
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
        varparam[i,0] = nlevel
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
        return ix
        
    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
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
                
                SPECMOD[:,i], dSPECMOD[:,i] = cls.calculate(
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

            SPECMOD[:], dSPECMOD[:] = cls.calculate(
                SPECMOD[:], 
                dSPECMOD[:],
                slice(ix,ix+3),
                forward_model.SpectroscopyX,
                A0,
                A1,
                A2
            )

        return ix + 3*NGEOM

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident : np.ndarray[[3],int],
            varparam : np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 3*varparam[0]


class Model667(SpectralModelBase):
    id : int = 667


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

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
            varfile : list[str],
            npro : int,
            nlocations : int,
            sxminfac : float,
        ) -> int:
        #******** dilution factor to account for thermal gradients thorughout exoplanet
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xfac = float(tmp[0])
        xfacerr = float(tmp[1])
        x0[ix] = xfac
        inum[ix] = 0 
        sx[ix,ix] = xfacerr**2.
        ix = ix + 1

        return ix

    @classmethod
    def calculate_from_subspecret(
            cls,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> int:
        #Model 667. Spectrum scaled by dilution factor to account for thermal gradients in planets
        #**********************************************************************************************

        xfactor = forward_model.Variables.XN[ix]
        spec = np.zeros(forward_model.SpectroscopyX.NWAVE)
        spec[:] = SPECMOD
        SPECMOD = cls.calculate(SPECMOD,xfactor)
        dSPECMOD = dSPECMOD * xfactor
        dSPECMOD[:,ix] = spec[:]
        ix = ix + 1


        return ix

    @classmethod
    def get_nxvar(
            cls,
            variables : "Variables_0",
            varident: np.ndarray[[3],int],
            varparam: np.ndarray[[3],int],
            NPRO : int,
            nlocations : int,
        ) -> int:
        return 1
