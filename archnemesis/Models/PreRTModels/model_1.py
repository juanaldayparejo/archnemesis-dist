

from typing import TYPE_CHECKING, Self, IO

import numpy as np


from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

import archnemesis.Data.constants as const
from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
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




class Model1(PreRTModelBase):
    """
        Variable deep abundance, fixed knee pressure and variable fractional scale height. 
    """
    id : int = 1

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector

            pknee : float,
            #   Knee pressure (atm)
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
            ModelParameter('ABU_DEEP', slice(0,1), 'Deep abundance'),
            ModelParameter('FSH', slice(1,2), 'Fractional scale height'),
        )

        self.pknee = pknee
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

            PKNEE : float,
            #   Knee pressure (atm)

            ABU_DEEP : float, 
            #   Deep abundance 
            
            FSH : float,
            #   Fractional scale height
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model1()

            DESCRIPTION :

                Variable deep abundance, fixed knee pressure and variable fractional scale height.    

            INPUTS :

                atm :: Python class defining the atmosphere
                ABU_DEEP :: Deep abundance
                FSH :: Fractional scale height
                PKNEE :: Knee pressure (atm)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(mparam,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model62(atm,p1,p2,p3,t0,alpha1,alpha2)

            MODIFICATION HISTORY : Juan Alday (18/12/2025)

        """        
        
        xfac = (1.0 - FSH) / FSH

        #d(xfac)/d(FSH)
        dxfac = -1.0 / FSH**2.

        #Finding the knee altitude 
        pknee_pa = PKNEE * 101325.   #Calculating knee pressure in Pa
        isort = np.argsort(atm.P)
        p_sorted = atm.P[isort]
        h_sorted = atm.H[isort]
        hknee = np.interp(pknee_pa, p_sorted, h_sorted) #metres

        #Calculating the scale height
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #Creating the new vertical profile and the functional derivatives
        xprof = np.zeros(atm.NP)
        xmap = np.zeros((2,atm.NP))   #Matrix with functional derivates of the parameters wrt the profiles
        jfsh = 0
        for j in range(atm.NP):

            #Above knee pressure
            if atm.P[j]>=pknee_pa:

                xprof[j] = ABU_DEEP
                xmap[0,j] = 1.0

            else:

                if jfsh == 0:
                    delh = atm.H[j] - hknee
                else:
                    delh = atm.H[j] - atm.H[j - 1]

                xprof[j]=xprof[j-1]*np.exp(-delh*xfac/scale[j])

                #Functional derivative of ABU_DEEP
                xmap[0,j] = xmap[0,j-1] * np.exp(-delh * xfac / scale[j])

                #Functional derivative of FSH
                xmap[1,j] = (
                    (-delh / scale[j])
                    * dxfac
                    * xprof[j-1]
                    * np.exp(-delh * xfac / scale[j])
                    + xmap[1,j-1]
                    * np.exp(-delh * xfac / scale[j])
                )

                jfsh = 1

                if xprof[j] < 1.0e-36:
                    xprof[j] = 1.0e-36

        #Updating atmosphere class
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            tmp = np.array(atm.VMR)
            tmp[:,atm_profile_idx] = xprof
            atm.edit_VMR(tmp)
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            atm.edit_T(xprof)        
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            tmp = np.array(atm.DUST)
            tmp[:,atm_profile_idx] = xprof
            atm.edit_DUST(tmp)
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            atm.PARAH2(xprof)
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            atm.FRAC(xprof)
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')
        
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
        #******** profile held as deep amount, fsh and knee pressure

        #Reading pknee
        s = f.readline().split()
        pknee = float(s[0])
        varparam[0] = pknee

        #Reading deep abundance
        s = f.readline().split()
        xdeep = float(s[0])
        edeep = float(s[1])

        if varident[0]==0: #Temperature
            x0[ix]=xdeep
            err=edeep
        else: #VMR, para-H2 or cloud
            if xdeep>0.0:
                x0[ix]=np.log(xdeep)
                lx[ix]=1
                inum[ix]=0  #We take the derivatives analytically
            else:
                raise ValueError("error in read_apr (Model 1) :: xdeep must be >0 if it applies to any parameter but temperature")
            err=edeep/xdeep
        sx[ix,ix]=err**2.
        ix += 1
        
        #Reading fractional scale height
        s = f.readline().split()
        fsh = float(s[0])
        efsh = float(s[1])
        if fsh>0.0:
            x0[ix] = np.log(fsh)
            lx[ix] = 1
            inum[ix] = 1 #We take the derivatives analytically
            sx[ix,ix] = (efsh/fsh)**2.
        else:
            raise ValueError('error in read_apr (Model 1) :: fsh must be > 0')
        ix += 1
        
        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, pknee)

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
        ix_0 = ix
        #******** profile held as deep amount, fsh and knee pressure  
        ix = ix + 2

        pknee = varparam[0]

        return cls(ix_0, ix-ix_0, pknee)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 1. profile held as deep amount, fsh and knee pressure  
        #***************************************************************

        atm = forward_model.AtmosphereX

        #Finding the atmospheric profile type and the index of the profile we want to modify
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        #Getting the parameters from the state vector
        xn_params = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        xn_params_1d = np.concatenate([np.atleast_1d(v) for v in xn_params])

        xdeep = xn_params[0] ; fsh = xn_params[1]
        
        #Modifying Atmosphere based on model parameters
        atm, xmap1 = self.calculate(
            atm, 
            atm_profile_type,
            atm_profile_idx,
            self.pknee, 
            xdeep, 
            fsh
        )
        
        #Calculating derivatives of the atmospheric profile with respect to the state vector
        for i in range(self.n_state_vector_entries):
            if forward_model.Variables.LX[self.state_vector_start+i] == 1: #If carried in log space, then the derivative is multiplied by the value of the parameter
                xmap1[i,:] = xmap1[i,:] * xn_params_1d[i]

        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1

        return