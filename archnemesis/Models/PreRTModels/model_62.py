
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter


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

class Model62(PreRTModelBase):
    """
        Temperature profile for exoplanets following the parameterisation of Madhusudhan & Seager (2009)   
    """
    id : int = 62

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
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
            ModelParameter('P1', slice(0,1), 'Pressure at interface of Layer 1-2'),
            ModelParameter('P2', slice(1,2), 'Pressure at middle of Layer 2'),
            ModelParameter('P3', slice(2,3), 'Pressure at interface of Layer 2-3'),
            ModelParameter('T0', slice(3,4), 'Top-of-atmosphere temperature (K)'),
            ModelParameter('alpha1', slice(4,5), 'Exponential parameter alpha in Layer 1'),
            ModelParameter('alpha2', slice(4,5), 'Exponential parameter alpha in Layer 2'),
        )
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            P1 : float, 
            #   Pressure at interface of Layer 1-2 (atm)
            
            P2 : float,
            #   Pressure at middle of Layer 2 (atm)
            
            P3 : float,
            #   Pressure at interface of Layer 2-3 (atm)
            
            T0 : float, 
            #   Top-of-atmosphere temperature (K)
            
            alpha1 : float, 
            #   Exponential parameter alpha in Layer 1
            
            alpha2 : float, 
            #   Exponential parameter alpha in Layer 2
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model62()

            DESCRIPTION :

                Temperature profile for exoplanets following the parameterisation of Madhusudhan & Seager (2009)   

            INPUTS :

                atm :: Python class defining the atmosphere
                P1 :: Pressure at interface of Layer 1-2 (atm)
                P2 :: Pressure at middle of Layer 2 (atm)
                P3 :: Pressure at interface of Layer 2-3 (atm)
                T0 :: Top-of-atmosphere temperature (K)
                alpha1 :: Exponential parameter alpha in Layer 1
                alpha2 :: Exponential parameter alpha in Layer 2

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
        
        TP = np.zeros(atm.NP)
        xmap = np.zeros((6,atm.NP))
        
        P = np.array(atm.P)  #Pa
        P0 = np.min(P)       #Pa
        P1 = P1 * 101325. ; P2 = P2 * 101325. ; P3 = P3 * 101325.   #Converting from atm to Pa
        
        T2 = ((1 / alpha1) * np.log10(P1 / P0)) ** 2 - ((1 / alpha2) * np.log10(P1 / P2)) ** 2 + T0
        T3 = ((1 / alpha2) * np.log10(P3 / P2)) ** 2 + T2
        
        for i in range(atm.NP):
            
            if P[i]>= P3:
                TP[i] = T3
            elif P[i] >= P1:
                TP[i] = ((1 / alpha2) * np.log10(P[i] / P2)) ** 2 + T2
            else:
                TP[i] = ((1 / alpha1) * np.log10(P[i] / P0)) ** 2 + T0

        # Numerical guard (prevents pathological values from upsetting hydrostatics)
        TP = np.clip(TP, 50.0, 6000.0)
        
        atm.edit_T(TP)
            
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
        #******** Temperature profile for exoplanets following the parameterisation of Madhusudhan & Seager (2009) 

        #Reading p1
        s = f.readline().split()
        p1 = float(s[0])
        p1_err = float(s[1])
        if p1>0.0:
            x0[ix] = np.log(p1)
            lx[ix] = 1
            inum[ix] = 1
            sx[ix,ix] = (p1_err/p1)**2.
        else:
            raise ValueError('error in read_apr :: p1 must be > 0')
        ix += 1
        
        #Reading p2
        s = f.readline().split()
        p2 = float(s[0])
        p2_err = float(s[1])
        if p2>0.0:
            x0[ix] = np.log(p2)
            lx[ix] = 1
            inum[ix] = 1
            sx[ix,ix] = (p2_err/p2)**2.
        else:
            raise ValueError('error in read_apr :: p2 must be > 0')
        ix += 1
        
        #Reading p3
        s = f.readline().split()
        p3 = float(s[0])
        p3_err = float(s[1])
        if p3>0.0:
            x0[ix] = np.log(p3)
            lx[ix] = 1
            inum[ix] = 1
            sx[ix,ix] = (p3_err/p3)**2.
        else:
            raise ValueError('error in read_apr :: p3 must be > 0')
        ix += 1

        #Reading t0
        s = f.readline().split()
        t0 = float(s[0])
        t0_err = float(s[1])
        if t0>0.0:
            x0[ix] = t0
            lx[ix] = 0
            inum[ix] = 1
            sx[ix,ix] = (t0_err)**2.
        else:
            raise ValueError('error in read_apr :: t0 must be > 0')
        ix += 1
        
        #Reading alpha1
        s = f.readline().split()
        alpha1 = float(s[0])
        alpha1_err = float(s[1])
        if alpha1>0.0:
            x0[ix] = np.log(alpha1)
            lx[ix] = 1
            inum[ix] = 1
            sx[ix,ix] = (alpha1_err/alpha1)**2.
        else:
            raise ValueError('error in read_apr :: alpha1 must be > 0')
        ix += 1
        
        #Reading alpha2
        s = f.readline().split()
        alpha2 = float(s[0])
        alpha2_err = float(s[1])
        if alpha2>0.0:
            x0[ix] = np.log(alpha2)
            lx[ix] = 1
            inum[ix] = 1
            sx[ix,ix] = (alpha2_err/alpha2)**2.
        else:
            raise ValueError('error in read_apr :: alpha2 must be > 0')
        ix += 1

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0)

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
        #******** Temperature profile for exoplanets following the parameterisation of Madhusudhan & Seager (2009)  
        ix = ix + 6

        return cls(ix_0, ix-ix_0)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 43. Temperature profile from double grey analytic formulation (Parmentier and Guillot (2014) and Line et al. (2013))  
        #***************************************************************

        atm = forward_model.AtmosphereX
        
        xn_params = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        p1 = xn_params[0] ; p2 = xn_params[1] ; p3 = xn_params[2] ; t0 = xn_params[3] ; alpha1 = xn_params[4] ; alpha2 = xn_params[5]
        
        atm = self.calculate(
            atm, 
            p1, p2, p3, t0, alpha1, alpha2
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:forward_model.AtmosphereX.NP] = 0.0
        
        return



