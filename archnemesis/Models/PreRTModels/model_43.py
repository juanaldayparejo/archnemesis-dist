
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

class Model43(PreRTModelBase):
    """
        Temperature profile from double grey analytic formulation (Parmentier and Guillot (2014) and Line et al. (2013))  
    """
    id : int = 43


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            T_star : float,
            #   Temperature of the host star (K)
            
            R_star : float,
            #   Star radius (km)
            
            sdist : float,
            #   Planet-star distance (km)
            
            T_int : float,
            #Internal temperature of the planet (K)
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, T_star, R_star, sdist, T_int)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model.
        # NOTE: It is best to define these in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        self.parameters = (
            ModelParameter('alpha', slice(0,1), 'Parameter alpha. Weighting between two streams'),
            ModelParameter('beta', slice(1,2), 'Parameter beta. Albedo/emissivity weight.'),
            ModelParameter('k_ir', slice(2,3), 'Parameter k_ir. Thermal IR opacity parameter. (units: cm2/g)'),
            ModelParameter('gammav1', slice(3,4), 'Ratio of visible stream 1 opacity to thermal opacity'),
            ModelParameter('gammav2', slice(4,5), 'Ratio of visible stream 2 opacity to thermal opacity'),
        )
        self.T_star = T_star
        self.R_star = R_star
        self.sdist = sdist
        self.T_int = T_int
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            alpha : float, 
            #   Parameter alpha. Weighting between two streams
            
            beta : float,
            #   Parameter beta. Albedo/emissivity weight
            
            k_ir : float,
            #   Parameter k_ir. Thermal IR opacity parameter. (units: cm2/g)
            
            gammav1 : float, 
            #   Ratio of visible stream 1 opacity to thermal opacity
            
            gammav2 : float, 
            #   Ratio of visible stream 2 opacity to thermal opacity
            
            T_star : float, 
            #   Star temperature (K)
            
            R_star : float, 
            #   Star radius (km)
            
            sdist : float,
            #   Planet-star distance (km)
            
            T_int : float,
            #   Internal temperature (K),
            
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model43()

            DESCRIPTION :

                Temperature profile from double grey analytic formulation 
                (Parmentier and Guillot (2014) and Line et al. (2013))  

            INPUTS :

                atm :: Python class defining the atmosphere
                alpha :: Parameter alpha. Weighting between two streams
                beta :: Parameter beta. Albedo/emissivity weight
                k_ir :: Parameter k_ir. Thermal IR opacity parameter. (units: cm2/g)
                gammav1 :: Ratio of visible stream 1 opacity to thermal opacity
                gammav2 :: Ratio of visible stream 2 opacity to thermal opacity
                tstar :: Star temperature (K)
                rstar :: Star radius (km)
                sdist :: Planet-star distance (km)
                tint :: Internal temperature (K),

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(mparam,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model43(atm,alpha,beta,k_ir,gammav1,gammav2,tstar,rstar,sdist,tint)

            MODIFICATION HISTORY : Juan Alday (18/12/2025)

        """        
        
        def e2(xin):
            """
            Routine for calculating the E2 exponential function and gradient
            
            Author: Pat Irwin (26/01/2021) - Original
                    Juan Alday (18/12/2025) - Translation into python
            """
            
            n = 100
            yl = np.array([0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,-2.58860e-08,
                        -2.58860e-08, -2.58860e-08, -2.58860e-08, -5.17719e-08,-5.17719e-08, -7.76579e-08, -7.76579e-08, -1.03544e-07,
                        -1.55316e-07, -2.07088e-07, -2.58860e-07, -3.36518e-07,-4.40062e-07, -5.43606e-07, -7.24808e-07, -9.31896e-07,
                        -1.21664e-06, -1.55316e-06, -2.01911e-06, -2.61449e-06,-3.39108e-06, -4.40064e-06, -5.66906e-06, -7.32579e-06,
                        -9.47437e-06, -1.22442e-05, -1.58166e-05, -2.04245e-05,-2.63527e-05, -3.39637e-05, -4.37754e-05, -5.64092e-05,
                        -7.26162e-05, -9.34584e-05, -0.000120179, -0.000154463,-0.000198384, -0.000254637, -0.000326597, -0.000418571,
                        -0.000536015, -0.000685872, -0.000876814, -0.00111995,-0.00142903, -0.00182161, -0.00231951, -0.00295015,
                        -0.00374779, -0.00475518, -0.00602551, -0.00762473,-0.00963471, -0.0121566, -0.0153151, -0.0192639,
                        -0.0241918, -0.0303303, -0.0379631, -0.0474377, -0.0591792,-0.0737073, -0.0916590, -0.113815, -0.141135, -0.174802,
                        -0.216279, -0.267385, -0.330392, -0.408156, -0.504287,-0.623379, -0.771314, -0.955672, -1.18628, -1.47592, -1.84134,
                        -2.30451, -2.89434, -3.64897, -4.61874, -5.87009, -7.49083,-9.59684, -12.3411, -15.9257, -20.6167, -26.7657, -34.8358])

            z = np.zeros(len(yl))
            x = np.zeros(len(yl))
            for i in range(100):
                z[i] = -10. + 12. * i / 100.
                x[i] = 10.**z[i]
                
            z1 = np.log10(xin)
            if z1<-10.:
                y=1.
                grad=-20.
            elif z1>1.89:
                y=0.
                grad=0.
            else:
                i = int( (z1+10.)/0.12 )
                if i==n:
                    i = n-2
                
                x1 = x[i]
                x2 = x[i+1]
                fx = (xin-x1)/(x2-x1)
                g1 = (yl[i+1] - yl[i])/(x2-x1)
                
                ylint = (1.-fx)*yl[i]+fx*yl[i+1]
                y=10.**ylint
                grad=y*np.log(10.)*g1
                
                #Numerical overflow fix
                if xin<1.0e-8:
                    grad=-20.
                    
            return y,grad 
         
        def calc_zeta(gamma,tau):
            """
            Subroutine to calculate the zeta function
            """
            
            c0 = 2./3. ; c1 = c0/gamma ; c2 = c1/gamma
            sarg = gamma * tau
            
            #Initial x
            x = c0 + c1 * (1.0 + (0.5 * sarg - 1.0) * np.exp(-sarg))
            
            y,grad = e2(sarg)
            
            #Final x
            zeta = x + c0 * gamma * (1.0 - 0.5 * tau**2.) * y
            
            #Derivative with respect to tau
            dzeta_dtau = c1 * ( -(0.5 * sarg - 1.) * gamma * np.exp(-sarg))
            dzeta_dtau += c1 * 0.5 * gamma * np.exp(-sarg)
            dzeta_dtau += c0 * gamma * (1.0 - 0.5 * tau**2.) * grad * gamma
            dzeta_dtau -= c0 * gamma * y * tau
            
            #Derivative with respect to gamma
            dzeta_dgamma = ( c1 * (-(0.5 * sarg -1.) * np.exp(-sarg) * tau + 0.5 * tau * np.exp(-sarg)))
            dzeta_dgamma -= c2 * (1.0 + (0.5 * sarg - 1.) * np.exp(-sarg))
            
            c3 = c0 * (1.0 - 0.5 * tau**2.)
            dzeta_dgamma += c3 * (gamma * grad * tau + y)
            
            return zeta, dzeta_dtau, dzeta_dgamma
        
        #Calculate the equilibrium temperature
        T_eq = T_star * np.sqrt(0.5*R_star/sdist)
        T_irr = beta * T_eq
        
        c1 = 3./4. * (T_int**4.)
        cx = 3./4. * (T_irr**4.)
        dcx_dbeta = 3. * (T_irr**3.) * T_eq
        
        #Calculate gravity at lowest level
        atm.calc_grav()
        G0 = atm.GRAV[0] #m s-2
        
        T_out = np.zeros(atm.NP)
        xmap = np.zeros((5,atm.NP))
        for i in range(atm.NP):
            
            tau = k_ir * atm.P[i] / G0 / 10.   #factor of 10 comes from change in units from SI to cgi
            x = c1*(2.0/3.0 + tau)
            
            #Calculating the zeta function
            zeta1,dz1_dtau,dz1_dgamma = calc_zeta(gammav1,tau)
            zeta2,dz2_dtau,dz2_dgamma = calc_zeta(gammav2,tau)
            
            x += cx*((1.-alpha)*zeta1 + alpha*zeta2)
            
            T_out[i] = x**0.25
            g1 = 0.25 * x**(-0.75)
            
            dx_dalpha = -cx * zeta1 + cx * zeta2
            dx_dbeta = ((1.0-alpha)*zeta1 + alpha*zeta2) * dcx_dbeta
            dx_dtau = cx*((1.-alpha)*dz1_dtau+alpha*dz2_dtau) + c1
            dx_dkir = dx_dtau*tau/k_ir
            dx_dg1 = cx*(1.0-alpha)*dz1_dgamma
            dx_dg2 = cx*alpha*dz2_dgamma
            
            xmap[0,i] = dx_dalpha * g1 * alpha
            xmap[1,i] = dx_dbeta * g1 * beta
            xmap[2,i] = dx_dkir * g1 * k_ir
            xmap[3,i] = dx_dg1 * g1 * gammav1
            xmap[4,i] = dx_dg2 * g1 * gammav2
            
        atm.edit_T(T_out)
            
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
        #******** Temperature profile from double grey analytic formulation (Parmentier and Guillot (2014) and Line et al. (2013))

        #Reading alpha
        s = f.readline().split()
        alpha = float(s[0])
        alpha_err = float(s[1])
        if alpha>0.0:
            x0[ix] = np.log(alpha)
            lx[ix] = 1
            inum[ix] = 0
            sx[ix,ix] = (alpha_err/alpha)**2.
        else:
            raise ValueError('error in read_apr :: alpha must be > 0')
        ix += 1
        
        #Reading beta
        s = f.readline().split()
        beta = float(s[0])
        beta_err = float(s[1])
        if beta>0.0:
            x0[ix] = np.log(beta)
            lx[ix] = 1
            inum[ix] = 0
            sx[ix,ix] = (beta_err/beta)**2.
        else:
            raise ValueError('error in read_apr :: beta must be > 0')
        ix += 1
        
        #Reading k_ir
        s = f.readline().split()
        k_ir = float(s[0])
        k_ir_err = float(s[1])
        if k_ir>0.0:
            x0[ix] = np.log(k_ir)
            lx[ix] = 1
            inum[ix] = 0
            sx[ix,ix] = (k_ir_err/k_ir)**2.
        else:
            raise ValueError('error in read_apr :: k_ir must be > 0')
        ix += 1

        #Reading gammav1
        s = f.readline().split()
        gammav1 = float(s[0])
        gammav1_err = float(s[1])
        if gammav1>0.0:
            x0[ix] = np.log(gammav1)
            lx[ix] = 1
            inum[ix] = 0
            sx[ix,ix] = (gammav1_err/gammav1)**2.
        else:
            raise ValueError('error in read_apr :: gammav1 must be > 0')
        ix += 1
        
        #Reading gammav2
        s = f.readline().split()
        gammav2 = float(s[0])
        gammav2_err = float(s[1])
        if gammav2>0.0:
            x0[ix] = np.log(gammav2)
            lx[ix] = 1
            inum[ix] = 0
            sx[ix,ix] = (gammav2_err/gammav2)**2.
        else:
            raise ValueError('error in read_apr :: gammav2 must be > 0')
        ix += 1


        #Reading the extra parameters
        s = f.readline().split()
        T_star = float(s[0])
        R_star = float(s[1])
        sdist = float(s[2])
        T_int = float(s[3])

        varparam[0] = T_star
        varparam[1] = R_star
        varparam[2] = sdist
        varparam[3] = T_int

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, T_star, R_star, sdist, T_int)

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
        #******** Temperature profile from double grey analytic formulation (Parmentier and Guillot (2014) and Line et al. (2013))  
        ix = ix + 5
        
        T_star = varparam[0]
        R_star = varparam[1]
        sdist = varparam[2]
        T_int = varparam[3]

        return cls(ix_0, ix-ix_0, T_star, R_star, sdist, T_int)

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
        alpha = xn_params[0] ; beta = xn_params[1] ; k_ir = xn_params[2] ; gammav1 = xn_params[3] ; gammav2 = xn_params[4]
        
        atm, xmap1 = self.calculate(
            atm, 
            alpha, beta, k_ir, gammav1, gammav2,
            self.T_star, self.R_star, self.sdist, self.T_int
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:forward_model.AtmosphereX.NP] = xmap1
        
        return


