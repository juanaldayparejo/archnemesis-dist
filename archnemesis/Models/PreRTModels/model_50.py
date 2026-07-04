
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

from archnemesis.enum import AtmosphericProfileTypeEnum

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

class Model50(PreRTModelBase):
    """
        In this model, the atmospheric parameters are modelled as continuous profiles
        multiplied by a scaling factor in linear space. Each element of the state vector
        corresponds to this scaling factor at each altitude level. This parameterisation
        allows the retrieval of negative VMRs.
    """
    id : int = 50


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileTypeEnum,
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
            ModelParameter('xprof', slice(0,n_state_vector_entries), 'Factors at each level to scale reference profile by', 'NUMBER'),
        )
        
        return

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
        #********* continuous profile of a scaling factor ************************
        ix = ix + npro
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


