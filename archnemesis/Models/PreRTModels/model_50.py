
from typing import TYPE_CHECKING, Self, IO, ClassVar

import numpy as np

from ._base import PreRTModelBase
from ..param import (
    StateParam, 
    ConstParam, 
    #VarParam,
)
import dataclasses as dc
from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


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

@dc.dataclass
class Model50(PreRTModelBase):
    """
        In this model, the atmospheric parameters are modelled as continuous profiles
        multiplied by a scaling factor in linear space. Each element of the state vector
        corresponds to this scaling factor at each altitude level. This parameterisation
        allows the retrieval of negative VMRs.
    """
    id : ClassVar[int] = 50

    full_profile: StateParam.using(slice(None), 'Factors at each level to scale reference profile by', 'PROFILE_TYPE') # noqa: F722 F821
    atm_profile_type: ConstParam.using(AtmosphericProfileTypeEnum, 'Atmospheric profile type this model applies to') # noqa: F722 F821
    input_file_type: ConstParam.using(ArchNemesisFileTypeEnum, 'Input file type') # noqa: F722 F821
    n_level: ConstParam.using(int, 'Number of levels in the profile') # noqa: F722 F821
    pressure: ConstParam.using(np.ndarray, 'Pressure at each entry of the profile') # noqa: F722 F821
    correlation_length: ConstParam.using(float, 'Correlation length of profile') # noqa: F722 F821


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
        
        # Model parameters are declared as `StateParam` class attributes.
        # No runtime `ModelParameter` tuple is required for the new API.
        return

    def calculate(self, atm,ipar,xprof,MakePlot=False):

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

        xprof = self.full_profile.v
        npro = len(xprof)
        if npro != atm.NP:
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
    def from_apr_file(
            cls,
            f: IO,
            varident: np.ndarray[[3], int],
            npro: int,
            ngas: int,
            ndust: int,
            nlocations: int,
            runname: str,
            sxminfac: float,
            input_file_type: ArchNemesisFileTypeEnum,
    ) -> "Model50":
        s = f.readline().split()
        f1 = open(s[0], 'r')
        tmp = np.fromstring(f1.readline().rsplit('!', 1)[0], sep=' ', count=2, dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref, xvals, xerrs = cls.read_apr_entries(f1, (float, float, float), nlevel)
        f1.close()

        instance = cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            input_file_type,
            nlevel,
            pref,
            clen,
        )
        instance.full_profile.log = instance.atm_profile_type.v != AtmosphericProfileTypeEnum.TEMPERATURE
        return instance


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


