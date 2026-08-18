
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from ._base import PreRTModelBase
from ..param import StateParam, ConstParam

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

@dc.dataclass(slots=True)
class Model110(PreRTModelBase):
    """
        In this model, the Venus cloud is parameterised using the model of Haus et al. (2016).
        In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes.
        In this parametersiation, we include the Haus cloud model as it is, but we allow the altitude of the cloud
        to vary according to the inputs.

        The units of the aerosol density are in m-3, so the extinction coefficients must not be normalised.
    """
    id: ClassVar[int] = 110

    z_offset: StateParam.using(slice(0,1), 'Offset in altitude (km) of the cloud with respect to the Haus et al. (2016) model.', 'km') # noqa: F722 F821
    atm_profile_type: ConstParam.using(AtmosphericProfileTypeEnum, 'Atmospheric profile type this model applies to') # noqa: F722 F821

    def calculate(self, atm, idust0, MakePlot=False):
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

        z_offset = self.z_offset.v

        zb1 = 49. + z_offset
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
    ) -> Self:
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 1)
        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )

        instance.z_offset.log = False

        return instance

    @classmethod
    def from_bookmark(
        cls,
        variables: "Variables_0",
        varident: np.ndarray[[3], int],
        varparam: np.ndarray[["mparam"], float],
        ix: int,
        npro: int,
        ngas: int,
        ndust: int,
        nlocations: int,
    ) -> Self:
        xvals = np.zeros(1)
        xerrs = np.zeros(1)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )


    def calculate_from_subprofretg(
        self,
        forward_model: "ForwardModel_0",
        ix: int,
        ipar: int,
        ivar: int,
        xmap: np.ndarray,
    ) -> None:
        atm = forward_model.AtmosphereX

        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)

        idust0 = int(abs(forward_model.Variables.VARIDENT[ivar, 0]) - 1)

        # Pull parameter values from the state vector into this instance
        self.pull_from_state_vector(forward_model.Variables.XN)

        # calculate uses the instance z_offset (or an explicit override)
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX, idust0)

        ix = ix + int(forward_model.Variables.NXVAR[ivar])


