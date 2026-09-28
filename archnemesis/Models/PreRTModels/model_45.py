
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter


from archnemesis.enum import AtmosphericProfileTypeEnum, GasEnum

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

class Model45(PreRTModelBase):
    """
        Irwin gas model: Variable deep tropospheric and stratospheric abundances,
        along with tropospheric humidity.

        Supports CH4 (gas 6), NH3 (gas 11), and H2S (gas 36).
        The H2S saturation law describes solid ice below 187.6 K.
        The three state-vector entries are ln(deep VMR), ln(humidity),
        and ln(stratospheric VMR), in that order.
    """
    
    id : int = 45

    # ln(saturation pressure / bar) = A + B / T.
    # CH4 and NH3 use the coefficients from the Fortran Irwin routines.
    _svp_coefficients = {
        GasEnum.CH4: (10.6815, -1163.83),
        GasEnum.NH3: (17.3471, -3930.55),
        GasEnum.H2S: (12.953, -2706.2),
    }

    def __init__(
            self, 
            state_vector_start : int, 
            n_state_vector_entries : int,
            atm_profile_type : AtmosphericProfileTypeEnum,
        ):
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)
        
        self.parameters = (
            ModelParameter('deep_vmr', slice(0,1), 'deep (tropospheric) gas volume mixing ratio', 'RATIO'),
            ModelParameter('humidity', slice(1,2), 'relative humidity of gas', 'RATIO'),
            ModelParameter('strato_vmr', slice(2,3), 'high (stratospheric) gas volume mixing ratio', 'RATIO'),
        )
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            atm_profile_type : AtmosphericProfileTypeEnum,
            atm_profile_idx : int | None,
            tropo, 
            humid, 
            strato, 
            MakePlot=True
        ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : Model45.calculate

            DESCRIPTION :
                Irwin gas model. Variable deep tropospheric and stratospheric abundances,
                along with tropospheric humidity. As in Fortran Model45, condensation is
                triggered at full saturation, then humidity scales the saturated
                abundance. The stratospheric and deep abundance caps follow.

            INPUTS :
                tropo :: Deep gas VMR
                humid :: Relative gas humidity in the troposphere
                strato :: Stratospheric gas VMR

            OUTPUTS :
                atm :: Updated atmosphere
                xnewgrad(3, NP) :: Derivatives of VMR with respect to
                    ln(tropo), ln(humid), and ln(strato), respectively.
                    These are derivatives within the selected branch; a
                    derivative across a branch discontinuity is not defined.
        """

        _lgr.debug(f'{atm_profile_type=} {atm_profile_idx=} {tropo=} {humid=} {strato=}')

        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            _msg = f'Model id={cls.id} is only defined for gas VMR profiles.'
            _lgr.error(_msg)
            raise ValueError(_msg)
            
        if atm_profile_idx is None:
            raise ValueError('Model45 requires a gas profile index')
        gas_id = atm.ID[atm_profile_idx]
        if gas_id not in cls._svp_coefficients:
            raise ValueError(f'Model45 is not set up for gas ID {gas_id}; expected 6, 11, or 36')
        svp_a, svp_b = cls._svp_coefficients[gas_id]

        NP = atm.NP

        xnew = np.zeros(NP)
        xnewgrad = np.zeros((3, NP))
        partial_pressure = np.zeros(NP)
        pbar = np.zeros(NP)
        psvp = np.zeros(NP)

        for i in range(NP):
            pbar[i] = atm.P[i] / 100000.0  # Convert Pascal to Bar

            # Calculate saturation pressure in bar for the selected gas.
            tmp = svp_a + svp_b / atm.T[i]
            psvp[i] = 1e-30 if tmp < -69.0 else np.exp(tmp)

            # 1. Start with Deep VMR
            partial_pressure[i] = tropo * pbar[i]
            active_parameter = 0

            # 2. Check Condensation (Humidity limit)
            if partial_pressure[i] / psvp[i] > 1.0:
                partial_pressure[i] = psvp[i] * humid
                active_parameter = 1

            # 3. Stratospheric Cap (Photochemistry/Depletion)
            # Cap VMR if pressure is low (< 0.1 bar)
            if pbar[i] < 0.1 and partial_pressure[i] / pbar[i] > strato:
                partial_pressure[i] = pbar[i] * strato
                active_parameter = 2

            # 4. Enforce Deep VMR (Deep Atmosphere > 0.5 bar)
            # Supersaturated humidity can exceed the deep abundance; apply
            # Model45's final cap and replace the active derivative as well.
            if pbar[i] > 0.5 and partial_pressure[i] / pbar[i] > tropo:
                partial_pressure[i] = pbar[i] * tropo
                active_parameter = 0

            xnew[i] = partial_pressure[i] / pbar[i]
            # Within each branch VMR is proportional to the active parameter,
            # so d(VMR)/d(ln(parameter)) equals VMR itself.
            xnewgrad[active_parameter, i] = xnew[i]

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
        if varident[0] not in cls._svp_coefficients:
            raise ValueError(f'Model45 is not set up for gas ID {varident[0]}; expected 6, 11, or 36')
        ix_0 = ix
        # Irwin gas model: deep VMR, humidity, stratospheric VMR.
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float')
        tropo = tmp[0]
        etropo = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float')
        humid = tmp[0]
        ehumid = tmp[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float')
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
        assert issubclass(cls, model_classification[0]), "Model base class mismatch"

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
        if varident[0] not in cls._svp_coefficients:
            raise ValueError(f'Model45 is not set up for gas ID {varident[0]}; expected 6, 11, or 36')
        ix_0 = ix
        # Irwin gas model bookmark.
        ix = ix + 3
        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class mismatch"
        return cls(ix_0, ix-ix_0, model_classification[1])

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        
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
