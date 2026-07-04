
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase

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

class Model202(PreRTModelBase):
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
        #********* simple scaling of telluric atmospheric profile ************************
        ix = ix + 1

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
        #Model 202. Scaling factor of telluric atmospheric profile
        #***************************************************************

        scafac = forward_model.Variables.XN[ix]
        varid1 = forward_model.Variables.VARIDENT[ivar,0] ; varid2 = forward_model.Variables.VARIDENT[ivar,1]
        if forward_model.TelluricX is not None:
            forward_model.TelluricX = self.calculate(forward_model.TelluricX,varid1,varid2,scafac)

        ix = ix + forward_model.Variables.NXVAR[ivar]


