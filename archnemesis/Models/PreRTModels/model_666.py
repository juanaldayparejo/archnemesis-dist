
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

from ..param import (
    StateParam, 
    #ConstParam, 
    VarParam,
)

from ._base import PreRTModelBase

from archnemesis.enum import ArchNemesisFileTypeEnum

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
class Model666(PreRTModelBase):
    """
        In this model, we retrieve the pressure at a given tangent height.
    """
    id : ClassVar[int] = 666
    
    pressure : StateParam.using(slice(0,1), "Pressure at tangent height", 'atm') # noqa: F722 F821
    
    tangent_height : VarParam.using(float, "Tangent height at which the pressure is retrieved", 'km') # noqa: F722 F821


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
        
        htan = cls.read_apr_entries(f, (float,))
        xvals, xerrs = cls.read_apr_value_error_pairs(f, 1)
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            htan
        )
        
        instance.pressure.num_diff = True
        
        return instance


    def calculate(self, Atmosphere, MakePlot=False):

        """
            FUNCTION NAME : model666()

            DESCRIPTION :

                Function defining the model parameterisation 666 in NEMESIS.
                In this model, we retrieve the pressure at a given tangent height.

            INPUTS :

                Atmosphere :: Atmosphere class
                htan :: Tangent height (km)
                ptan :: Pressure at tangent height (atm)

            OPTIONAL INPUTS: None

            OUTPUTS :
                
                Atmosphere :: Updated Atmosphere class with recomputed pressure levels

            CALLING SEQUENCE:

                Atmosphere = model666(Atmosphere,htan,ptan)

            MODIFICATION HISTORY : Juan Alday (15/02/2023)

        """
        htan = self.tangent_height.v
        ptan = self.pressure.v
        
        hpre = Atmosphere.H
        ppre = Atmosphere.P
    
        _lgr.info(f'Calculating model 666 with htan={htan} km and ptan={ptan} atm')

        Atmosphere.adjust_hydrostatP(htan*1.0e3,ptan*101325.)

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(ppre,hpre/1.0e3,label='Uncorrected')
            ax1.plot(Atmosphere.P,Atmosphere.H/1.0e3,label='Corrected')
            ax1.legend()
            ax1.set_xlabel('Pressure (Pa)')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_xscale('log')
            plt.tight_layout()

        return Atmosphere

    

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
        
        if varident[2] != cls.id:
            raise ValueError('error in Model666.from_bookmark() :: wrong model id')

        htan = varparam[0]

        return cls((0,0), htan)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 666. Retrieval of pressure at a given tangent height
        #***************************************************************
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX)



