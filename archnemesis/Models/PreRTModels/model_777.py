
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

from ..param import (
    StateParam, 
    #ConstParam, 
    #VarParam,
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
class Model777(PreRTModelBase):
    """
        In this model, we apply a correction to the tangent heights listed on the 
        Measurement class
    """
    id : ClassVar[int] = 777
    
    height_correction : StateParam.using(slice(0,1), 'Correction to the tangent heights', 'km') # noqa: F722 F821

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
    
        xvals, xerrs = cls.read_apr_value_error_pairs(f, 1)
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
        )
        
        instance.height_correction.num_diff = True
        
        return instance


    def calculate(self, Measurement,MakePlot=False):

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
        hcorr = self.height_correction.v

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
        return cls((0,0))

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 777. Retrieval of tangent height corrections
        #***************************************************************
        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX)


