from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

from ..param import (
    StateParam, 
    ConstParam, 
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
    from archnemesis.Scatter_0 import Scatter_0

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
class Model887(PreRTModelBase):
    """
        In this model, the cross-section spectrum of IDUST is changed given the parameters in 
        the state vector
    """
    id : ClassVar[int] = 887

    extinction_cross_section : StateParam.using(slice(0,None), 'Extinction cross section values') # noqa: F722 F821
    
    correlation_length : ConstParam[float].using('Correlation length between the wavelengths/wavenumbers') # noqa: F722 F821
    wave_points : ConstParam[np.ndarray].using('Wavenumbers/wavelengths of the extinction cross-section values.') # noqa: F722 F821
    
    nspec : VarParam[int].using('Number of spectral points (must be the same as the *.xsc file)') # noqa: F722 F821
    aerosol_id : VarParam[int].using('Aerosol ID number for the dust we are operating upon') # noqa: F722 F821
    
    

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
    
        nspec, icloud, clen = cls.read_apr_entries(f, (int, int, float))
        
        waves, xvals, xerrs = cls.read_apr_entries(f, (float, float, float), nspec)
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            clen,
            waves,
            nspec,
            icloud
        )
        
        instance.extinction_cross_section.num_diff = True
        
        return instance

    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["mx","mx"],float],
            sxminfac : float,
    ):
        """
        Model 887 requires a custom covariance matrix calculation
        """
        self.push_to_covariance_matrix_with_correlation_length(
            sx,
            sxminfac,
            correlation_lengths = (0, self.correlation_length.v),
            distances = self.wave_points.v,
        )



    def calculate(
            self, 
            Scatter : "Scatter_0",
            MakePlot=False
    ) -> "Scatter_0":

        """
            FUNCTION NAME : model887()

            DESCRIPTION :

                Function defining the model parameterisation 887 in NEMESIS.
                In this model, the cross-section spectrum of IDUST is changed given the parameters in 
                the state vector

            INPUTS :

                Scatter :: Python class defining the spectral properties of aerosols in the atmosphere
                xsc :: New cross-section spectrum of aerosol IDUST
                idust :: Index of the aerosol to be changed (from 0 to NDUST-1)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model887(Scatter,xsc,idust)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        xsc = self.extinction_cross_section.v
        idust = self.aerosol_id.v

        if len(xsc)!=Scatter.NWAVE:
            raise ValueError('error in model 887 :: Cross-section array must be defined at the same wavelengths as in .xsc')
        else:
            kext = np.zeros([Scatter.NWAVE,Scatter.DUST])
            kext[:,:] = Scatter.KEXT
            kext[:,idust] = xsc[:]
            Scatter.KEXT = kext

        if MakePlot==True:
            fig,ax1=plt.subplots(1,1,figsize=(10,3))
            ax1.semilogy(Scatter.WAVE,Scatter.KEXT[:,idust])
            ax1.grid()
            if Scatter.ISPACE==1:
                ax1.set_xlabel(r'Wavelength ($\mu$m)')
            else:
                ax1.set_xlabel(r'Wavenumber (cm$^{-1}$')
            plt.tight_layout()
            plt.show()
        
        return Scatter
    
    
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
        raise NotImplementedError()


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
    ) -> None:
        forward_model.ScatterX = self.calculate(forward_model.ScatterX)


