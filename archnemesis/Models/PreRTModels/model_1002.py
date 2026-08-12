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

from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.Atmosphere_0 import Atmosphere_0
    #from archnemesis.ForwardModel_0 import ForwardModel_0

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
class Model1002(PreRTModelBase):
    """ 
        In this model, the atmospheric parameters are scaled using a single factor with 
        respect to the vertical profiles in the reference atmosphere.
        
        The model is applied simultaneously in different planet locations
    """
    id : ClassVar[int] = 1002

    scaling_factors: StateParam.using(slice(0,None), 'Scaling factors at each location', 'NUMBER') # noqa: F722 F821
    
    atm_profile_type: ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to') # noqa: F722 F821
    lats: ConstParam[np.ndarray].using('Latitude of each location') # noqa: F722 F821
    lons: ConstParam[np.ndarray].using('Longitude of each location') # noqa: F722 F821
    correlation_length: ConstParam[float].using('Correlation length (degrees)') # noqa: F722 F821
    angular_distance : ConstParam[np.ndarray].using('Angular distances between points', 'degrees') # noqa: F722 F821
    
    n_locations: VarParam[int].using('Number of locations') # noqa: F722 F821
    
    
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
    
        nlocs, clen = cls.read_apr_entries(f, (int, int))
        lats, lons, xvals, xerrs = cls.read_apr_entries(f, (float, float, float, float), nlocs)
        
        
        lats_r = lats * np.pi/180.0
        lons_r = lons * np.pi/180.0
        
        one = np.ones((nlocs,nlocs),dtype=float)
        angular_distance = np.arccos(
            np.sin(lats_r[None,:]) @ np.sin(lats_r[:,None]) 
            + np.cos(lats_r[None,:]) @ np.cos(lats_r[:,None])*(one*lons_r[None,:] - one(lons_r[:,None]))
        ) * 180.0 / np.pi
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            lats,
            lons,
            clen,
            angular_distance,
            nlocs,
        )
        
        instance.scaling_factors.log = False
        
        return instance


    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["mx","mx"],float],
            sxminfac : float,
    ):
        """
        Model 444 requires a custom covariance matrix calculation
        """
        self.push_to_covariance_matrix_with_correlation_length(
            sx,
            sxminfac,
            correlation_lengths = (0, self.correlation_length.v),
            distances = self.angular_distance.v,
        )



    def calculate(
            self, 
            atm : "Atmosphere_0",
            atm_profile_type : AtmosphericProfileTypeEnum,
            atm_profile_idx : int,
            MakePlot=False
    ):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 1002 in NEMESIS.

                This is the same as model 2, but applied simultaneously in different planet locations
                In this model, the atmospheric parameters are scaled using a single factor with 
                respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum denoting what part of the atmosphere is being retrieved
            
                atm_profile_idx :: Index of the atmospheric type.

            OPTIONAL INPUTS: None

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(nlocations,ngas+2+ncont,npro,nlocations) :: Matrix of relating funtional derivatives to 
                                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model1002(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (19/04/2023)

        """
        scf = self.scaling_factors.v
        
        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))
        #xmap1 = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))

        if len(scf)!=atm.NLOCATIONS:
            raise ValueError('error in model 1002 :: The number of scaling factors must be the same as the number of locations in Atmosphere')

        if atm.NLOCATIONS<=1:
            raise ValueError('error in model 1002 :: This model can be applied only if NLOCATIONS>1')

        x1 = np.zeros((atm.NP,atm.NLOCATIONS))
        xref = np.zeros((atm.NP,atm.NLOCATIONS))
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            xref[:,:] = atm.VMR[:,atm_profile_idx,:]
            x1[:,:] = atm.VMR[:,atm_profile_idx,:] * scf[:]
            atm.VMR[:,atm_profile_idx,:] =  x1
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            xref[:] = atm.T[:,:]
            x1[:] = np.transpose(np.transpose(atm.T[:,:]) * scf[:])
            atm.T[:,:] = x1 
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            xref[:] = atm.DUST[:,atm_profile_idx,:]
            x1[:] = np.transpose(np.transpose(atm.DUST[:,atm_profile_idx,:]) * scf[:])
            atm.DUST[:,atm_profile_idx,:] = x1
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            xref[:] = atm.PARAH2[:,:]
            x1[:] = np.transpose(np.transpose(atm.PARAH2[:,:]) * scf)
            atm.PARAH2[:,:] = x1
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            xref[:] = atm.FRAC[:,:]
            x1[:] = np.transpose(np.transpose(atm.FRAC[:,:]) * scf)
            atm.FRAC[:,:] = x1


        #This calculation takes a long time for big arrays
        #for j in range(atm.NLOCATIONS):
        #    xmap[j,ipar,:,j] = xref[:,j]


        if MakePlot==True:

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig,ax1 = plt.subplots(1,1,figsize=(6,4))
            im1 = ax1.scatter(atm.LONGITUDE,atm.LATITUDE,c=scf,cmap='jet',vmin=scf.min(),vmax=scf.max())
            ax1.grid()
            ax1.set_xlabel('Longitude / deg')
            ax1.set_ylabel('Latitude / deg')
            ax1.set_xlim(-180.,180.)
            ax1.set_ylim(-90.,90.)
            ax1.set_title('Model 1002')

            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar1 = plt.colorbar(im1, cax=cax)
            cbar1.set_label('Scaling factor')

            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #******** scaling of atmospheric profiles at multiple locations (linear scale)

        nlocs = int(varparam[0])
        if nlocs != nlocations:
            raise ValueError('error in model 1002 :: number of locations must be the same as in Surface and Atmosphere')

        instance = cls.from_arrays(
            np.zeros((nlocs,)),
            np.zeros((nlocs,)),
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            np.zeros((nlocs,)),#lats,
            np.zeros((nlocs,)),#lons,
            0,#clen,
            nlocs,
        )
        instance.scaling_factors.log = False
        
        return instance

