
from typing import TYPE_CHECKING, Self, ClassVar, Any, IO
import dataclasses as dc

import numpy as np


from ._base import PreRTModelBase
from ..param import StateParam, ConstParam, VarParam

from archnemesis.Scatter_0 import kk_new_sub

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





@dc.dataclass(slots=True)
class Model444(PreRTModelBase):
    """
    Allows for retrieval of the particle size distribution and imaginary refractive index.
    """
    
    id : ClassVar[int] = 444

    particle_size_distribution_params : StateParam.using(slice(0,2), 'Values that define the particle size distribution')
    imaginary_ref_idx                 : StateParam.using(slice(2,None), 'Imaginary refractive index of the particle size distribution')
    
    haze_file_path      : ConstParam[str].using("Path to the file that contains haze parameters")
    
    aerosol_species_idx : VarParam[int].using('Index of the aerosol species the imaginary refactive index is varying for')
    scattering_type_id  : VarParam[int].using('Type of scattering used in calculations')
    n_waves             : VarParam[int].using('Number of wavenumbers for the imaginary refractive index')
    haze_waves          : VarParam[np.ndarray].using('Wavenumbers of the imaginary refractive index points')
    haze_wave_norm      : VarParam[float].using('Wavenumber to normalise extinction cross section spectrum to')
    haze_wave_ref       : VarParam[float].using('Reference wavenumber for normal component')
    haze_wave_ref_rri   : VarParam[float].using('Real component of refractive index at reference wavenumber')
    correlation_length  : VarParam[float].using('Correlation length of imaginary refractive index')

    
    
    
    @staticmethod
    def read_haze_file(fpath) -> tuple[np.ndarray, np.ndarray, tuple[Any,...]]:
        """
        Reads details from haze file at `fpath`
        
        Returns: (
            state vector, 
            covariance vector,
            constparams and varparams
        )
        """
        with open(fpath, 'r') as f:
            """
            File contents:
            ```
            mean radius of particle size distribution, error
            variance of size distribution, error
            number of wavelengths, correlation length
            reference wavelength, real part of refractive index at reference wavelength
            wavenumber to normalise extinction cross section spectrum to
            wavenumber, imaginary refractive index, error
            ...       , ...                       , ...
            ```
            """
            xvals = []
            xerrs = []
            for j in range(2):
                xval, xerr = (float(a) for a in f.readline().split()[:2])
                xvals.append(np.log(xval))
                xerrs.append((xerr/xval)**2)
            
            n_waves, clen = f.readline().split('!')[0].split()
            vref, nreal_ref = f.readline().split('!')[0].split()
            v_od_norm = f.readline().split('!')[0]
            
            n_waves = int(n_waves)
            clen = float(clen)
            vref = float(vref)
            nreal_ref = float(nreal_ref)
            v_od_norm = float(v_od_norm)
            
            
            haze_waves = np.zeros(n_waves)
            
            for j in range(int(n_waves)):
                v, xval, xerr = (float(x) for x in f.readline().split()[:3])
                haze_waves[j] = v
                xvals.append(np.log(xval))
                xerrs.append((xerr/xval)**2)
            
            return (
                np.array(xvals),
                np.array(xerrs),
                (
                    n_waves,
                    np.array(haze_waves),
                    v_od_norm,
                    vref,
                    nreal_ref,
                    clen,
                ),
            )
    
    @classmethod
    def from_apr_file(
            cls,
            f : IO,
            varident : np.ndarray[[3],int],
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
            runname : str,
            sxminfac : float,
    ) -> Self:
        haze_file = f.readline().split()[0]
        
        (
            xvals, 
            xerrs,
            (
                n_waves,
                haze_waves,
                v_od_norm,
                vref,
                nreal_ref,
                clen,
            )
        ) = cls.read_haze_file(haze_file)
        
        aerosol_species_idx = varident[1]-1
        
        scattering_type_id = 1 # Should add a way to alter this value from the input files.
    
        return cls.from_arrays(
            xvals,
            xerrs,
            haze_file,
            aerosol_species_idx, 
            scattering_type_id,
            n_waves,
            haze_waves,
            v_od_norm,
            vref,
            nreal_ref,
            clen,
        )
    
    
    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["mx","mx"],float],
            sxminfac : float,
    ):
        """
        Model 444 requires a custom covariance matrix calculation
        """
        sx_part = sx[self.state_vector_slice, self.state_vector_slice]
        sx_part[...] = 0.0
        
        sx_part[
            self.particle_size_distribution_params.slice, 
            self.particle_size_distribution_params.slice
        ] = np.diag((self.particle_size_distribution_params.e / self.particle_size_distribution_params.v)**2)
        
        sx_part2 = sx_part[self.imaginary_ref_idx.slice, self.imaginary_ref_idx.slice]
        
        sx_part2[...] = np.diag((self.imaginary_ref_idx.e / self.imaginary_ref_idx.v)**2)
        
        
        if self.correlation_length.v > 0:
        
            one = np.ones((self.n_waves, self.n_waves))
            distance_matrix = (one * self.haze_waves[None,:]) - (one * self.haze_waves[:,None])
            cor_dist_mat = np.exp(-1*np.abs(distance_matrix / self.correlation_length.v))
            
            variance = (self.imaginary_ref_idx.e / self.imaginary_ref_idx.v)**2
            
            sx_part2[cor_dist_mat >= sxminfac] = ((variance[:,None] @ variance[None,:])*cor_dist_mat)[cor_dist_mat >= sxminfac]



    def calculate(self, Scatter : "Scatter_0"):
        a = np.exp(self.particle_size_distribution_params.v[0])
        b = np.exp(self.particle_size_distribution_params.v[1])
        
        iscat = self.scattering_type_id.v
        if iscat == 1:
            pars = (a,b,(1-3*b)/b)
        elif iscat == 2:
            pars = (a,b,0)
        elif iscat == 4:
            pars = (a,0,0)
        else:
            _lgr.warning(f'ISCAT = {iscat} not implemented for model 444 yet! Defaulting to iscat = 1.')
            pars = (a,b,(1-3*b)/b)

        Scatter.WAVER = self.haze_waves.v
        Scatter.REFIND_IM = np.exp(self.imaginary_ref_idx.v)
        reference_nreal = self.haze_wave_ref_rri.v
        reference_wave = self.haze_wave_ref.v
        normalising_wave = self.haze_wave_norm.v
        
        idust = self.aerosol_species_idx.v
        
        if len(Scatter.REFIND_IM) == 1:
            Scatter.REFIND_IM = Scatter.REFIND_IM * np.ones_like(Scatter.WAVER)

        Scatter.REFIND_REAL = kk_new_sub(np.array(Scatter.WAVER), np.array(Scatter.REFIND_IM), reference_wave, reference_nreal)

        Scatter.makephase(idust, iscat, pars)

        xextnorm = np.interp(normalising_wave,Scatter.WAVE,Scatter.KEXT[:,idust])
        Scatter.KEXT[:,idust] = Scatter.KEXT[:,idust]/xextnorm
        Scatter.KSCA[:,idust] = Scatter.KSCA[:,idust]/xextnorm
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
        ix_0 = ix
        #******** model for retrieving an aerosol particle size distribution and imaginary refractive index spectrum
        _lgr.warn(f"{cls.__name__}.from_bookmark(...) only sets model parameters that have been stored in `varident`, `varparam`. Therefore it cannot set `haze_params['WAVE']` at the moment as those values are in an external file whose name is not stored in those locations. Use with caution.")
        
        #haze_waves = []
        for j in range(2):
            ix = ix + 1

        nwave = varparam[0] - 2
        #clen = varparam[1]
        vref = varparam[2]
        nreal_ref = varparam[3]
        v_od_norm = varparam[4]
        
        haze_params = dict()
        haze_params['NX'] = nwave
        #haze_params['WAVE'] = haze_waves    !This needs to be fixed!
        haze_params['NREAL'] = float(nreal_ref)
        haze_params['WAVE_REF'] = float(vref)
        haze_params['WAVE_NORM'] = float(v_od_norm)

        for j in range(int(nwave)):
            ix = ix + 1

        aerosol_species_idx = varident[1]-1
        scattering_type_id = 1 # Should add a way to alter this value from the input files.

        return cls(ix_0, ix-ix_0, haze_params, aerosol_species_idx, scattering_type_id)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
    ) -> None:
        self.pull_from_state_vector(forward_model.Variables.XN)
        forward_model.ScatterX = self.calculate(forward_model.ScatterX)

