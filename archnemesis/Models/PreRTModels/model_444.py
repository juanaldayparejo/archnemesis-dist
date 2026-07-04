
from typing import TYPE_CHECKING, Self, IO, Any

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

from archnemesis.Scatter_0 import kk_new_sub

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

class Model444(PreRTModelBase):
    """
        Allows for retrieval of the particle size distribution and imaginary refractive index.
    """
    
    id : int = 444


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            haze_params : dict[str,Any],
            #   Optical constants for the aerosol species (haze) this model represents
            
            aerosol_species_index : int,
            #   Index of the aerosol species that this model pertains to
            
            scattering_type_id : int,
            #   The scattering type this model uses
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('particle_size_distribution_params', slice(0,2), 'Values that define the particle size distribution'),
            ModelParameter('imaginary_ref_idx', slice(2,None), 'Imaginary refractive index of the particle size distribution'),
        )
        
        # Store model-specific constants on the model instance for easy access later
        self.haze_params = haze_params
        self.aerosol_species_idx = aerosol_species_index
        self.scattering_type_id = scattering_type_id


    @classmethod
    def calculate(
            cls, 
            Scatter : "Scatter_0",
            #   Scatter_0 instance of the retrieval setup we are calculating this model for
            
            idust : int,
            #   Aerosol species index we are calculating this model for
            
            iscat : int,
            #   scattering type we are using for this mode. NOTE: this is always set to 1 for now
            
            xprof : np.ndarray[["nparam"],float],
            #   The slice of the state vector that parameters of this model are held in.
            
            haze_params : dict[str,Any] ,
            #   A dictionary of constants for the aerosol being represented by this model.
            
        ) -> "Scatter_0":
        """
            FUNCTION NAME : model444()

            DESCRIPTION :

                Function defining the model parameterisation 444 in NEMESIS.

                Allows for retrieval of the particle size distribution and imaginary refractive index.

            INPUTS :

                Scatter :: Python class defining the scattering parameters
                idust :: Index of the aerosol distribution to be modified (from 0 to NDUST-1)
                iscat :: Flag indicating the particle size distribution
                xprof :: Contains the size distribution parameters and imaginary refractive index
                haze_params :: Read from 444 file. Contains relevant constants.

            OPTIONAL INPUTS:


            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model444(Scatter,idust,iscat,xprof,haze_params)

            MODIFICATION HISTORY : Joe Penn (11/9/2024)

        """   
        _lgr.debug(f'{idust=} {iscat=} {xprof=} {type(xprof)=}')
        for item in ('WAVE', 'NREAL', 'WAVE_REF', 'WAVE_NORM'):
            _lgr.debug(f'haze_params[{item}] : {type(haze_params[item])} = {haze_params[item]}')

        a = np.exp(xprof[0])
        b = np.exp(xprof[1])
        if iscat == 1:
            pars = (a,b,(1-3*b)/b)
        elif iscat == 2:
            pars = (a,b,0)
        elif iscat == 4:
            pars = (a,0,0)
        else:
            _lgr.warning(f'ISCAT = {iscat} not implemented for model 444 yet! Defaulting to iscat = 1.')
            pars = (a,b,(1-3*b)/b)

        Scatter.WAVER = haze_params['WAVE']
        Scatter.REFIND_IM = np.exp(xprof[2:])
        reference_nreal = haze_params['NREAL']
        reference_wave = haze_params['WAVE_REF']
        normalising_wave = haze_params['WAVE_NORM']
        if len(Scatter.REFIND_IM) == 1:
            Scatter.REFIND_IM = Scatter.REFIND_IM * np.ones_like(Scatter.WAVER)

        Scatter.REFIND_REAL = kk_new_sub(np.array(Scatter.WAVER), np.array(Scatter.REFIND_IM), reference_wave, reference_nreal)


        Scatter.makephase(idust, iscat, pars)

        xextnorm = np.interp(normalising_wave,Scatter.WAVE,Scatter.KEXT[:,idust])
        Scatter.KEXT[:,idust] = Scatter.KEXT[:,idust]/xextnorm
        Scatter.KSCA[:,idust] = Scatter.KSCA[:,idust]/xextnorm
        return Scatter


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
        #******** model for retrieving an aerosol particle size distribution and imaginary refractive index spectrum
        
        _lgr.debug(f'{ix=}')
        s = f.readline().split()    
        haze_f = open(s[0],'r')
        haze_waves = []
        for j in range(2):
            line = haze_f.readline().split()
            xai, xa_erri = line[:2]

            x0[ix] = np.log(float(xai))
            lx[ix] = 1
            sx[ix,ix] = (float(xa_erri)/float(xai))**2.

            ix = ix + 1
        _lgr.debug(f'{ix=}')

        nwave, clen = haze_f.readline().split('!')[0].split()
        vref, nreal_ref = haze_f.readline().split('!')[0].split()
        v_od_norm = haze_f.readline().split('!')[0]
        _lgr.debug(f'{nwave=} {clen=} {vref=} {nreal_ref=} {v_od_norm=}')

        for j in range(int(nwave)):
            line = haze_f.readline().split()
            v, xai, xa_erri = line[:3]

            x0[ix] = np.log(float(xai))
            lx[ix] = 1
            sx[ix,ix] = (float(xa_erri)/float(xai))**2.

            ix = ix + 1
            haze_waves.append(float(v))

            if float(clen) < 0:
                break
        _lgr.debug(f'{ix=}')

        aerosol_species_idx = varident[1]-1

        haze_params = dict()
        haze_params['NX'] = 2+len(haze_waves)
        haze_params['WAVE'] = haze_waves
        haze_params['NREAL'] = float(nreal_ref)
        haze_params['WAVE_REF'] = float(vref)
        haze_params['WAVE_NORM'] = float(v_od_norm)

        varparam[0] = 2+len(haze_waves)
        varparam[1] = float(clen)
        varparam[2] = float(vref)
        varparam[3] = float(nreal_ref)
        varparam[4] = float(v_od_norm)

        if float(clen) > 0:
            for j in range(int(nwave)):
                for k in range(int(nwave)):

                    delv = haze_waves[k]-haze_waves[j]
                    arg = abs(delv/float(clen))
                    xfac = np.exp(-arg)
                    if xfac >= sxminfac:
                        sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                        sx[ix+k,ix+j] = sx[ix+j,ix+k]
        _lgr.debug(f'{ix=}')
        
        scattering_type_id = 1 # Should add a way to alter this value from the input files.

        return cls(ix_0, ix-ix_0, haze_params, aerosol_species_idx, scattering_type_id)


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
        
        # NOTE:
        # ix is not required as we have stored that information on the model instance
        # ipar is ignored for this model
        # ivar is ignored for this model
        # xmap is ignored for this model
        forward_model.ScatterX = self.calculate(
            forward_model.ScatterX,
            self.aerosol_species_idx,
            self.scattering_type_id,
            self.get_state_vector_slice(forward_model.Variables.XN),
            self.haze_params
        )


