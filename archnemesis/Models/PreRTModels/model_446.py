
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase

from archnemesis.helpers import h5py_helper

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

class Model446(PreRTModelBase):
    """
        In this model, we change the extinction coefficient and single scattering albedo 
        of a given aerosol population based on its particle size, and based on the extinction 
        coefficients tabulated in a look-up table
    """
    id : int = 446

    def __init__(
            self, 
            state_vector_start : int, 
            n_state_vector_entries : int,
            lookup_table_fpath : str,
        ):
        """
        Initialise an instance of the model.
        
        ## ARGUMENTS ##
            
            state_vector_start : int
                The index of the first entry of the model parameters in the state vector
            
            n_state_vector_entries : int
                The number of model parameters that are stored in the state vector
            
            lookup_table_fpath: str
                path to the lookup table of extinction coefficient vs particle size
                
        
        ## RETURNS ##
            An initialised instance of this object
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        self.lookup_table_fpath : str = lookup_table_fpath


    @classmethod
    def calculate(cls, Scatter,idust,wavenorm,xwave,rsize,lookupfile,MakePlot=False):

        """
            FUNCTION NAME : model446()

            DESCRIPTION :

                Function defining the model parameterisation 446 in NEMESIS.

                In this model, we change the extinction coefficient and single scattering albedo 
                of a given aerosol population based on its particle size, and based on the extinction 
                coefficients tabulated in a look-up table

            INPUTS :

                Scatter :: Python class defining the scattering parameters
                idust :: Index of the aerosol distribution to be modified (from 0 to NDUST-1)
                wavenorm :: Flag indicating if the extinction coefficient needs to be normalised to a given wavelength (1 if True)
                xwave :: If wavenorm=1, then this indicates the normalisation wavelength/wavenumber
                rsize :: Particle size at which to interpolate the extinction cross section
                lookupfile :: Name of the look-up file storing the extinction cross section data

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Scatter :: Updated Scatter class

            CALLING SEQUENCE:

                Scatter = model446(Scatter,idust,wavenorm,xwave,rsize,lookupfile)

            MODIFICATION HISTORY : Juan Alday (25/11/2021)

        """

        import h5py
        from scipy.interpolate import interp1d

        #Reading the look-up table file
        with h5py.File(lookupfile,'r') as f:

            #NWAVE = h5py_helper.retrieve_data(f, 'NWAVE', np.int32)
            NSIZE = h5py_helper.retrieve_data(f, 'NSIZE', np.int32)

            WAVE = h5py_helper.retrieve_data(f, 'WAVE', np.array)
            REFF = h5py_helper.retrieve_data(f, 'REFF', np.array)

            KEXT = h5py_helper.retrieve_data(f, 'KEXT', np.array)      #(NWAVE,NSIZE)
            SGLALB = h5py_helper.retrieve_data(f, 'SGLALB', np.array)  #(NWAVE,NSIZE)

        #First we interpolate to the wavelengths in the Scatter class
        sext = interp1d(WAVE,KEXT,axis=0)
        KEXT1 = sext(Scatter.WAVE)
        salb = interp1d(WAVE,SGLALB,axis=0)
        SGLALB1 = salb(Scatter.WAVE)

        #Second we interpolate to the required particle size
        if rsize<REFF.min():
            rsize =REFF.min()
        if rsize>REFF.max():
            rsize=REFF.max()

        sext = interp1d(REFF,KEXT1,axis=1)
        KEXTX = sext(rsize)
        salb = interp1d(REFF,SGLALB1,axis=1)
        SGLALBX = salb(rsize)

        #Now check if we need to normalise the extinction coefficient
        if wavenorm==1:
            snorm = interp1d(Scatter.WAVE,KEXTX)
            vnorm = snorm(xwave)

            KEXTX[:] = KEXTX[:] / vnorm

        KSCAX = SGLALBX * KEXTX

        #Now we update the Scatter class with the required results
        Scatter.KEXT[:,idust] = KEXTX[:]
        Scatter.KSCA[:,idust] = KSCAX[:]
        Scatter.SGLALB[:,idust] = SGLALBX[:]

        f.close()

        if MakePlot==True:

            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6),sharex=True)

            for i in range(NSIZE):

                ax1.plot(WAVE,KEXT[:,i])
                ax2.plot(WAVE,SGLALB[:,i])

            ax1.plot(Scatter.WAVE,Scatter.KEXT[:,idust],c='black')
            ax2.plot(Scatter.WAVE,Scatter.SGLALB[:,idust],c='black')

            if Scatter.ISPACE==0:
                label='Wavenumber (cm$^{-1}$)'
            else:
                label=r'Wavelength ($\mu$m)'
            ax2.set_xlabel(label)
            ax1.set_xlabel('Extinction coefficient')
            ax2.set_xlabel('Single scattering albedo')

            ax1.set_facecolor('lightgray')
            ax2.set_facecolor('lightgray')

            plt.tight_layout()

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
        #******** model for retrieving an aerosol particle size distribution from a tabulated look-up table

        #This model changes the extinction coefficient of a given aerosol population based on 
        #the extinction coefficient look-up table stored in a separate file. 

        #The look-up table specifies the extinction coefficient as a function of particle size, and 
        #the parameter in the state vector is the particle size

        #The look-up table must have the format specified in Models/Models.py (model446)

        s = f.readline().split()
        aerosol_id = int(s[0])    #Aerosol population (from 0 to NDUST-1)
        wavenorm = int(s[1])      #If 1 - then the extinction coefficient will be normalised at a given wavelength

        xwave = 0.0
        if wavenorm==1:
            xwave = float(s[2])   #If 1 - wavelength at which to normalise the extinction coefficient

        varparam[0] = aerosol_id
        varparam[1] = wavenorm
        varparam[2] = xwave

        #Read the name of the look-up table file
        s = f.readline().split()
        fnamex = s[0]

        #Reading the particle size and its a priori error
        s = f.readline().split()
        lx[ix] = 0
        inum[ix] = 1
        x0[ix] = float(s[0])
        sx[ix,ix] = (float(s[1]))**2.

        ix = ix + 1

        return cls(ix_0, ix-ix_0, fnamex)


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
        #******** model for retrieving an aerosol particle size distribution from a tabulated look-up table

        #This model changes the extinction coefficient of a given aerosol population based on 
        #the extinction coefficient look-up table stored in a separate file. 

        #The look-up table specifies the extinction coefficient as a function of particle size, and 
        #the parameter in the state vector is the particle size

        #The look-up table must have the format specified in Models/Models.py (model446)
        
        _lgr.warn(f"{cls.__name__}.from_bookmark(...) only sets model parameters that have been stored in `varident`, `varparam`, and the state vector. Therefore it cannot return the original value for `fnamex` at the moment. Use with caution.")
        #aerosol_id = varparam[0]
        #wavenorm = varparam[1]
        #xwave = varparam[2]

        fnamex = ""   #This needs to be fixed!

        ix = ix + 1

        return cls(ix_0, ix-ix_0, fnamex)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 446. model for retrieving the particle size distribution based on the data in a look-up table
        #***************************************************************

        #This model fits the particle size distribution based on the optical properties at different sizes
        #tabulated in a pre-computed look-up table. What this model does is to interpolate the optical 
        #properties based on those tabulated.

        idust0 = int(forward_model.Variables.VARPARAM[ivar,0])
        wavenorm = int(forward_model.Variables.VARPARAM[ivar,1])
        xwave = forward_model.Variables.VARPARAM[ivar,2]
        rsize = forward_model.Variables.XN[ix]

        forward_model.ScatterX = self.calculate(forward_model.ScatterX,idust0,wavenorm,xwave,rsize,self.lookup_table_fpath,MakePlot=False)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


