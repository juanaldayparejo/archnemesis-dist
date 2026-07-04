
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

from archnemesis.helpers.maths_helper import ngauss

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

class Model229(PreRTModelBase):
    """
        Model for representing the double-Gaussian parameterisation of the instrument lineshape for
        retrievals from the Atmospheric Chemistry Suite aboard the ExoMars Trace Gas Orbiter
    """
    id : int = 229
    
    def __init__(
            self, 
            state_vector_start : int = 0, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int = 7,
            #   Number of parameters for this model stored in the state vector
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('A0', slice(0,1), 'Wavenumber offset of main at lowest wavenumber','cm-1'),
            ModelParameter('A1', slice(1,2), 'Wavenumber offset of main at wavenumber in the middle','cm-1'),
            ModelParameter('A2', slice(2,3), 'Wavenumber offset of main at highest wavenumber','cm-1'),
            ModelParameter('DELDG', slice(3,4), 'Offset of the second gaussian with respect to the first one (assumed spectrally constant)','cm-1'),
            ModelParameter('FWHM', slice(4,5), 'FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units)','cm-1'),
            ModelParameter('AMP1', slice(5,6), 'Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber'),
            ModelParameter('AMP2', slice(6,7), 'Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var)'),
        )


    @classmethod
    def calculate(cls, Measurement,par1,par2,par3,par4,par5,par6,par7,MakePlot=False):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 229 in NEMESIS.
                In this model, the ILS of the measurement is defined from every convolution wavenumber
                using the double-Gaussian parameterisation created for analysing ACS MIR spectra

            INPUTS :

                Measurement :: Python class defining the Measurement
                par1 :: Wavenumber offset of main at lowest wavenumber
                par2 :: Wavenumber offset of main at wavenumber in the middle
                par3 :: Wavenumber offset of main at highest wavenumber 
                par4 :: Offset of the second gaussian with respect to the first one (assumed spectrally constant)
                par5 :: FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units)
                par6 :: Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
                par7 :: Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                Updated Measurement class

            CALLING SEQUENCE:

                Measurement = model229(Measurement,par1,par2,par3,par4,par5,par6,par7)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        #Calculating the parameters for each spectral point
        nconv = Measurement.NCONV[0]
        vconv1 = Measurement.VCONV[0:nconv,0]
        ng = 2

        # 1. Wavenumber offset of the two gaussians
        #    We divide it in two sections with linear polynomials     
        iconvmid = int(nconv/2.)
        wavemax = vconv1[nconv-1]
        wavemin = vconv1[0]
        wavemid = vconv1[iconvmid]
        offgrad1 = (par2 - par1)/(wavemid-wavemin)
        offgrad2 = (par2 - par3)/(wavemid-wavemax)
        offset = np.zeros([nconv,ng])
        for i in range(iconvmid):
            offset[i,0] = (vconv1[i] - wavemin) * offgrad1 + par1
            offset[i,1] = offset[i,0] + par4
        for i in range(nconv-iconvmid):
            offset[i+iconvmid,0] = (vconv1[i+iconvmid] - wavemax) * offgrad2 + par3
            offset[i+iconvmid,1] = offset[i+iconvmid,0] + par4

        # 2. FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
        fwhm = np.zeros([nconv,ng])
        fwhml = par5 / wavemin**2.0
        for i in range(nconv):
            fwhm[i,0] = fwhml * (vconv1[i])**2.
            fwhm[i,1] = fwhm[i,0]

        # 3. Amplitde of the second gaussian with respect to the main one
        amp = np.zeros([nconv,ng])
        ampgrad = (par7 - par6)/(wavemax-wavemin)
        for i in range(nconv):
            amp[i,0] = 1.0
            amp[i,1] = (vconv1[i] - wavemin) * ampgrad + par6

        #Running for each spectral point
        nfil = np.zeros(nconv,dtype='int32')
        mfil1 = 200
        vfil1 = np.zeros([mfil1,nconv])
        afil1 = np.zeros([mfil1,nconv])
        for i in range(nconv):

            #determining the lowest and highest wavenumbers to calculate
            xlim = 0.0
            xdist = 5.0 
            for j in range(ng):
                xcen = offset[i,j]
                xmin = abs(xcen - xdist*fwhm[i,j]/2.)
                if xmin > xlim:
                    xlim = xmin
                xmax = abs(xcen + xdist*fwhm[i,j]/2.)
                if xmax > xlim:
                    xlim = xmax

            #determining the wavenumber spacing we need to sample properly the gaussians
            xsamp = 7.0   #number of points we require to sample one HWHM 
            xhwhm = 10000.0
            for j in range(ng):
                xhwhmx = fwhm[i,j]/2. 
                if xhwhmx < xhwhm:
                    xhwhm = xhwhmx
            deltawave = xhwhm/xsamp
            np1 = 2.0 * xlim / deltawave
            npx = int(np1) + 1

            #Calculating the ILS in this spectral point
            iamp = np.zeros([ng])
            imean = np.zeros([ng])
            ifwhm = np.zeros([ng])
            fun = np.zeros([npx])
            xwave = np.linspace(vconv1[i]-deltawave*(npx-1)/2.,vconv1[i]+deltawave*(npx-1)/2.,npx)        
            for j in range(ng):
                iamp[j] = amp[i,j]
                imean[j] = offset[i,j] + vconv1[i]
                ifwhm[j] = fwhm[i,j]

            fun = ngauss(npx,xwave,ng,iamp,imean,ifwhm)  
            nfil[i] = npx
            vfil1[0:nfil[i],i] = xwave[:]
            afil1[0:nfil[i],i] = fun[:]

        mfil = nfil.max()
        vfil = np.zeros([mfil,nconv])
        afil = np.zeros([mfil,nconv])
        for i in range(nconv):
            vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
            afil[0:nfil[i],i] = afil1[0:nfil[i],i]

        Measurement.NFIL = nfil
        Measurement.VFIL = vfil
        Measurement.AFIL = afil

        if MakePlot==True:

            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))

            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax1.set_ylabel(r'f($\nu$)')
            ax1.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()

            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax2.set_ylabel(r'f($\nu$)')
            ax2.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()

            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax3.set_ylabel(r'f($\nu$)')
            ax3.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()

            plt.tight_layout()
            plt.show()

        return Measurement


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
        #******** model for retrieving the ILS in ACS MIR solar occultation observations

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #wavenumber offset at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #wavenumber offset at wavenumber in the middle
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #wavenumber offset at highest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        return cls(ix_0, ix-ix_0)

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
        #******** model for retrieving the ILS in ACS MIR solar occultation observations
        ix = ix + 7

        return cls(ix_0, ix-ix_0)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
        #***************************************************************

        par1 = forward_model.Variables.XN[ix]
        par2 = forward_model.Variables.XN[ix+1]
        par3 = forward_model.Variables.XN[ix+2]
        par4 = forward_model.Variables.XN[ix+3]
        par5 = forward_model.Variables.XN[ix+4]
        par6 = forward_model.Variables.XN[ix+5]
        par7 = forward_model.Variables.XN[ix+6]

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,par1,par2,par3,par4,par5,par6,par7)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


