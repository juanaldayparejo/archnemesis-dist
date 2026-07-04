
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
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

class Model230(PreRTModelBase):
    """
        In this model, the ILS of the measurement is defined from every convolution wavenumber
        using the double-Gaussian parameterisation created for analysing ACS MIR spectra.
        However, we can define several spectral windows where the ILS is different
    """
    id : int = 230


    @classmethod
    def calculate(cls, Measurement,nwindows,liml,limh,par,MakePlot=False):

        """
            FUNCTION NAME : model230()

            DESCRIPTION :

                Function defining the model parameterisation 230 in NEMESIS.
                In this model, the ILS of the measurement is defined from every convolution wavenumber
                using the double-Gaussian parameterisation created for analysing ACS MIR spectra.
                However, we can define several spectral windows where the ILS is different

            INPUTS :

                Measurement :: Python class defining the Measurement
                nwindows :: Number of spectral windows in which to fit the ILS
                liml(nwindows) :: Low wavenumber limit of each spectral window
                limh(nwindows) :: High wavenumber limit of each spectral window
                par(0,nwindows) :: Wavenumber offset of main at lowest wavenumber for each window
                par(1,nwindows) :: Wavenumber offset of main at wavenumber in the middle for each window
                par(2,nwindows) :: Wavenumber offset of main at highest wavenumber for each window
                par(3,nwindows) :: Offset of the second gaussian with respect to the first one (assumed spectrally constant) for each window
                par(4,nwindows) :: FWHM of the main gaussian at lowest wavenumber (assumed to be constat in wavelength units) for each window
                par(5,nwindows) :: Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber for each window
                par(6,nwindows) :: Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear var) for each window

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated Measurement class

            CALLING SEQUENCE:

                Measurement = model230(Measurement,nwindows,liml,limh,par)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        #Calculating the parameters for each spectral point
        nconv = Measurement.NCONV[0]
        vconv2 = Measurement.VCONV[0:nconv,0]
        ng = 2


        nfil2 = np.zeros(nconv,dtype='int32')
        mfil2 = 200
        vfil2 = np.zeros([mfil2,nconv])
        afil2 = np.zeros([mfil2,nconv])

        ivtot = 0
        for iwindow in range(nwindows):

            #Calculating the wavenumbers at which each spectral window applies
            ivwin = np.where( (vconv2>=liml[iwindow]) & (vconv2<=limh[iwindow]) )
            ivwin = ivwin[0]

            vconv1 = vconv2[ivwin]
            nconv1 = len(ivwin)


            par1 = par[0,iwindow]
            par2 = par[1,iwindow]
            par3 = par[2,iwindow]
            par4 = par[3,iwindow]
            par5 = par[4,iwindow]
            par6 = par[5,iwindow]
            par7 = par[6,iwindow]

            # 1. Wavenumber offset of the two gaussians
            #    We divide it in two sections with linear polynomials     
            iconvmid = int(nconv1/2.)
            wavemax = vconv1[nconv1-1]
            wavemin = vconv1[0]
            wavemid = vconv1[iconvmid]
            offgrad1 = (par2 - par1)/(wavemid-wavemin)
            offgrad2 = (par2 - par3)/(wavemid-wavemax)
            offset = np.zeros([nconv,ng])
            for i in range(iconvmid):
                offset[i,0] = (vconv1[i] - wavemin) * offgrad1 + par1
                offset[i,1] = offset[i,0] + par4
            for i in range(nconv1-iconvmid):
                offset[i+iconvmid,0] = (vconv1[i+iconvmid] - wavemax) * offgrad2 + par3
                offset[i+iconvmid,1] = offset[i+iconvmid,0] + par4

            # 2. FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
            fwhm = np.zeros([nconv1,ng])
            fwhml = par5 / wavemin**2.0
            for i in range(nconv1):
                fwhm[i,0] = fwhml * (vconv1[i])**2.
                fwhm[i,1] = fwhm[i,0]

            # 3. Amplitde of the second gaussian with respect to the main one
            amp = np.zeros([nconv1,ng])
            ampgrad = (par7 - par6)/(wavemax-wavemin)
            for i in range(nconv1):
                amp[i,0] = 1.0
                amp[i,1] = (vconv1[i] - wavemin) * ampgrad + par6


            #Running for each spectral point
            nfil = np.zeros(nconv1,dtype='int32')
            mfil1 = 200
            vfil1 = np.zeros([mfil1,nconv1])
            afil1 = np.zeros([mfil1,nconv1])
            for i in range(nconv1):

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



            nfil2[ivtot:ivtot+nconv1] = nfil[:]
            vfil2[0:mfil1,ivtot:ivtot+nconv1] = vfil1[0:mfil1,:]
            afil2[0:mfil1,ivtot:ivtot+nconv1] = afil1[0:mfil1,:]

            ivtot = ivtot + nconv1

        if ivtot!=nconv:
            raise ValueError('error in model 230 :: The spectral windows must cover the whole measured spectral range')

        mfil = nfil2.max()
        vfil = np.zeros([mfil,nconv])
        afil = np.zeros([mfil,nconv])
        for i in range(nconv):
            vfil[0:nfil2[i],i] = vfil2[0:nfil2[i],i]
            afil[0:nfil2[i],i] = afil2[0:nfil2[i],i]

        Measurement.NFIL = nfil2
        Measurement.VFIL = vfil
        Measurement.AFIL = afil

        if MakePlot==True:

            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))

            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax1.set_ylabel(r'f($\nu$)')
            ax1.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()

            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax2.set_ylabel(r'f($\nu$)')
            ax2.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()

            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil2[ix],ix],afil[0:nfil2[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)')
            ax3.set_ylabel(r'f($\nu$)')
            ax3.set_xlim([vfil[0:nfil2[ix],ix].min(),vfil[0:nfil2[ix],ix].max()])
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
        #******** model for retrieving multiple ILS (different spectral windows) in ACS MIR solar occultation observations

        s = f.readline().split()
        f1 = open(s[0],'r')
        s = f1.readline().split()
        nwindows = int(s[0])
        varparam[0] = nwindows
        liml = np.zeros(nwindows)
        limh = np.zeros(nwindows)
        for iwin in range(nwindows):
            s = f1.readline().split()
            liml[iwin] = float(s[0])
            limh[iwin] = float(s[1])
            varparam[2*iwin+1] = liml[iwin]
            varparam[2*iwin+2] = limh[iwin]

        par = np.zeros((7,nwindows))
        parerr = np.zeros((7,nwindows))
        for iw in range(nwindows):
            for j in range(7):
                s = f1.readline().split()
                par[j,iw] = float(s[0])
                parerr[j,iw] = float(s[1])
                x0[ix] = par[j,iw]
                sx[ix,ix] = (parerr[j,iw])**2.
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
        #******** model for retrieving multiple ILS (different spectral windows) in ACS MIR solar occultation observations
        nwindows = varparam[0]
        for iw in range(nwindows):
            for j in range(7):
                ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 230. Retrieval of multiple instrument line shapes for ACS-MIR
        #***************************************************************

        nwindows = int(forward_model.Variables.VARPARAM[ivar,0])
        liml = np.zeros(nwindows)
        limh = np.zeros(nwindows)
        i0 = 1
        for iwin in range(nwindows):
            liml[iwin] = forward_model.Variables.VARPARAM[ivar,i0]
            limh[iwin] = forward_model.Variables.VARPARAM[ivar,i0+1]
            i0 = i0 + 2

        par1 = np.zeros((7,nwindows))
        for iwin in range(nwindows):
            for jwin in range(7):
                par1[jwin,iwin] = forward_model.Variables.XN[ix]
                ix = ix + 1

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,nwindows,liml,limh,par1)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


