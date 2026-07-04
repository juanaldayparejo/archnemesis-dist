
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase

from archnemesis.helpers.maths_helper import ngauss

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

class Model228(PreRTModelBase):
    """
        In this model, the wavelength calibration of a given spectrum is performed, as well as the fit
        of a double Gaussian ILS suitable for ACS MIR solar occultation observations

        The wavelength calibration is performed such that the first wavelength or wavenumber is given by V0. 
        Then the rest of the wavelengths of the next data points are calculated by calculating the wavelength
        step between data points given by dV = C0 + C1*data_number + C2*data_number, where data_number 
        is an array going from 0 to NCONV-1.

        The ILS is fit using the approach of Alday et al. (2019, A&A). In this approach, the parameters to fit
        the ILS are the Offset of the second gaussian with respect to the first one (P0), the FWHM of the main 
        gaussian (P1), Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber (P2)
        , Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (P3), and
        a linear variation of the relative amplitude.
    """
    id : int = 228


    @classmethod
    def calculate(cls, Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3,MakePlot=False):

        """
            FUNCTION NAME : model228()

            DESCRIPTION :

                Function defining the model parameterisation 228 in NEMESIS.

                In this model, the wavelength calibration of a given spectrum is performed, as well as the fit
                of a double Gaussian ILS suitable for ACS MIR solar occultation observations

                The wavelength calibration is performed such that the first wavelength or wavenumber is given by V0. 
                Then the rest of the wavelengths of the next data points are calculated by calculating the wavelength
                step between data points given by dV = C0 + C1*data_number + C2*data_number, where data_number 
                is an array going from 0 to NCONV-1.

                The ILS is fit using the approach of Alday et al. (2019, A&A). In this approach, the parameters to fit
                the ILS are the Offset of the second gaussian with respect to the first one (P0), the FWHM of the main 
                gaussian (P1), Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber (P2)
                , Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (P3), and
                a linear variation of the relative amplitude.

            INPUTS :

                Measurement :: Python class defining the Measurement
                Spectroscopy :: Python class defining the Spectroscopy
                V0 :: Wavelength/Wavenumber of the first data point
                C0,C1,C2 :: Coefficients to calculate the step size in wavelength/wavenumbers between data points
                P0,P1,P2,P3 :: Parameters used to define the double Gaussian ILS of ACS MIR

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated Measurement and Spectroscopy classes

            CALLING SEQUENCE:

                Measurement,Spectroscopy = model228(Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3)

            MODIFICATION HISTORY : Juan Alday (20/12/2021)

        """

        #1.: Defining the new wavelength array
        ##################################################

        nconv = Measurement.NCONV[0]
        vconv1 = np.zeros(nconv)
        vconv1[0] = V0

        xx = np.linspace(0,nconv-2,nconv-1)
        dV = C0 + C1*xx + C2*(xx)**2.

        for i in range(nconv-1):
            vconv1[i+1] = vconv1[i] + dV[i]

        for i in range(Measurement.NGEOM):
            Measurement.VCONV[0:Measurement.NCONV[i],i] = vconv1[:]

        #2.: Calculating the new ILS function based on the new convolution wavelengths
        ###################################################################################

        ng = 2 #Number of gaussians to include

        #Wavenumber offset of the two gaussians
        offset = np.zeros([nconv,ng])
        offset[:,0] = 0.0
        offset[:,1] = P0

        #FWHM for the two gaussians (assumed to be constant in wavelength, not in wavenumber)
        fwhm = np.zeros([nconv,ng])
        fwhml = P1 / vconv1[0]**2.0
        for i in range(nconv):
            fwhm[i,0] = fwhml * (vconv1[i])**2.
            fwhm[i,1] = fwhm[i,0]

        #Amplitde of the second gaussian with respect to the main one
        amp = np.zeros([nconv,ng])
        ampgrad = (P3 - P2)/(vconv1[nconv-1]-vconv1[0])
        for i in range(nconv):
            amp[i,0] = 1.0
            amp[i,1] = (vconv1[i] - vconv1[0]) * ampgrad + P2

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

        #3. Defining new calculations wavelengths and reading again lbl-tables in correct range
        ###########################################################################################

        #Spectroscopy.read_lls(Spectroscopy.RUNNAME)
        #Measurement.wavesetc(Spectroscopy,IGEOM=0)
        #Spectroscopy.read_tables(wavemin=Measurement.WAVE.min(),wavemax=Measurement.WAVE.max())

        return Measurement,Spectroscopy


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
        #******** model for retrieving the ILS and Wavelength calibration in ACS MIR solar occultation observations

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #V0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #C0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #C1
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #C2
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P0 - Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P1 - FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P2 - Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files   #P3 - Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
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
        #******** model for retrieving the ILS and Wavelength calibration in ACS MIR solar occultation observations
        ix = ix + 8

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 228. Retrieval of instrument line shape for ACS-MIR and wavelength calibration
        #**************************************************************************************

        V0 = forward_model.Variables.XN[ix]
        C0 = forward_model.Variables.XN[ix+1]
        C1 = forward_model.Variables.XN[ix+2]
        C2 = forward_model.Variables.XN[ix+3]
        P0 = forward_model.Variables.XN[ix+4]
        P1 = forward_model.Variables.XN[ix+5]
        P2 = forward_model.Variables.XN[ix+6]
        P3 = forward_model.Variables.XN[ix+7]

        forward_model.MeasurementX,forward_model.SpectroscopyX = self.calculate(forward_model.MeasurementX,forward_model.SpectroscopyX,V0,C0,C1,C2,P0,P1,P2,P3)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


