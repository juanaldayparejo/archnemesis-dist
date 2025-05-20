from __future__ import annotations #  for 3.9 compatability

from .ModelBase import *


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



class NonAtmosphericModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with anything but the Atmosphere component.
    """
    name : str = 'name of non-atmospheric model should be overwritten in subclass'
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        return varident[0]==cls.id
    
    
    def patch_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        """
        Patches values of components based upon values of model parameters in the state vector. Called from ForwardModel_0::subprofretg.
        """
        _lgr.info(f'Model id {self.id} method "patch_from_subprofretg" does nothing...')
    
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Updated spectra based upon values of model parameters in the state vector. Called from ForwardModel_0::subspecret.
        """
        _lgr.info(f'Model id {self.id} method "calculate_from_subspecret" does nothing...')
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    @abc.abstractmethod
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError(f'calculate_from_subprofretg must be implemented for all NonAtmospheric models')


class Model228(NonAtmosphericModelBase):
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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** model for retrieving the ILS and Wavelength calibration in ACS MIR solar occultation observations

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #V0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C0
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C1
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #C2
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P0 - Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P1 - FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P2 - Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #P3 - Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
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

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model229(NonAtmosphericModelBase):
    id : int = 229


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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** model for retrieving the ILS in ACS MIR solar occultation observations

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at wavenumber in the middle
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #wavenumber offset at highest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Offset of the second gaussian with respect to the first one (assumed spectrally constant)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #FWHM of the main gaussian (assumed to be constant in wavelength units)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at lowest wavenumber
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #Relative amplitude of the second gaussian with respect to the gaussian at highest wavenumber (linear variation)
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 0
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

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model230(NonAtmosphericModelBase):
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

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model444(NonAtmosphericModelBase):
    """
        Allows for retrieval of the particle size distribution and imaginary refractive index.
    """
    
    id : int = 444


    def __init__(
            self, 
            i_state_vector_start : int, 
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
        super().__init__(i_state_vector_start, n_state_vector_entries)
        
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


class Model446(NonAtmosphericModelBase):
    id : int = 446

    def __init__(
            self, 
            i_state_vector_start : int, 
            n_state_vector_entries : int,
            lookup_table_fpath : str,
        ):
        """
        Initialise an instance of the model.
        
        ## ARGUMENTS ##
            
            i_state_vector_start : int
                The index of the first entry of the model parameters in the state vector
            
            n_state_vector_entries : int
                The number of model parameters that are stored in the state vector
            
            lookup_table_fpath: str
                path to the lookup table of extinction coefficient vs particle size
                
        
        ## RETURNS ##
            An initialised instance of this object
        """
        super().__init__(i_state_vector_start, n_state_vector_entries)
        
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
        f = h5py.File(lookupfile,'r')

        NWAVE = np.int32(f.get('NWAVE'))
        NSIZE = np.int32(f.get('NSIZE'))

        WAVE = np.array(f.get('WAVE'))
        REFF = np.array(f.get('REFF'))

        KEXT = np.array(f.get('KEXT'))      #(NWAVE,NSIZE)
        SGLALB = np.array(f.get('SGLALB'))  #(NWAVE,NSIZE)

        f.close()

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

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model447(NonAtmosphericModelBase):
    id : int = 447


    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        _lgr.warning(f'Model id = {cls.id} is not implemented yet, so it will never be chosen as a valid model')
        return False


    @classmethod
    def calculate(cls, Measurement,v_doppler):

        """
            FUNCTION NAME : model447()

            DESCRIPTION :

                Function defining the model parameterisation 447 in NEMESIS.
                In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
                is common to all geometries, but in the future it will be updated so that each measurement
                can have a different Doppler velocity (in order to retrieve wind speeds).

            INPUTS :

                Measurement :: Python class defining the measurement
                v_doppler :: Doppler velocity (km/s)

            OPTIONAL INPUTS: none

            OUTPUTS :

                Measurement :: Updated measurement class with the correct Doppler velocity

            CALLING SEQUENCE:

                Measurement = model447(Measurement,v_doppler)

            MODIFICATION HISTORY : Juan Alday (25/07/2023)

        """

        Measurement.V_DOPPLER = v_doppler

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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** model for retrieving the Doppler shift

        #Read the Doppler velocity and its uncertainty
        s = f.readline().split()
        v_doppler = float(s[0])     #km/s
        v_doppler_err = float(s[1]) #km/s

        #Filling the state vector and a priori covariance matrix with the doppler velocity
        lx[ix] = 0
        x0[ix] = v_doppler
        sx[ix,ix] = (v_doppler_err)**2.
        inum[ix] = 1

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
        raise NotImplementedError


class Model500(NonAtmosphericModelBase):
    id : int = 500


    @classmethod
    def calculate(cls, k_cia, waven, icia, vlo, vhi, nbasis, amplitudes):
        """
            FUNCTION NAME : model500()

            DESCRIPTION :

                Function defining the model parameterisation 500.
                This allows the retrieval of CIA opacity with a gaussian basis.
                Assumes a constant P/T dependence.

            INPUTS :

                cia :: CIA class

                icia :: CIA pair to be modelled

                vlo :: Lower wavenumber bound

                vhi :: Upper wavenumber bound

                nbasis :: Number of gaussians in the basis

                amplitudes :: Amplitudes of each gaussian


            OUTPUTS :

                cia :: Updated CIA class
                xmap :: Gradient (not implemented)

            CALLING SEQUENCE:

                cia,xmap = model500(cia, icia, nbasis, amplitudes)

            MODIFICATION HISTORY : Joe Penn (14/01/25)

        """

        ilo = np.argmin(np.abs(waven-vlo))
        ihi = np.argmin(np.abs(waven-vhi))
        width = (ihi - ilo)/nbasis          # Width of the Gaussian functions
        centers = np.linspace(ilo, ihi, int(nbasis))

        def gaussian_basis(x, centers, width):
            return np.exp(-((x[:, None] - centers[None, :])**2) / (2 * width**2))

        x = np.arange(ilo,ihi+1)

        G = gaussian_basis(x, centers, width)
        gaussian_cia = G @ amplitudes

        k_cia = k_cia * 0

        k_cia[icia,:,:,ilo:ihi+1] = gaussian_cia

        xmap = np.zeros(1)
        return k_cia,xmap


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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix

        s = f.readline().split()
        amp_f = open(s[0],'r')

        tmp = np.fromfile(amp_f,sep=' ',count=2,dtype='float')

        nbasis = int(tmp[0])
        clen = float(tmp[1])

        amp = np.zeros([nbasis])
        eamp = np.zeros([nbasis])

        for j in range(nbasis):
            tmp = np.fromfile(amp_f,sep=' ',count=2,dtype='float')
            amp[j] = float(tmp[0])
            eamp[j] = float(tmp[1])

            lx[ix+j] = 1
            x0[ix+j] = np.log(amp[j])
            sx[ix+j,ix+j] = ( eamp[j]/amp[j]  )**2.

        for j in range(nbasis):
            for k in range(nbasis):

                deli = j-k
                arg = abs(deli/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        varparam[0] = nbasis
        ix = ix + nbasis

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:

        icia = forward_model.Variables.VARIDENT[ivar,1]

        if forward_model.Measurement.ISPACE == WaveUnit.Wavelength_um:
            vlo = 1e4/(forward_model.SpectroscopyX.WAVE.max())
            vhi = 1e4/(forward_model.SpectroscopyX.WAVE.min())
        else:
            vlo = forward_model.SpectroscopyX.WAVE.min()
            vhi = forward_model.SpectroscopyX.WAVE.max()

        nbasis = forward_model.Variables.VARPARAM[ivar,0]
        amplitudes = np.exp(forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]])*1e-40

        new_k_cia, xmap1 = self.calculate(forward_model.CIA.K_CIA.copy(), forward_model.CIA.WAVEN, icia, vlo, vhi, nbasis, amplitudes)

        forward_model.CIA.K_CIA = new_k_cia
        forward_model.CIAX.K_CIA = new_k_cia

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model777(NonAtmosphericModelBase):
    id : int = 777


    @classmethod
    def calculate(cls, Measurement,hcorr,MakePlot=False):

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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** tangent height correction
        s = f.readline().split()
        hcorr = float(s[0])
        herr = float(s[1])

        x0[ix] = hcorr
        sx[ix,ix] = herr**2.
        inum[ix] = 1

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
        #Model 777. Retrieval of tangent height corrections
        #***************************************************************

        hcorr = forward_model.Variables.XN[ix]

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,hcorr)

        ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model887(NonAtmosphericModelBase):
    id : int = 887


    @classmethod
    def calculate(cls, Scatter,xsc,idust,MakePlot=False):

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
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        ix_0 = ix
        #******** Cloud x-section spectrum

        #Read in number of points, cloud id, and correlation between elements.
        s = f.readline().split()
        nwv = int(s[0]) #number of spectral points (must be the same as in .xsc)
        icloud = int(s[1])  #aerosol ID
        clen = float(s[2])  #Correlation length (in wavelengths/wavenumbers)

        varparam[0] = nwv
        varparam[1] = icloud

        #Read the wavelengths and the extinction cross-section value and error
        wv = np.zeros(nwv)
        xsc = np.zeros(nwv)
        err = np.zeros(nwv)
        for iw in range(nwv):
            s = f.readline().split()
            wv[iw] = float(s[0])
            xsc[iw] = float(s[1])
            err[iw] = float(s[2])
            if xsc[iw]<=0.0:
                raise ValueError('error in read_apr :: Cross-section in model 887 must be greater than 0')

        #It is important to check that the wavelengths in .apr and in .xsc are the same
        Aero0 = Scatter_0()
        Aero0.read_xsc(runname)
        for iw in range(Aero0.NWAVE):
            if (wv[iw]-Aero0.WAVE[iw])>0.01:
                raise ValueError('error in read_apr :: Number of wavelengths in model 887 must be the same as in .xsc')

        #Including the parameters in state vector and covariance matrix
        for j in range(nwv):
            x0[ix+j] = np.log(xsc[j])
            lx[ix+j] = 1
            inum[ix+j] = 1
            sx[ix+j,ix+j] = (err[j]/xsc[j])**2.

        for j in range(nwv):
            for k in range(nwv):
                delv = wv[j] - wv[k]
                arg = abs(delv/clen)
                xfac = np.exp(-arg)
                if xfac>0.001:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        jxsc = ix

        ix = ix + nwv

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError

