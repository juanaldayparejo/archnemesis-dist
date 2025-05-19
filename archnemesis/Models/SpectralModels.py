
from .ModelBase import *


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



class SpectralModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the calculated spectrum in the forward model.
    """
    name : str = 'name of spectral model should be overwritten in subclass'
    
    
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
    
    
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        """
        Updated values of components based upon values of model parameters in the state vector. Called from ForwardModel_0::subprofretg.
        """
        _lgr.info(f'Model id {self.id} method "calculate_from_subprofretg" does nothing...')
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        raise NotImplementedError(f'calculate_from_subspecret should be implemented for all Spectral models')


class Model231(SpectralModelBase):
    id : int = 231
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
            igeom_slices : tuple[slice,...],
            meas : "Measurement_0",
            NGEOM : int, 
            NDEGREE : int, 
            COEFF : np.ndarray[['NDEGREE+1','NGEOM'],float]
        ) -> tuple[np.ndarray[['NCONV','NGEOM'],float], np.ndarray[['NCONV','NGEOM','NX'],float]]:
        
        for i in range(meas.NGEOM):
            T = COEFF[i]

            WAVE0 = meas.VCONV[0,i]
            spec = np.array(SPECMOD[0:meas.NCONV[i],i])

            #Changing the state vector based on this parameterisation
            POL = np.zeros(meas.NCONV[i])
            for j in range(NDEGREE+1):
                POL[:] = POL[:] + T[j]*(meas.VCONV[0:meas.NCONV[i],i]-WAVE0)**j

            SPECMOD[0:meas.NCONV[i],i] *= POL[:]

            #Changing the rest of the gradients based on the impact of this parameterisation
            dSPECMOD[0:meas.NCONV[i],i,:] *= POL[:,None]

            #Defining the analytical gradients for this parameterisation
            dspecmod_part = dSPECMOD[0:meas.NCONV[i],i,igeom_slices[i]]
            for j in range(NDEGREE+1):
                dspecmod_part[:,j] = spec * (meas.VCONV[0:meas.NCONV[i],i]-WAVE0)**j

        return SPECMOD, dSPECMOD
    
    
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
        #******** multiplication of calculated spectrum by polynomial function (following polynomial of degree N)

        #The computed spectra is multiplied by R = R0 * POL
        #Where the polynomial function POL depends on the wavelength given by:
        # POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='int')
        nlevel = int(tmp[0])
        ndegree = int(tmp[1])
        varparam[0] = nlevel
        varparam[1] = ndegree
        for ilevel in range(nlevel):
            tmp = f1.readline().split()
            for ic in range(ndegree+1):
                r0 = float(tmp[2*ic])
                err0 = float(tmp[2*ic+1])
                x0[ix] = r0
                sx[ix,ix] = (err0)**2.
                inum[ix] = 0
                ix = ix + 1
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        NGEOM = int(forward_model.Variables.VARPARAM[ivar,0])
        NDEGREE = int(forward_model.Variables.VARPARAM[ivar,1])
        COEFF = np.array(forward_model.Variables.XN[ix : ix+(NDEGREE+1)*forward_model.Measurement.NGEOM]).reshape((NDEGREE+1,forward_model.Measurement.NGEOM))
        
        igeom_slices = tuple(slice(ix+(NDEGREE+1)*igeom, ix+(NDEGREE+1)*(igeom+1)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))
        
        SPECMOD[...], dSPECMOD[...] = self.calculate(SPECMOD, dSPECMOD, igeom_slices, forward_model.Measurement, NGEOM, NDEGREE, COEFF)


# [JD] This uses forward_model.SpectroscopyX, whereas Model231 uses forward_model.Measurement, is this correct?
class Model2310(SpectralModelBase):
    id : int = 2310
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
            igeom_slices : tuple[slice,...],
            Spectroscopy : "Spectroscopy_0",
            NGEOM : int, 
            NDEGREE : int, 
            COEFF : np.ndarray[['NDEGREE+1','NGEOM'],float],
            lowin : np.ndarray[['NWINDOWS'],float],
            hiwin : np.ndarray[['NWINDOWS'],float],
        ) -> tuple[np.ndarray[['NCONV','NGEOM'],float], np.ndarray[['NCONV','NGEOM','NX'],float]]:
        
        for IWIN in range(lowin.size):
            ivin = np.where( (Spectroscopy.WAVE>=lowin[IWIN]) & (Spectroscopy.WAVE<hiwin[IWIN]) )[0]
            nvin = len(ivin)
        
            for i in range(NGEOM):
                T = COEFF[i]

                WAVE0 = Spectroscopy.WAVE[ivin].min()
                spec = np.array(SPECMOD[ivin,i])

                #Changing the state vector based on this parameterisation
                POL = np.zeros(nvin)
                for j in range(NDEGREE+1):
                    POL[:] = POL[:] + T[j]*(Spectroscopy.WAVE[ivin]-WAVE0)**j

                SPECMOD[ivin,i] *=  POL[:]

                #Changing the rest of the gradients based on the impact of this parameterisation
                dSPECMOD[ivin,i,:] *= POL[:,None]
                

                #Defining the analytical gradients for this parameterisation
                dspecmod_part = dSPECMOD[ivin,i,igeom_slices[i]]
                for j in range(NDEGREE+1):
                    dspecmod_part[:,j] = spec[:] * (Spectroscopy.WAVE[ivin]-WAVE0)**j

        return SPECMOD, dSPECMOD
    
    
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
        """
        Continuum addition to transmission spectra using a varying scaling factor (following polynomial of degree N)
        in several spectral windows 

        The computed spectra is multiplied by R = R0 * (T0 + POL)
        Where the polynomial function POL depends on the wavelength given by:
        POL = A0 + A1*(WAVE-WAVE0) + A2*(WAVE-WAVE0)**2. + ...
        """
        
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=3,dtype='int')
        nlevel = int(tmp[0])
        ndegree = int(tmp[1])
        nwindows = int(tmp[2])
        varparam[0] = nlevel
        varparam[1] = ndegree
        varparam[2] = nwindows

        i0 = 0
        #Defining the boundaries of the spectral windows
        for iwin in range(nwindows):
            tmp = f1.readline().split()
            varparam[3+i0] = float(tmp[0])
            i0 = i0 + 1
            varparam[3+i0] = float(tmp[1])
            i0 = i0 + 1

        #Reading the coefficients for the polynomial in each geometry and spectral window
        for iwin in range(nwindows):
            for ilevel in range(nlevel):
                tmp = np.fromfile(f1,sep=' ',count=2*(ndegree+1),dtype='float')
                for ic in range(ndegree+1):
                    r0 = float(tmp[2*ic])
                    err0 = float(tmp[2*ic+1])
                    x0[ix] = r0
                    sx[ix,ix] = (err0)**2.
                    inum[ix] = 0
                    ix = ix + 1
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 2310. Scaling of spectra using a varying scaling factor (following a polynomial of degree N)
        in multiple spectral windows
        """

        #NGEOM = int(forward_model.Variables.VARPARAM[ivar,0])
        NGEOM = forward_model.MeasurementX.NGEOM
        NDEGREE = int(forward_model.Variables.VARPARAM[ivar,1])
        NWINDOWS = int(forward_model.Variables.VARPARAM[ivar,2])

        lowin = np.zeros(NWINDOWS)
        hiwin = np.zeros(NWINDOWS)
        i0 = 0
        for IWIN in range(NWINDOWS):
            lowin[IWIN] = float(forward_model.Variables.VARPARAM[ivar,3+i0])
            i0 = i0 + 1
            hiwin[IWIN] = float(forward_model.Variables.VARPARAM[ivar,3+i0])
            i0 = i0 + 1
        
        COEFF = np.array(forward_model.Variables.XN[ix : ix+(NDEGREE+1)*NGEOM]).reshape((NDEGREE+1,NGEOM))
        
        igeom_slices = tuple(slice(ix+(NDEGREE+1)*igeom, ix+(NDEGREE+1)*(igeom+1)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))

        SPECMOD[...], dSPECMOD[...] = self.calculate(
            SPECMOD, 
            dSPECMOD, 
            igeom_slices, 
            forward_model.SpectroscopyX,
            NGEOM, 
            NDEGREE, 
            COEFF,
            lowin,
            hiwin
        )


class Model232(SpectralModelBase):
    id : int = 232
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            igeom_slice : slice,
            Spectroscopy : "Spectroscopy_0",
            TAU0 : float,
            ALPHA : float,
            WAVE0 : float,
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        spec = np.array(SPECMOD)
        factor = np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) )

        #Changing the state vector based on this parameterisation
        SPECMOD *= factor

        #Changing the rest of the gradients based on the impact of this parameterisation
        dSPECMOD *= factor[:,None]

        #Defining the analytical gradients for this parameterisation
        dspecmod_part = SPECMOD[:,igeom_slice]
        dspecmod_part[:,0] = spec[:] * ( -((Spectroscopy.WAVE/WAVE0)**(-ALPHA)) * np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) ) )
        dspecmod_part[:,1] = spec[:] * TAU0 * np.exp ( -TAU0 * (Spectroscopy.WAVE/WAVE0)**(-ALPHA) ) * np.log(Spectroscopy.WAVE/WAVE0) * (Spectroscopy.WAVE/WAVE0)**(-ALPHA)

        return SPECMOD, dSPECMOD
    
    
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
        """
        Continuum addition to transmission spectra using the Angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """
        s = f.readline().split()
        wavenorm = float(s[0])                    

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
        nlevel = int(tmp[0])
        varparam[0] = nlevel
        varparam[1] = wavenorm
        for ilevel in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=4,dtype='float')
            r0 = float(tmp[0])   #Opacity level at wavenorm
            err0 = float(tmp[1])
            r1 = float(tmp[2])   #Angstrom coefficient
            err1 = float(tmp[3])
            x0[ix] = r0
            sx[ix,ix] = (err0)**2.
            x0[ix+1] = r1
            sx[ix+1,ix+1] = err1**2.
            inum[ix] = 0
            inum[ix+1] = 0                        
            ix = ix + 2
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 232. Continuum addition to transmission spectra using the angstrom coefficient

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( - TAU0 * (WAVE/WAVE0)**-ALPHA )
        Where the parameters to fit are TAU0 and ALPHA
        """

        #The effect of this model takes place after the computation of the spectra in CIRSrad!
        if int(forward_model.Variables.NXVAR[ivar]/2)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 232 :: The number of levels for the addition of continuum must be the same as NGEOM')

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(2), ix+(igeom+1)*(2)) for igeom, nconv in enumerate(forward_model.Measurement.NCONV))

        if NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):
                TAU0 = forward_model.Variables.XN[ix]
                ALPHA = forward_model.Variables.XN[ix+1]
                WAVE0 = forward_model.Variables.VARPARAM[ivar,1]
                
                SPECMOD[:,i], dSPECMOD[:,i] = self.calculate(
                    SPECMOD[:,i], 
                    dSPECMOD[:,i], 
                    igeom_slices[i], 
                    forward_model.SpectroscopyX,
                    TAU0,
                    ALPHA,
                    WAVE0
                )

        else:
            T0 = forward_model.Variables.XN[ix]
            ALPHA = forward_model.Variables.XN[ix+1]
            WAVE0 = forward_model.Variables.VARPARAM[ivar,1]
            
            _lgr.warning(f'It looks like there is no calculation for NGEOM=1 for model id = {self.id}')


class Model233(SpectralModelBase):
    id : int = 233
    
    
    @classmethod
    def calculate(
            cls, 
            SPECMOD : np.ndarray[['NCONV'],float],
            dSPECMOD : np.ndarray[['NCONV','NX'],float],
            igeom_slice : slice,
            Spectroscopy : "Spectroscopy_0",
            A0 : float,
            A1 : float,
            A2 : float,
        ) -> tuple[np.ndarray[['NCONV'],float], np.ndarray[['NCONV','NX'],float]]:
        
        spec = np.array(SPECMOD)

        #Calculating the aerosol opacity at each wavelength
        TAU = np.exp(A0 + A1 * np.log(Spectroscopy.WAVE) + A2 * np.log(Spectroscopy.WAVE)**2.)

        #Changing the state vector based on this parameterisation
        SPECMOD *= np.exp ( -TAU )

        #Changing the rest of the gradients based on the impact of this parameterisation
        dSPECMOD *= np.exp ( -TAU )
        
        #Defining the analytical gradients for this parameterisation
        dspecmod_part = SPECMOD[:,igeom_slice]
        dspecmod_part[:,0] = spec[:] * (-TAU) * np.exp(-TAU)
        dspecmod_part[:,1] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Spectroscopy.WAVE)
        dspecmod_part[:,2] = spec[:] * (-TAU) * np.exp(-TAU) * np.log(Spectroscopy.WAVE)**2.

        return SPECMOD, dSPECMOD
    
    
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
        """
        Aerosol opacity modelled with a variable angstrom coefficient. Applicable to transmission spectra.

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.                  
        """

        #Reading the file where the a priori parameters are stored
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=1,dtype='int')
        nlevel = int(tmp[0])
        varparam[0] = nlevel
        for ilevel in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=6,dtype='float')
            a0 = float(tmp[0])   #A0
            err0 = float(tmp[1])
            a1 = float(tmp[2])   #A1
            err1 = float(tmp[3])
            a2 = float(tmp[4])   #A2
            err2 = float(tmp[5])
            x0[ix] = a0
            sx[ix,ix] = (err0)**2.
            x0[ix+1] = a1
            sx[ix+1,ix+1] = err1**2.
            x0[ix+2] = a2
            sx[ix+2,ix+2] = err2**2.
            inum[ix] = 0
            inum[ix+1] = 0    
            inum[ix+2] = 0                  
            ix = ix + 3
        return cls(ix_0, ix-ix_0)
        
    
    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        """
        Model 232. Continuum addition to transmission spectra using a variable angstrom coefficient (Schuster et al., 2006 JGR)
        ***************************************************************

        The computed transmission spectra is multiplied by TRANS = TRANS0 * NP.EXP( -TAU_AERO )
        Where the aerosol opacity is modelled following

         np.log(TAU_AERO) = a0 + a1 * np.log(WAVE) + a2 * np.log(WAVE)**2.

        The coefficient a2 accounts for a curvature in the angstrom coefficient used in model 232. Note that model
        233 converges to model 232 when a2=0.

        The effect of this model takes place after the computation of the spectra in CIRSrad!
        """
        
        if int(forward_model.Variables.NXVAR[ivar]/3)!=forward_model.MeasurementX.NGEOM:
            raise ValueError('error using Model 233 :: The number of levels for the addition of continuum must be the same as NGEOM')

        NGEOM = forward_model.MeasurementX.NGEOM
        igeom_slices = tuple(slice(ix+igeom*(3), ix+(igeom+1)*(3)) for igeom in range(NGEOM))


        if forward_model.MeasurementX.NGEOM>1:
            for i in range(forward_model.MeasurementX.NGEOM):

                A0 = forward_model.Variables.XN[ix]
                A1 = forward_model.Variables.XN[ix+1]
                A2 = forward_model.Variables.XN[ix+2]
                
                SPECMOD[:,i], dSPECMOD[:,i] = self.calculate(
                    SPECMOD[:,i], 
                    dSPECMOD[:,i],
                    igeom_slices[i],
                    forward_model.SpectroscopyX,
                    A0,
                    A1,
                    A2
                )

        else:
            A0 = forward_model.Variables.XN[ix]
            A1 = forward_model.Variables.XN[ix+1]
            A2 = forward_model.Variables.XN[ix+2]

            SPECMOD[:], dSPECMOD[:] = self.calculate(
                SPECMOD[:], 
                dSPECMOD[:],
                slice(ix,ix+3),
                forward_model.SpectroscopyX,
                A0,
                A1,
                A2
            )


class Model667(SpectralModelBase):
    id : int = 667


    @classmethod
    def calculate(cls, Spectrum,xfactor,MakePlot=False):

        """
            FUNCTION NAME : model667()

            DESCRIPTION :

                Function defining the model parameterisation 667 in NEMESIS.
                In this model, the output spectrum is scaled using a dillusion factor to account
                for strong temperature gradients in exoplanets

            INPUTS :

                Spectrum :: Modelled spectrum 
                xfactor :: Dillusion factor

            OPTIONAL INPUTS: None

            OUTPUTS :

                Spectrum :: Modelled spectrum scaled by the dillusion factor

            CALLING SEQUENCE:

                Spectrum = model667(Spectrum,xfactor)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        Spectrum = Spectrum * xfactor

        return Spectrum


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
        #******** dilution factor to account for thermal gradients thorughout exoplanet
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xfac = float(tmp[0])
        xfacerr = float(tmp[1])
        x0[ix] = xfac
        inum[ix] = 0 
        sx[ix,ix] = xfacerr**2.
        ix = ix + 1

        return cls(ix_0, ix-ix_0)


    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        #Model 667. Spectrum scaled by dilution factor to account for thermal gradients in planets
        #**********************************************************************************************

        xfactor = forward_model.Variables.XN[ix]
        spec = np.zeros(forward_model.SpectroscopyX.NWAVE)
        spec[:] = SPECMOD
        SPECMOD = self.calculate(SPECMOD,xfactor)
        dSPECMOD = dSPECMOD * xfactor
        dSPECMOD[:,ix] = spec[:]
        ix = ix + 1

