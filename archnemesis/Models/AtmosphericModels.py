
from .ModelBase import *


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



class AtmosphericModelBase(ModelBase):
    """
    Abstract base class of all parameterised models used by ArchNemesis that interact with the Atmosphere component.
    """
    
    name : str = 'name of atmospheric model should be overwritten in subclass'
    
    
    @classmethod
    def is_varident_valid(
            cls,
            varident : np.ndarray[[3],int],
        ) -> bool:
        return varident[2]==cls.id
    
    
    ## Abstract methods below this line, subclasses must implement all of these methods ##
    
    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError(f'calculate_from_subprofretg should be implemented for all Atmospheric models')


class Modelm1(AtmosphericModelBase):
    id : int = -1


    @classmethod
    def calculate(cls, atm,ipar,xprof):
        """
        FUNCTION NAME : modelm1()

        DESCRIPTION :

            Function defining the model parameterisation -1 in NEMESIS.
            In this model, the aerosol profiles is modelled as a continuous profile in units
            of particles perModelm1 gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
            are in particles per gram of atmosphere

        INPUTS :

            atm :: Python class defining the atmosphere

            ipar :: Atmospheric parameter to be changed
                    (0 to NVMR-1) :: Gas VMR
                    (NVMR) :: Temperature
                    (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                    (NVMR+NDUST) :: Para-H2
                    (NVMR+NDUST+1) :: Fractional cloud coverage

            xprof(npro) :: Atmospheric aerosol profile in particles/cm3

        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated

        OUTPUTS :

            atm :: Updated atmospheric class
            xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                elements in state vector

        CALLING SEQUENCE:

            atm,xmap = modelm1(atm,ipar,xprof)

        MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model -1 :: Number of levels in atmosphere does not match and profile')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([npro,npar,npro])

        if ipar<atm.NVMR:  #Gas VMR
            raise ValueError('error :: Model -1 is just compatible with aerosol populations (Gas VMR given)')
        elif ipar==atm.NVMR: #Temperature
            raise ValueError('error :: Model -1 is just compatible with aerosol populations (Temperature given)')
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            x1 = np.exp(xprof)
            if jtmp<atm.NDUST:
                atm.DUST_UNITS_FLAG[jtmp] = -1
                atm.DUST[:,jtmp] = x1 #* 1000. * rho
            elif jtmp==atm.NDUST:
                raise ValueError('error :: Model -1 is just compatible with aerosol populations')
            elif jtmp==atm.NDUST+1:
                raise ValueError('error :: Model -1 is just compatible with aerosol populations')

        for j in range(npro):
            xmap[0:npro,ipar,j] = x1[:] #* 1000. * rho
        return atm, xmap


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
        #* continuous cloud, but cloud retrieved as particles/cm3 rather than
        #* particles per gram to decouple it from pressure.
        #********* continuous particles/cm3 profile ************************
        if varident[0] >= 0:
            raise ValueError('error in read_apr_nemesis :: model -1 type is only for use with aerosols')

        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref = np.zeros([nlevel])
        ref = np.zeros([nlevel])
        eref = np.zeros([nlevel])
        for j in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
            pref[j] = float(tmp[0])
            ref[j] = float(tmp[1])
            eref[j] = float(tmp[2])

            lx[ix+j] = 1
            x0[ix+j] = np.log(ref[j])
            sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        f1.close()

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]
        ix = ix + nlevel

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        jtmp = ipar - (forward_model.AtmosphereX.NVMR+1)
        if forward_model.Variables.VARPARAM[ivar,0]\
            and ipar > forward_model.AtmosphereX.NVMR\
            and jtmp < forward_model.AtmosphereX.NDUST: # Fortran true so flip aerosol model

            forward_model.AtmosphereX,xmap1 = Model0.calculate(forward_model.AtmosphereX,ipar,xprof)
        else:
            forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)

        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


    def patch_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model0(AtmosphericModelBase):
    id : int = 0


    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 0 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                in which each element of the state vector corresponds to the atmospheric profile 
                at each altitude level

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                xprof(npro) :: Atmospheric profile

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model0(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 0 :: Number of levels in atmosphere does not match and profile')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([npro,npar,npro])

        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            x1 = np.exp(xprof)
            vmr = np.zeros([atm.NP,atm.NVMR])
            vmr[:,:] = atm.VMR
            vmr[:,jvmr] = x1
            atm.edit_VMR(vmr)
            for j in range(npro):
                xmap[j,ipar,j] = x1[j]
        elif ipar==atm.NVMR: #Temperature
            x1 = xprof
            atm.edit_T(x1)
            for j in range(npro):
                xmap[j,ipar,j] = 1.
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            x1 = np.exp(xprof)
            if jtmp<atm.NDUST: #Dust in m-3
                dust = np.zeros([atm.NP,atm.NDUST])
                dust[:,:] = atm.DUST
                dust[:,jtmp] = x1
                atm.edit_DUST(dust)
                for j in range(npro):
                    xmap[j,ipar,j] = x1[j]
            elif jtmp==atm.NDUST:
                atm.PARAH2 = x1
                for j in range(npro):
                    xmap[j,ipar,j] = x1[j]
            elif jtmp==atm.NDUST+1:
                atm.FRAC = x1
                for j in range(npro):
                    xmap[j,ipar,j] = x1[j]

        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #********* continuous profile ************************
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref = np.zeros([nlevel])
        ref = np.zeros([nlevel])
        eref = np.zeros([nlevel])
        for j in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
            pref[j] = float(tmp[0])
            ref[j] = float(tmp[1])
            eref[j] = float(tmp[2])
        f1.close()

        if varident[0] == 0:  # *** temperature, leave alone ****
            x0[ix:ix+nlevel] = ref[:]
            for j in range(nlevel):
                sx[ix+j,ix+j] = eref[j]**2.
                if varident[1] == -1: #Gradients computed numerically
                    inum[ix+j] = 1

        else:                   #**** vmr, cloud, para-H2 , fcloud, take logs ***
            for j in range(nlevel):
                lx[ix+j] = 1
                x0[ix+j] = np.log(ref[j])
                sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 0. Continuous profile
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        jtmp = ipar - (forward_model.AtmosphereX.NVMR+1)
        if (forward_model.Variables.VARPARAM[ivar,0] != 0
                and ipar > forward_model.AtmosphereX.NVMR
                and jtmp < forward_model.AtmosphereX.NDUST
            ): # Fortran true so flip aerosol model
            forward_model.AtmosphereX,xmap1 = Modelm1.calculate(forward_model.AtmosphereX,ipar,xprof)
        else:
            forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)

        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model2(AtmosphericModelBase):
    id : int = 2


    @classmethod
    def calculate(cls, atm,ipar,scf,MakePlot=False):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
                In this model, the atmospheric parameters are scaled using a single factor with 
                respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scf :: Scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([1,npar,atm.NP])

        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:] = atm.VMR[:,jvmr]
            x1[:] = atm.VMR[:,jvmr] * scf
            atm.VMR[:,jvmr] =  x1
        elif ipar==atm.NVMR: #Temperature
            xref[:] = atm.T[:]
            x1[:] = atm.T[:] * scf
            atm.T[:] = x1 
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST:
                xref[:] = atm.DUST[:,jtmp]
                x1[:] = atm.DUST[:,jtmp] * scf
                atm.DUST[:,jtmp] = x1
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2
                x1[:] = atm.PARAH2 * scf
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC
                x1[:] = atm.FRAC * scf
                atm.FRAC = x1

        xmap[0,ipar,:] = xref[:]

        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #**** model 2 - Simple scaling factor of reference profile *******
        #Read in scaling factor

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        x0[ix] = float(tmp[0])
        sx[ix,ix] = (float(tmp[1]))**2.

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
        #Model 2. Scaling factor
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix])
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model3(AtmosphericModelBase):
    id : int = 3


    @classmethod
    def calculate(cls, atm,ipar,scf,MakePlot=False):

        """
            FUNCTION NAME : model3()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
                In this model, the atmospheric parameters are scaled using a single factor 
                in logscale with respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scf :: Log scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([1,npar,atm.NP])

        x1 = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            x1[:] = atm.VMR[:,jvmr] * np.exp(scf)
            atm.VMR[:,jvmr] =  x1 
        elif ipar==atm.NVMR: #Temperature
            x1[:] = atm.T[:] * np.exp(scf)
            atm.T[:] = x1 
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST:
                x1[:] = atm.DUST[:,jtmp] * np.exp(scf)
                atm.DUST[:,jtmp] = x1
            elif jtmp==atm.NDUST:
                x1[:] = atm.PARAH2 * np.exp(scf)
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                x1[:] = atm.FRAC * np.exp(scf)
                atm.FRAC = x1

        xmap[0,ipar,:] = x1[:]

        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        #**** model 3 - Exponential scaling factor of reference profile *******
        #Read in scaling factor

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xfac = float(tmp[0])
        err = float(tmp[1])

        if xfac > 0.0:
            x0[ix] = np.log(xfac)
            lx[ix] = 1
            sx[ix,ix] = ( err/xfac ) **2.
        else:
            raise ValueError('Error in read_apr_nemesis().  xfac must be > 0')

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
        #Model 3. Log scaling factor
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix])
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model9(AtmosphericModelBase):
    id : int = 9


    @classmethod
    def calculate(cls, atm,ipar,href,fsh,tau,MakePlot=False):

        """
            FUNCTION NAME : model9()

            DESCRIPTION :

                Function defining the model parameterisation 9 in NEMESIS.
                In this model, the profile (cloud profile) is represented by a value
                at a certain height, plus a fractional scale height. Below the reference height 
                the profile is set to zero, while above it the profile decays exponentially with
                altitude given by the fractional scale height. In addition, this model scales
                the profile to give the requested integrated cloud optical depth.

            INPUTS :

                atm :: Python class defining the atmosphere

                href :: Base height of cloud profile (km)

                fsh :: Fractional scale height (km)

                tau :: Total integrated column density of the cloud (m-2)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model9(atm,ipar,href,fsh,tau)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        from scipy.integrate import simpson
        from archnemesis.Data.gas_data import const

        #Checking that profile is for aerosols
        if(ipar<=atm.NVMR):
            raise ValueError('error in model 9 :: This model is defined for aerosol profiles only')

        if(ipar>atm.NVMR+atm.NDUST):
            raise ValueError('error in model 9 :: This model is defined for aerosol profiles only')


        #Calculating the actual atmospheric scale height in each level
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros([3,npar,atm.NP])
        for itest in range(4):

            xdeep = tau
            xfsh = fsh
            hknee = href

            if itest==0:
                dummy = 1
            elif itest==1: #For calculating the gradient wrt tau
                dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xdeep = np.exp( np.log(tau) + dx )
            elif itest==2: #For calculating the gradient wrt fsh
                dx = 0.05 * np.log(fsh)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xfsh = np.exp( np.log(fsh) + dx )
            elif itest==3: #For calculating the gradient wrt href
                dx = 0.05 * href
                if dx==0.0:
                    dx = 0.1
                hknee = href + dx

            #Initialising some arrays
            ND = np.zeros(atm.NP)   #Dust density (m-3)

            #Calculating the density in each level
            jfsh = -1
            if atm.H[0]/1.0e3>=hknee:
                jfsh = 1
                ND[0] = 1.

            for jx in range(atm.NP-1):
                j = jx + 1
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfsh

                if atm.H[j]/1.0e3>=hknee:

                    if jfsh<0:
                        ND[j]=1.0
                        jfsh = 1
                    else:
                        ND[j]=ND[j-1]*np.exp(-delh/xfac)


            for j in range(atm.NP):
                if(atm.H[j]/1.0e3<hknee):
                    if(atm.H[j+1]/1.0e3>=hknee):
                        ND[j] = ND[j] * (1.0 - (hknee*1.0e3-atm.H[j])/(atm.H[j+1]-atm.H[j]))
                    else:
                        ND[j] = 0.0

            #Calculating column density (m-2) by integrating the number density (m-3) over column (m)
            #Note that when doing the layering, the total column density in the atmosphere might not be
            #exactly the same as in xdeep due to misalignments at the boundaries of the cloud
            totcol = simpson(ND,x=atm.H)
            ND = ND / totcol * xdeep

            if itest==0:
                xprof[:] = ND[:]
            else:
                xmap[itest-1,ipar,:] = (ND[:]-xprof[:])/dx

        icont = ipar - (atm.NVMR+1)
        atm.DUST[0:atm.NP,icont] = xprof

        return atm,xmap


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
        #******** cloud profile held as total optical depth plus
        #******** base height and fractional scale height. Below the knee
        #******** pressure the profile is set to zero - a simple
        #******** cloud in other words!
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        hknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xfsh = tmp[0]
        efsh = tmp[1]

        if xdeep>0.0:
            x0[ix] = np.log(xdeep)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        err = edeep/xdeep
        sx[ix,ix] = err**2.

        ix = ix + 1

        if xfsh>0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

        err = efsh/xfsh
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = hknee
        #inum[ix] = 1
        sx[ix,ix] = eknee**2.

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
        #Model 9. Simple cloud represented by base height, fractional scale height
            #and the total integrated cloud density
        #***************************************************************

        tau = np.exp(forward_model.Variables.XN[ix])    #Integrated dust column-density
        fsh = np.exp(forward_model.Variables.XN[ix+1])  #Fractional scale height
        href = forward_model.Variables.XN[ix+2]         #Base height (km)

        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,href,fsh,tau)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model32(AtmosphericModelBase):
    id : int = 32


    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'tau' : slice(0,1),
            'frac_scale_height' : slice(1,2),
            'p_ref' : slice(2,3),
        }


    @classmethod
    def calculate(cls, atm,ipar,pref,fsh,tau,MakePlot=False):

        """
            FUNCTION NAME : model32()

            DESCRIPTION :

                Function defining the model parameterisation 32 in NEMESIS.
                In this model, the profile (cloud profile) is represented by a value
                at a certain pressure level, plus a fractional scale height which defines an exponential
                drop of the cloud at higher altitudes. Below the pressure level, the cloud is set 
                to exponentially decrease with a scale height of 1 km. 


            INPUTS :

                atm :: Python class defining the atmosphere
                pref :: Base pressure of cloud profile (atm)
                fsh :: Fractional scale height (km)
                tau :: Total integrated column density of the cloud (m-2) or cloud optical depth (if kext is normalised)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model32(atm,ipar,pref,fsh,tau)

            MODIFICATION HISTORY : Juan Alday (29/05/2024)

        """
        _lgr.debug(f'{ipar=} {pref=} {tau=}')

        from scipy.integrate import simpson
        from archnemesis.Data.gas_data import const

        #Checking that profile is for aerosols
        if(ipar<=atm.NVMR):
            raise ValueError('error in model 32 :: This model is defined for aerosol profiles only')

        if(ipar>atm.NVMR+atm.NDUST):
            raise ValueError('error in model 32 :: This model is defined for aerosol profiles only')

        icont = ipar - (atm.NVMR+1)   #Index of the aerosol population we are modifying

        #Calculating the actual atmospheric scale height in each level
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        #This gradient is calcualted numerically (in this function) as it is too hard otherwise
        xprof = np.zeros(atm.NP)
        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((3,npar,atm.NP))
        for itest in range(4):

            xdeep = tau
            xfsh = fsh
            pknee = pref
            if itest==0:
                dummy = 1
            elif itest==1: #For calculating the gradient wrt tau
                dx = 0.05 * np.log(tau)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xdeep = np.exp( np.log(tau) + dx )
            elif itest==2: #For calculating the gradient wrt fsh
                dx = 0.05 * np.log(fsh)  #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                xfsh = np.exp( np.log(fsh) + dx )
            elif itest==3: #For calculating the gradient wrt pref
                dx = 0.05 * np.log(pref) #In the state vector this variable is passed in log-scale
                if dx==0.0:
                    dx = 0.1
                pknee = np.exp( np.log(pref) + dx )

            #Getting the altitude level based on the pressure/height relation
            isort = np.argsort(atm.P)
            hknee = np.interp(pknee,atm.P[isort]/101325.,atm.H[isort])  #metres

            #Initialising some arrays
            ND = np.zeros(atm.NP)   #Dust density (m-3)
            OD = np.zeros(atm.NP)   #Column density (m-2)
            Q = np.zeros(atm.NP)    #Specific density (particles/gram of atmosphere)


            #Finding the levels in the atmosphere that span pknee
            jknee = -1
            for j in range(atm.NP-1):
                if((atm.P[j]/101325. >= pknee) & (atm.P[j+1]/101325.< pknee)):
                    jknee = j


            if jknee < 0:
                jknee = 0

            #Calculating cloud density at the first level occupied by the cloud
            delh = atm.H[jknee+1] - hknee   #metres
            xfac = 0.5 * (scale[jknee]+scale[jknee+1]) * xfsh  #metres
            ND[jknee+1] = np.exp(-delh/xfac)


            delh = hknee - atm.H[jknee]  #metres
            xf = 1000.  #The cloud below is set to decrease with a scale height of 1 km
            ND[jknee] = np.exp(-delh/xf)

            #Calculating the cloud density above this level
            for j in range(jknee+2,atm.NP):
                delh = atm.H[j] - atm.H[j-1]
                xfac = scale[j] * xfsh
                ND[j] = ND[j-1] * np.exp(-delh/xfac)

            #Calculating the cloud density below this level
            for j in range(0,jknee):
                delh = atm.H[jknee] - atm.H[j]
                xf = 1000.    #The cloud below is set to decrease with a scale height of 1 km
                ND[j] = np.exp(-delh/xf)

            #Now that we have the initial cloud number density (m-3) we can just divide by the mass density to get specific density
            Q[:] = ND[:] / rho[:] / 1.0e3 #particles per gram of atm

            #Now we integrate the optical thickness (calculate column density essentially)
            OD[atm.NP-1] = ND[atm.NP-1] * (scale[atm.NP-1] * xfsh * 1.0e2)  #the factor 1.0e2 is for converting from m to cm
            jfsh = -1
            for j in range(atm.NP-2,-1,-1):
                if j>jknee:
                    delh = atm.H[j+1] - atm.H[j]   #m
                    xfac = scale[j] * xfsh
                    OD[j] = OD[j+1] + (ND[j] - ND[j+1]) * xfac * 1.0e2
                elif j==jknee:
                    delh = atm.H[j+1] - hknee
                    xfac = 0.5 * (scale[j]+scale[j+1])*xfsh
                    OD[j] = OD[j+1] + (1. - ND[j+1]) * xfac * 1.0e2
                    xfac = 1000.
                    OD[j] = OD[j] + (1.0 - ND[j]) * xfac * 1.0e2
                else:
                    delh = atm.H[j+1] - atm.H[j]
                    xfac = 1000.
                    OD[j] = OD[j+1] + (ND[j+1]-ND[j]) * xfac * 1.0e2

            ODX = OD[0]

            #Now we normalise the specific density profile
            #This should be done later to make this totally secure
            for j in range(atm.NP):
                OD[j] = OD[j] * xdeep / ODX
                ND[j] = ND[j] * xdeep / ODX
                Q[j] = Q[j] * xdeep / ODX
                if Q[j]>1.0e10:
                    Q[j] = 1.0e10
                if Q[j]<1.0e-36:
                    Q[j] = 1.0e-36

                #if ND[j]>1.0e10:
                #    ND[j] = 1.0e10
                #if ND[j]<1.0e-36:
                #    ND[j] = 1.0e-36

            if itest==0:  #First iteration, using the values in the state vector
                xprof[:] = Q[:]
            else:  #Next iterations used to calculate the derivatives
                xmap[itest-1,ipar,:] = (Q[:] - xprof[:])/dx

        #Now updating the atmosphere class with the new profile
        atm.DUST[:,icont] = xprof[:]
        _lgr.debug(f'{xprof=}')

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(atm.DUST[:,icont],atm.P/101325.)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max()/101325.,atm.P.min()/101325.)
            ax1.set_xlabel('Cloud density (m$^{-3}$)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()

        atm.DUST_RENORMALISATION[icont] = tau  #Adding flag to ensure that the dust optical depth is tau

        return atm,xmap


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
        #******** cloud profile is represented by a value at a 
        #******** variable pressure level and fractional scale height.
        #******** Below the knee pressure the profile is set to drop exponentially.

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xfsh = tmp[0]
        efsh = tmp[1]

        #optical depth
        if varident[0]==0:
            #temperature - leave alone
            x0[ix] = xdeep
            err = edeep
        else:
            if xdeep>0.0:
                x0[ix] = np.log(xdeep)
                lx[ix] = 1
                err = edeep/xdeep
                #inum[ix] = 1
            else:
                raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        sx[ix,ix] = err**2.

        ix = ix + 1

        #cloud fractional scale height
        if xfsh>0.0:
            x0[ix] = np.log(xfsh)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xfsh (cloud fractional scale height) must be positive')

        err = efsh/xfsh
        sx[ix,ix] = err**2.

        ix = ix + 1

        #cloud pressure level
        if pknee>0.0:
            x0[ix] = np.log(pknee)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter pknee (cloud pressure level) must be positive')

        err = eknee/pknee
        sx[ix,ix] = err**2.

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
        #Model 32. Cloud profile is represented by a value at a variable
            #pressure level and fractional scale height.
            #Below the knee pressure the profile is set to drop exponentially.
        #***************************************************************
        tau = np.exp(forward_model.Variables.XN[ix])   #Base pressure (atm)
        fsh = np.exp(forward_model.Variables.XN[ix+1])  #Integrated dust column-density (m-2) or opacity
        pref = np.exp(forward_model.Variables.XN[ix+2])  #Fractional scale height
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,pref,fsh,tau)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model45(AtmosphericModelBase):
    id : int = 45


    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'deep_vmr' : slice(0,1),
            'humidity' : slice(1,2),
            'strato_vmr' : slice(2,3),
        }


    @classmethod
    def calculate(cls, atm, ipar, tropo, humid, strato, MakePlot=True):

        """
            FUNCTION NAME : model45()

            DESCRIPTION :

                Irwin CH4 model. Variable deep tropospheric and stratospheric abundances,
                along with tropospheric humidity.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                tropo :: Deep methane VMR

                humid :: Relative methane humidity in the troposphere

                strato :: Stratospheric methane VMR

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model45(atm, ipar, tropo, humid, strato)

            MODIFICATION HISTORY : Joe Penn (09/10/2024)

        """

        _lgr.debug(f'{ipar=} {tropo=} {humid=} {strato=}')

        SCH40 = 10.6815
        SCH41 = -1163.83
        # psvp is in bar
        NP = atm.NP

        xnew = np.zeros(NP)
        xnewgrad = np.zeros(NP)
        pch4 = np.zeros(NP)
        pbar = np.zeros(NP)
        psvp = np.zeros(NP)

        for i in range(NP):
            pbar[i] = atm.P[i] /100000#* 1.013
            tmp = SCH40 + SCH41 / atm.T[i]
            psvp[i] = 1e-30 if tmp < -69.0 else np.exp(tmp)

            pch4[i] = tropo * pbar[i]
            if pch4[i] / psvp[i] > 1.0:
                pch4[i] = psvp[i] * humid

            if pbar[i] < 0.1 and pch4[i] / pbar[i] > strato:
                pch4[i] = pbar[i] * strato

            if pbar[i] > 0.5 and pch4[i] / pbar[i] > tropo:
                pch4[i] = pbar[i] * tropo
                xnewgrad[i] = 1.0

            xnew[i] = pch4[i] / pbar[i]

        _lgr.debug(f'{xnew=}')
        atm.VMR[:, ipar] = xnew

        return atm, xnewgrad


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
        #******** Irwin CH4 model. Represented by tropospheric and stratospheric methane 
        #******** abundances, along with methane humidity. 
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        tropo = tmp[0]
        etropo = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        humid = tmp[0]
        ehumid = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        strato = tmp[0]
        estrato = tmp[1]



        x0[ix] = np.log(tropo)
        lx[ix] = 1
        err = etropo/tropo
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = np.log(humid)
        lx[ix] = 1
        err = ehumid/humid
        sx[ix,ix] = err**2.

        ix = ix + 1

        x0[ix] = np.log(strato)
        lx[ix] = 1
        err = estrato/strato
        sx[ix,ix] = err**2.

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
        #Model 45. Irwin CH4 model. Variable deep tropospheric and stratospheric abundances,
            #along with tropospheric humidity.
        #***************************************************************
        tropo = np.exp(forward_model.Variables.XN[ix])   # Deep tropospheric abundance
        humid = np.exp(forward_model.Variables.XN[ix+1])  # Humidity
        strato = np.exp(forward_model.Variables.XN[ix+2])  # Stratospheric abundance
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX, ipar, tropo, humid, strato)
        xmap[ix] = xmap1

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model47(AtmosphericModelBase):
    id : int = 47


    @property
    def parameter_slices(self) -> dict[str, slice]:
        return {
            'tau' : slice(0,1),
            'p_ref' : slice(1,2),
            'fwhm' : slice(2,3),
        }


    @classmethod
    def calculate(cls, atm, ipar, tau, pref, fwhm, MakePlot=False):

        """
            FUNCTION NAME : model47()

            DESCRIPTION :

                Profile is represented by a Gaussian with a specified optical thickness centred
                at a variable pressure level plus a variable FWHM (log press).

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                tau :: Integrated optical thickness.

                pref :: Mean pressure (atm) of the cloud.

                fwhm :: FWHM of the log-Gaussian.

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model47(atm, ipar, tau, pref, fwhm)

            MODIFICATION HISTORY : Joe Penn (08/10/2024)

        """
        _lgr.debug(f'{ipar=} {tau=} {pref=} {fwhm=}')

        from archnemesis.Data.gas_data import const

        # First, check that the profile is for aerosols
        if ipar <= atm.NVMR:
            raise ValueError('Error in model47: This model is defined for aerosol profiles only')

        if ipar > atm.NVMR + atm.NDUST:
            raise ValueError('Error in model47: This model is defined for aerosol profiles only')

        icont = ipar - (atm.NVMR + 1)   # Index of the aerosol population we are modifying

        # Calculate atmospheric properties
        R = const["R"]
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)
        rho = atm.calc_rho()*1e-3    #density (kg/m3)

        # Convert pressures to atm
        P = atm.P / 101325.0  # Pressure in atm

        # Compute Y0 = np.log(pref)
        Y0 = np.log(pref)

        # Compute XWID, the standard deviation of the Gaussian
        XWID = fwhm

        # Initialize arrays
        Q = np.zeros(atm.NP)
        ND = np.zeros(atm.NP)
        OD = np.zeros(atm.NP)
        X1 = np.zeros(atm.NP)

        XOD = 0.0

        for j in range(atm.NP):
            Y = np.log(P[j])

            # Compute Q[j]
            Q[j] = 1.0 / (XWID * np.sqrt(np.pi)) * np.exp(-((Y - Y0) / XWID) ** 2)  #Q is specific density in particles per gram of atm

            # Compute ND[j]
            ND[j] = Q[j] * (rho[j] / 1.0e3) #ND is m-3

            # Compute OD[j]
            OD[j] = ND[j] * scale[j] * 1e5  # The factor 1e5 converts m to cm

            # Check for NaN or small values
            if np.isnan(OD[j]) or OD[j] < 1e-36:
                OD[j] = 1e-36
            if np.isnan(Q[j]) or Q[j] < 1e-36:
                Q[j] = 1e-36

            XOD += OD[j]

            X1[j] = Q[j]

        # Empirical correction to XOD
        XOD = XOD * 0.25

        # Rescale Q[j]
        for j in range(atm.NP):
            X1[j] = Q[j] * tau / XOD  # XDEEP is tau

            # Check for NaN or small values
            if np.isnan(X1[j]) or X1[j] < 1e-36:
                X1[j] = 1e-36

        # Now compute the Jacobian matrix xmap
        npar = atm.NVMR + 2 + atm.NDUST  # Assuming this is the total number of parameters
        xmap = np.zeros((3, npar, atm.NP))

        for j in range(atm.NP):
            Y = np.log(P[j])

            # First parameter derivative: xmap[0, ipar, j] = X1[j] / tau
            xmap[0, ipar, j] = X1[j] / tau  # XDEEP is tau

            # Derivative of X1[j] with respect to Y0 (pref)
            xmap[1, ipar, j] = 2.0 * (Y - Y0) / XWID ** 2 * X1[j]

            # Derivative of X1[j] with respect to XWID (fwhm)
            xmap[2, ipar, j] = (2.0 * ((Y - Y0) ** 2) / XWID ** 3 - 1.0 / XWID) * X1[j]

        # Update the atmosphere class with the new profile
        atm.DUST[:, icont] = X1[:]
        _lgr.debug(f'{X1=}')

        if MakePlot:
            fig, ax1 = plt.subplots(1, 1, figsize=(3, 4))
            ax1.plot(atm.DUST[:, icont], atm.P / 101325.0)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_ylim(atm.P.max() / 101325.0, atm.P.min() / 101325.0)
            ax1.set_xlabel('Cloud density (particles/kg)')
            ax1.set_ylabel('Pressure (atm)')
            ax1.grid()
            plt.tight_layout()
            plt.show()

        atm.DUST_RENORMALISATION[icont] = tau   #Adding flag to ensure that the dust optical depth is tau

        return atm, xmap


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
        #******** cloud profile is represented by a peak optical depth at a 
        #******** variable pressure level and a Gaussian profile with FWHM (in log pressure)

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xdeep = tmp[0]
        edeep = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        pknee = tmp[0]
        eknee = tmp[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        xwid = tmp[0]
        ewid = tmp[1]

        #total optical depth
        if varident[0]==0:
            #temperature - leave alone
            x0[ix] = xdeep
            err = edeep
        else:
            if xdeep>0.0:
                x0[ix] = np.log(xdeep)
                lx[ix] = 1
                err = edeep/xdeep
                #inum[ix] = 1
            else:
                raise ValueError('error in read_apr() :: Parameter xdeep (total atmospheric aerosol column) must be positive')

        sx[ix,ix] = err**2.

        ix = ix + 1

        #pressure level of the cloud
        if pknee>0.0:
            x0[ix] = np.log(pknee)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter pknee (cloud pressure level) must be positive')

        err = eknee/pknee
        sx[ix,ix] = err**2.

        ix = ix + 1

        #fwhm of the gaussian function describing the cloud profile
        if xwid>0.0:
            x0[ix] = np.log(xwid)
            lx[ix] = 1
            #inum[ix] = 1
        else:
            raise ValueError('error in read_apr() :: Parameter xwid (width of the cloud gaussian profile) must be positive')

        err = ewid/xwid
        sx[ix,ix] = err**2.

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
        #Model 47. Profile is represented by a Gaussian with a specified optical thickness centred
            #at a variable pressure level plus a variable FWHM (log press) in height.
        #***************************************************************
        tau = np.exp(forward_model.Variables.XN[ix])   #Integrated dust column-density (m-2) or opacity
        pref = np.exp(forward_model.Variables.XN[ix+1])  #Base pressure (atm)
        fwhm = np.exp(forward_model.Variables.XN[ix+2])  #FWHM
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX, ipar, tau, pref, fwhm)
        
        _lgr.debug(f'{self.n_state_vector_entries=}')
        _lgr.debug(f'{forward_model.Variables.NXVAR[ivar]=}')
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model49(AtmosphericModelBase):
    id : int = 49


    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 49 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                 in linear space. This parameterisation allows the retrieval of negative VMRs.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                xprof(npro) :: Scaling factor at each altitude level

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model50(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (08/06/2022)

        """

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 49 :: Number of levels in atmosphere and scaling factor profile does not match')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((npro,npar,npro))

        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:] = atm.VMR[:,jvmr]
            x1[:] = xprof
            vmr = np.zeros((atm.NP,atm.NVMR))
            vmr[:,:] = atm.VMR
            vmr[:,jvmr] = x1[:]
            atm.edit_VMR(vmr)
        elif ipar==atm.NVMR: #Temperature
            xref = atm.T
            x1 = xprof
            atm.edit_T(x1)
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST: #Dust in m-3
                xref[:] = atm.DUST[:,jtmp]
                x1[:] = xprof
                dust = np.zeros((atm.NP,atm.NDUST))
                dust[:,:] = atm.DUST
                dust[:,jtmp] = x1
                atm.edit_DUST(dust)
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2
                x1[:] = xprof
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC
                x1[:] = xprof
                atm.FRAC = x1

        for j in range(npro):
            xmap[j,ipar,j] = 1.

        return atm,xmap


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
        #********* continuous profile in linear scale ************************
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref = np.zeros([nlevel])
        ref = np.zeros([nlevel])
        eref = np.zeros([nlevel])
        for j in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
            pref[j] = float(tmp[0])
            ref[j] = float(tmp[1])
            eref[j] = float(tmp[2])
        f1.close()

        #inum[ix:ix+nlevel] = 1
        x0[ix:ix+nlevel] = ref[:]
        for j in range(nlevel):
            sx[ix+j,ix+j] = eref[j]**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 50. Continuous profile in linear scale
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model50(AtmosphericModelBase):
    id : int = 50


    @classmethod
    def calculate(cls, atm,ipar,xprof,MakePlot=False):

        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 50 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                multiplied by a scaling factor in linear space. Each element of the state vector
                corresponds to this scaling factor at each altitude level. This parameterisation
                allows the retrieval of negative VMRs.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                xprof(npro) :: Scaling factor at each altitude level

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model50(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (08/06/2022)

        """

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 50 :: Number of levels in atmosphere and scaling factor profile does not match')

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((npro,npar,npro))

        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:] = atm.VMR[:,jvmr]
            x1[:] = atm.VMR[:,jvmr] * xprof
            vmr = np.zeros((atm.NP,atm.NVMR))
            vmr[:,:] = atm.VMR
            vmr[:,jvmr] = x1[:]
            atm.edit_VMR(vmr)
        elif ipar==atm.NVMR: #Temperature
            xref = atm.T
            x1 = atm.T * xprof
            atm.edit_T(x1)
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST: #Dust in m-3
                xref[:] = atm.DUST[:,jtmp]
                x1[:] = atm.DUST[:,jtmp] * xprof
                dust = np.zeros((atm.NP,atm.NDUST))
                dust[:,:] = atm.DUST
                dust[:,jtmp] = x1
                atm.edit_DUST(dust)
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2
                x1[:] = atm.PARAH2 * xprof
                atm.PARAH2 = x1
            elif jtmp==atm.NDUST+1:
                xref[:] = atm.FRAC
                x1[:] = atm.FRAC * xprof
                atm.FRAC = x1

        for j in range(npro):
            xmap[j,ipar,j] = xref[j]

        return atm,xmap


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
        #********* continuous profile of a scaling factor ************************
        s = f.readline().split()
        f1 = open(s[0],'r')
        tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
        nlevel = int(tmp[0])
        if nlevel != npro:
            raise ValueError('profiles must be listed on same grid as .prf')
        clen = float(tmp[1])
        pref = np.zeros([nlevel])
        ref = np.zeros([nlevel])
        eref = np.zeros([nlevel])
        for j in range(nlevel):
            tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
            pref[j] = float(tmp[0])
            ref[j] = float(tmp[1])
            eref[j] = float(tmp[2])
        f1.close()

        x0[ix:ix+nlevel] = ref[:]
        for j in range(nlevel):
            sx[ix+j,ix+j] = eref[j]**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        ix = ix + nlevel
        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 50. Continuous profile of scaling factors
        #***************************************************************

        xprof = np.zeros(forward_model.Variables.NXVAR[ivar])
        xprof[:] = forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,xprof)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model51(AtmosphericModelBase):
    id : int = 51


    @classmethod
    def calculate(cls, atm, ipar, scale, scale_gas, scale_iso):
        """
            FUNCTION NAME : model51()

            DESCRIPTION :

                Function defining the model parameterisation 51 (49 in NEMESIS).
                In this model, the profile is scaled using a single factor with 
                respect to a reference profile.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scale :: Scaling factor
                scale_gas :: Reference gas
                scale_iso :: Reference isotope

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        npar = atm.NVMR+2+atm.NDUST

        iref_vmr = np.where((atm.ID == scale_gas)&(atm.ISO == scale_iso))[0][0]
        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)

        xref[:] = atm.VMR[:,iref_vmr]
        x1[:] = xref * scale
        atm.VMR[:,ipar] = x1

        xmap = np.zeros([1,npar,atm.NP])

        xmap[0,ipar,:] = xref[:]

        return atm,xmap


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
        #********* multiple of different profile ************************
        prof = np.fromfile(f,sep=' ',count=2,dtype='int')
        profgas = prof[0]
        profiso = prof[1]
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        scale = tmp[0]
        escale = tmp[1]

        varparam[1] = profgas
        varparam[2] = profiso
        x0[ix] = np.log(scale)
        lx[ix] = 1
        err = escale/scale
        sx[ix,ix] = err**2.

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
        #Model 51. Scaling of a reference profile
        #***************************************************************                
        scale = np.exp(forward_model.Variables.XN[ix])
        scale_gas, scale_iso = forward_model.Variables.VARPARAM[ivar,1:3]
        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,scale,scale_gas,scale_iso)
        xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP] = xmap1[:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model110(AtmosphericModelBase):
    id : int = 110


    @classmethod
    def calculate(cls, atm, idust0, z_offset):
        """
            FUNCTION NAME : model110()

            DESCRIPTION :

                Function defining the model parameterisation 110.
                In this model, the Venus cloud is parameterised using the model of Haus et al. (2016).
                In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes.
                In this parametersiation, we include the Haus cloud model as it is, but we allow the altitude of the cloud
                to vary according to the inputs.

                The units of the aerosol density are in m-3, so the extinction coefficients must not be normalised.


            INPUTS :

                atm :: Python class defining the atmosphere

                idust0 :: Index of the first aerosol population in the atmosphere class to be changed,
                          but it will indeed affect four aerosol populations.
                          Thus atm.NDUST must be at least 4.

                z_offset :: Offset in altitude (km) of the cloud with respect to the Haus et al. (2016) model.

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        h = atm.H/1.0e3
        nh = len(h)

        if atm.NDUST<idust0+4:
            raise ValueError('error in model 110 :: The cloud model requires at least 4 modes')

        #Cloud mode 1
        ###################################################

        zb1 = 49. + z_offset #Lower base of peak altitude (km)
        zc1 = 16.            #Layer thickness of constant peak particle (km)
        Hup1 = 3.5           #Upper scale height (km)
        Hlo1 = 1.            #Lower scale height (km)
        n01 = 193.5          #Particle number density at zb (cm-3)

        N1 = 3982.04e5       #Total column particle density (cm-2)
        tau1 = 3.88          #Total column optical depth at 1 um

        n1 = np.zeros(nh)

        ialt1 = np.where(h<zb1)
        ialt2 = np.where((h<=(zb1+zc1)) & (h>=zb1))
        ialt3 = np.where(h>(zb1+zc1))

        n1[ialt1] = n01 * np.exp( -(zb1-h[ialt1])/Hlo1 )
        n1[ialt2] = n01
        n1[ialt3] = n01 * np.exp( -(h[ialt3]-(zb1+zc1))/Hup1 )

        #Cloud mode 2
        ###################################################

        zb2 = 65. + z_offset  #Lower base of peak altitude (km)
        zc2 = 1.0             #Layer thickness of constant peak particle (km)
        Hup2 = 3.5            #Upper scale height (km)
        Hlo2 = 3.             #Lower scale height (km)
        n02 = 100.            #Particle number density at zb (cm-3)

        N2 = 748.54e5         #Total column particle density (cm-2)
        tau2 = 7.62           #Total column optical depth at 1 um

        n2 = np.zeros(nh)

        ialt1 = np.where(h<zb2)
        ialt2 = np.where((h<=(zb2+zc2)) & (h>=zb2))
        ialt3 = np.where(h>(zb2+zc2))

        n2[ialt1] = n02 * np.exp( -(zb2-h[ialt1])/Hlo2 )
        n2[ialt2] = n02
        n2[ialt3] = n02 * np.exp( -(h[ialt3]-(zb2+zc2))/Hup2 )

        #Cloud mode 2'
        ###################################################

        zb2p = 49. + z_offset   #Lower base of peak altitude (km)
        zc2p = 11.              #Layer thickness of constant peak particle (km)
        Hup2p = 1.0             #Upper scale height (km)
        Hlo2p = 0.1             #Lower scale height (km)
        n02p = 50.              #Particle number density at zb (cm-3)

        N2p = 613.71e5          #Total column particle density (cm-2)
        tau2p = 9.35            #Total column optical depth at 1 um

        n2p = np.zeros(nh)

        ialt1 = np.where(h<zb2p)
        ialt2 = np.where((h<=(zb2p+zc2p)) & (h>=zb2p))
        ialt3 = np.where(h>(zb2p+zc2p))

        n2p[ialt1] = n02p * np.exp( -(zb2p-h[ialt1])/Hlo2p )
        n2p[ialt2] = n02p
        n2p[ialt3] = n02p * np.exp( -(h[ialt3]-(zb2p+zc2p))/Hup2p )

        #Cloud mode 3
        ###################################################

        zb3 = 49. + z_offset    #Lower base of peak altitude (km)
        zc3 = 8.                #Layer thickness of constant peak particle (km)
        Hup3 = 1.0              #Upper scale height (km)
        Hlo3 = 0.5              #Lower scale height (km)
        n03 = 14.               #Particle number density at zb (cm-3)

        N3 = 133.86e5           #Total column particle density (cm-2)
        tau3 = 14.14            #Total column optical depth at 1 um

        n3 = np.zeros(nh)

        ialt1 = np.where(h<zb3)
        ialt2 = np.where((h<=(zb3+zc3)) & (h>=zb3))
        ialt3 = np.where(h>(zb3+zc3))

        n3[ialt1] = n03 * np.exp( -(zb3-h[ialt1])/Hlo3 )
        n3[ialt2] = n03
        n3[ialt3] = n03 * np.exp( -(h[ialt3]-(zb3+zc3))/Hup3 )


        new_dust = np.zeros((atm.NP,atm.NDUST))

        new_dust[:,:] = atm.DUST[:,:]
        new_dust[:,idust0] = n1[:] * 1.0e6 #Converting from cm-3 to m-3
        new_dust[:,idust0+1] = n2[:] * 1.0e6
        new_dust[:,idust0+2] = n2p[:] * 1.0e6
        new_dust[:,idust0+3] = n3[:] * 1.0e6

        atm.edit_DUST(new_dust)

        return atm


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
        #******** model for Venus cloud following Haus et al. (2016) with altitude offset

        if varident[0]>0:
            raise ValueError('error in read_apr model 110 :: VARIDENT[0] must be negative to be associated with the aerosols')

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #z_offset
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
        #Model 110. Venus cloud model from Haus et al. (2016) with altitude offset
        #************************************************************************************  

        offset = forward_model.Variables.XN[ix]   #altitude offset in km
        idust0 = np.abs(forward_model.Variables.VARIDENT[ivar,0])-1  #Index of the first cloud mode                
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,idust0,offset)

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model111(AtmosphericModelBase):
    id : int = 111


    @classmethod
    def calculate(cls, atm, idust0, so2_deep, so2_top, z_offset):
        """
            FUNCTION NAME : model111()

            DESCRIPTION :

                Function defining the model parameterisation 111.

                This is a parametersiation for the Venus cloud following the model of Haus et al. (2016) (same as model 110),
                but also includes a parametersiation for the SO2 profiles, whose mixing ratio is tightly linked to the
                altitude of the cloud.

                In this model, the cloud is made of a mixture of H2SO2+H2O droplets, with four different modes, and we allow the 
                variation of the cloud altitude. The units of the aerosol density are in m-3, so the extinction coefficients must 
                not be normalised.

                In the case of the SO2 profile, it is tightly linked to the altitude of the cloud, as the mixing ratio
                of these species greatly decreases within the cloud due to condensation and photolysis. This molecule is
                modelled by defining its mixing ratio below and above the cloud, and the mixing ratio is linearly interpolated in
                log-scale within the cloud.

            INPUTS :

                atm :: Python class defining the atmosphere

                idust0 :: Index of the first aerosol population in the atmosphere class to be changed,
                          but it will indeed affect four aerosol populations.
                          Thus atm.NDUST must be at least 4.

                so2_deep :: SO2 volume mixing ratio below the cloud
                so2_top :: SO2 volume mixing ratio above the cloud

                z_offset :: Offset in altitude (km) of the cloud with respect to the Haus et al. (2016) model.

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model111(atm,idust0,so2_deep,so2_top,z_offset)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        h = atm.H/1.0e3
        nh = len(h)

        if atm.NDUST<idust0+4:
            raise ValueError('error in model 111 :: The cloud model requires at least 4 modes')

        iSO2 = np.where( (atm.ID==9) & (atm.ISO==0) )[0]
        if len(iSO2)==0:
            raise ValueError('error in model 111 :: SO2 must be defined in atmosphere class')
        else:
            iSO2 = iSO2[0]

        #Cloud mode 1
        ###################################################

        zb1 = 49. + z_offset #Lower base of peak altitude (km)
        zc1 = 16.            #Layer thickness of constant peak particle (km)
        Hup1 = 3.5           #Upper scale height (km)
        Hlo1 = 1.            #Lower scale height (km)
        n01 = 193.5          #Particle number density at zb (cm-3)

        N1 = 3982.04e5       #Total column particle density (cm-2)
        tau1 = 3.88          #Total column optical depth at 1 um

        n1 = np.zeros(nh)

        ialt1 = np.where(h<zb1)
        ialt2 = np.where((h<=(zb1+zc1)) & (h>=zb1))
        ialt3 = np.where(h>(zb1+zc1))

        n1[ialt1] = n01 * np.exp( -(zb1-h[ialt1])/Hlo1 )
        n1[ialt2] = n01
        n1[ialt3] = n01 * np.exp( -(h[ialt3]-(zb1+zc1))/Hup1 )

        #Cloud mode 2
        ###################################################

        zb2 = 65. + z_offset  #Lower base of peak altitude (km)
        zc2 = 1.0             #Layer thickness of constant peak particle (km)
        Hup2 = 3.5            #Upper scale height (km)
        Hlo2 = 3.             #Lower scale height (km)
        n02 = 100.            #Particle number density at zb (cm-3)

        N2 = 748.54e5         #Total column particle density (cm-2)
        tau2 = 7.62           #Total column optical depth at 1 um

        n2 = np.zeros(nh)

        ialt1 = np.where(h<zb2)
        ialt2 = np.where((h<=(zb2+zc2)) & (h>=zb2))
        ialt3 = np.where(h>(zb2+zc2))

        n2[ialt1] = n02 * np.exp( -(zb2-h[ialt1])/Hlo2 )
        n2[ialt2] = n02
        n2[ialt3] = n02 * np.exp( -(h[ialt3]-(zb2+zc2))/Hup2 )

        #Cloud mode 2'
        ###################################################

        zb2p = 49. + z_offset   #Lower base of peak altitude (km)
        zc2p = 11.              #Layer thickness of constant peak particle (km)
        Hup2p = 1.0             #Upper scale height (km)
        Hlo2p = 0.1             #Lower scale height (km)
        n02p = 50.              #Particle number density at zb (cm-3)

        N2p = 613.71e5          #Total column particle density (cm-2)
        tau2p = 9.35            #Total column optical depth at 1 um

        n2p = np.zeros(nh)

        ialt1 = np.where(h<zb2p)
        ialt2 = np.where((h<=(zb2p+zc2p)) & (h>=zb2p))
        ialt3 = np.where(h>(zb2p+zc2p))

        n2p[ialt1] = n02p * np.exp( -(zb2p-h[ialt1])/Hlo2p )
        n2p[ialt2] = n02p
        n2p[ialt3] = n02p * np.exp( -(h[ialt3]-(zb2p+zc2p))/Hup2p )

        #Cloud mode 3
        ###################################################

        zb3 = 49. + z_offset    #Lower base of peak altitude (km)
        zc3 = 8.                #Layer thickness of constant peak particle (km)
        Hup3 = 1.0              #Upper scale height (km)
        Hlo3 = 0.5              #Lower scale height (km)
        n03 = 14.               #Particle number density at zb (cm-3)

        N3 = 133.86e5           #Total column particle density (cm-2)
        tau3 = 14.14            #Total column optical depth at 1 um

        n3 = np.zeros(nh)

        ialt1 = np.where(h<zb3)
        ialt2 = np.where((h<=(zb3+zc3)) & (h>=zb3))
        ialt3 = np.where(h>(zb3+zc3))

        n3[ialt1] = n03 * np.exp( -(zb3-h[ialt1])/Hlo3 )
        n3[ialt2] = n03
        n3[ialt3] = n03 * np.exp( -(h[ialt3]-(zb3+zc3))/Hup3 )


        new_dust = np.zeros((atm.NP,atm.NDUST))

        new_dust[:,:] = atm.DUST[:,:]
        new_dust[:,idust0] = n1[:] * 1.0e6 #Converting from cm-3 to m-3
        new_dust[:,idust0+1] = n2[:] * 1.0e6
        new_dust[:,idust0+2] = n2p[:] * 1.0e6
        new_dust[:,idust0+3] = n3[:] * 1.0e6

        atm.edit_DUST(new_dust)


        #SO2 vmr profile
        ####################################################

        cloud_bottom = zb1
        cloud_top = zb1 + 20. #Assuming the cloud extends 20 km above the base
        SO2grad = (np.log(so2_top)-np.log(so2_deep))/(cloud_top-cloud_bottom)  #dVMR/dz (km-1)

        #Calculating SO2 profile
        so2 = np.zeros(nh)
        ibelow = np.where(h<cloud_bottom)[0]
        iabove = np.where(h>cloud_top)[0]
        icloud = np.where((h>=cloud_bottom) & (h<=cloud_top))[0]

        so2[ibelow] = so2_deep
        so2[iabove] = so2_top
        so2[icloud] = np.exp(np.log(so2_deep) + SO2grad*(h[icloud]-cloud_bottom))

        #Updating SO2 profile in atmosphere class
        atm.update_gas(9,0,so2)

        return atm


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
        #******** model for Venus cloud and SO2 vmr profile with altitude offset

        if varident[0]>0:
            raise ValueError('error in read_apr model 111 :: VARIDENT[0] must be negative to be associated with the aerosols')

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #z_offset
        x0[ix] = float(tmp[0])
        sx[ix,ix] = float(tmp[1])**2.
        lx[ix] = 0
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #SO2_deep
        so2_deep = float(tmp[0])
        so2_deep_err = float(tmp[1])
        x0[ix] = np.log(so2_deep)
        sx[ix,ix] = (so2_deep_err/so2_deep)**2.
        lx[ix] = 1
        inum[ix] = 1
        ix = ix + 1

        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')   #SO2_top
        so2_top = float(tmp[0])
        so2_top_err = float(tmp[1])
        x0[ix] = np.log(so2_top)
        sx[ix,ix] = (so2_top_err/so2_top)**2.
        lx[ix] = 1
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
        #Model 110. Venus cloud model and SO2 vmr profile with altitude offset
        #************************************************************************************  

        offset = forward_model.Variables.XN[ix]   #altitude offset in km
        so2_deep = np.exp(forward_model.Variables.XN[ix+1])   #SO2 vmr below the cloud
        so2_top = np.exp(forward_model.Variables.XN[ix+2])   #SO2 vmr above the cloud

        idust0 = np.abs(forward_model.Variables.VARIDENT[ivar,0])-1  #Index of the first cloud mode                
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,idust0,so2_deep,so2_top,offset)

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model202(AtmosphericModelBase):
    id : int = 202


    @classmethod
    def calculate(cls, telluric,varid1,varid2,scf):

        """
            FUNCTION NAME : model202()

            DESCRIPTION :

                Function defining the model parameterisation 202 in NEMESIS.
                In this model, the telluric atmospheric profile is multiplied by a constant 
                scaling factor

            INPUTS :

                telluric :: Python class defining the telluric atmosphere

                varid1,varid2 :: The first two values of the Variable ID. They follow the 
                                 same convention as in Model parameterisation 2

                scf :: Scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                telluric :: Updated Telluric class

            CALLING SEQUENCE:

                telluric = model52(telluric,varid1,varid2,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2025)

        """

        x1 = np.zeros(telluric.Atmosphere.NP)
        xref = np.zeros(telluric.Atmosphere.NP)

        if(varid1==0): #Temperature

            xref[:] = telluric.Atmosphere.T[:]
            x1[:] = telluric.Atmosphere.T[:] * scf
            telluric.Atmosphere.T[:] = x1[:]

        elif(varid1>0): #Gaseous abundance

            jvmr = -1
            for j in range(telluric.Atmosphere.NVMR):
                if((telluric.Atmosphere.ID[j]==varid1) & (telluric.Atmosphere.ISO[j]==varid2)):
                    jvmr = j
            if jvmr==-1:
                print('Required ID :: ',varid1,varid2)
                print('Avaiable ID and ISO :: ',telluric.Atmosphere.ID,telluric.Atmosphere.ISO)
                raise ValueError('error in model 202 :: The required gas is not found in Telluric atmosphere')

            xref[:] = telluric.Atmosphere.VMR[:,jvmr]
            x1[:] = telluric.Atmosphere.VMR[:,jvmr] * scf
            telluric.Atmosphere.VMR[:,jvmr] = x1[:]

        else:
            raise ValueError('error in model 202 :: The retrieved parameter has to be either temperature or a gaseous abundance')

        return telluric


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
        #********* simple scaling of telluric atmospheric profile ************************
        tmp = np.fromfile(f,sep=' ',count=2,dtype='float')
        x0[ix] = float(tmp[0])
        sx[ix,ix] = (float(tmp[1]))**2.
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
        #Model 202. Scaling factor of telluric atmospheric profile
        #***************************************************************

        scafac = forward_model.Variables.XN[ix]
        varid1 = forward_model.Variables.VARIDENT[ivar,0] ; varid2 = forward_model.Variables.VARIDENT[ivar,1]
        if forward_model.TelluricX is not None:
            forward_model.TelluricX = self.calculate(forward_model.TelluricX,varid1,varid2,scafac)

        ix = ix + forward_model.Variables.NXVAR[ivar]


class Model1002(AtmosphericModelBase):
    id : int = 1002


    @classmethod
    def calculate(cls, atm,ipar,scf,MakePlot=False):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 1002 in NEMESIS.

                This is the same as model 2, but applied simultaneously in different planet locations
                In this model, the atmospheric parameters are scaled using a single factor with 
                respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scf(nlocations) :: Scaling factors at the different locations

            OPTIONAL INPUTS: None

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(nlocations,ngas+2+ncont,npro,nlocations) :: Matrix of relating funtional derivatives to 
                                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model1002(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (19/04/2023)

        """

        npar = atm.NVMR+2+atm.NDUST
        xmap = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))
        xmap1 = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))

        if len(scf)!=atm.NLOCATIONS:
            raise ValueError('error in model 1002 :: The number of scaling factors must be the same as the number of locations in Atmosphere')

        if atm.NLOCATIONS<=1:
            raise ValueError('error in model 1002 :: This model can be applied only if NLOCATIONS>1')

        x1 = np.zeros((atm.NP,atm.NLOCATIONS))
        xref = np.zeros((atm.NP,atm.NLOCATIONS))
        if ipar<atm.NVMR:  #Gas VMR
            jvmr = ipar
            xref[:,:] = atm.VMR[:,jvmr,:]
            x1[:,:] = atm.VMR[:,jvmr,:] * scf[:]
            atm.VMR[:,jvmr,:] =  x1
        elif ipar==atm.NVMR: #Temperature
            xref[:] = atm.T[:,:]
            x1[:] = np.transpose(np.transpose(atm.T[:,:]) * scf[:])
            atm.T[:,:] = x1 
        elif ipar>atm.NVMR:
            jtmp = ipar - (atm.NVMR+1)
            if jtmp<atm.NDUST:
                xref[:] = atm.DUST[:,jtmp,:]
                x1[:] = np.transpose(np.transpose(atm.DUST[:,jtmp,:]) * scf[:])
                atm.DUST[:,jtmp,:] = x1
            elif jtmp==atm.NDUST:
                xref[:] = atm.PARAH2[:,:]
                x1[:] = np.transpose(np.transpose(atm.PARAH2[:,:]) * scf)
                atm.PARAH2[:,:] = x1
            elif jtmp==atm.NDUST+1:
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
        #******** scaling of atmospheric profiles at multiple locations (linear scale)

        s = f.readline().split()

        #Reading file with the a priori information
        f1 = open(s[0],'r') 
        s = np.fromfile(f1,sep=' ',count=2,dtype='float')   #nlocations and correlation length
        nlocs = int(s[0])   #number of locations
        clen = int(s[1])    #correlation length (degress)

        if nlocs != nlocations:
            raise ValueError('error in model 1002 :: number of locations must be the same as in Surface and Atmosphere')

        lats = np.zeros(nlocs)
        lons = np.zeros(nlocs)
        sfactor = np.zeros(nlocs)
        efactor = np.zeros(nlocs)
        for iloc in range(nlocs):

            s = np.fromfile(f1,sep=' ',count=4,dtype='float')   
            lats[iloc] = float(s[0])    #latitude of the location
            lons[iloc] = float(s[1])    #longitude of the location
            sfactor[iloc] = float(s[2])   #scaling value
            efactor[iloc] = float(s[3])   #uncertainty in scaling value

        f1.close()

        #Including the parameters in the state vector
        varparam[0] = nlocs
        iparj = 1
        for iloc in range(nlocs):
            #Including surface temperature in the state vector
            x0[ix+iloc] = sfactor[iloc]
            sx[ix+iloc,ix+iloc] = efactor[iloc]**2.0
            lx[ix+iloc] = 0     #linear scale
            inum[ix+iloc] = 0   #analytical calculation of jacobian


        #Defining the correlation between surface pixels 
        for j in range(nlocs):
            s1 = np.sin(lats[j]/180.*np.pi)
            s2 = np.sin(lats/180.*np.pi)
            c1 = np.cos(lats[j]/180.*np.pi)
            c2 = np.cos(lats/180.*np.pi)
            c3 = np.cos( (lons[j]-lons)/180.*np.pi )
            psi = np.arccos( s1*s2 + c1*c2*c3 ) / np.pi * 180.   #angular distance (degrees)
            arg = abs(psi/clen)
            xfac = np.exp(-arg)
            for k in range(nlocs):
                if xfac[k]>0.001:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac[k]
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]

        jsurf = ix

        ix = ix + nlocs

        return cls(ix_0, ix-ix_0)


    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 1002. Scaling factors at multiple locations
        #***************************************************************

        forward_model.AtmosphereX,xmap1 = self.calculate(forward_model.AtmosphereX,ipar,forward_model.Variables.XN[ix:ix+forward_model.Variables.NXVAR[ivar]],MakePlot=False)
        #This calculation takes a long time for big arrays
        #xmap[ix:ix+forward_model.Variables.NXVAR[ivar],:,0:forward_model.AtmosphereX.NP,0:forward_model.AtmosphereX.NLOCATIONS] = xmap1[:,:,:,:]

        ix = ix + forward_model.Variables.NXVAR[ivar]


