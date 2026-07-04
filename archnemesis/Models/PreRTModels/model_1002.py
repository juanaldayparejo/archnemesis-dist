
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase

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

class Model1002(PreRTModelBase):
    """ 
        In this model, the atmospheric parameters are scaled using a single factor with 
        respect to the vertical profiles in the reference atmosphere.
        
        The model is applied simultaneously in different planet locations
    """
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
        #xmap1 = np.zeros((atm.NLOCATIONS,npar,atm.NP,atm.NLOCATIONS))

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
            ngas : int,
            ndust : int,
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
        #iparj = 1
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

        #jsurf = ix

        ix = ix + nlocs

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])


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
        #******** scaling of atmospheric profiles at multiple locations (linear scale)

        nlocs = varparam[0]
        if nlocs != nlocations:
            raise ValueError('error in model 1002 :: number of locations must be the same as in Surface and Atmosphere')

        ix = ix + nlocs

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])


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


