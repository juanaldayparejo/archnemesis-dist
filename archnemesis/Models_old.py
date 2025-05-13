# NAME:
#       Models.py (archNEMESIS)
#
# DESCRIPTION:
#
#	This library contains functions to change the Nemesis atmospheric model parameters based on the different parameterisations      
#
# CATEGORY:
#
#	NEMESIS
# 
# MODIFICATION HISTORY: Juan Alday 15/07/2021

from archnemesis import *
import numpy as np
import matplotlib.pyplot as plt
import os


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)

###############################################################################################

def modelm1(atm,ipar,xprof,MakePlot=False):
    
    """
        FUNCTION NAME : modelm1()
        
        DESCRIPTION :
        
            Function defining the model parameterisation -1 in NEMESIS.
            In this model, the aerosol profiles is modelled as a continuous profile in units
            of particles per gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
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
#             rho = atm.calc_rho()  #kg/m3
            atm.DUST_UNITS_FLAG[jtmp] = -1
            atm.DUST[:,jtmp] = x1 #* 1000. * rho
        elif jtmp==atm.NDUST:
            raise ValueError('error :: Model -1 is just compatible with aerosol populations')
        elif jtmp==atm.NDUST+1:
            raise ValueError('error :: Model -1 is just compatible with aerosol populations')
    
    for j in range(npro):
        xmap[0:npro,ipar,j] = x1[:] #* 1000. * rho
        

    if MakePlot==True:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(7,5))

        for i in range(atm.NDUST):
            ax1.semilogx(atm.DUST[:,i]*rho,atm.H/1000.)
            ax2.semilogx(atm.DUST[:,i],atm.H/1000.)

        ax1.grid()
        ax2.grid()
        ax1.set_xlabel('Aerosol density (particles per cm$^{-3}$)')
        ax1.set_ylabel('Altitude (km)')
        ax2.set_xlabel('Aerosol density (particles per gram of atm)')
        ax2.set_ylabel('Altitude (km)')
        plt.tight_layout()
        plt.show()

    return atm,xmap


###############################################################################################

def model0(atm,ipar,xprof,MakePlot=False):
    
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


###############################################################################################

def model2(atm,ipar,scf,MakePlot=False):
    
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


###############################################################################################

def model3(atm,ipar,scf,MakePlot=False):
    
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


###############################################################################################

def model9(atm,ipar,href,fsh,tau,MakePlot=False):
    
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


###############################################################################################

def model32(atm,ipar,pref,fsh,tau,MakePlot=False):
    
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


###############################################################################################
def model45(atm, ipar, tropo, humid, strato, MakePlot=True):

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


###############################################################################################

def model47(atm, ipar, tau, pref, fwhm, MakePlot=False):
    
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

###############################################################################################

def model49(atm,ipar,xprof,MakePlot=False):
    
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


###############################################################################################

def model50(atm,ipar,xprof,MakePlot=False):
    
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

#################################################################################################################################

def model51(atm, ipar, scale, scale_gas, scale_iso):
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

#################################################################################################################################

def model110(atm, idust0, z_offset):
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


#################################################################################################################################

def model111(atm, idust0, so2_deep, so2_top, z_offset):
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


###############################################################################################


def model202(telluric,varid1,varid2,scf):
    
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


###############################################################################################

def model228(Measurement,Spectroscopy,V0,C0,C1,C2,P0,P1,P2,P3,MakePlot=False):
    
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


###############################################################################################

def model229(Measurement,par1,par2,par3,par4,par5,par6,par7,MakePlot=False):
    
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


###############################################################################################

def model230(Measurement,nwindows,liml,limh,par,MakePlot=False):
    
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


###############################################################################################
def model444(Scatter,idust,iscat,xprof,haze_params):
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
        _lgr.debug(f'haze_params[{item},{idust}] = {type(haze_params[item,idust])} {haze_params[item,idust]}')
    
    a = np.exp(xprof[0])
    b = np.exp(xprof[1])
    if iscat == 1:
        pars = (a,b,(1-3*b)/b)
    elif iscat == 2:
        pars = (a,b,0)
    elif iscat == 4:
        pars = (a,0,0)
    else:
        print(f'WARNING: ISCAT = {iscat} not implemented for model 444 yet! Defaulting to iscat = 1.')
        pars = (a,b,(1-3*b)/b)
    
    Scatter.WAVER = haze_params['WAVE',idust]
    Scatter.REFIND_IM = np.exp(xprof[2:])
    reference_nreal = haze_params['NREAL',idust]
    reference_wave = haze_params['WAVE_REF',idust]
    normalising_wave = haze_params['WAVE_NORM',idust]
    if len(Scatter.REFIND_IM) == 1:
        Scatter.REFIND_IM = Scatter.REFIND_IM * np.ones_like(Scatter.WAVER)
        
    Scatter.REFIND_REAL = kk_new_sub(np.array(Scatter.WAVER), np.array(Scatter.REFIND_IM), reference_wave, reference_nreal)
    
    
    Scatter.makephase(idust, iscat, pars)
    
    xextnorm = np.interp(normalising_wave,Scatter.WAVE,Scatter.KEXT[:,idust])
    Scatter.KEXT[:,idust] = Scatter.KEXT[:,idust]/xextnorm
    Scatter.KSCA[:,idust] = Scatter.KSCA[:,idust]/xextnorm
    return Scatter

###############################################################################################

def model446(Scatter,idust,wavenorm,xwave,rsize,lookupfile,MakePlot=False):
    
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


###############################################################################################

def model447(Measurement,v_doppler):
    
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

def model500(k_cia, waven, icia, vlo, vhi, nbasis, amplitudes):
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


###############################################################################################

def model667(Spectrum,xfactor,MakePlot=False):
    
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


###############################################################################################

def model777(Measurement,hcorr,MakePlot=False):
    
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


###############################################################################################

def model887(Scatter,xsc,idust,MakePlot=False):
    
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
    

###############################################################################################

def model1002(atm,ipar,scf,MakePlot=False):
    
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



########################################################################################################################
#EXTRA FUNCTIONS
########################################################################################################################

def ngauss(npx,x,ng,iamp,imean,ifwhm,MakePlot=False):


    """
        FUNCTION NAME : ngauss()
        
        DESCRIPTION : 

            Create a function which is the sum of multiple gaussians
 
        INPUTS :
      
            npx :: Number of points in x-array
            x(npx) :: Array specifying the points at which the function must be calculated
            ng :: Number of gaussians
            iamp(ng) :: Amplitude of each of the gaussians
            imean(ng) :: Center x-point of the gaussians
            ifwhm(ng) :: FWHM of the gaussians

        OPTIONAL INPUTS: none
        
        OUTPUTS : 

            fun(npx) :: Function at each x-point

        CALLING SEQUENCE:
        
            fun = ngauss(npx,x,ng,iamp,imean,ifwhm)
 
        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    fun  = np.zeros([npx])
    isigma = ifwhm/(2.0*np.sqrt(2.*np.log(2.)))
    for i in range(npx):
        for j in range(ng):
            fun[i] = fun[i] + iamp[j] * np.exp( -(x[i]-imean[j])**2.0/(2.0*isigma[j]**2.)  )


    #Make plot if keyword is specified
    if MakePlot == True:
        axis_font = {'size':'20'}
        cm = plt.cm.get_cmap('RdYlBu')
        fig = plt.figure(figsize=(15,8))
        wavemin = x.min()
        wavemax = x.max()
        ax = plt.axes()
        ax.set_xlim(wavemin,wavemax)
        ax.tick_params(labelsize=20)
        ax.ticklabel_format(useOffset=False)
        plt.xlabel('x',**axis_font)
        plt.ylabel('f(x)',**axis_font)
        im = ax.plot(x,fun)
        plt.grid()
        plt.show()    
    
    return fun