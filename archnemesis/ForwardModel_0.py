from __future__ import annotations #  for 3.9 compatability
from archnemesis import *
from archnemesis.Models import Models
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
from multiprocessing import Pool
from joblib import Parallel, delayed
import sys

from archnemesis.enums import (
    AtmosphericProfileFormatEnum,
    PathCalc,
    SpectralCalculationMode,
    ScatteringCalculationMode,
    SpectraUnit,
    LowerBoundaryCondition,
    WaveUnit,
    PathObserverPointing,
    RayleighScatteringMode,
)

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 15 2022

@author: juanalday

Forward Model Class.
"""

class ForwardModel_0:

    def __init__(self, runname='wasp121', Atmosphere=None, Surface=None,
        Measurement=None, Spectroscopy=None, Stellar=None, Scatter=None,
        CIA=None, Layer=None, Variables=None, Telluric=None, adjust_hydrostat=True):

        """Forward Model class

        The Forward Model class compiles different function required to perform the radiative transfer calculations.
        It takes as arguments the different Reference Classes, which provide all the required information to compute
        the forward model. The Forward Model class then modifies and uses this information to perform the calculations.


        Inputs (Reference classes)
        ------
        @class Atmosphere:,
            Class defining the Atmosphere
        @class Surface:,
            Class defining the Surface
        @class Measurement:,
            Class defining the Measurement
        @class Scatter:,
            Class defining the Scatter
        @class Spectroscopy:,
            Class defining the Spectroscopy
        @class Stellar:,
            Class defining the Stellar
        @class CIA:,
            Class defining the CIA
        @class Layer:,
            Class defining the Layer
        @class Variables:,
            Class defining the Variables
        @class Telluric:,
            Class defining the Telluric
        @log adjust_hydrostat:,
            Flag indicating whether the re-adjustment of the pressure or altitude levels
            based on the hydrostatic equilibrium equation must be performed or not.

        Attributes (Attribute classes)
        ----------

        Note: The classes appear to be duplicated, some of them having an X in the end.
        The main difference between these two types is that ones are the reference classes
        given as an input, and which are not varied through the Forward Model. The other ones
        are the reference classes modified by the model paramterisations and varied through
        the calculations to calculate a specific forward model

        @attribute AtmosphereX:
            Class defining the Atmospheric profile for a particular forward model
        @attribute SurfaceX:
            Class defining the Surface for a particular forward model
        @attribute MeasurementX:
            Class defining the Measurement for a particular forward model
        @attribute ScatterX:
            Class defining the Scatter for a particular forward model
        @attribute CIAX:
            Class defining the CIA for a particular forward model
        @attribute SpectroscopyX:
            Class defining the Spectroscopy for a particular forward model
        @attribute StellarX:
            Class defining the Stellar for a particular forward model
        @attribute LayerX:
            Class defining the Layer for a particular forward model
        @attribute PathX:
            Class defining the Path for a particular forward model
        @attribute TelluricX:
            Class defining the Telluric for a particular forward model

        Methods
        -------

        Forward Models and Jacobeans
        ##########################################

            ForwardModel_0.nemesisfm()
            ForwardModel_0.nemesisfmg()
            ForwardModel_0.nemesisSOfm()
            ForwardModel_0.nemesisSOfmg()
            ForwardModel_0.nemesisLfm()
            ForwardModel_0.nemesisLfmg()
            ForwardModel_0.nemesisCfm()
            ForwardModel_0.nemesisMAPfm()
            ForwardModel_0.jacobian_nemesis(nemesisSO=False)

        Mapping models into reference classes
        ##########################################

            ForwardModel_0.subprofretg()
            ForwardModel_0.subspecret()

        Path calculation and geometry
        ##########################################

            ForwardModel_0.select_Measurement()
            ForwardModel_0.select_location()
            ForwardModel_0.calc_path()
            ForwardModel_0.calc_pathg()
            ForwardModel_0.calc_path_SO()
            ForwardModel_0.calc_pathg_SO()
            ForwardModel_0.calc_path_L()
            ForwardModel_0.calc_pathg_L()

        Radiative transfer calculations
        ##########################################

            ForwardModel_0.CIRSrad()
            ForwardModel_0.CIRSradg()
            ForwardModel_0.calc_tau_cia()
            ForwardModel_0.calc_tau_dust()
            ForwardModel_0.calc_tau_gas()
            ForwardModel_0.calc_tau_rayleigh()
            ForwardModel_0.calc_brdf_matrix()

        Multiple scattering routines
        ###########################################

            ForwardModel_0.scloud11wave()
            ForwardModel_0.scloud11flux()
            ForwardModel_0.streamflux()
            
        Error checks
        ###########################################

            ForwardModel_0.check_gas_spec_atm()

        """

        from copy import deepcopy

        self.runname = runname

        #Building the reference classes into the Forward Model class
        self.Atmosphere = Atmosphere
        self.Surface = Surface
        self.Measurement = Measurement
        self.Scatter = Scatter
        self.Spectroscopy = Spectroscopy
        self.CIA = CIA
        self.Stellar = Stellar
        self.Variables = Variables
        self.Layer = Layer
        self.Telluric = Telluric
        self.adjust_hydrostat=adjust_hydrostat

        #Creating extra class to hold the variables class in each permutation of the Jacobian Matrix
        self.Variables1 = deepcopy(Variables)

        #Creating extra classes to store the parameters for a particular forward model
        self.AtmosphereX = deepcopy(Atmosphere)
        self.SurfaceX = deepcopy(Surface)
        self.MeasurementX = deepcopy(Measurement)
        self.ScatterX = deepcopy(Scatter)
        self.SpectroscopyX = deepcopy(Spectroscopy)
        self.CIAX = deepcopy(CIA)
        self.StellarX = deepcopy(Stellar)
        self.LayerX = deepcopy(Layer)
        self.TelluricX = deepcopy(Telluric)
        self.PathX = None


    ###############################################################################################
    ###############################################################################################
    # CALCULATIONS OF FORWARD MODELS AND JACOBEANS
    ###############################################################################################
    ###############################################################################################

    ###############################################################################################

    def select_nemesis_fm(
            self,
            nemesisSO : bool = False,
            nemesisL : bool = False,
            analytical_gradient : bool = False,
        ) -> Callable[[],np.ndarray] | Callable[[],tuple[np.ndarray,np.ndarray]]:
        """
        Selects the correct method to calculate the nemesis forward model based on passed flags.
        """
        method = None
        
        
        if nemesisSO:
            method = self.nemesisSOfmg if analytical_gradient else self.nemesisSOfm
        elif nemesisL:
            method = self.nemesisLfmg if analytical_gradient else self.nemesisLfm
        else:
            method = self.nemesisfmg if analytical_gradient else self.nemesisfm
        
        if method is None:
            raise RuntimeError('Could not select method to use when calculating nemesis forward model.')
        
        return method
        

    def nemesisfm(self):

        """
            FUNCTION NAME : nemesisfm()

            DESCRIPTION : This function computes a forward model

            INPUTS : none

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisfm()

            MODIFICATION HISTORY : Juan Alday (14/03/2022)

        """

        from copy import copy, deepcopy
        
        #Errors and checks
        if self.Atmosphere.NLOCATIONS!=1:
            raise ValueError('error in nemesisfm :: archNEMESIS has not been setup for dealing with multiple locations yet')
            
        if self.Surface.NLOCATIONS!=1:
            raise ValueError('error in nemesisfm :: archNEMESIS has not been setup for dealing with multiple locations yet')

        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        SPECONV = np.zeros(self.Measurement.MEAS.shape) #Initalise the array where the spectra will be stored (NWAVE,NGEOM)
        for IGEOM in range(self.Measurement.NGEOM):

            #Calculating new wave array            
            self.Measurement.build_ils(IGEOM=IGEOM)
            wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=IGEOM)
                
            #Reading tables in the required wavelength range
            self.SpectroscopyX = deepcopy(self.Spectroscopy)
            self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

            #Initialise array for averaging spectra (if required by NAV>1)
            SPEC = np.zeros(self.SpectroscopyX.NWAVE)
            WGEOMTOT = 0.0
            for IAV in range(self.Measurement.NAV[IGEOM]):

                #Selecting the relevant Measurement
                self.select_Measurement(IGEOM,IAV)

                #Making copy of classes to avoid overwriting them
                self.AtmosphereX = deepcopy(self.Atmosphere)
                self.ScatterX = deepcopy(self.Scatter)
                self.StellarX = deepcopy(self.Stellar)
                self.SurfaceX = deepcopy(self.Surface)
                self.LayerX = deepcopy(self.Layer)
                self.CIAX = deepcopy(self.CIA)
                self.TelluricX = deepcopy(self.Telluric)
                flagh2p = False

                #Updating the required parameters based on the current geometry
                if self.MeasurementX.EMISS_ANG[0,0]>=0.0:
                    self.ScatterX.SOL_ANG = self.MeasurementX.SOL_ANG[0,0]
                    self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]
                    self.ScatterX.AZI_ANG = self.MeasurementX.AZI_ANG[0,0]
                else:
                    self.ScatterX.SOL_ANG = self.MeasurementX.TANHE[0,0]
                    self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]

                #Changing the different classes taking into account the parameterisations in the state vector
                xmap = self.subprofretg()
                
                #Calling gsetpat to split the new reference atmosphere and calculate the path
                self.LayerX.DUST_UNITS_FLAG = self.AtmosphereX.DUST_UNITS_FLAG
                self.calc_path()
                
                #Calling CIRSrad to perform the radiative transfer calculations
                SPEC1X = self.CIRSrad()

                if self.PathX.NPATH>1:  #If the calculation type requires several paths for a given geometry (e.g. netflux calculation)
                    SPEC1 = np.zeros((self.PathX.NPATH*self.SpectroscopyX.NWAVE,1))  #We linearise all paths into 1 measurement
                    ip = 0
                    for iPath in range(self.PathX.NPATH):
                        SPEC1[ip:ip+self.SpectroscopyX.NWAVE,0] = SPEC1X[:,iPath]
                else:
                    SPEC1 = SPEC1X

                #Averaging the spectra in case NAV>1
                if self.Measurement.NAV[IGEOM]>=1:
                    SPEC[:] = SPEC[:] + self.Measurement.WGEOM[IGEOM,IAV] * SPEC1[:,0]
                    WGEOMTOT = WGEOMTOT + self.Measurement.WGEOM[IGEOM,IAV]
                else:
                    SPEC[:] = SPEC1[:,0]

            
            #Applying the Telluric transmission if it exists
            if self.TelluricX is not None:
                
                #Looking for the calculation wavelengths
                wavecalc_min_tel,wavecalc_max_tel = self.Measurement.calc_wave_range(apply_doppler=False,IGEOM=IGEOM)
                self.TelluricX.Spectroscopy.read_tables(wavemin=wavecalc_min_tel,wavemax=wavecalc_max_tel)
                
                #Calculating the telluric transmission
                WAVE_TELLURIC,TRANSMISSION_TELLURIC = self.TelluricX.calc_transmission()
            
                #Interpolating the telluric transmission to the wavelengths of the planetary spectrum
                wavecorr = self.MeasurementX.correct_doppler_shift(self.SpectroscopyX.WAVE)
                TRANSMISSION_TELLURICx = np.interp(wavecorr,WAVE_TELLURIC,TRANSMISSION_TELLURIC)
                
                #Applying the telluric transmission to the planetary spectrum
                SPEC *= TRANSMISSION_TELLURICx
                
            
            #Convolving the spectra with the Instrument line shape
            if self.SpectroscopyX.ILBL == SpectralCalculationMode.K_TABLES: #k-tables
                if os.path.exists(self.runname+'.fwh')==True:
                    FWHMEXIST=self.runname
                else:
                    FWHMEXIST=''

                SPECONV1 = self.Measurement.conv(self.SpectroscopyX.WAVE,SPEC,IGEOM=IGEOM,FWHMEXIST='')

            elif self.SpectroscopyX.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES: #LBL-tables
                SPECONV1 = self.Measurement.lblconv(self.SpectroscopyX.WAVE,SPEC,IGEOM=IGEOM)

            SPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM] = SPECONV1[0:self.Measurement.NCONV[IGEOM]]
            
            #Normalising measurement to a given wavelength if required
            if self.Measurement.IFORM == SpectraUnit.Normalised_radiance:
                SPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM] /= np.interp(self.Measurement.VNORM,self.Measurement.VCONV[0:self.Measurement.NCONV[IGEOM],IGEOM],SPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM])

        #Applying any changes to the computed spectra required by the state vector
        dSPECONV = np.zeros((self.Measurement.NCONV.max(),self.Measurement.NGEOM,self.Variables.NX))
        SPECONV,dSPECONV = self.subspecret(SPECONV,dSPECONV)

        return SPECONV


    ###############################################################################################

    def nemesisfmg(self):

        """
            FUNCTION NAME : nemesisfmg()

            DESCRIPTION : This function computes a forward model and the analytical gradients

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra
                dSPECMOD(NCONV,NGEOM,NX) :: Gradients of the spectra in each geometry with respect to the elements
                                            in the state vector

            CALLING SEQUENCE:

                nemesisfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from copy import deepcopy
        
        #Errors and checks
        if self.Atmosphere.NLOCATIONS!=1:
            raise ValueError('error in nemesisfm :: archNEMESIS has not been setup for dealing with multiple locations yet')
            
        if self.Surface.NLOCATIONS!=1:
            raise ValueError('error in nemesisfm :: archNEMESIS has not been setup for dealing with multiple locations yet')

        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        #Estimating the number of calculations that will need to be computed to model the spectra
        #included in the Measurement class (taking into account al geometries and averaging points)
        NCALC = np.sum(self.Measurement.NAV)
        SPECONV = np.zeros(self.Measurement.MEAS.shape) #Initalise the array where the spectra will be stored (NWAVE,NGEOM)
        dSPECONV = np.zeros((self.Measurement.NCONV.max(),self.Measurement.NGEOM,self.Variables.NX)) #Initalise the array where the gradients will be stored (NWAVE,NGEOM,NX)
        for IGEOM in range(self.Measurement.NGEOM):

            #Calculating new wave array            
            self.Measurement.build_ils(IGEOM=IGEOM)
            wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=IGEOM)
                
            #Reading tables in the required wavelength range
            self.SpectroscopyX = deepcopy(self.Spectroscopy)
            self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

            #Initialise array for averaging spectra (if required by NAV>1)
            SPEC = np.zeros(self.SpectroscopyX.NWAVE)
            dSPEC = np.zeros((self.SpectroscopyX.NWAVE,self.Variables.NX))
            WGEOMTOT = 0.0
            for IAV in range(self.Measurement.NAV[IGEOM]):

                #Selecting the relevant Measurement
                self.select_Measurement(IGEOM,IAV)

                #Making copy of classes to avoid overwriting them
                self.AtmosphereX = deepcopy(self.Atmosphere)
                self.ScatterX = deepcopy(self.Scatter)
                self.StellarX = deepcopy(self.Stellar)
                self.SurfaceX = deepcopy(self.Surface)
                self.LayerX = deepcopy(self.Layer)
                self.CIAX = deepcopy(self.CIA)
                self.TelluricX = deepcopy(self.Telluric)
                flagh2p = False

                #Updating the required parameters based on the current geometry
                if self.MeasurementX.EMISS_ANG[0,0]>=0.0:
                    self.ScatterX.SOL_ANG = self.MeasurementX.SOL_ANG[0,0]
                    self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]
                    self.ScatterX.AZI_ANG = self.MeasurementX.AZI_ANG[0,0]
                else:
                    self.ScatterX.SOL_ANG = self.MeasurementX.TANHE[0,0]
                    self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]

                #Changing the different classes taking into account the parameterisations in the state vector
                xmap = self.subprofretg()
                
                self.LayerX.DUST_UNITS_FLAG = self.AtmosphereX.DUST_UNITS_FLAG

                #Calling gsetpat to split the new reference atmosphere and calculate the path
                self.calc_pathg()

                #Calling CIRSrad to perform the radiative transfer calculations
                #SPEC1,dSPEC3,dTSURF = CIRSradg(self.runname,self.Variables,self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
                SPEC1,dSPEC3,dTSURF = self.CIRSradg()

                #Mapping the gradients from Layer properties to Profile properties
                _lgr.info('Mapping gradients from Layer to Profile')
                #Calculating the elements from NVMR+2+NDUST that need to be mapped
                incpar = []
                for i in range(self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST):
                    if np.mean(xmap[:,i,:])!=0.0:
                        incpar.append(i)

                if len(incpar)>0:
                    dSPEC2 = map2pro(dSPEC3,self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.PathX.NLAYIN,self.PathX.LAYINC,self.LayerX.DTE,self.LayerX.DAM,self.LayerX.DCO,INCPAR=incpar)
                else:
                    dSPEC2 = np.zeros((self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH))
                del dSPEC3

                #Mapping the gradients from Profile properties to elements in state vector
                _lgr.info('Mapping gradients from Profile to State Vector')
                dSPEC1 = map2xvec(dSPEC2,self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.Variables.NX,xmap)
                #(NWAVE,NPATH,NX)
                del dSPEC2

                #Adding the temperature surface gradient if required
                if self.Variables.JSURF>=0:
                    dSPEC1[:,0,self.Variables.JSURF] = dTSURF[:,0]

                #Averaging the spectra in case NAV>1
                if self.Measurement.NAV[IGEOM]>=1:
                    SPEC[:] = SPEC[:] + self.Measurement.WGEOM[IGEOM,IAV] * SPEC1[:,0]
                    dSPEC[:,:] = dSPEC[:,:] + self.Measurement.WGEOM[IGEOM,IAV] * dSPEC1[:,0,:]
                    WGEOMTOT = WGEOMTOT + self.Measurement.WGEOM[IGEOM,IAV]
                else:
                    SPEC[:] = SPEC1[:,0]
                    dSPEC[:,:] = dSPEC1[:,0,:]

            #Applying the Telluric transmission if it exists
            if self.TelluricX is not None:
                                         
                #Looking for the calculation wavelengths
                wavecalc_min_tel,wavecalc_max_tel = self.Measurement.calc_wave_range(apply_doppler=False,IGEOM=IGEOM)
                self.TelluricX.Spectroscopy.read_tables(wavemin=wavecalc_min_tel,wavemax=wavecalc_max_tel)
                
                #Calculating the telluric transmission
                WAVE_TELLURIC,TRANSMISSION_TELLURIC = self.TelluricX.calc_transmission()
            
                #Interpolating the telluric transmission to the wavelengths of the planetary spectrum 
                wavecorr = self.MeasurementX.correct_doppler_shift(self.SpectroscopyX.WAVE)
                TRANSMISSION_TELLURICx = np.interp(wavecorr,WAVE_TELLURIC,TRANSMISSION_TELLURIC)
                
                #Applying the telluric transmission to the planetary spectrum
                SPEC *= TRANSMISSION_TELLURICx
                dSPEC[:,:] = (dSPEC[:,:].T * TRANSMISSION_TELLURICx).T 

            #Convolving the spectra with the Instrument line shape
            if self.Spectroscopy.ILBL == SpectralCalculationMode.K_TABLES: #k-tables

                if os.path.exists(self.runname+'.fwh')==True:
                    FWHMEXIST=self.runname
                else:
                    FWHMEXIST=''

                SPECONV1,dSPECONV1 = self.Measurement.convg(self.SpectroscopyX.WAVE,SPEC,dSPEC,IGEOM=IGEOM,FWHMEXIST='')

            elif self.Spectroscopy.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES: #LBL-tables

                SPECONV1,dSPECONV1 = self.Measurement.lblconvg(self.SpectroscopyX.WAVE,SPEC,dSPEC,IGEOM=IGEOM)

            SPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM] = SPECONV1[0:self.Measurement.NCONV[IGEOM]]
            dSPECONV[0:self.Measurement.NCONV[IGEOM],IGEOM,:] = dSPECONV1[0:self.Measurement.NCONV[IGEOM],:]

        #Applying any changes to the spectra required by the state vector
        SPECONV,dSPECONV = self.subspecret(SPECONV,dSPECONV)

        return SPECONV,dSPECONV

    ###############################################################################################

    def nemesisSOfm(self):

        """
            FUNCTION NAME : nemesisSOfm()

            DESCRIPTION : This function computes a forward model for a solar occultation observation

            INPUTS : none

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisSOfm()

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from scipy import interpolate
        from copy import deepcopy

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.Variables1 = deepcopy(self.Variables)
        self.MeasurementX = deepcopy(self.Measurement)
        self.AtmosphereX = deepcopy(self.Atmosphere)
        self.ScatterX = deepcopy(self.Scatter)
        self.StellarX = deepcopy(self.Stellar)
        self.SurfaceX = deepcopy(self.Surface)
        self.LayerX = deepcopy(self.Layer)
        self.SpectroscopyX = deepcopy(self.Spectroscopy)
        self.CIAX = deepcopy(self.CIA)
        flagh2p = False

        #Errors and checks
        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        #Defining spectral range         
        self.Measurement.build_ils(IGEOM=0) 
        wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=None)
            
        #Reading tables in the required wavelength range
        self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = False

        #Mapping variables into different classes
        xmap = self.subprofretg()

        #Calculating the atmospheric paths
        self.LayerX.DUST_UNITS_FLAG = self.AtmosphereX.DUST_UNITS_FLAG
        self.calc_path_SO()
        BASEH_TANHE = np.zeros(self.PathX.NPATH)
        for i in range(self.PathX.NPATH):
            BASEH_TANHE[i] = self.LayerX.BASEH[self.PathX.LAYINC[int(self.PathX.NLAYIN[i]/2),i]]/1.0e3

        #Calling CIRSrad to calculate the spectra
        SPECOUT = self.CIRSrad()

        #Interpolating the spectra to the correct altitudes defined in Measurement
        SPECMOD = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM])
        for i in range(self.MeasurementX.NGEOM):

            #Find altitudes above and below the actual tangent height
            ibase = np.argmin(np.abs(BASEH_TANHE-self.MeasurementX.TANHE[i]))
            base0 = BASEH_TANHE[ibase]/1.0e3
            if base0<=self.MeasurementX.TANHE[i]:
                ibasel = ibase
                ibaseh = ibase + 1
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            if ibaseh>self.PathX.NPATH-1:
                SPECMOD[:,i] = SPECOUT[:,ibasel]
            else:
                fhl = (self.MeasurementX.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
                fhh = (BASEH_TANHE[ibaseh]-self.MeasurementX.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

                SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)


        #Convolving the spectrum with the instrument line shape
        _lgr.info('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL == SpectralCalculationMode.K_TABLES:
            SPECONV,dSPECONV = self.MeasurementX.convg(self.SpectroscopyX.WAVE,SPECMOD,dSPECMOD,IGEOM='All')
        elif self.SpectroscopyX.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES:
            SPECONV = self.MeasurementX.lblconv(self.SpectroscopyX.WAVE,SPECMOD,IGEOM='All')
            dSPECONV = np.zeros([self.MeasurementX.NCONV.max(),self.MeasurementX.NGEOM,self.Variables.NX])

        #Applying any changes to the spectra required by the state vector
        SPECONV,dSPECONV = self.subspecret(SPECONV,dSPECONV)

        #Fitting the polynomial baseline with python
        #basem = self.MeasurementX.MEAS / SPECONV
        
        #ndegree = 2
        #for it in range(self.MeasurementX.NGEOM):
        #    pcoef = np.polyfit(self.MeasurementX.VCONV[:,it]-self.MeasurementX.VCONV[0,it],basem[:,it],ndegree)
        
        #    baseline_fit = np.zeros(self.MeasurementX.NCONV[it])
        #    for ideg in range(ndegree+1):
        #        baseline_fit[:] = baseline_fit[:] + pcoef[ideg] * (self.MeasurementX.VCONV[:,it]-self.MeasurementX.VCONV[0,it])**(ndegree-ideg)
 
        #    SPECONV[:,it] = SPECONV[:,it] * baseline_fit

        return SPECONV


    ###############################################################################################

    def nemesisSOfmg(self):

        """
            FUNCTION NAME : nemesisSOfmg()

            DESCRIPTION : This function computes a forward model for a solar occultation observation and the gradients
                       of the transmission spectrum with respect to the elements in the state vector

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra
                dSPECMOD(NCONV,NGEOM,NX) :: Derivatives of each spectrum in each geometry with
                                        respect to the elements of the state vector

            CALLING SEQUENCE:

                nemesisSOfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from scipy import interpolate
        from copy import deepcopy

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.Variables1 = deepcopy(self.Variables)
        self.MeasurementX = deepcopy(self.Measurement)
        self.AtmosphereX = deepcopy(self.Atmosphere)
        self.ScatterX = deepcopy(self.Scatter)
        self.StellarX = deepcopy(self.Stellar)
        self.SurfaceX = deepcopy(self.Surface)
        self.LayerX = deepcopy(self.Layer)
        self.SpectroscopyX = deepcopy(self.Spectroscopy)
        self.CIAX = deepcopy(self.CIA)
        flagh2p = False

        #Errors and checks
        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        #Defining spectral range         
        self.Measurement.build_ils(IGEOM=0) 
        wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=None)
            
        #Reading tables in the required wavelength range
        self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = False

        #Mapping variables into different classes
        xmap = self.subprofretg()

        #Calculating the atmospheric paths
        self.calc_pathg_SO()
        BASEH_TANHE = np.zeros(self.PathX.NPATH)
        for i in range(self.PathX.NPATH):
            BASEH_TANHE[i] = self.LayerX.BASEH[self.PathX.LAYINC[int(self.PathX.NLAYIN[i]/2),i]]/1.0e3


        #Calling CIRSrad to calculate the spectra
        _lgr.info('Running CIRSradg')
        #SPECOUT,dSPECOUT2,dTSURF = CIRSradg(self.runname,self.Variables,self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
        SPECOUT,dSPECOUT2,dTSURF = self.CIRSradg()

        #Mapping the gradients from Layer properties to Profile properties
        _lgr.info('Mapping gradients from Layer to Profile')
        #Calculating the elements from NVMR+2+NDUST that need to be mapped
        incpar = []
        for i in range(self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST):
            if np.mean(xmap[:,i,:])!=0.0:
                incpar.append(i)

        dSPECOUT1 = map2pro(dSPECOUT2,self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.PathX.NLAYIN,self.PathX.LAYINC,self.LayerX.DTE,self.LayerX.DAM,self.LayerX.DCO,INCPAR=incpar)
        #(NWAVE,NVMR+2+NDUST,NPRO,NPATH)
        del dSPECOUT2

        #Mapping the gradients from Profile properties to elements in state vector
        _lgr.info('Mapping gradients from Profile to State Vector')
        dSPECOUT = map2xvec(dSPECOUT1,self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.Variables.NX,xmap)
        #(NWAVE,NPATH,NX)
        del dSPECOUT1

        #Interpolating the spectra to the correct altitudes defined in Measurement
        SPECMOD = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM])
        dSPECMOD = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM,self.Variables.NX])
        for i in range(self.MeasurementX.NGEOM):

            #Find altitudes above and below the actual tangent height
            ibase = np.argmin(np.abs(BASEH_TANHE-self.MeasurementX.TANHE[i]))
            base0 = BASEH_TANHE[ibase]
            
            if base0<=self.MeasurementX.TANHE[i]:
                ibasel = ibase
                ibaseh = ibase + 1
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            if ibaseh>self.PathX.NPATH-1:
                SPECMOD[:,i] = SPECOUT[:,ibasel]
                dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]
            else:
                fhl = (self.MeasurementX.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
                fhh = (BASEH_TANHE[ibaseh]-self.MeasurementX.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

                SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)
                dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]*(1.-fhl) + dSPECOUT[:,ibaseh,:]*(1.-fhh)

        #Convolving the spectrum with the instrument line shape
        _lgr.info('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL == SpectralCalculationMode.K_TABLES:
            SPECONV,dSPECONV = self.MeasurementX.convg(self.SpectroscopyX.WAVE,SPECMOD,dSPECMOD,IGEOM='All')
        elif self.SpectroscopyX.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES:
            SPECONV,dSPECONV = self.MeasurementX.lblconvg(self.SpectroscopyX.WAVE,SPECMOD,dSPECMOD,IGEOM='All')

        #Calculating the gradients of any parameterisations involving the convolution
        dSPECONV = self.subspeconv(self.SpectroscopyX.WAVE,SPECMOD,dSPECONV)
        
        #Applying any changes to the spectra required by the state vector
        SPECONV,dSPECONV = self.subspecret(SPECONV,dSPECONV)
        
        #Fitting the polynomial baseline with python
        #basem = self.MeasurementX.MEAS / SPECONV
        
        #ndegree = 2
        #for it in range(self.MeasurementX.NGEOM):
        #    pcoef = np.polyfit(self.MeasurementX.VCONV[:,it]-self.MeasurementX.VCONV[0,it],basem[:,it],ndegree)
        
        #    baseline_fit = np.zeros(self.MeasurementX.NCONV[it])
        #    for ideg in range(ndegree+1):
        #        baseline_fit[:] = baseline_fit[:] + pcoef[ideg] * (self.MeasurementX.VCONV[:,it]-self.MeasurementX.VCONV[0,it])**(ndegree-ideg)
                
        #    ix = 0
        #    for ivar in range(self.Variables.NVAR):
            
        #        for j in range(self.Variables.NXVAR[ivar]):
        #            dSPECONV[:,it,ix] = dSPECONV[:,it,ix] * baseline_fit[:]
        #            ix = ix + 1

        #    SPECONV[:,it] = SPECONV[:,it] * baseline_fit
        
        return SPECONV,dSPECONV


    ###############################################################################################

    def nemesisLfm(self):

        """
            FUNCTION NAME : nemesisLfm()

            DESCRIPTION : This function computes a forward model for a limb geometry (thermal emission)
            
                          This forward model is optimised so that all tangent altitudes are calculated simulatneously,
                           substantially increasing the computational speed of the forward model

            INPUTS : none

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisLfm()

            MODIFICATION HISTORY : Juan Alday (08/05/2025)

        """

        from scipy import interpolate
        from copy import deepcopy

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.Variables1 = deepcopy(self.Variables)
        self.MeasurementX = deepcopy(self.Measurement)
        self.AtmosphereX = deepcopy(self.Atmosphere)
        self.ScatterX = deepcopy(self.Scatter)
        self.StellarX = deepcopy(self.Stellar)
        self.SurfaceX = deepcopy(self.Surface)
        self.LayerX = deepcopy(self.Layer)
        self.SpectroscopyX = deepcopy(self.Spectroscopy)
        self.CIAX = deepcopy(self.CIA)
        flagh2p = False

        #Errors and checks
        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        #Defining spectral range         
        self.Measurement.build_ils(IGEOM=0) 
        wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=None)
            
        #Reading tables in the required wavelength range
        self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = False

        #Mapping variables into different classes
        xmap = self.subprofretg()

        #Calculating the atmospheric paths
        self.LayerX.DUST_UNITS_FLAG = self.AtmosphereX.DUST_UNITS_FLAG
        self.calc_path_L()
        BASEH_TANHE = np.zeros(self.PathX.NPATH)
        for i in range(self.PathX.NPATH):
            BASEH_TANHE[i] = self.LayerX.BASEH[self.PathX.LAYINC[int(self.PathX.NLAYIN[i]/2),i]]/1.0e3

        #Calling CIRSrad to calculate the spectra
        SPECOUT = self.CIRSrad()

        #Interpolating the spectra to the correct altitudes defined in Measurement
        SPECMOD = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM])
        for i in range(self.MeasurementX.NGEOM):

            #Find altitudes above and below the actual tangent height
            ibase = np.argmin(np.abs(BASEH_TANHE-self.MeasurementX.TANHE[i]))
            base0 = BASEH_TANHE[ibase]/1.0e3
            if base0<=self.MeasurementX.TANHE[i]:
                ibasel = ibase
                ibaseh = ibase + 1
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            if ibaseh>self.PathX.NPATH-1:
                SPECMOD[:,i] = SPECOUT[:,ibasel]
            else:
                fhl = (self.MeasurementX.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
                fhh = (BASEH_TANHE[ibaseh]-self.MeasurementX.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

                SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)

        #Convolving the spectrum with the instrument line shape
        _lgr.info('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL==0:
            SPECONV = self.MeasurementX.conv(self.SpectroscopyX.WAVE,SPECMOD,IGEOM='All')
        elif self.SpectroscopyX.ILBL==2:
            SPECONV = self.MeasurementX.lblconv(self.SpectroscopyX.WAVE,SPECMOD,IGEOM='All')
        dSPECONV = np.zeros([self.MeasurementX.NCONV.max(),self.MeasurementX.NGEOM,self.Variables.NX])

        #Applying any changes to the spectra required by the state vector
        SPECONV,dSPECONV = self.subspecret(SPECONV,dSPECONV)

        return SPECONV


    ###############################################################################################

    def nemesisLfmg(self):

        """
            FUNCTION NAME : nemesisSOfmg()

            DESCRIPTION : This function computes a forward model for a limb observation (thermal emission) and the gradients
                       of the radiance spectrum with respect to the elements in the state vector

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra
                dSPECMOD(NCONV,NGEOM,NX) :: Derivatives of each spectrum in each geometry with
                                        respect to the elements of the state vector

            CALLING SEQUENCE:

                nemesisLfmg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from scipy import interpolate
        from copy import deepcopy

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.Variables1 = deepcopy(self.Variables)
        self.MeasurementX = deepcopy(self.Measurement)
        self.AtmosphereX = deepcopy(self.Atmosphere)
        self.ScatterX = deepcopy(self.Scatter)
        self.StellarX = deepcopy(self.Stellar)
        self.SurfaceX = deepcopy(self.Surface)
        self.LayerX = deepcopy(self.Layer)
        self.SpectroscopyX = deepcopy(self.Spectroscopy)
        self.CIAX = deepcopy(self.CIA)
        flagh2p = False

        #Errors and checks
        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        #Defining spectral range         
        self.Measurement.build_ils(IGEOM=0) 
        wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=None)
            
        #Reading tables in the required wavelength range
        self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = False

        #Mapping variables into different classes
        xmap = self.subprofretg()

        #Calculating the atmospheric paths
        self.calc_pathg_L()
        BASEH_TANHE = np.zeros(self.PathX.NPATH)
        for i in range(self.PathX.NPATH):
            BASEH_TANHE[i] = self.LayerX.BASEH[self.PathX.LAYINC[int(self.PathX.NLAYIN[i]/2),i]]/1.0e3


        #Calling CIRSrad to calculate the spectra
        _lgr.info('Running CIRSradg')
        #SPECOUT,dSPECOUT2,dTSURF = CIRSradg(self.runname,self.Variables,self.MeasurementX,self.AtmosphereX,self.SpectroscopyX,self.ScatterX,self.StellarX,self.SurfaceX,self.CIAX,self.LayerX,self.PathX)
        SPECOUT,dSPECOUT2,dTSURF = self.CIRSradg()

        #Mapping the gradients from Layer properties to Profile properties
        _lgr.info('Mapping gradients from Layer to Profile')
        #Calculating the elements from NVMR+2+NDUST that need to be mapped
        incpar = []
        for i in range(self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST):
            if np.mean(xmap[:,i,:])!=0.0:
                incpar.append(i)

        dSPECOUT1 = map2pro(dSPECOUT2,self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.PathX.NLAYIN,self.PathX.LAYINC,self.LayerX.DTE,self.LayerX.DAM,self.LayerX.DCO,INCPAR=incpar)
        #(NWAVE,NVMR+2+NDUST,NPRO,NPATH)
        del dSPECOUT2

        #Mapping the gradients from Profile properties to elements in state vector
        _lgr.info('Mapping gradients from Profile to State Vector')
        dSPECOUT = map2xvec(dSPECOUT1,self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR,self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.PathX.NPATH,self.Variables.NX,xmap)
        #(NWAVE,NPATH,NX)
        del dSPECOUT1

        #Interpolating the spectra to the correct altitudes defined in Measurement
        SPECMOD = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM])
        dSPECMOD = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM,self.Variables.NX])
        for i in range(self.MeasurementX.NGEOM):

            #Find altitudes above and below the actual tangent height
            ibase = np.argmin(np.abs(BASEH_TANHE-self.MeasurementX.TANHE[i]))
            base0 = BASEH_TANHE[ibase]
            
            if base0<=self.MeasurementX.TANHE[i]:
                ibasel = ibase
                ibaseh = ibase + 1
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            if ibaseh>self.PathX.NPATH-1:
                SPECMOD[:,i] = SPECOUT[:,ibasel]
                dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]
            else:
                fhl = (self.MeasurementX.TANHE[i]-BASEH_TANHE[ibasel])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])
                fhh = (BASEH_TANHE[ibaseh]-self.MeasurementX.TANHE[i])/(BASEH_TANHE[ibaseh]-BASEH_TANHE[ibasel])

                SPECMOD[:,i] = SPECOUT[:,ibasel]*(1.-fhl) + SPECOUT[:,ibaseh]*(1.-fhh)
                dSPECMOD[:,i,:] = dSPECOUT[:,ibasel,:]*(1.-fhl) + dSPECOUT[:,ibaseh,:]*(1.-fhh)

        #Convolving the spectrum with the instrument line shape
        _lgr.info('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL==0:
            SPECONV,dSPECONV = self.MeasurementX.convg(self.SpectroscopyX.WAVE,SPECMOD,dSPECMOD,IGEOM='All')
        elif self.SpectroscopyX.ILBL==2:
            SPECONV,dSPECONV = self.MeasurementX.lblconvg(self.SpectroscopyX.WAVE,SPECMOD,dSPECMOD,IGEOM='All')

        #Calculating the gradients of any parameterisations involving the convolution
        dSPECONV = self.subspeconv(self.SpectroscopyX.WAVE,SPECMOD,dSPECONV)
        
        #Applying any changes to the spectra required by the state vector
        SPECONV,dSPECONV = self.subspecret(SPECONV,dSPECONV)
        
        return SPECONV,dSPECONV


###############################################################################################

    def nemesisCfm(self):

        """
            FUNCTION NAME : nemesisCfm()

            DESCRIPTION : This function computes a forward model for an upward-looking or downward-looking 
                          instrument observing the atmosphere at different viewing angles

            INPUTS : none

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisCfm()

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from scipy import interpolate
        from copy import deepcopy

        #First we change the reference atmosphere taking into account the parameterisations in the state vector
        self.Variables1 = deepcopy(self.Variables)
        self.MeasurementX = deepcopy(self.Measurement)
        self.AtmosphereX = deepcopy(self.Atmosphere)
        self.ScatterX = deepcopy(self.Scatter)
        self.StellarX = deepcopy(self.Stellar)
        self.SurfaceX = deepcopy(self.Surface)
        self.LayerX = deepcopy(self.Layer)
        self.SpectroscopyX = deepcopy(self.Spectroscopy)
        self.CIAX = deepcopy(self.CIA)
        flagh2p = False

        #Errors and checks
        self.check_gas_spec_atm()
        self.check_wave_range_consistency()
        
        #Defining spectral range         
        self.Measurement.build_ils(IGEOM=0) 
        wavecalc_min,wavecalc_max = self.Measurement.calc_wave_range(apply_doppler=True,IGEOM=None)
            
        #Reading tables in the required wavelength range
        self.SpectroscopyX.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)

        #Setting up flag not to re-compute levels based on hydrostatic equilibrium (unless pressure or tangent altitude are retrieved)
        self.adjust_hydrostat = True

        #Mapping variables into different classes
        xmap = self.subprofretg()
        
        #Selecting the first angle to calculate the path (the actual geometry will be carried with the Measurement class)
        self.ScatterX.SOL_ANG = self.MeasurementX.SOL_ANG[0,0]
        self.ScatterX.EMISS_ANG = self.MeasurementX.EMISS_ANG[0,0]
        self.ScatterX.AZI_ANG = self.MeasurementX.AZI_ANG[0,0]

        #Calculating the atmospheric paths
        self.calc_path_C()
        
        #Calling CIRSrad to calculate the spectra
        SPECOUT = self.CIRSrad()

        #Applying any changes to the spectra required by the state vector
        dSPECOUT = np.zeros([self.SpectroscopyX.NWAVE,self.MeasurementX.NGEOM,self.Variables.NX])
        SPECOUT,dSPECOUT = self.subspecret(SPECOUT,dSPECOUT)

        #Convolving the spectrum with the instrument line shape
        _lgr.info('Convolving spectra and gradients with instrument line shape')
        if self.SpectroscopyX.ILBL == SpectralCalculationMode.K_TABLES:
            SPECONV,dSPECONV = self.MeasurementX.convg(self.SpectroscopyX.WAVE,SPECOUT,dSPECOUT,IGEOM='All')
        elif self.SpectroscopyX.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES:
            SPECONV = self.MeasurementX.lblconv(self.SpectroscopyX.WAVE,SPECOUT,IGEOM='All')

        return SPECONV


    ###############################################################################################

    def nemesisMAPfm(self,NCores=1):

        """
            FUNCTION NAME : nemesisMAPfm()

            DESCRIPTION : This function computes a forward model of a map with several pixels. 
                          The method implemented here assumes that ALL forward models required to
                          construct the map coincide exactly with the locations at which the Atmosphere
                          and Surface are defined. Namely, this means that:

                            - Measurement.FLAT = Atmosphere.LATITUDE = Surface.LATITUDE
                            - Measurement.FLON = Atmosphere.LONGITUDE = Surface.LONGITUDE
                            
                          The only exception for FLAT and FLON not to be equal to the points in the Atmosphere
                          and Surface is if they are equal to -999. In this case, that particular forward model
                          will be zero at all wavelengths. This exception is included to properly reconstruct
                          the FOV of the instrument, in the case that it includes points outside the planet's disk.
                          
                          In order to optimise the computations, a forward model is calculated at every LOCATION
                          defined in Atmosphere/Surface (or at every unique FLAT/FLON). The forward models are
                          then combined as required to perform the convolution with the Point Spread Function (WGEOM)
                          of the instrument.

            INPUTS : none

            OPTIONAL INPUTS:
            
                NCores :: Number of cores to use to compute the forward models in different locations in parallel

            OUTPUTS :

                SPECMOD(NCONV,NGEOM) :: Modelled spectra

            CALLING SEQUENCE:

                ForwardModel.nemesisMAPfm()

            MODIFICATION HISTORY : Juan Alday (18/04/2023)

        """

        from copy import copy
        
        #Errors and checkcs
        self.check_gas_spec_atm()
        
        #Checking that all FLAT and FLON points exist in the Atmosphere and Surface
        for iGEOM in range(self.Measurement.NGEOM):
            for iAV in range(self.Measurement.NAV[iGEOM]):
                
                #If FLAT is nan it means measurement is outside the disk
                if np.isnan(self.Measurement.FLAT[iGEOM,iAV])==False:
                
                    iex = np.where( (self.Atmosphere.LATITUDE==self.Measurement.FLAT[iGEOM,iAV]) & (self.Atmosphere.LONGITUDE==self.Measurement.FLON[iGEOM,iAV]) &
                                (self.Surface.LATITUDE==self.Measurement.FLAT[iGEOM,iAV]) & (self.Surface.LONGITUDE==self.Measurement.FLON[iGEOM,iAV]))[0]
                
                    if len(iex)==0:
                        raise ValueError('error in nemesisMAPfm :: All FLAT/FLON points for the forward model must coincide with the locations in Atmosphere and Surface')
                    
        #Calculating a forward model for each LOCATION on the planet
        SPEC = np.zeros((self.Spectroscopy.NWAVE,self.Atmosphere.NLOCATIONS))  #Modelled spectra at each of the locations
        
        if NCores==1:    #Only one available core
        
            for ISPEC in range(self.Atmosphere.NLOCATIONS):
                
                _lgr.info(f"nemesisMAPfm :: Calculating spectrum {(ISPEC,'of ',self.Atmosphere.NLOCATIONS)}")
                SPEC[:,ISPEC] = calc_spectrum_location(ISPEC,self.Atmosphere,self.Surface,self.Measurement,self.Scatter,self.Spectroscopy,self.CIA,self.Stellar,self.Variables,self.Layer)
                
        else:            #Parallel computation of the forward models

            _lgr.info('nemesisMAPfm :: Calculating spectra at different locations in parallel')

            #ray.init(num_cpus=NCores)
            #SPECtot_ids = []
            #for ISPEC in range(self.Atmosphere.NLOCATIONS):
            #    SPECtot_ids.append(calc_spectrum_location_parallel.remote(ISPEC,self.Atmosphere,self.Surface,self.Measurement,self.Scatter,self.Spectroscopy,self.CIA,self.Stellar,self.Variables,self.Layer))
            
            #Block until the results have finished and get the results.
            #SPECtot1 = ray.get(SPECtot_ids)
            #for ix in range(self.Atmosphere.NLOCATIONS):
            #    SPEC[0:self.Spectroscopy.NWAVE,ix] = SPECtot1[ix]
            #ray.shutdown()
            
            
        #Convolving the spectra with the point spread function (WGEOM) for each geometry
        _lgr.info('nemesisMAPfm :: Convolving the measurements with the Point Spread Function')
        SPECMOD = np.zeros((self.Spectroscopy.NWAVE,self.Measurement.NGEOM))
        for iGEOM in range(self.Measurement.NGEOM):
            
            #Going through each point within the instantaneous FOV
            for iAV in range(self.Measurement.NAV[iGEOM]):
                
                if((np.isnan(self.Measurement.FLAT[iGEOM,iAV])==False) and (np.isnan(self.Measurement.FLON[iGEOM,iAV])==False)):
                    iloc = np.where((self.Atmosphere.LATITUDE==self.Measurement.FLAT[iGEOM,iAV]) & (self.Atmosphere.LONGITUDE==self.Measurement.FLON[iGEOM,iAV]))[0]
                    SPECMOD[:,iGEOM] = SPECMOD[:,iGEOM] + SPEC[:,iloc[0]] * self.Measurement.WGEOM[iGEOM,iAV]
        
            SPECMOD[:,iGEOM] = SPECMOD[:,iGEOM] / np.sum(self.Measurement.WGEOM[iGEOM,0:self.Measurement.NAV[iGEOM]])
            
            
        #Applying any changes to the spectra required by the state vector
        self.MeasurementX = copy(self.Measurement)
        dSPECMOD = np.zeros((self.Spectroscopy.NWAVE,self.Measurement.NGEOM,self.Variables.NX))
        SPECPSF,dSPEC = self.subspecret(SPECMOD,dSPECMOD,IGEOM=None)
        
        #Convolving the spectrum with the instrument line shape
        _lgr.info('nemesisMAPfm :: Convolving spectra and gradients with instrument line shape')
        if self.Spectroscopy.ILBL == SpectralCalculationMode.K_TABLES:
            SPECONV,dSPECONV = self.MeasurementX.convg(SPECMOD,dSPECMOD,IGEOM='All')
        elif self.Spectroscopy.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES:
            SPECONV,dSPECONV = self.MeasurementX.lblconvg(SPECMOD,dSPECMOD,IGEOM='All')

        return SPECONV


    def chunked_execution(self, args):
        
        """
            FUNCTION NAME : chunked_execution()

            DESCRIPTION :

                This function takes chunks from the parallel execution in jacobian_nemesis and
                sends distributes jobs within the chunks to execute_fm.

            MODIFICATION HISTORY : Joe Penn (9/07/2024)

        """
        
        start, end, xnx, ixrun, nemesisSO, nemesisL, YNtot, nfm = args
        results = np.copy(YNtot)  # Local copy to prevent conflicts
        for ifm in range(start, end):
            inp = (ifm, nfm, xnx, ixrun, nemesisSO, nemesisL, results)
            results = self.execute_fm(inp)
        return start, results

    def execute_fm(self, inp):
        
        """
            FUNCTION NAME : execute_fm()

            DESCRIPTION :

                This function is used to compute the forward models for jacobian_nemesis.
                Print outputs from the forward models are supressed to avoid too much output.

            MODIFICATION HISTORY : Joe Penn (9/07/2024)

        """
        import archnemesis.cfg.logs
        
        # Unpack input tuple
        ifm, nfm, xnx, ixrun, nemesisSO, nemesisL, YNtot = inp
        # ifm - index of forward model
        # nfm - number of forward models
        # xnx - array holding state vectors for all parallel forward models
        # ixrun - seems to be the number of state vector entries for this forward model (unsure about this)
        # nemesisSO - boolean flag to perform solar occultation forward model
        # nemesisL - boolean flag to perform limb forward model
        # YNtot - modelled spectra for all parallel forward models
        
        # define variables
        SPECMOD = None
        
        
        # Find the method to use when modelling the spectrum
        nemesis_method = self.select_nemesis_fm(nemesisSO, nemesisL, analytical_gradient=False)
        
        _lgr.info(f'Calculating forward model {ifm+1}/{nfm}')
        
        # load state vector with state for this specific forward model
        self.Variables.XN = xnx[:, ixrun[ifm]]
        
        
        # Put as little as possible in here so we only have to handle a small subset of state adjustment
        try:
            # Turn off warning and below logging so we are not flooded with output
            archnemesis.cfg.logs.push_packagewide_level(logging.ERROR)
            
            # model the spectrum
            SPECMOD = nemesis_method()
        finally:
            # Stop disabling logging levels
            archnemesis.cfg.logs.pop_packagewide_level()
        
        
        
        if SPECMOD is not None:
            #Only Re-shape calculated spectrum into the shape of the measurement vector if the calculation completed
            ik = 0
            for igeom in range(self.Measurement.NGEOM):
                YNtot[ik:ik+self.Measurement.NCONV[igeom],ifm] = SPECMOD[0:self.Measurement.NCONV[igeom],igeom]
                ik += self.Measurement.NCONV[igeom]
            
            _lgr.info(f'Calculated forward model {ifm+1}/{nfm}')
        else:
            # If calculation failed, throw an error.
            raise RuntimeError(f'Something went wrong when calculating forward model {ifm+1}/{nfm}. Modelled spectra was not calculated.')
            
        return YNtot
    
    ###############################################################################################

    def jacobian_nemesis(self,NCores=1,nemesisSO=False,nemesisL=False):

        """

            FUNCTION NAME : jacobian_nemesis()

            DESCRIPTION :

                This function calculates the Jacobian matrix by calling nx+1 times nemesisSOfm().
                This routine is set up so that each forward model is calculated in parallel,
                increasing the computational speed of the code

            INPUTS :

                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated
                NCores :: Number of cores that can be used to parallelise the calculation of the jacobian matrix

            OUTPUTS :

                YN(NY) :: New measurement vector
                KK(NY,NX) :: Jacobian matrix

            CALLING SEQUENCE:

                YN,KK = jacobian_nemesis(Variables,Measurement,Atmosphere,Scatter,Stellar,Surface,CIA,Layer)

            MODIFICATION HISTORY : Joe Penn (9/07/2024)

        """

        from copy import deepcopy


        #################################################################################
        # Making some calculations for storing all the arrays
        #################################################################################

        #Constructing state vector after perturbation of each of the elements and storing in matrix

        self.Variables.calc_DSTEP() #Calculating the step size for the perturbation of each element
        
        #nxn = self.Variables.NX+1
        xnx = np.zeros([self.Variables.NX,self.Variables.NX+1], dtype=float)
        
        xnx[:,0] = self.Variables.XN
        xnx[:,1:] = np.repeat(self.Variables.XN[:,None], self.Variables.NX, axis=1) + np.diag(self.Variables.DSTEP)
        zeros_mask = xnx[:,1:] == 0
        xnx[:,1:][zeros_mask] = 0.05
        


        #################################################################################
        # Calculating the first forward model and the analytical part of Jacobian
        #################################################################################

        #self.Variables.NUM[:] = 1     #Uncomment for trying numerical differentiation
        if self.Scatter.ISCAT != ScatteringCalculationMode.THERMAL_EMISSION:
            self.Variables.NUM[:] = 1  #If scattering is present, gradients are calculated numerically
        
        
        ian1 = np.where(self.Variables.NUM==0)  #Gradients calculated using CIRSradg
        ian1 = ian1[0]

        iYN = 0
        KK = np.zeros([self.Measurement.NY,self.Variables.NX])

        if len(ian1)>0:

            _lgr.info('Calculating analytical part of the Jacobian :: Calling nemesisfmg ')
            nemesis_method = self.select_nemesis_fm(nemesisSO, nemesisL, analytical_gradient=True)

            SPECMOD,dSPECMOD = nemesis_method()
                
            #Re-shaping the arrays to create the measurement vector and Jacobian matrix
            YN = np.zeros(self.Measurement.NY)
            ik = 0
            for igeom in range(self.Measurement.NGEOM):
                YN[ik:ik+self.Measurement.NCONV[igeom]] = SPECMOD[0:self.Measurement.NCONV[igeom],igeom]
                KK[ik:ik+self.Measurement.NCONV[igeom],:] = dSPECMOD[0:self.Measurement.NCONV[igeom],igeom,:]
                ik += self.Measurement.NCONV[igeom]

            iYN = 1 #Indicates that some of the gradients and the measurement vector have already been caculated

        #################################################################################
        # Calculating all the required forward models for numerical differentiation
        #################################################################################

        inum1 = np.where( (self.Variables.NUM==1) & (self.Variables.FIX==0) ) # only do numerical differentiation for elements of state vector that can vary (i.e. FIX==0, 'not fixed')
        inum = inum1[0]

        if iYN==0:
            nfm = len(inum) + 1  #Number of forward models to run to calculate the Jacobian and measurement vector
            ixrun = np.zeros(nfm,dtype='int32')
            ixrun[0] = 0
            ixrun[1:nfm] = inum[:] + 1
        else:
            nfm = len(inum)  #Number of forward models to run to calculate the Jacobian
            ixrun = np.zeros(nfm,dtype='int32')
            ixrun[0:nfm] = inum[:] + 1


        #Calling the forward model nfm times to calculate the measurement vector for each case
        YNtot = np.zeros((self.Measurement.NY,nfm))

        
        if nfm>0:
            _lgr.info('Calculating numerical part of the Jacobian :: running '+str(nfm)+' forward models ')
            
            # Splitting into chunks and parallelising
            NCores = min(NCores,nfm)
            base_chunk_size = nfm // NCores
            remainder = nfm % NCores

            chunks = [(i * base_chunk_size + min(i, remainder),
                       (i + 1) * base_chunk_size + min(i + 1, remainder),
                       xnx, ixrun, nemesisSO, nemesisL, YNtot, nfm) for i in range(NCores)]

             #with Pool(NCores) as pool:
                 #results = pool.map(self.chunked_execution, chunks)

            results = Parallel(n_jobs=NCores)(
                delayed(self.chunked_execution)(chunk) for chunk in chunks
            )
            # Reorder and combine results based on their starting index
            ordered_results = sorted(results, key=lambda x: x[0])
            YNtot = np.sum(np.stack([res[1] for res in ordered_results]), axis=0)

            if iYN==0:
                YN = np.zeros(self.Measurement.NY)
                YN[:] = YNtot[0:self.Measurement.NY,0]

        #################################################################################
        # Calculating the Jacobian matrix
        #################################################################################

        for i in range(len(inum)):

            if iYN==0:
                ifm = i + 1
            else:
                ifm = i

            xn1 = self.Variables.XN[inum[i]] * 1.05
            if xn1==0.0:
                xn1=0.05
            if self.Variables.FIX[inum[i]] == 0:
                KK[:,inum[i]] = (YNtot[:,ifm]-YN)/(xn1-self.Variables.XN[inum[i]])

        return YN,KK



    ###############################################################################################
    ###############################################################################################
    # MAPPING THE STATE VECTOR (MODEL PARAMETERISATIONS) INTO THE REFERENCE CLASSES
    ###############################################################################################
    ###############################################################################################


    ###############################################################################################

    def _get_ipar(self, varident : np.ndarray[[3],int]) -> None | int:
        """
        Calculates the value of 'ipar', an integer that encodes which
        atmospheric profile is to be retrieved. Returns 'None' if the
        parameterised model is not an atmospheric one.
        """
        if ((varident[2]<100) 
                    or ((varident[2]>=1000) and (varident[2]<=1100))
                ):
            if varident[0]==0:     #Temperature is to be retrieved
                ipar = self.AtmosphereX.NVMR
            elif varident[0]>0:    #Gas VMR is to be retrieved
                jvmr = np.nonzero( (np.array(self.AtmosphereX.ID)==varident[0]) & (np.array(self.AtmosphereX.ISO)==varident[1]) )[0]
                assert len(jvmr)==1, 'Cannot have more than one gas VMR retrieved at once'
                ipar = int(jvmr[0])
            elif varident[0]<0: # aerosol species density is to be retrieved
                jcont = -int(varident[0])
                ipar = self.AtmosphereX.NVMR + jcont
            return ipar
        else:
            return None

    def subprofretg(self):

        """
        FUNCTION NAME : subprogretg()

        DESCRIPTION : Updates the reference classes based on the variables and parameterisations in the
                      state vector. Changes to other parameters in the model based on the variables
                      and parameterisations in the state vector are also performed here. However,
                      the functional derivatives to these other parameters are not included since
                      they cannot be determined analytically.

        INPUTS : none

        OPTIONAL INPUTS: none

        OUTPUTS :

            xmap(maxv,ngas+2+ncont,npro) :: Matrix relating functional derivatives calculated
                                             by CIRSRADG to the elements of the state vector.
                                             Elements of XMAP are the rate of change of
                                             the profile vectors (i.e. temperature, vmr prf
                                             files) with respect to the change in the state
                                             vector elements. So if X1(J) is the modified
                                             temperature,vmr,clouds at level J to be
                                             written out to runname.prf or aerosol.prf then
                                            XMAP(K,L,J) is d(X1(J))/d(XN(K)) and where
                                            L is the identifier (1 to NGAS+1+2*NCONT)

        CALLING SEQUENCE:

            xmap = ForwardModel.subprofretg()

        MODIFICATION HISTORY : Juan Alday (15/03/2022)

        """

        #Checking if hydrostatic equilibrium needs to be used for any parameterisation
        if self.Variables.JPRE!=-1:
            self.adjust_hydrostat = True
        if self.Variables.JTAN!=-1:
            self.adjust_hydrostat = True
        #Modify profile via hydrostatic equation to make sure the atm is in hydrostatic equilibrium
        if self.adjust_hydrostat==True:
            if self.Variables.JPRE==-1:
                jhydro = 0
                #Then we modify the altitude levels and keep the pressures fixed
                
                self.AtmosphereX.adjust_hydrostatH()
                self.AtmosphereX.calc_grav()   #Updating the gravity values at the new heights
            else:
                #Then we modifify the pressure levels and keep the altitudes fixed
                jhydro = 1
                for i in range(self.Variables.NVAR):
                    if self.Variables.VARIDENT[i,0]==666:
                        htan = self.Variables.VARPARAM[i,0] * 1000.
                ptan = np.exp(self.Variables.XN[self.Variables.JPRE]) * 101325.
                self.AtmosphereX.adjust_hydrostatP(htan,ptan)

        #Calculate atmospheric density
        rho = self.AtmosphereX.calc_rho() #kg/m3


        # NOTE: instead of having two different versions of `xmap`, just use the multiple location version.
        #Initialising xmap
        if self.AtmosphereX.NLOCATIONS==1:
            # `xmap` is functional derivatives of state vector w.r.t profiles for each location
            # let:
            #   k = index of state_vector
            #   l = index of profile
            #   j = index of point in the l^th profile (all profiles are on the same height/pressure grid
            #   i = index of the location
            # then
            #   xmap[k,l,j,i] = d[profile_j]/d[state_vector_k] for the l^th profile at the i^th location
            
            # `xmap` shape is defined as follows:
            # shape = (
            #   number of values in the state vector, 
            #   number of volume mixing ratios in atmosphere 
            #       + number of aerosol profiles in atmosphere
            #       + 1 for para h2 fraction profile in atmosphere
            #       + 1 for fractional cloud cover profile in atmosphere,
            #   number of points in atmosphere profiles (i.e. number of height levels, at which pressure, temperature, etc. is defined),
            #   number of locations
            # )
            xmap = np.zeros((self.Variables.NX,self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST,self.AtmosphereX.NP))
        else:
            #raise ValueError('error in subprofretg :: subprofretg has not been upgraded yet to deal with multiple locations')
            xmap = np.zeros((self.Variables.NX,self.AtmosphereX.NVMR+2+self.AtmosphereX.NDUST,self.AtmosphereX.NP,self.AtmosphereX.NLOCATIONS))

        #Going through the different variables an updating the atmosphere accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):
            
            
            ipar = self._get_ipar(self.Variables.VARIDENT[ivar])


            #Model parameterisation applies to atmospheric parameters in multiple locations
            if ((self.Variables.VARIDENT[ivar,2]>=1000) 
                    and (self.Variables.VARIDENT[ivar,2]<=1100)
                    and (self.AtmosphereX.NLOCATION <= 1)
                ):
                    raise ValueError('error in subprofretg :: Models 1000-1100 are meant to be used for models of atmospheric properties in multiple locations')


            # Calculate state vector for the model
            self.Variables.models[ivar].calculate_from_subprofretg(self, ix, ipar, ivar, xmap)
            ix += self.Variables.models[ivar].n_state_vector_entries


        #Now check if any gas in the retrieval saturates

        if self.AtmosphereX.AMFORM == AtmosphericProfileFormatEnum.CALC_MOLECULAR_WEIGHT_SCALE_VMR_TO_ONE:
            #Find the gases whose vmr is retrieved so that we do not adjust them
            ISCALE = np.ones(self.AtmosphereX.NVMR,dtype='int32')
            for ivar in range(self.Variables.NVAR):
                if self.Variables.VARIDENT[ivar,0]>0:
                    #Then it is gas parameterisation
                    igas = np.where( (self.AtmosphereX.ID==self.Variables.VARIDENT[ivar,0]) & (self.AtmosphereX.ISO==self.Variables.VARIDENT[ivar,1]) )[0]
                    if len(igas)==1:
                        ISCALE[igas] = 0
                    elif len(igas)>1:
                        raise ValueError('error :: There are several parameterisations affecting the same gas')
                        
            self.AtmosphereX.adjust_VMR(ISCALE=ISCALE)
            self.AtmosphereX.calc_molwt()

        
        
        #Re-scale H/P based on the hydrostatic equilibrium equation
        if self.adjust_hydrostat==True:
            if jhydro==0:
                #Then we modify the altitude levels and keep the pressures fixed
                self.AtmosphereX.adjust_hydrostatH()
                self.AtmosphereX.calc_grav()   #Updating the gravity values at the new heights
            else:
                #Then we modifify the pressure levels and keep the altitudes fixed
                self.AtmosphereX.adjust_hydrostatP(htan,ptan)


        #Patch for model -1, since the aerosol density is defined in particles per gram of atm (depends on the density)
        #Going through the different variables an updating the atmosphere accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):

            ipar = self._get_ipar(self.Variables.VARIDENT[ivar])

            #Model parameterisation applies to atmospheric parameters in multiple locations
            if ((self.Variables.VARIDENT[ivar,2]>=1000) 
                    and (self.Variables.VARIDENT[ivar,2]<=1100)
                    and (self.AtmosphereX.NLOCATION <= 1)
                ):
                    raise ValueError('error in subprofretg :: Models 1000-1100 are meant to be used for models of atmospheric properties in multiple locations')

            # Patch state vector for the model
            self.Variables.models[ivar].patch_from_subprofretg(self, ix, ipar, ivar, xmap)
            ix += self.Variables.models[ivar].n_state_vector_entries
        
        return xmap


    ###############################################################################################

    def subspecret(self,SPECMOD,dSPECMOD):

        """
        FUNCTION NAME : subspecret()

        DESCRIPTION : Performs any required changes to the modelled spectra based on the parameterisations
                      included in the state vector. These changes can include for example the superposition
                      of diffraction orders in an AOTF spectrometer or the scaling of the spectra to account
                      for hemispheric assymmetries in exoplanet retrievals.

        INPUTS :

            Measurement :: Python class defining the observation
            Variables :: Python class defining the parameterisations and state vector
            SPECMOD(NCONV,NGEOM) :: Modelled spectrum in each geometry (convolved with ILS)
            dSPECMOD(NCONV,NGEOM,NX) :: Modelled gradients in each geometry (convolved with ILS)

        OPTIONAL INPUTS: None
        
        OUTPUTS :

            SPECMOD :: Updated modelled spectrum
            dSPECMOD :: Updated gradients

        CALLING SEQUENCE:

            SPECMOD,dSPECMOD = subspecret(SPECMOD,dSPECMOD)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)

        """

        #Going through the different variables an updating the spectra and gradients accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):

            self.Variables.models[ivar].calculate_from_subspecret(self, ix, ivar, SPECMOD, dSPECMOD)
            ix += self.Variables.models[ivar].n_state_vector_entries

        return SPECMOD,dSPECMOD

    ###############################################################################################

    def subspeconv(self,WAVE,SPECMOD,dSPECONV):

        """
        FUNCTION NAME : subspeconv()

        DESCRIPTION : Calculate the gradients for any model parameterisation that involves the convolution
                       of the modelled spectrum with the instrument lineshape. These parameterisations can 
                       include for example the retrieval of the spectral resolution of the instrument function,
                       in case it is not well characterised. 

        INPUTS :

            Measurement :: Python class defining the observation
            Variables :: Python class defining the parameterisations and state vector
            WAVE(NWAVE) :: Calculation wavelengths or wavenumbers
            SPECMOD(NWAVE,NGEOM) :: Modelled spectrum in each geometry (not yet convolved with ILS)
            dSPECONV(NCONV,NGEOM,NX) :: Modelled gradients in each geometry (previously convolved with ILS)

        OPTIONAL INPUTS: none

        OUTPUTS :

            dSPECONV :: Updated gradients in each geometry

        CALLING SEQUENCE:

            SPECONV = subspecret(Measurement,Variables,SPECMOD,dSPECONV)

        MODIFICATION HISTORY : Juan Alday (15/07/2022)

        """

        #Going through the different variables an updating the spectra and gradients accordingly
        ix = 0
        for ivar in range(self.Variables.NVAR):

            if self.Variables.VARIDENT[ivar,0]==229:
                #Model 229. Retrieval of instrument line shape for ACS-MIR (v2)
                #***************************************************************

                #Getting the reference values for the ILS parameterisation
                par1 = self.Variables.XN[ix]
                par2 = self.Variables.XN[ix+1]
                par3 = self.Variables.XN[ix+2]
                par4 = self.Variables.XN[ix+3]
                par5 = self.Variables.XN[ix+4]
                par6 = self.Variables.XN[ix+5]
                par7 = self.Variables.XN[ix+6]

                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par6,par7)

                #Performing first convolution of the spectra
                SPECONV_ref = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')

                #Going through each of the parameters to calculate the gradients

                par11 = self.Variables.XN[ix]*1.05
                self.MeasurementX = model229(self.MeasurementX,par11,par2,par3,par4,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix] = (SPECONV1-SPECONV_ref)/(par11-par1)

                par21 = self.Variables.XN[ix+1]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par21,par3,par4,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+1] = (SPECONV1-SPECONV_ref)/(par21-par2)

                par31 = self.Variables.XN[ix+2]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par31,par4,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+2] = (SPECONV1-SPECONV_ref)/(par31-par3)

                par41 = self.Variables.XN[ix+3]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par41,par5,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+3] = (SPECONV1-SPECONV_ref)/(par41-par4)

                par51 = self.Variables.XN[ix+4]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par51,par6,par7)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+4] = (SPECONV1-SPECONV_ref)/(par51-par5)

                par61 = self.Variables.XN[ix+5]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par61,par7)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+5] = (SPECONV1-SPECONV_ref)/(par61-par6)

                par71 = self.Variables.XN[ix+6]*1.05
                self.MeasurementX = model229(self.MeasurementX,par1,par2,par3,par4,par5,par6,par71)
                SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                dSPECONV[:,:,ix+6] = (SPECONV1-SPECONV_ref)/(par71-par7)

                ipar = -1
                ix = ix + self.Variables.NXVAR[ivar]

            elif self.Variables.VARIDENT[ivar,0]==230:
                #Model 230. Retrieval of multiple instrument line shapes for ACS-MIR (multiple spectral windows)
                #***************************************************************

                #Getting reference values and calculating the reference convolved spectrum
                nwindows = int(self.Variables.VARPARAM[ivar,0])
                liml = np.zeros(nwindows)
                limh = np.zeros(nwindows)
                i0 = 1
                for iwin in range(nwindows):
                    liml[iwin] = self.Variables.VARPARAM[ivar,i0]
                    limh[iwin] = self.Variables.VARPARAM[ivar,i0+1]
                    i0 = i0 + 2

                par1 = np.zeros((7,nwindows))
                il = 0
                for iwin in range(nwindows):
                    for jwin in range(7):
                        par1[jwin,iwin] = self.Variables.XN[ix+il]
                        il = il + 1

                self.MeasurementX = model230(self.MeasurementX,nwindows,liml,limh,par1)

                #Performing first convolution of the spectra
                SPECONV_ref = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')

                il = 0
                for iwin in range(nwindows):
                    for jwin in range(7):
                        par2 = np.zeros(par1.shape)
                        par2[:,:] = par1[:,:]
                        par2[jwin,iwin] = par1[jwin,iwin] * 1.05

                        self.MeasurementX = model230(self.MeasurementX,nwindows,liml,limh,par2)

                        SPECONV1 = self.MeasurementX.lblconv(WAVE,SPECMOD,IGEOM='All')
                        dSPECONV[:,:,ix+il] = (SPECONV1-SPECONV_ref)/(par2[jwin,iwin]-par1[jwin,iwin])

                        il = il + 1

                ix = ix + self.Variables.NXVAR[ivar]

            else:
                ix = ix + self.Variables.NXVAR[ivar]

        return dSPECONV

    ###############################################################################################
    ###############################################################################################
    # PATH CALCULATION AND DEFINITION OF GEOMETRY
    ###############################################################################################
    ###############################################################################################


    ###############################################################################################

    def select_Measurement(self,IGEOM,IAV):

        """
            FUNCTION NAME : select_Measurement()

            DESCRIPTION : This function fills the MeasurementX class with the information about
                          a specific measurement that wants to be modelled

            INPUTS :

                IGEOM :: Integer defining the number of the geometry (from 0 to NGEOM - 1)
                IAV :: Integer defining the number of the averaging point for the geometry (from 0 to NAV(IGEOM))

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated Measurement1 class

            CALLING SEQUENCE:

                ForwardModel.select_Measurement(IGEOM,IAV)

            MODIFICATION HISTORY : Juan Alday (25/08/2022)

        """

        self.MeasurementX.NGEOM = 1
        self.MeasurementX.FWHM = self.Measurement.FWHM
        self.MeasurementX.IFORM = SpectraUnit(self.Measurement.IFORM)
        self.MeasurementX.ISPACE = WaveUnit(self.Measurement.ISPACE)

        #Selecting the measurement and spectral points
        NCONV = np.zeros(self.MeasurementX.NGEOM,dtype='int32')
        VCONV = np.zeros((self.Measurement.NCONV[IGEOM],self.MeasurementX.NGEOM))
        MEAS = np.zeros((self.Measurement.NCONV[IGEOM],self.MeasurementX.NGEOM))
        ERRMEAS = np.zeros((self.Measurement.NCONV[IGEOM],self.MeasurementX.NGEOM))
        
        NCONV[0] = self.Measurement.NCONV[IGEOM]
        self.MeasurementX.NCONV = NCONV

        VCONV[:,0] = self.Measurement.VCONV[0:NCONV[0],IGEOM]
        self.MeasurementX.edit_VCONV(VCONV)
        
        MEAS[:,0] = self.Measurement.MEAS[0:NCONV[0],IGEOM]
        self.MeasurementX.edit_MEAS(MEAS)
        
        ERRMEAS[:,0] = self.Measurement.ERRMEAS[0:NCONV[0],IGEOM]
        self.MeasurementX.edit_ERRMEAS(ERRMEAS)
        
        #Selecting the geometry
        if self.Measurement.EMISS_ANG[IGEOM,IAV]>=0:
            NAV = np.ones(self.MeasurementX.NGEOM,dtype='int32')
            FLAT = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            FLON = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            WGEOM = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            SOL_ANG = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            EMISS_ANG = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            AZI_ANG = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            
            FLAT[0,0] = self.Measurement.FLAT[IGEOM,IAV]
            FLON[0,0] = self.Measurement.FLON[IGEOM,IAV]
            WGEOM[0,0] = self.Measurement.WGEOM[IGEOM,IAV]
            AZI_ANG[0,0] = self.Measurement.AZI_ANG[IGEOM,IAV]
            SOL_ANG[0,0] = self.Measurement.SOL_ANG[IGEOM,IAV]
            EMISS_ANG[0,0] = self.Measurement.EMISS_ANG[IGEOM,IAV]
            
            self.MeasurementX.NAV = NAV
            self.MeasurementX.edit_FLAT(FLAT)
            self.MeasurementX.edit_FLON(FLON)
            self.MeasurementX.edit_WGEOM(WGEOM)
            self.MeasurementX.edit_AZI_ANG(AZI_ANG)
            self.MeasurementX.edit_SOL_ANG(SOL_ANG)
            self.MeasurementX.edit_EMISS_ANG(EMISS_ANG)
        else:
            NAV = np.ones(self.MeasurementX.NGEOM,dtype='int32')
            FLAT = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            FLON = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            WGEOM = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            TANHE = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            EMISS_ANG = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            AZI_ANG = np.zeros((self.MeasurementX.NGEOM,NAV[0]))
            
            FLAT[0,0] = self.Measurement.FLAT[IGEOM,IAV]
            FLON[0,0] = self.Measurement.FLON[IGEOM,IAV]
            WGEOM[0,0] = self.Measurement.WGEOM[IGEOM,IAV]
            TANHE[0,0] = self.Measurement.TANHE[IGEOM,IAV]
            EMISS_ANG[0,0] = self.Measurement.EMISS_ANG[IGEOM,IAV]
            
            self.MeasurementX.NAV = NAV
            self.MeasurementX.edit_FLAT(FLAT)
            self.MeasurementX.edit_FLON(FLON)
            self.MeasurementX.edit_WGEOM(WGEOM)
            self.MeasurementX.edit_TANHE(TANHE)
            self.MeasurementX.edit_EMISS_ANG(EMISS_ANG)

        self.MeasurementX.LATITUDE = self.MeasurementX.FLAT[0,0]
        self.MeasurementX.LONGITUDE = self.MeasurementX.FLON[0,0]


    ###############################################################################################

    def select_location(self,ILOC):

        """
            FUNCTION NAME : select_location()

            DESCRIPTION : This function fills the AtmosphereX and SurfaceX classes with the information
                          about the specific location where the forward model wants to be performed

            INPUTS :

                ILOC :: Integer defining the number of the location (from 0 to NLOCATION - 1)

            OPTIONAL INPUTS: none

            OUTPUTS :

                Updated MeasurementX and SurfaceX classes
            CALLING SEQUENCE:

                ForwardModel.select_location(ILOC)

            MODIFICATION HISTORY : Juan Alday (20/04/2023)

        """
        
        #Selecting the required atmosphere
        #################################################################
        
        self.AtmosphereX.NLOCATIONS = 1
        
        self.AtmosphereX.LATITUDE = self.Atmosphere.LATITUDE[ILOC]
        self.AtmosphereX.LONGITUDE = self.Atmosphere.LONGITUDE[ILOC]
        self.AtmosphereX.RADIUS = self.Atmosphere.RADIUS[ILOC]
        self.AtmosphereX.edit_H(self.Atmosphere.H[:,ILOC])
        self.AtmosphereX.edit_P(self.Atmosphere.P[:,ILOC])
        self.AtmosphereX.edit_T(self.Atmosphere.T[:,ILOC])
        self.AtmosphereX.edit_VMR(self.Atmosphere.VMR[:,:,ILOC])
        self.AtmosphereX.GRAV = self.Atmosphere.GRAV[:,ILOC]
        self.AtmosphereX.MOLWT = self.Atmosphere.MOLWT[:,ILOC]
        
        if self.Atmosphere.NDUST>0:
            self.AtmosphereX.edit_DUST(self.Atmosphere.DUST[:,:,ILOC])
            
        
        #Selecting the required surface
        ##################################################################

        if self.SurfaceX.GASGIANT==False: #Checking if there is surface

            self.SurfaceX.NLOCATIONS = 1
            
            self.SurfaceX.LATITUDE = self.Surface.LATITUDE[ILOC]
            self.SurfaceX.LONGITUDE = self.Surface.LONGITUDE[ILOC]
            self.SurfaceX.TSURF = self.Surface.TSURF[ILOC]
            
            self.SurfaceX.edit_EMISSIVITY(self.Surface.EMISSIVITY[:,ILOC])
            
            #Checking if it is a Hapke surface
            if self.SurfaceX.LOWBC==LowerBoundaryCondition.HAPKE:
            
                self.SurfaceX.edit_SGLALB(self.Surface.SGLALB[:,ILOC])
                self.SurfaceX.edit_BS0(self.Surface.BS0[:,ILOC])
                self.SurfaceX.edit_hs(self.Surface.hs[:,ILOC])
                self.SurfaceX.edit_BC0(self.Surface.BC0[:,ILOC])
                self.SurfaceX.edit_hc(self.Surface.hc[:,ILOC])
                self.SurfaceX.edit_K(self.Surface.K[:,ILOC])
                self.SurfaceX.edit_ROUGHNESS(self.Surface.ROUGHNESS[:,ILOC])
                self.SurfaceX.edit_G1(self.Surface.G2[:,ILOC])
                self.SurfaceX.edit_G2(self.Surface.G1[:,ILOC])
                self.SurfaceX.edit_F(self.Surface.F[:,ILOC])
            
        
        #Selecting the required Layer
        ##################################################
        
        self.LayerX.RADIUS = self.Layer.RADIUS[ILOC]

    ###############################################################################################

    def calc_path(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                    different parameters in the Path class are changed to perform correctly
                    the radiative transfer calculations

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_path(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from archnemesis import AtmCalc_0,Path_0

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        Layer.LAYANG = 0.0
        if Scatter.EMISS_ANG<0.0:
            Layer.LAYHT = Scatter.SOL_ANG * 1.0e3
            Layer.LAYANG = 90.0

        Layer.calc_layering(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID, VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)
        
        #Setting the flags for the Path and calculation types
        ##############################################################################

        
        path_calc = PathCalc.PLANCK_FUNCTION_AT_BIN_CENTRE
        path_observer_pointing = PathObserverPointing.DISK 
        
        if Scatter.EMISS_ANG >= 0.0:
            path_observer_pointing = PathObserverPointing.NADIR
            angle = Scatter.EMISS_ANG
            botlay = 0
        else:
            path_observer_pointing = PathObserverPointing.LIMB
            angle = 90.0
            botlay = 0

        if Scatter.ISCAT == ScatteringCalculationMode.THERMAL_EMISSION:
            if Measurement.IFORM == SpectraUnit.Atmospheric_transmission:
                pass
            else:
                path_calc |= PathCalc.THERMAL_EMISSION
        elif Scatter.ISCAT == ScatteringCalculationMode.MULTIPLE_SCATTERING:
            path_calc |= PathCalc.MULTIPLE_SCATTERING
        elif Scatter.ISCAT == ScatteringCalculationMode.INTERNAL_RADIATION_FIELD:
            path_calc |= PathCalc.MULTIPLE_SCATTERING | PathCalc.NEAR_LIMB
        elif Scatter.ISCAT == ScatteringCalculationMode.SINGLE_SCATTERING_PLANE_PARALLEL:
            path_calc |= PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
        elif Scatter.ISCAT == ScatteringCalculationMode.SINGLE_SCATTERING_SPHERICAL:
            path_calc |= PathCalc.SINGLE_SCATTERING_SPHERICAL
        elif Scatter.ISCAT == ScatteringCalculationMode.INTERNAL_NET_FJLUX:
            angle = 0.0
            path_calc |= PathCalc.MULTIPLE_SCATTERING | PathCalc.NET_FLUX
        elif Scatter.ISCAT == ScatteringCalculationMode.DOWNWARD_BOTTOM_FLUX:
            angle = 0.0
            path_calc |= PathCalc.MULTIPLE_SCATTERING | PathCalc.DOWNWARD_FLUX
        else:
            raise ValueError('error in calc_path :: selected ISCAT has not been implemented yet')


        #_lgr.info(PRESS/101235.)
        #raise ValueError()


        #Performing the calculation of the atmospheric path
        ##############################################################################

        #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
        NCALC = 1    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        iAtmCalc = AtmCalc_0(
            Layer,
            path_observer_pointing=path_observer_pointing,
            IPZEN=ZenithAngleOrigin.BOTTOM,
            BOTLAY=botlay,
            ANGLE=angle,
            EMISS_ANG=Scatter.EMISS_ANG,
            SOL_ANG=Scatter.SOL_ANG,
            AZI_ANG=Scatter.AZI_ANG,
            path_calc=path_calc,
        )
        AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_pathg(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_pathg()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                    different parameters in the Path class are changed to perform correctly
                    the radiative transfer calculations. This version also computes the matrices relating
                    the properties of each layer (Layer) with the properties of the input profiles (Atmosphere)

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_pathg(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from archnemesis import AtmCalc_0,Path_0
        import numpy as np

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        Layer.LAYANG = 0.0
        if Scatter.EMISS_ANG<0.0:
            Layer.LAYHT = Scatter.SOL_ANG * 1.0e3
            Layer.LAYANG = 90.0

        Layer.calc_layeringg(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)

        #Setting the flags for the Path and calculation types
        ##############################################################################
        
        path_calc = PathCalc.PLANCK_FUNCTION_AT_BIN_CENTRE
        path_observer_pointing = PathObserverPointing.DISK

        if Scatter.EMISS_ANG>=0.0:
            path_observer_pointing = PathObserverPointing.NADIR
            angle=Scatter.EMISS_ANG
            botlay=0
        else:
            path_observer_pointing = PathObserverPointing.LIMB
            angle=90.0
            botlay=0
        
        if Scatter.ISCAT == ScatteringCalculationMode.THERMAL_EMISSION:
            if Measurement.IFORM == SpectraUnit.Atmospheric_transmission:
                pass
            else:
                path_calc |= PathCalc.THERMAL_EMISSION
        elif Scatter.ISCAT == ScatteringCalculationMode.MULTIPLE_SCATTERING:
            path_calc |= PathCalc.MULTIPLE_SCATTERING
        elif Scatter.ISCAT == ScatteringCalculationMode.INTERNAL_RADIATION_FIELD:
            path_calc |= PathCalc.MULTIPLE_SCATTERING | PathCalc.NEAR_LIMB
        elif Scatter.ISCAT == ScatteringCalculationMode.SINGLE_SCATTERING_PLANE_PARALLEL:
            path_calc |= PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
        elif Scatter.ISCAT == ScatteringCalculationMode.SINGLE_SCATTERING_SPHERICAL:
            path_calc |= PathCalc.SINGLE_SCATTERING_SPHERICAL
        else:
            raise ValueError('error in calc_pathg :: selected ISCAT has not been implemented yet')


        #Performing the calculation of the atmospheric path
        ##############################################################################

        #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
        NCALC = 1    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        iAtmCalc = AtmCalc_0(
            Layer,
            path_observer_pointing = path_observer_pointing,
            BOTLAY=botlay,
            ANGLE=angle,
            IPZEN=ZenithAngleOrigin.BOTTOM,
            EMISS_ANG=Scatter.EMISS_ANG,
            SOL_ANG=Scatter.SOL_ANG,
            AZI_ANG=Scatter.AZI_ANG,
            path_calc=path_calc,
        )
        AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_path_SO(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path_SO()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                      different parameters in the Path class are changed to perform correctly
                      the radiative transfer calculations

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            ForwardModel.calc_path_SO()

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from archnemesis import AtmCalc_0,Path_0

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        #Based on the new reference atmosphere, we split the atmosphere into layers
        #In solar occultation LAYANG = 90.0
        Layer.LAYANG = 90.0
        
        #Calculating the atmospheric layering
        Layer.calc_layering(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID, VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)

        #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
        #############################################################################################################

        #Calculating the required paths that need to be calculated
        ITANHE = []
        for igeom in range(Measurement.NGEOM):

            ibase = np.argmin(np.abs(Layer.BASEH/1.0e3-Measurement.TANHE[igeom]))
            base0 = Layer.BASEH[ibase]/1.0e3
            
            if base0<=Measurement.TANHE[igeom]:
                ibasel = ibase
                ibaseh = ibase + 1
                if ibaseh==Layer.NLAY:
                    ibaseh = ibase
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            ITANHE.append(ibasel)
            ITANHE.append(ibaseh)

        ITANHE = np.unique(ITANHE)

        NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        for ICALC in range(NCALC):
            iAtmCalc = AtmCalc_0(
                Layer,
                path_observer_pointing=PathObserverPointing.LIMB,
                BOTLAY=ITANHE[ICALC],
                ANGLE=90.0,
                IPZEN=ZenithAngleOrigin.BOTTOM,
                path_calc=PathCalc.PLANCK_FUNCTION_AT_BIN_CENTRE,
            )
            AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)


    ###############################################################################################

    def calc_pathg_SO(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_pathg_SO()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                  different parameters in the Path class are changed to perform correctly
                  the radiative transfer calculations. This version also computes the matrices relating
                  the properties of each layer (Layer) with the properties of the input profiles (Atmosphere)

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_pathg(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from archnemesis import AtmCalc_0,Path_0

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX


        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        #Based on the new reference atmosphere, we split the atmosphere into layers
        #In solar occultation LAYANG = 90.0
        Layer.LAYANG = 90.0

        #Calculating the atmospheric layering
        Layer.calc_layeringg(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)

        #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
        #############################################################################################################

        #Calculating the required paths that need to be calculated
        ITANHE = []
        for igeom in range(Measurement.NGEOM):

            ibase = np.argmin(np.abs(Layer.BASEH/1.0e3-Measurement.TANHE[igeom]))
            base0 = Layer.BASEH[ibase]/1.0e3
            
            if base0<=Measurement.TANHE[igeom]:
                ibasel = ibase
                ibaseh = ibase + 1
                if ibaseh==Layer.NLAY:
                    ibaseh = ibase
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            ITANHE.append(ibasel)
            ITANHE.append(ibaseh)

        ITANHE = np.unique(ITANHE)

        NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        for ICALC in range(NCALC):
            iAtmCalc = AtmCalc_0(
                Layer,
                path_observer_pointing = PathObserverPointing.LIMB,
                BOTLAY=ITANHE[ICALC],
                ANGLE=90.0,
                IPZEN=ZenithAngleOrigin.BOTTOM,
                path_calc = PathCalc.PLANCK_FUNCTION_AT_BIN_CENTRE,
            )
            AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_path_L(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path_L()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                      different parameters in the Path class are changed to perform correctly
                      the radiative transfer calculations

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            ForwardModel.calc_path_L()

        MODIFICATION HISTORY : Juan Alday (08/05/2025)
        """

        from archnemesis import AtmCalc_0,Path_0

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        #Based on the new reference atmosphere, we split the atmosphere into layers
        #In solar occultation LAYANG = 90.0
        Layer.LAYANG = 90.0
        
        #Calculating the atmospheric layering
        Layer.calc_layering(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID, VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)

        #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
        #############################################################################################################

        #Calculating the required paths that need to be calculated
        ITANHE = []
        for igeom in range(Measurement.NGEOM):

            ibase = np.argmin(np.abs(Layer.BASEH/1.0e3-Measurement.TANHE[igeom]))
            base0 = Layer.BASEH[ibase]/1.0e3
            
            if base0<=Measurement.TANHE[igeom]:
                ibasel = ibase
                ibaseh = ibase + 1
                if ibaseh==Layer.NLAY:
                    ibaseh = ibase
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            ITANHE.append(ibasel)
            ITANHE.append(ibaseh)

        ITANHE = np.unique(ITANHE)

        NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        for ICALC in range(NCALC):
            #iAtmCalc = AtmCalc_0(Layer,LIMB=True,BOTLAY=ITANHE[ICALC],ANGLE=90.0,IPZEN=0,THERM=True)
            iAtmCalc = AtmCalc_0(
                Layer,
                path_observer_pointing = PathObserverPointing.LIMB,
                BOTLAY=ITANHE[ICALC],
                ANGLE=90.0,
                IPZEN=ZenithAngleOrigin.BOTTOM,
                path_calc = PathCalc.THERMAL_EMISSION,
            )
            AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_pathg_L(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_pathg_L()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                  different parameters in the Path class are changed to perform correctly
                  the radiative transfer calculations. This version also computes the matrices relating
                  the properties of each layer (Layer) with the properties of the input profiles (Atmosphere)

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_pathg(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from archnemesis import AtmCalc_0,Path_0

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX


        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        #Based on the new reference atmosphere, we split the atmosphere into layers
        #In solar occultation LAYANG = 90.0
        Layer.LAYANG = 90.0

        #Calculating the atmospheric layering
        Layer.calc_layeringg(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID,VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)

        #Based on the atmospheric layerinc, we calculate each required atmospheric path to model the measurements
        #############################################################################################################

        #Calculating the required paths that need to be calculated
        ITANHE = []
        for igeom in range(Measurement.NGEOM):

            ibase = np.argmin(np.abs(Layer.BASEH/1.0e3-Measurement.TANHE[igeom]))
            base0 = Layer.BASEH[ibase]/1.0e3
            
            if base0<=Measurement.TANHE[igeom]:
                ibasel = ibase
                ibaseh = ibase + 1
                if ibaseh==Layer.NLAY:
                    ibaseh = ibase
            else:
                ibasel = ibase - 1
                ibaseh = ibase

            ITANHE.append(ibasel)
            ITANHE.append(ibaseh)

        ITANHE = np.unique(ITANHE)

        NCALC = len(ITANHE)    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        for ICALC in range(NCALC):
            #iAtmCalc = AtmCalc_0(Layer,LIMB=True,BOTLAY=ITANHE[ICALC],ANGLE=90.0,IPZEN=0,THERM=True)
            iAtmCalc = AtmCalc_0(
                Layer,
                path_observer_pointing = PathObserverPointing.LIMB,
                BOTLAY=ITANHE[ICALC],
                ANGLE=90.0,
                IPZEN=ZenithAngleOrigin.BOTTOM,
                path_calc = PathCalc.THERMAL_EMISSION,
            )
            AtmCalc_List.append(iAtmCalc)

        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)

    ###############################################################################################

    def calc_path_C(self,Atmosphere=None,Scatter=None,Measurement=None,Layer=None):

        """
        FUNCTION NAME : calc_path_C()

        DESCRIPTION : Based on the flags read in the different NEMESIS files (e.g., .fla, .set files),
                    different parameters in the Path class are changed to perform correctly
                    the radiative transfer calculations
                    
                    Version defined for an observer looking either up or down at different angles
                    For example:
                        - Observer on the surface looking up at different geometries (sky brightness)
                        - Observer on space looking down at different parts of the planet (assumed to be the same atm and surface across the whole planet)

        INPUTS : None

        OPTIONAL INPUTS:

            Atmosphere :: Python class defining the reference atmosphere (Default : self.AtmosphereX)
            Scatter :: Python class defining the parameters required for scattering calculations (Default : self.ScatterX)
            Measurement :: Python class defining the measurements and observations (Default : self.MeasurementX)
            Layer :: Python class defining the atmospheric layering scheme for the calculation (Default : self.LayerX)

        OUTPUTS :

            self.PathX :: Python class defining the calculation type and the path

        CALLING SEQUENCE:

            Layer,Path = calc_path_C(Atmosphere,Scatter,Layer)

        MODIFICATION HISTORY : Juan Alday (15/03/2021)
        """

        from archnemesis import AtmCalc_0,Path_0

        #Initialise variables
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Scatter is None:
            Scatter = self.ScatterX
        if Measurement is None:
            Measurement = self.MeasurementX
        if Layer is None:
            Layer = self.LayerX
            
        #Checking that all emission angles in Measurement are set for an upward-looking instrument
        emi = Measurement.EMISS_ANG[:,0]
        if all(value > 90 for value in emi):
            _lgr.info('calc_path_C :: All geometries are upward-looking.')
        elif all(value < 90 for value in emi):
            _lgr.info('calc_path_C :: All geometries are downward-looking.')
        else:
            raise ValueError('error in calc_path_C :: All geometries must be either upward-looking or downward-loong in this version (i.e. EMISS_ANG>90 or EMISS_ANG<90)')  
        
        #Checking that multiple scattering is turned on
        if Scatter.ISCAT != ScatteringCalculationMode.MULTIPLE_SCATTERING:
            raise ValueError(f'error in calc_path_C :: This version of the code is meant to use multiple scattering (ISCAT={ScatteringCalculationMode(1)})')

        #Checking that there is only 1 NAV per geometry
        for iGEOM in range(Measurement.NGEOM):
            if Measurement.NAV[iGEOM]>1:
                raise ValueError('error in calc_path_C :: In this version we only allow 1 NAV per geometry')

        #Checking that the solar zenith angle is the same in all geometries
        #sza = np.unique(Measurement.SOL_ANG[:,0])
        #if len(sza)>1:
        #    raise ValueError('error in calc_path_COMBINED :: The solar zenith angle is expected to be the same for all geometries')
        

        Scatter.EMISS_ANG = Measurement.EMISS_ANG[0,0]
        Scatter.SOL_ANG = Measurement.SOL_ANG[0,0]
        Scatter.AZI_ANG = Measurement.AZI_ANG[0,0]

        #Based on the new reference atmosphere, we split the atmosphere into layers
        ################################################################################

        #Limb or nadir observation?
        #Is observation at limb? (coded with -ve emission angle where sol_ang is then the tangent altitude)

        Layer.LAYANG = 0.0
        if Scatter.EMISS_ANG<0.0:
            Layer.LAYHT = Scatter.SOL_ANG * 1.0e3
            Layer.LAYANG = 90.0

        #Calculating the atmospheric layering
        Layer.calc_layering(H=Atmosphere.H,P=Atmosphere.P,T=Atmosphere.T, ID=Atmosphere.ID, VMR=Atmosphere.VMR, DUST=Atmosphere.DUST, PARAH2=Atmosphere.PARAH2, MOLWT=Atmosphere.MOLWT)

        #Setting the flags for the Path and calculation types
        ############################################################################## 

        
        path_observer_pointing = PathObserverPointing.DISK
        path_calc = PathCalc.MULTIPLE_SCATTERING | PathCalc.PLANCK_FUNCTION_AT_BIN_CENTRE
        #Nadir observation
        if Scatter.EMISS_ANG>=0.0:
            path_observer_pointing = PathObserverPointing.NADIR 
            angle=Scatter.EMISS_ANG


        #Performing the calculation of the atmospheric path
        ##############################################################################

        #Based on the atmospheric layering, we calculate each atmospheric path (at each tangent height)
        NCALC = Measurement.NGEOM    #Number of calculations (geometries) to be performed
        AtmCalc_List = []
        
        for iGEOM in range(Measurement.NGEOM):
            iAtmCalc = AtmCalc_0(
                Layer,
                path_observer_pointing=path_observer_pointing,
                BOTLAY=0,
                ANGLE=0.,
                IPZEN=ZenithAngleOrigin.BOTTOM,
                EMISS_ANG=Measurement.EMISS_ANG[iGEOM, 0],
                SOL_ANG=Measurement.SOL_ANG[iGEOM, 0],
                AZI_ANG=Measurement.AZI_ANG[iGEOM, 0],
                path_calc=path_calc,
            )
            AtmCalc_List.append(iAtmCalc)
            
        #We initialise the total Path class, indicating that the calculations can be combined
        self.PathX = Path_0(AtmCalc_List,COMBINE=True)


    ###############################################################################################
    ###############################################################################################
    # RADIATIVE TRANSFER
    ###############################################################################################
    ###############################################################################################


    ###############################################################################################

    def CIRSrad(self):

        """
            FUNCTION NAME : CIRSrad()

            DESCRIPTION : This function computes the spectrum given the calculation type

            INPUTS :

                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations
                Path :: Python class defining the calculation type and the path

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECOUT(Spectroscopy.NWAVE,Path.NPATH) :: Output spectrum (non-convolved) in the units given by IMOD

            CALLING SEQUENCE:

                SPECOUT = CIRSrad(Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        #import matplotlib as matplotlib
        from scipy import interpolate
        from copy import copy

        #Initialise some arrays
        ###################################

        #Calculating the vertical opacity of each layer
        ######################################################
        ######################################################
        ######################################################
        ######################################################

        #There will be different kinds of opacities:
        #   Line opacity due to gaseous absorption (K-tables or LBL-tables)
        #   Continuum opacity due to aerosols coming from the extinction coefficient
        #   Continuum opacity from different gases like H, NH3 (flags in .fla file)
        #   Collision-Induced Absorption
        #   Scattering opacity derived from the particle distribution and the single scattering albedo.
        #        For multiple scattering, this is passed to scattering routines


        #Calculating the gaseous line opacity in each layer
        ########################################################################################################
        if self.SpectroscopyX.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:  #LBL-table

            TAUGAS = np.zeros((self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.LayerX.NLAY,self.SpectroscopyX.NGAS))  #Vertical opacity of each gas in each layer

            #Calculating the cross sections for each gas in each layer
            k = self.SpectroscopyX.calc_klbl(self.LayerX.NLAY,self.LayerX.PRESS/101325.,self.LayerX.TEMP,WAVECALC=self.SpectroscopyX.WAVE)

            for i in range(self.SpectroscopyX.NGAS):
                IGAS = self.AtmosphereX.locate_gas(self.SpectroscopyX.ID[i],self.SpectroscopyX.ISO[i])

                #Calculating vertical column density in each layer
                VLOSDENS = self.LayerX.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #cm-2

                #Calculating vertical opacity for each gas in each layer
                TAUGAS[:,0,:,i] = k[:,:,i] * VLOSDENS

            #Combining the gaseous opacity in each layer
            TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)
            #Removing necessary data to save memory
            del k

        elif self.SpectroscopyX.ILBL==SpectralCalculationMode.K_TABLES:    #K-table
            
            #Calculating the k-coefficients for each gas in each layer
            k_gas = self.SpectroscopyX.calc_k(self.LayerX.NLAY,self.LayerX.PRESS/101325.,self.LayerX.TEMP,WAVECALC=self.SpectroscopyX.WAVE) # (NWAVE,NG,NLAY,NGAS) 
            f_gas = np.zeros((self.SpectroscopyX.NGAS,self.LayerX.NLAY))
            utotl = np.zeros(self.LayerX.NLAY)
            for i in range(self.SpectroscopyX.NGAS):
                IGAS = self.AtmosphereX.locate_gas(self.SpectroscopyX.ID[i],self.SpectroscopyX.ISO[i])

                #When using gradients
                f_gas[i,:] = self.LayerX.AMOUNT[:,IGAS] * 1.0e-4 * 1.0e-20  #Vertical column density of the radiatively active gases in cm-2

            #Combining the k-distributions of the different gases in each layer
            k_layer = k_overlap(self.SpectroscopyX.DELG,k_gas,f_gas)

            #Calculating the opacity of each layer
            TAUGAS = k_layer #(NWAVE,NG,NLAY)

            #Removing necessary data to save memory
            del k_gas
            del k_layer
            self.SpectroscopyX.K = None

        else:
            raise ValueError(f'error in CIRSrad :: ILBL must be either {SpectralCalculationMode(0)} or {SpectralCalculationMode(2)}')
        self.LayerX.TAUGAS = TAUGAS
        
        #Calculating the continuum absorption by gaseous species
        #################################################################################################################

        #Computes a polynomial approximation to any known continuum spectra for a particular gas over a defined wavenumber region.
        #raise ValueError
        #To be done

        #Calculating the vertical opacity by CIA
        #################################################################################################################

        if self.CIAX==None:
            TAUCIA = np.zeros((self.SpectroscopyX.NWAVE,self.LayerX.NLAY))
            #dTAUCIA = np.zeros((self.SpectroscopyX.NWAVE,self.LayerX.NLAY,7))
            _lgr.info('CIRSrad :: CIA not included in calculations')
        else:
            _lgr.info('CIRSrad :: Calculating CIA opacities')
            TAUCIA,dTAUCIA = self.calc_tau_cia() #(NWAVE,NLAY);(NWAVE,NLAY,7)
            self.LayerX.TAUCIA = TAUCIA
            
            #Removing CIA since it is no longer needed 
            self.CIAX = None
            
        #Calculating the vertical opacity by Rayleigh scattering
        #################################################################################################################

        TAURAY,dTAURAY = self.calc_tau_rayleigh(MakePlot=False)  #(NWAVE,NLAY)
        self.LayerX.TAURAY = TAURAY

        #Calculating the vertical opacity by aerosols from the extinction coefficient and single scattering albedo
        #################################################################################################################

        TAUDUST1,TAUCLSCAT,dTAUDUST1,dTAUCLSCAT = self.calc_tau_dust() #(NWAVE,NLAYER,NDUST)

        #Calculating the total optical depth for the aerosols
        TAUDUST1 = np.clip(np.nan_to_num(TAUDUST1),0,1e20)
        
        _lgr.info(f"CIRSrad :: Aerosol optical depths at  {(self.SpectroscopyX.WAVE[0],' :: ',np.sum(TAUDUST1[0,:,:],axis=0))}")
        
        #Adding the opacity by the different dust populations
        TAUDUST = np.sum(TAUDUST1,2)  #(NWAVE,NLAYER) Absorption + Scattering
        TAUSCAT = np.sum(TAUCLSCAT,2)  #(NWAVE,NLAYER) Scattering

        self.LayerX.TAUDUST = TAUDUST
        self.LayerX.TAUSCAT = TAUSCAT
        self.LayerX.TAUCLSCAT = TAUCLSCAT

        del TAUDUST1

        #Combining the different kinds of opacity in each layer
        ########################################################################################################
        TAUTOT = np.zeros(TAUGAS.shape) #(NWAVE,NG,NLAY)
        for ig in range(self.SpectroscopyX.NG):
            TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]
        
        
        self.LayerX.TAUTOT = TAUTOT
        del TAUTOT,TAUGAS,TAUCIA,TAUDUST,TAURAY

        #Step through the different number of paths and calculate output spectrum
        ############################################################################

        #Output paths may be:
        #	      Imod
        #		0	(Atm) Pure transmission
        #		1	(Atm) Absorption (useful for small transmissions)
        #		2	(Atm) Emission. Planck function evaluated at each
        #				wavenumber. NOT SUPPORTED HERE.
        #		3	(Atm) Emission. Planck function evaluated at bin
        #				center.
        #		8	(Combined Cell,Atm) The product of two
        #				previous output paths.
        #		11	(Atm) Contribution function.
        #		13	(Atm) SCR Sideband
        #		14	(Atm) SCR Wideband
        #		15	(Atm) Multiple scattering (multiple models)
        #		16	(Atm) Single scattering approximation.
        #		21	(Atm) Net flux calculation (thermal)
        #		22	(Atm) Limb scattering calculation
        #		23	(Atm) Limb scattering calculation using precomputed
        #			      internal radiation field.
        #		24	(Atm) Net flux calculation (scattering)
        #		25	(Atm) Upwards flux (internal) calculation (scattering)
        #		26	(Atm) Upwards flux (top) calculation (scattering)
        #		27	(Atm) Downwards flux (bottom) calculation (scattering)
        #		28	(Atm) Single scattering approximation (spherical)

        IMODM  = np.unique(self.PathX.IMOD)
        assert IMODM.size==1, 'CIRSrad :: IMODM should be a single value, multiple path calculation types not supported yet'

        IMODM = PathCalc(IMODM[0])  #Convert to enum
        
        _lgr.info(f'CIRSrad :: IMODM = {IMODM!r}')
        
        if not (PathCalc.ABSORBTION 
                | PathCalc.THERMAL_EMISSION 
                | PathCalc.MULTIPLE_SCATTERING
                | PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL
            ) & IMODM:  #Pure transmission

            #Calculating the line-of-sight opacities
            TAUTOT_LAYINC = self.LayerX.TAUTOT[:,:,self.PathX.LAYINC[:,:]] * self.PathX.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)

            #Calculating the total opacity over the path
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

            #Pure transmission spectrum 
            SPECOUT = np.exp(-(TAUTOT_PATH))  #(NWAVE,NG,NPATH)

            xfac = 1.0
            if self.MeasurementX.IFORM==SpectraUnit.Atmospheric_transmission:
                self.StellarX.calc_solar_flux()
                #Interpolating to the calculation wavelengths
                f = interpolate.interp1d(self.StellarX.VCONV,self.StellarX.SOLFLUX)
                solflux = f(self.SpectroscopyX.WAVE)
                xfac = solflux
                for ipath in range(self.PathX.NPATH):
                    for ig in range(self.SpectroscopyX.NG):
                        SPECOUT[:,ig,ipath] = SPECOUT[:,ig,ipath] * xfac

        elif PathCalc.ABSORBTION in IMODM: #Absorbtion (useful for small transmissions) 

            #Calculating the line-of-sight opacities
            TAUTOT_LAYINC = self.LayerX.TAUTOT[:,:,self.PathX.LAYINC[:,:]] * self.PathX.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)

            #Calculating the total opacity over the path
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

            #Absorption spectrum (useful for small transmissions)
            SPECOUT = 1.0 - np.exp(-(TAUTOT_PATH)) #(NWAVE,NG,NPATH)

        elif PathCalc.THERMAL_EMISSION in IMODM: #Thermal emission from planet 

            _lgr.info('CIRSrad :: Performing thermal emission calculation')

            #Calculating the line-of-sight opacities
            TAUTOT_LAYINC = self.LayerX.TAUTOT[:,:,self.PathX.LAYINC[:,:]] * self.PathX.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)

            #Defining the units of the output spectrum
            xfac = np.ones(self.SpectroscopyX.NWAVE) 
            if self.MeasurementX.IFORM==SpectraUnit.FluxRatio:
                xfac *= np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(self.StellarX.VCONV,self.StellarX.SOLSPEC)
                solpspec = f(self.SpectroscopyX.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec

            #Interpolating the emissivity of the surface to the correct wavelengths
            if self.SurfaceX.TSURF>0.0:
                f = interpolate.interp1d(self.SurfaceX.VEM,self.SurfaceX.EMISSIVITY)
                EMISSIVITY = f(self.SpectroscopyX.WAVE)
            else:
                EMISSIVITY = np.zeros(self.SpectroscopyX.NWAVE)
            
            #Calculating the contribution from surface reflectance
            if( 
                (self.StellarX.SOLEXIST is True) 
                and (self.SurfaceX.GASGIANT is False) 
                and (self.SurfaceX.LOWBC != LowerBoundaryCondition.THERMAL) ):
                
                #Calculating solar flux at top of the atmosphere
                self.StellarX.calc_solar_flux()
                f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLFLUX)
                SOLFLUX = f(self.SpectroscopyX.WAVE)
                
                #Calculating the surface reflectance
                if self.SurfaceX.LOWBC == LowerBoundaryCondition.LAMBERTIAN: #Lambertian reflection
                    
                    ALBEDO = np.zeros(self.SpectroscopyX.NWAVE)
                    ALBEDO[:] = 1.0 - EMISSIVITY[:] if self.SurfaceX.GALB < 0.0 else self.SurfaceX.GALB
                    
                    REFLECTANCE = np.zeros(self.SpectroscopyX.NWAVE)
                    #REFLECTANCE[:] = self.SurfaceX.calc_Lambert_BRDF(ALBEDO,self.ScatterX.SOL_ANG)[:,0]
                
            else:
                SOLFLUX = np.zeros(self.SpectroscopyX.NWAVE)
                REFLECTANCE = np.zeros(self.SpectroscopyX.NWAVE)
            
            #Calculating the spectra
            SPECOUT = np.zeros((self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.PathX.NPATH))
            for ipath in range(self.PathX.NPATH):
                NLAYIN = self.PathX.NLAYIN[ipath]
                EMTEMP = self.PathX.EMTEMP[0:NLAYIN,ipath]
                EMPRESS = self.LayerX.PRESS[self.PathX.LAYINC[0:NLAYIN,ipath]]
                
                SPECOUT[:,:,ipath] = calc_thermal_emission_spectrum(self.MeasurementX.ISPACE,self.SpectroscopyX.WAVE,TAUTOT_LAYINC[:,:,0:NLAYIN,ipath],EMTEMP,EMPRESS,self.SurfaceX.TSURF,EMISSIVITY,SOLFLUX,REFLECTANCE,self.PathX.SOL_ANG[ipath],self.PathX.EMISS_ANG[ipath])
        
                #Changing the units of the spectra
                SPECOUT[:,:,ipath] = (SPECOUT[:,:,ipath].T * xfac).T

        elif PathCalc.SINGLE_SCATTERING_PLANE_PARALLEL in IMODM: #Single scattering calculation

            _lgr.info('CIRSrad :: Performing single scattering calculation')
 
            #Calculating the line-of-sight opacities
            TAUTOT_LAYINC = self.LayerX.TAUTOT[:,:,self.PathX.LAYINC[:,:]] * self.PathX.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)

            #Obtaining the phase function of each aerosol at the scattering angle if single scattering
            sol_ang = self.PathX.SOL_ANG     #(NPATH)
            emiss_ang = self.PathX.EMISS_ANG #(NPATH)
            azi_ang = self.PathX.AZI_ANG     #(NPATH)

            #Calculating cos(alpha), where alpha is the scattering angle 
            calpha = np.sin(sol_ang / 180. * np.pi) * np.sin(emiss_ang / 180. * np.pi) * np.cos( azi_ang/180.*np.pi - np.pi ) - \
                    np.cos(emiss_ang / 180. * np.pi) * np.cos(sol_ang / 180. * np.pi)
                    
            #Calculating the phase function for each aerosol type
            phase_function = np.zeros((self.SpectroscopyX.NWAVE,self.PathX.NPATH,self.ScatterX.NDUST+1))
            phase_function[:,:,0:self.ScatterX.NDUST] = self.ScatterX.calc_phase(np.arccos(calpha)/np.pi*180.,self.SpectroscopyX.WAVE)  
            phase_function[:,:,self.ScatterX.NDUST] = self.ScatterX.calc_phase_ray(np.arccos(calpha)/np.pi*180.)
            phase_function = np.transpose(phase_function,(0,2,1)) #(NWAVE,NDUST+1,NPATH) 

            # Single scattering albedo
            omega = np.zeros((self.SpectroscopyX.NWAVE, self.SpectroscopyX.NG, self.LayerX.NLAY))
            iin = np.where(self.LayerX.TAUTOT > 0.0)
            if iin[0].size > 0:
                omega[iin[0], iin[1], iin[2]] = (
                    (self.LayerX.TAURAY[iin[0], iin[2]] + self.LayerX.TAUSCAT[iin[0], iin[2]]) /
                    self.LayerX.TAUTOT[iin[0], iin[1], iin[2]]
                    )

            #Solar flux at the top of the atmosphere
            if self.StellarX.SOLEXIST is True:  
                self.StellarX.calc_solar_flux()
                solar = np.interp(self.SpectroscopyX.WAVE,self.StellarX.WAVE,self.StellarX.SOLFLUX)
            else:
                solar = np.zeros(self.SpectroscopyX.NWAVE)

            #Defining the units of the output spectrum
            xfac = np.ones(self.SpectroscopyX.NWAVE)
            if self.MeasurementX.IFORM==SpectraUnit.FluxRatio:
                xfac *= np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(self.StellarX.VCONV,self.StellarX.SOLSPEC)
                solpspec = f(self.SpectroscopyX.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec

            #Surface emissivity
            if self.SurfaceX.TSURF>0.0:
                f = interpolate.interp1d(self.SurfaceX.VEM,self.SurfaceX.EMISSIVITY)
                EMISSIVITY = f(self.SpectroscopyX.WAVE)
            else:
                EMISSIVITY = np.zeros(self.SpectroscopyX.NWAVE)
                
            #Surface reflectance
            if self.SurfaceX.LOWBC != LowerBoundaryCondition.THERMAL:
                BRDF = self.SurfaceX.calc_BRDF(self.SpectroscopyX.WAVE,self.PathX.SOL_ANG,self.PathX.EMISS_ANG,self.PathX.AZI_ANG) #(NWAVE,NPATH)
            else:
                BRDF = np.zeros((self.SpectroscopyX.NWAVE,self.PathX.NPATH))

            #Looping over path
            SPECOUT = np.zeros((self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.PathX.NPATH))
            for ipath in range(self.PathX.NPATH):
                                
                #Average phsae function for each layer
                phasex = np.zeros((self.SpectroscopyX.NWAVE,self.ScatterX.NDUST+1,self.LayerX.NLAY))
                
                phasex[:,0:self.ScatterX.NDUST,:] = np.transpose((phase_function[:,0:self.ScatterX.NDUST,ipath] * np.transpose(self.LayerX.TAUCLSCAT[:,:,:],axes=(1,0,2))),axes=(1,2,0))
                phasex[:,self.ScatterX.NDUST,:] = np.transpose(phase_function[:,self.ScatterX.NDUST,ipath] * np.transpose(self.LayerX.TAURAY[:,:]))
                phase = np.sum(phasex,axis=1) #(NWAVE,NLAY)
                phase[phase>0] = phase[phase>0] / (self.LayerX.TAURAY[phase>0] + self.LayerX.TAUSCAT[phase>0])

                #Selecting properties across the path
                NLAYIN = self.PathX.NLAYIN[ipath]
                EMTEMP = self.PathX.EMTEMP[0:NLAYIN,ipath]
                EMPHASE = phase[:,self.PathX.LAYINC[0:NLAYIN,ipath]]
                EMOMEGA = omega[:,:,self.PathX.LAYINC[0:NLAYIN,ipath]]

                #Calculating the spectrum
                SPECOUT[:,:,ipath] = calc_singlescatt_plane_spectrum(self.MeasurementX.ISPACE,self.SpectroscopyX.WAVE,TAUTOT_LAYINC[:,:,0:NLAYIN,ipath],EMTEMP,EMOMEGA,EMPHASE,self.SurfaceX.TSURF,EMISSIVITY,BRDF[:,ipath],solar,sol_ang[ipath],emiss_ang[ipath])
        
                #Changing the units of the spectra
                SPECOUT[:,:,ipath] = (SPECOUT[:,:,ipath].T * xfac).T

        elif (PathCalc.DOWNWARD_FLUX | PathCalc.MULTIPLE_SCATTERING) in IMODM: #Downwards flux (bottom) calculation (scattering)

            _lgr.info('CIRSrad :: Downwards flux calculation at the bottom of the atmosphere')

            #The codes below calculates the downwards flux
            #spectrum in units of W cm-2 (cm-1)-1 or W cm-2 um-1.

            #Calculating spectrum
            SPECOUT = np.zeros((self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.PathX.NPATH))
            for ipath in range(self.PathX.NPATH):

                #Calculating the solar flux at the top of the atmosphere
                solar = np.zeros(self.SpectroscopyX.NWAVE)
                if self.StellarX.SOLEXIST==True:
                    self.StellarX.calc_solar_flux()
                    f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLFLUX)
                    solar[:] = f(self.SpectroscopyX.WAVE)  #W cm-2 (cm-1)-1 or W cm-2 um-1


                #Defining the units of the output spectrum
                xfac = 1.
                if self.MeasurementX.IFORM==SpectraUnit.FluxRatio:
                    xfac=np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2.
                    f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLSPEC)
                    solpspec = f(self.SpectroscopyX.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                    xfac = xfac / solpspec
                elif self.MeasurementX.IFORM==SpectraUnit.Integrated_spectral_power:
                    xfac=np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2.

                #Calculating the radiance at the boundaries of each layer
                #Uplf(NWAVE,NG,NMU,NLAY,NF)   Donward radiance in the bottom boundary of each layer
                #Umif(NWAVE,NG,NMU,NLAY,NF)   Upward radiance in the top boundary of each layer
                Uplf,Umif = self.scloud11flux(self.ScatterX,self.SurfaceX,self.LayerX,self.MeasurementX,solar,diffuse=True)

                #Calculating the fluxes at the boundaries of each layer
                fup,fdown = self.streamflux(self.LayerX.NLAY,self.ScatterX.NMU,self.ScatterX.MU,self.ScatterX.WTMU,Umif,Uplf)  #(NWAVE,NG,NLAY)

                #Getting the downward flux at the bottom layer 
                SPECOUT[:,:,ipath] = fdown[:,:,0]*xfac

        elif PathCalc.MULTIPLE_SCATTERING in IMODM: #Multiple scattering calculation

            _lgr.info('CIRSrad :: Performing multiple scattering calculation')
            _lgr.info(f"CIRSrad :: NF =  {(self.ScatterX.NF,'; NMU = ',self.ScatterX.NMU,'; NPHI = ',self.ScatterX.NPHI)}")
            _lgr.debug(f'{self.PathX.EMISS_ANG=} {self.PathX.SOL_ANG=} {self.PathX.AZI_ANG=}')


            #Calculating the solar flux at the top of the atmosphere
            solar = np.zeros(self.SpectroscopyX.NWAVE)
            if self.StellarX.SOLEXIST==True:
                self.StellarX.calc_solar_flux()
                f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLFLUX)
                solar[:] = f(self.SpectroscopyX.WAVE)  #W cm-2 (cm-1)-1 or W cm-2 um-1 

            #Defining the units of the output spectrum
            xfac = 1.
            if self.MeasurementX.IFORM==SpectraUnit.FluxRatio:
                xfac=np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLSPEC)
                solpspec = f(self.SpectroscopyX.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec
            elif self.MeasurementX.IFORM==SpectraUnit.Integrated_spectral_power:
                xfac=np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2. 
            
            #Calculating the radiance
            SPECOUT = self.scloud11wave(self.SpectroscopyX.WAVE,self.ScatterX,self.SurfaceX,self.LayerX,self.MeasurementX,self.PathX, solar)

        else:
            raise NotImplementedError(f'error in CIRSrad :: Calculation type "{IMODM}" not included in CIRSrad')

        #Now integrate over g-ordinates
        SPECOUT = np.tensordot(SPECOUT, self.SpectroscopyX.DELG, axes=([1],[0])) #NWAVE,NPATH
        
        return SPECOUT


    ###############################################################################################

    def CIRSradg(self):

        """
            FUNCTION NAME : CIRSradg()

            DESCRIPTION : This function computes the spectrum given the calculation type

            INPUTS :

                runname :: Name of the Nemesis run
                Variables :: Python class defining the parameterisations and state vector
                Measurement :: Python class defining the measurements
                Atmosphere :: Python class defining the reference atmosphere
                Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
                Scatter :: Python class defining the parameters required for scattering calculations
                Stellar :: Python class defining the stellar spectrum
                Surface :: Python class defining the surface
                CIA :: Python class defining the Collision-Induced-Absorption cross-sections
                Layer :: Python class defining the layering scheme to be applied in the calculations
                Path :: Python class defining the calculation type and the path

            OPTIONAL INPUTS: none

            OUTPUTS :

                SPECOUT(Spectroscopy.NWAVE,Path.NPATH) :: Output spectrum (non-convolved) in the units given by IMOD

            CALLING SEQUENCE:

                SPECOUT = CIRSradg(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)

            MODIFICATION HISTORY : Juan Alday (25/07/2021)

        """

        from scipy import interpolate
        #from NemesisPy import nemesisf
        from copy import copy
        

        #Calculating the vertical opacity of each self.LayerX
        ######################################################
        ######################################################
        ######################################################
        ######################################################

        #There will be different kinds of opacities:
        #   Continuum opacity due to aerosols coming from the extinction coefficient
        #   Continuum opacity from different gases like H, NH3 (flags in .fla file)
        #   Collision-Induced Absorption
        #   Scattering opacity derived from the particle distribution and the single scattering albedo.
        #        For multiple scattering, this is passed to scattering routines
        #   Line opacity due to gaseous absorption (K-tables or LBL-tables)


        #Defining the matrices where the derivatives will be stored
        dTAUCON = np.zeros((self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR+2+self.ScatterX.NDUST,self.LayerX.NLAY)) #(NWAVE,NLAY,NGAS+2+NDUST)
        dTAUSCA = np.zeros((self.SpectroscopyX.NWAVE,self.AtmosphereX.NVMR+2+self.ScatterX.NDUST,self.LayerX.NLAY)) #(NWAVE,NLAY,NGAS+2+NDUST)

        #Calculating the continuum absorption by gaseous species
        #################################################################################################################

        #Computes a polynomial approximation to any known continuum spectra for a particular gas over a defined wavenumber region.

        #To be done

        #Calculating the vertical opacity by self.CIAX
        #################################################################################################################

        if self.CIAX==None:
            TAUCIA = np.zeros((self.SpectroscopyX.NWAVE,self.LayerX.NLAY))
            dTAUCIA = np.zeros((self.SpectroscopyX.NWAVE,self.LayerX.NLAY,7))
            _lgr.info('CIRSrad :: self.CIAX not included in calculations')
        else:
            _lgr.info('CIRSradg :: Calculating self.CIAX opacity')
            TAUCIA,dTAUCIA = self.calc_tau_cia() #(NWAVE,NLAY);(NWAVE,NLAY,NVMR+2)
            self.LayerX.TAUCIA = TAUCIA

            dTAUCON[:,0:self.AtmosphereX.NVMR,:] = dTAUCON[:,0:self.AtmosphereX.NVMR,:] + np.transpose(np.transpose(dTAUCIA[:,:,0:self.AtmosphereX.NVMR],axes=(2,0,1)) / (self.LayerX.TOTAM.T),axes=(1,0,2)) #dTAUCIA/dAMOUNT (m2)
            dTAUCON[:,self.AtmosphereX.NVMR,:] = dTAUCON[:,self.AtmosphereX.NVMR,:] + dTAUCIA[:,:,self.AtmosphereX.NVMR]  #dTAUCIA/dT

            flagh2p = False
            if flagh2p==True:
                dTAUCON[:,self.AtmosphereX.NVMR+1+self.ScatterX.NDUST,:] = dTAUCON[:,self.AtmosphereX.NVMR+1+self.ScatterX.NDUST,:] + dTAUCIA[:,:,6]  #dTAUCIA/dPARA-H2

        #Calculating the vertical opacity by Rayleigh scattering
        #################################################################################################################

        TAURAY,dTAURAY = self.calc_tau_rayleigh()
        self.LayerX.TAURAY = TAURAY

        for i in range(self.AtmosphereX.NVMR):
            dTAUCON[:,i,:] = dTAUCON[:,i,:] + dTAURAY[:,:] #dTAURAY/dAMOUNT (m2)

        #Calculating the vertical opacity by aerosols from the extinction coefficient and single scattering albedo
        #################################################################################################################

        _lgr.info('CIRSradg :: Calculating DUST opacity')
        TAUDUST1,TAUCLSCAT,dTAUDUST1,dTAUCLSCAT = self.calc_tau_dust() #(NWAVE,NLAYER,NDUST)

        #Adding the opacity by the different dust populations
        TAUDUST = np.sum(TAUDUST1,2)  #(NWAVE,NLAYER)
        TAUSCAT = np.sum(TAUCLSCAT,2)  #(NWAVE,NLAYER)

        for i in range(self.ScatterX.NDUST):
            dTAUCON[:,self.AtmosphereX.NVMR+1+i,:] = dTAUCON[:,self.AtmosphereX.NVMR+1+i,:] + dTAUDUST1[:,:,i]  #dTAUDUST/dAMOUNT (m2)
            dTAUSCA[:,self.AtmosphereX.NVMR+1+i,:] = dTAUSCA[:,self.AtmosphereX.NVMR+1+i,:] + dTAUCLSCAT[:,:,i]

        #Calculating the total optical depth for the aerosols
        _lgr.info(f"CIRSrad :: Aerosol optical depths at  {(self.SpectroscopyX.WAVE[0],' :: ',np.sum(TAUDUST1[0,:,:],axis=0))}")

        #Calculating the gaseous line opacity in each self.LayerX
        ########################################################################################################

        _lgr.info('CIRSradg :: Calculating GAS opacity')
        if self.SpectroscopyX.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES:  #LBL-table

            TAUGAS = np.zeros([self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.LayerX.NLAY,self.SpectroscopyX.NGAS])  #Vertical opacity of each gas in each self.LayerX
            dTAUGAS = np.zeros([self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.AtmosphereX.NVMR+2+self.ScatterX.NDUST,self.LayerX.NLAY])

            #Calculating the cross sections for each gas in each self.LayerX
            k,dkdT = self.SpectroscopyX.calc_klblg(self.LayerX.NLAY,self.LayerX.PRESS/101325.,self.LayerX.TEMP)

            for i in range(self.SpectroscopyX.NGAS):
                IGAS = np.where( (self.AtmosphereX.ID==self.SpectroscopyX.ID[i]) & (self.AtmosphereX.ISO==self.SpectroscopyX.ISO[i]) )
                IGAS = IGAS[0]

                #Calculating vertical column density in each self.LayerX
                VLOSDENS = self.LayerX.AMOUNT[:,IGAS].T * 1.0e-20   #m-2

                #Calculating vertical opacity for each gas in each self.LayerX
                TAUGAS[:,0,:,i] = k[:,:,i] * 1.0e-4 * VLOSDENS
                dTAUGAS[:,0,IGAS[0],:] = k[:,:,i] * 1.0e-4 * 1.0e-20  #dTAUGAS/dAMOUNT (m2)
                dTAUGAS[:,0,self.AtmosphereX.NVMR,:] = dTAUGAS[:,0,self.AtmosphereX.NVMR,:] + dkdT[:,:,i] * 1.0e-4 * VLOSDENS #dTAUGAS/dT

            #Combining the gaseous opacity in each self.LayerX
            TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)

        elif self.SpectroscopyX.ILBL == SpectralCalculationMode.K_TABLES:    #K-table

            dTAUGAS = np.zeros([self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.AtmosphereX.NVMR+2+self.ScatterX.NDUST,self.LayerX.NLAY])

            #Calculating the k-coefficients for each gas in each self.LayerX
            k_gas,dkgasdT = self.SpectroscopyX.calc_kg(self.LayerX.NLAY,self.LayerX.PRESS/101325.,self.LayerX.TEMP) # (NWAVE,NG,NLAY,NGAS)
            
            f_gas = np.zeros([self.SpectroscopyX.NGAS,self.LayerX.NLAY])
            utotl = np.zeros(self.LayerX.NLAY)
            for i in range(self.SpectroscopyX.NGAS):
                IGAS = np.where( (self.AtmosphereX.ID==self.SpectroscopyX.ID[i]) & (self.AtmosphereX.ISO==self.SpectroscopyX.ISO[i]) )[0][0]
                f_gas[i,:] = self.LayerX.AMOUNT[:,IGAS] * 1.0e-4 * 1.0e-20  #Vertical column density of the radiatively active gases in cm-2

            #Combining the k-distributions of the different gases in each self.LayerX, as well as their gradients
            k_layer,dk_layer = k_overlapg(self.SpectroscopyX.DELG,k_gas,dkgasdT,f_gas)

            #Calculating the opacity of each self.LayerX
            TAUGAS = k_layer #(NWAVE,NG,NLAY)

            #Calculating the gradients of each self.LayerX and for each gas
            for i in range(self.SpectroscopyX.NGAS):
                IGAS = np.where( (self.AtmosphereX.ID==self.SpectroscopyX.ID[i]) & (self.AtmosphereX.ISO==self.SpectroscopyX.ISO[i]) )
                IGAS = IGAS[0]
                dTAUGAS[:,:,IGAS[0],:] = dk_layer[:,:,:,i] * 1.0e-4 * 1.0e-20  #dTAU/dAMOUNT (m2)

            dTAUGAS[:,:,self.AtmosphereX.NVMR,:] = dk_layer[:,:,:,self.SpectroscopyX.NGAS] #dTAU/dT

        else:
            raise NotImplementedError(f'error in CIRSrad :: ILBL must be either {SpectralCalculationMode(0)} or {SpectralCalculationMode(2)}')
        
        #Combining the different kinds of opacity in each self.LayerX
        ########################################################################################################

        _lgr.info('CIRSradg :: Calculating TOTAL opacity')
        TAUTOT = np.zeros(TAUGAS.shape) #(NWAVE,NG,NLAY)
        dTAUTOT = np.zeros(dTAUGAS.shape) #(NWAVE,NG,NVMR+2+NDUST,NLAY)
        for ig in range(self.SpectroscopyX.NG):
            TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]
            dTAUTOT[:,ig,:,:] = (dTAUGAS[:,ig,:,:] + dTAUCON[:,:,:]) #dTAU/dAMOUNT (m2) or dTAU/dK (K-1)
        
        self.LayerX.TAUTOT = TAUTOT
        del TAUGAS,TAUCIA,TAUDUST,TAURAY
        del dTAUGAS,dTAUCON

        #Calculating the line-of-sight opacities
        #################################################################################################################

        _lgr.info('CIRSradg :: Calculating TOTAL line-of-sight opacity')
        TAUTOT_LAYINC = TAUTOT[:,:,self.PathX.LAYINC[:,:]] * self.PathX.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)
        dTAUTOT_LAYINC = dTAUTOT[:,:,:,self.PathX.LAYINC[:,:]] * self.PathX.SCALE[:,:] #(NWAVE,NG,NGAS+2+NDUST,NLAYIN,NPATH)

        #Step through the different number of paths and calculate output spectrum
        ############################################################################

        #Output paths may be:
        #	      Imod
        #		0	(Atm) Pure transmission
        #		1	(Atm) Absorption (useful for small transmissions) 
        #		2	(Atm) Emission. Planck function evaluated at each
        #				wavenumber. NOT SUPPORTED HERE.
        #		3	(Atm) Emission. Planck function evaluated at bin
        #				center.
        #		8	(Combined Cell,Atm) The product of two
        #				previous output paths.
        #		11	(Atm) Contribution function.
        #		13	(Atm) SCR Sideband
        #		14	(Atm) SCR Wideband
        #		15	(Atm) Multiple scattering (multiple models)
        #		16	(Atm) Single scattering approximation.
        #		21	(Atm) Net flux calculation (thermal)
        #		22	(Atm) Limb scattering calculation
        #		23	(Atm) Limb scattering calculation using precomputed
        #			      internal radiation field.
        #		24	(Atm) Net flux calculation (scattering)
        #		25	(Atm) Upwards flux (internal) calculation (scattering)
        #		26	(Atm) Upwards flux (top) calculation (scattering)
        #		27	(Atm) Downwards flux (bottom) calculation (scattering)
        #		28	(Atm) Single scattering approximation (spherical)

        
        IMODM = np.unique(self.PathX.IMOD)
        assert IMODM.size == 1, 'CIRSradg :: More than one calculation type in the same self.PathX. This is not supported yet'

        IMODM = PathCalc(IMODM[0]) #Convert to PathCalc enum

        _lgr.info(f'CIRSradg :: IMODM = {IMODM!r}')
 
        SPECOUT = np.zeros([self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.PathX.NPATH])
        dSPECOUT = np.zeros([self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.AtmosphereX.NVMR+2+self.ScatterX.NDUST,self.PathX.NLAYIN.max(),self.PathX.NPATH])
        dTSURF = np.zeros((self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.PathX.NPATH))

        if not (PathCalc.ABSORBTION 
                | PathCalc.THERMAL_EMISSION 
            ) & IMODM:  #Pure transmission spectrum

            _lgr.info('CIRSradg :: Calculating TRANSMISSION')
            #Calculating the total opacity over the self.PathX
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)
            del TAUTOT_LAYINC

            #Pure transmission spectrum
            SPECOUT = np.exp(-(TAUTOT_PATH))  #(NWAVE,NG,NPATH)
            #del TAUTOT_PATH

            xfac = np.ones(self.SpectroscopyX.NWAVE)
            if self.MeasurementX.IFORM==SpectraUnit.Atmospheric_transmission:  #If IFORM=4 we should multiply the transmission by solar flux
                self.StellarX.calc_solar_flux()
                #Interpolating to the calculation wavelengths
                f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLFLUX)
                solflux = f(self.SpectroscopyX.WAVE)
                xfac = solflux
                for ipath in range(self.PathX.NPATH):
                    for ig in range(self.SpectroscopyX.NG):
                        SPECOUT[:,ig,ipath] = SPECOUT[:,ig,ipath] * xfac


            _lgr.info('CIRSradg :: Calculating GRADIENTS')
            dSPECOUT = np.transpose(-SPECOUT * np.transpose(dTAUTOT_LAYINC,axes=[2,3,0,1,4]),axes=[2,3,0,1,4])
            del dTAUTOT_LAYINC
            del TAUTOT_PATH


        elif PathCalc.ABSORBTION in IMODM: #Absorption spectrum (useful for small transmissions)
            _lgr.info('CIRSradg :: Calculating ABSORPTION')
            #Calculating the total opacity over the self.PathX
            TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

            #Absorption spectrum (useful for small transmissions)
            SPECOUT = 1.0 - np.exp(-(TAUTOT_PATH)) #(NWAVE,NG,NPATH)


        elif PathCalc.THERMAL_EMISSION in IMODM: #Thermal emission from planet
            _lgr.info('CIRSradg :: Calculating THERMAL_EMISSION')
            #Defining the units of the output spectrum
            xfac = np.ones(self.SpectroscopyX.NWAVE)
            if self.MeasurementX.IFORM==SpectraUnit.FluxRatio:
                xfac*=np.pi*4.*np.pi*((self.AtmosphereX.RADIUS)*1.0e2)**2.
                f = interpolate.interp1d(self.StellarX.WAVE,self.StellarX.SOLSPEC)
                solpspec = f(self.SpectroscopyX.WAVE)  #self.StellarX power spectrum (W (cm-1)-1 or W um-1)
                xfac = xfac / solpspec

            #Interpolating the emissivity of the self.SurfaceX to the calculation wavelengths
            if self.SurfaceX.TSURF>0.0:
                f = interpolate.interp1d(self.SurfaceX.VEM,self.SurfaceX.EMISSIVITY)
                EMISSIVITY = f(self.SpectroscopyX.WAVE)
            else:
                EMISSIVITY = np.zeros(self.SpectroscopyX.NWAVE)
                
            #Calculating the spectra
            for ipath in range(self.PathX.NPATH):
                NLAYIN = self.PathX.NLAYIN[ipath]
                EMTEMP = self.PathX.EMTEMP[0:NLAYIN,ipath]
                EMPRESS = self.LayerX.PRESS[self.PathX.LAYINC[0:NLAYIN,ipath]]
                
                SPECOUT[:,:,ipath],dSPECOUT[:,:,:,0:NLAYIN,ipath],dTSURF[:,:,ipath] = calc_thermal_emission_spectrumg(self.MeasurementX.ISPACE,self.SpectroscopyX.WAVE,TAUTOT_LAYINC[:,:,0:NLAYIN,ipath],dTAUTOT_LAYINC[:,:,:,0:NLAYIN,ipath],self.AtmosphereX.NVMR,EMTEMP,EMPRESS,self.SurfaceX.TSURF,EMISSIVITY)
        
                #Changing the units of the spectra and gradients
                SPECOUT[:,:,ipath] = (SPECOUT[:,:,ipath].T * xfac).T
                dTSURF[:,:,ipath] = (dTSURF[:,:,ipath].T * xfac).T
                dSPECOUT[:,:,:,:,ipath] = np.transpose(np.transpose(dSPECOUT[:,:,:,:,ipath],axes=[1,2,3,0])*xfac,axes=[3,0,1,2])

        #Now integrate over g-ordinates
        _lgr.info('CIRSradg :: Integrading over g-ordinates')
        SPECOUT = np.tensordot(SPECOUT, self.SpectroscopyX.DELG, axes=([1],[0])) #NWAVE,NPATH
        dSPECOUT = np.tensordot(dSPECOUT, self.SpectroscopyX.DELG, axes=([1],[0])) #(WAVE,NGAS+2+NDUST,NLAYIN,NPATH)
        dTSURF = np.tensordot(dTSURF, self.SpectroscopyX.DELG, axes=([1],[0])) #NWAVE,NPATH
        
        return SPECOUT,np.nan_to_num(dSPECOUT),dTSURF


###############################################################################################
    def calc_tau_cia(self,ISPACE=None,WAVEC=None,CIA=None,Atmosphere=None,Layer=None,MakePlot=False):
        """
        Calculate the CIA opacity in each atmospheric layer
        This is the new version developed for archNEMESIS (more versatile in terms of CIA pairs included)
        
        @param ISPACE: int
            Flag indicating whether the calculation must be performed in wavenumbers (0) or wavelength (1)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param CIA: class
            Python class defining the CIA cross sections
        @param Atmosphere: class
            Python class defining the reference atmosphere
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAUCIA(NWAVE,NLAY) :: CIA optical depth in each atmospheric layer
        dTAUCIA(NWAVE,NLAY,NVMR+2) :: Rate of change of CIA optical depth with:
                                 (0 to NVMR-1) Gaseous VMRs
                                 (NVMR) Temperature
                                 (NVMR+1) para-H2 fraction
        """

        from scipy import interpolate
        from archnemesis.CIA_0 import co2cia,n2h2cia,n2n2cia

       #Initialising variables
        if ISPACE is None:
            ISPACE = WaveUnit(self.MeasurementX.ISPACE)
        if WAVEC is None:
            WAVEC = self.SpectroscopyX.WAVE
        if CIA is None:
            CIA = self.CIAX
        if Atmosphere is None:
            Atmosphere = self.AtmosphereX
        if Layer is None:
            Layer = self.LayerX
            
        #Calculating the volume mixing ratios of each species in each layer
        q = np.transpose(Layer.PP.T / Layer.PRESS) #(NLAY,NVMR)
        
        #Calculating index of some specific species
        ico2 = -1
        ih2 = -1
        ihe = -1
        ich4 = -1
        in2 = -1
        for i in range(Atmosphere.NVMR):

            if Atmosphere.ID[i]==Gas.H2:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    ih2 = i

            if Atmosphere.ID[i]==Gas.He:
                ihe = i

            if Atmosphere.ID[i]==Gas.N2:
                in2 = i

            if Atmosphere.ID[i]==Gas.CH4:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    ich4 = i

            if Atmosphere.ID[i]==Gas.CO2:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    ico2 = i
        
        #Calculating which pairs depend on the ortho/para-H2 ratio
        INORMALD = CIA.locate_INORMAL_pairs()
        
        #Calculating the factor to be multiplied by the cross sections to get total optical depth
        TOTAM = Layer.TOTAM * 1.0e-4 #Total column density in each layer (cm-2)
        XLEN = Layer.DELH * 1.0e2 #Height of each layer (cm)
        XFAC = TOTAM**2. / XLEN   #molec^2 cm-5, which multiplied by cross sections in cm5 molec-2 gives unitless optical depth
        
        #Defining the calculation wavenumbers
        if ISPACE==WaveUnit.Wavenumber_cm:
            WAVEN = WAVEC
        elif ISPACE==WaveUnit.Wavelength_um:
            WAVEN = 1.e4/WAVEC
            isort = np.argsort(WAVEN)
            WAVEN = WAVEN[isort]

        if((WAVEN.min()<CIA.WAVEN.min()) or (WAVEN.max()>CIA.WAVEN.max())):
            _lgr.warning(' in CIA :: Calculation wavelengths expand a larger range than in CIA table')
            
       #calculating the CIA opacity at the correct temperature and wavenumber
        NWAVEC = len(WAVEC)   #Number of calculation wavelengths
        tau_cia_layer = np.zeros((NWAVEC,Layer.NLAY))
        dtau_cia_layer = np.zeros((NWAVEC,Layer.NLAY,Atmosphere.NVMR+2)) #gradients are calculated wrt each of the gas vmrs, temperature and para-H2 fraction
        for ilay in range(Layer.NLAY):

            #Interpolating to the correct temperature
            temp1 = Layer.TEMP[ilay]
            it = np.argmin(np.abs(CIA.TEMP-temp1))
            temp0 = CIA.TEMP[it]

            if CIA.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    temp1 = CIA.TEMP[it]
                    itl = 0
                    ithi = 1
                else:
                    itl = it - 1

            elif CIA.TEMP[it]<temp1:
                itl = it
                if it==CIA.NT-1:
                    temp1 = CIA.TEMP[it]
                    ithi = CIA.NT - 1
                    itl = CIA.NT - 2
                else:
                    ithi = it + 1
            
            frac1 = Layer.FRAC[ilay]
            ip = np.argmin(np.abs(CIA.FRAC-frac1))
            frac0 = CIA.FRAC[ip]
            
            if CIA.FRAC[ip]>=frac1:
                iphi = ip
                if ip==0:
                    frac1 = CIA.FRAC[ip]
                    ipl = 0
                    iphi = 1
                else:
                    ipl = ip - 1

            elif CIA.FRAC[ip]<frac1:
                ipl = ip
                if ip==CIA.NPARA-1:
                    temp1 = CIA.FRAC[ip]
                    iphi = CIA.NPARA - 1
                    ipl = CIA.NPARA - 2
                else:
                    iphi = ip + 1
            
            if CIA.NPARA == 0:
                ipl = 0
                iphi = 0
                
            # Extracting the CIA coefficients for the 4 surrounding points
            ktloplo = CIA.K_CIA[:, ipl, itl, :]
            kthiplo = CIA.K_CIA[:, ipl, ithi, :]
            ktlophi = CIA.K_CIA[:, iphi, itl, :]
            kthiphi = CIA.K_CIA[:, iphi, ithi, :]

            # Interpolation factors for temperature
            fhl_temp = (temp1 - CIA.TEMP[itl]) / (CIA.TEMP[ithi] - CIA.TEMP[itl])
            fhh_temp = (CIA.TEMP[ithi] - temp1) / (CIA.TEMP[ithi] - CIA.TEMP[itl])
            dfhldT = 1.0 / (CIA.TEMP[ithi] - CIA.TEMP[itl])
            dfhhdT = -1.0 / (CIA.TEMP[ithi] - CIA.TEMP[itl])

            # Interpolation factors for para fraction
            if len(CIA.FRAC) > 1:
                fhl_frac = (frac1 - CIA.FRAC[ipl]) / (CIA.FRAC[iphi] - CIA.FRAC[ipl])
                fhh_frac = (CIA.FRAC[iphi] - frac1) / (CIA.FRAC[iphi] - CIA.FRAC[ipl])
                dfhldF = 1.0 / (CIA.FRAC[iphi] - CIA.FRAC[ipl])
                dfhhdF = -1.0 / (CIA.FRAC[iphi] - CIA.FRAC[ipl])
            else:
                fhl_frac = 0.5
                fhh_frac = 0.5
                dfhldF = 0.0
                dfhhdF = 0.0

            # Final interpolation for kt and dktdT considering both temperature and para fraction
            ktlo = ktloplo * fhh_temp + kthiplo * fhl_temp
            kthi = ktlophi * fhh_temp + kthiphi * fhl_temp

            kt = ktlo * fhh_frac + kthi * fhl_frac
            # Derivative with respect to temperature
            dktdT = (kthi - ktlo) * dfhldT

            # Derivative with respect to fraction
            dktdF = (kthi - ktlo) * dfhldF + (ktlophi - ktloplo) * dfhhdF
                        
            #Cheking that interpolation can be performed to the calculation wavenumbers
            if( (CIA.WAVEN.min()<=WAVEN.min()) & (CIA.WAVEN.max()>=WAVEN.max()) ):
                
                inwave1 = np.where( (WAVEN>=CIA.WAVEN.min()) & (WAVEN<=CIA.WAVEN.max()) )[0]
                
                sum1 = np.zeros(NWAVEC)  #Temporary array to store the contribution from all CIA pairs
                for ipair in range(CIA.NPAIR):
                    
                    #Getting the indices of the two gases in the CIA pair
                    igas1 = np.where( Atmosphere.ID==CIA.IPAIRG1[ipair] )[0]
                    igas2 = np.where( Atmosphere.ID==CIA.IPAIRG2[ipair] )[0]
                    
                    if len(igas1)>1:
                        #raise ValueError('error in calc_tau_cia :: CIA does not currently allow the calculation of the CIA contribution from different isotopes.')
                        igas1 = np.where( (Atmosphere.ID==CIA.IPAIRG1[ipair]) & (Atmosphere.ISO==1) )[0] #Selecting the most abundant isotope only
                
                    if len(igas2)>1:
                        #raise ValueError('error in calc_tau_cia :: CIA does not currently allow the calculation of the CIA contribution from different isotopes.')
                        igas2 = np.where( (Atmosphere.ID==CIA.IPAIRG2[ipair]) & (Atmosphere.ISO==1) )[0] #Selecting the most abundant isotope only
                
                    
                    if((len(igas1)==1) & (len(igas2)==1)):
                        
                        #Both gases are defined in the atmosphere and therefore we can have CIA absorption
                        igas1 = igas1[0]
                        igas2 = igas2[0]
                        
                        
                        #Interpolating the CIA cross sections to the correct wavenumbers
                        k_cia = np.zeros(NWAVEC)
                        dkdT_cia = np.zeros(NWAVEC)
                        
                        f = interpolate.interp1d(CIA.WAVEN,kt[ipair,:])
                        k_cia[inwave1] = f(WAVEN[inwave1])
                        f = interpolate.interp1d(CIA.WAVEN,dktdT[ipair,:])
                        dkdT_cia[inwave1] = f(WAVEN[inwave1])
                
                        if INORMALD[ipair]==True:
                            #This pair depends on the INORMAL flag and is used only if the flag is true
                        
                            if CIA.INORMALT[ipair]==CIA.INORMAL:
                            
                                sum1[:] = sum1[:] + k_cia[:] * q[ilay,igas1] * q[ilay,igas2]
                                
                                dtau_cia_layer[:,ilay,igas1] = dtau_cia_layer[:,ilay,igas1] + q[ilay,igas2] * k_cia[:]
                                dtau_cia_layer[:,ilay,igas2] = dtau_cia_layer[:,ilay,igas2] + q[ilay,igas1] * k_cia[:]
                                dtau_cia_layer[:,ilay,Atmosphere.NVMR-2] = dtau_cia_layer[:,ilay,Atmosphere.NVMR-2] + dkdT_cia[:] * q[ilay,igas1] * q[ilay,igas2]
                                
                        else:
                            
                            #This pair does not depend in the INORMAL flag
                            sum1[:] = sum1[:] + k_cia[:] * q[ilay,igas1] * q[ilay,igas2]
                            
                            dtau_cia_layer[:,ilay,igas1] = dtau_cia_layer[:,ilay,igas1] + q[ilay,igas2] * k_cia[:]
                            dtau_cia_layer[:,ilay,igas2] = dtau_cia_layer[:,ilay,igas2] + q[ilay,igas1] * k_cia[:]
                            dtau_cia_layer[:,ilay,Atmosphere.NVMR-2] = dtau_cia_layer[:,ilay,Atmosphere.NVMR-2] + dkdT_cia[:] * q[ilay,igas1] * q[ilay,igas2]
                            

                #Look up CO2-CO2 CIA coefficients (external)
                if ico2!=-1:
                    k_co2 = co2cia(WAVEN)
                    sum1[:] = sum1[:] + k_co2[:] * q[ilay,ico2] * q[ilay,ico2]
                    dtau_cia_layer[:,ilay,ico2] = dtau_cia_layer[:,ilay,ico2] + 2.*q[ilay,ico2]*k_co2[:]

                #Look up N2-N2 NIR CIA coefficients (external)
                if in2!=-1:
                    k_n2n2 = n2n2cia(WAVEN)
                    sum1[:] = sum1[:] + k_n2n2[:] * q[ilay,in2] * q[ilay,in2]
                    dtau_cia_layer[:,ilay,in2] = dtau_cia_layer[:,ilay,in2] + 2.*q[ilay,in2]*k_n2n2[:]

                #Look up N2-H2 NIR CIA coefficients (external)
                if((in2!=-1) & (ih2!=-1)):
                    k_n2h2 = n2h2cia(WAVEN)
                    sum1[:] = sum1[:] + k_n2h2[:] * q[ilay,in2] * q[ilay,ih2]
                    dtau_cia_layer[:,ilay,ih2] = dtau_cia_layer[:,ilay,ih2] + q[ilay,in2] * k_n2h2[:]
                    dtau_cia_layer[:,ilay,in2] = dtau_cia_layer[:,ilay,in2] + q[ilay,ih2] * k_n2h2[:]

                tau_cia_layer[:,ilay] = sum1[:] * XFAC[ilay]
                dtau_cia_layer[:,ilay,:] = dtau_cia_layer[:,ilay,:] * XFAC[ilay]
                
        if ISPACE==WaveUnit.Wavelength_um:
            tau_cia_layer[:,:] = tau_cia_layer[isort,:]
            dtau_cia_layer[:,:,:] = dtau_cia_layer[isort,:,:]

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for ilay in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_cia_layer[:,ilay])
            ax1.grid()
            plt.tight_layout()
            plt.show()
            
        return tau_cia_layer,dtau_cia_layer

###############################################################################################
    def calc_tau_dust(self, WAVEC=None, Scatter=None, Layer=None, MakePlot=False):
        """
        Calculate the aerosol opacity in each atmospheric layer

        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Scatter: class
            Scatter:: Python class defining the optical properties of the aerosols in the atmosphere
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAUDUST(NWAVE,NLAY,NDUST) :: Aerosol opacity for each aerosol type and each layer (from extinction coefficients)
        TAUCLSCAT(NWAVE,NLAY,NDUST) :: Aerosol scattering opacity for each aerosol type and each layer
        dTAUDUSTdq(NWAVE,NLAY,NDUST) :: Rate of change of the aerosol opacity with the dust abundance
        dTAUCLSCATdq(NWAVE,NLAY,NDUST) :: Rate of change of the aerosol scattering opacity with dust abundance
        """

        # Initialising variables
        if WAVEC is None:
            WAVEC = self.SpectroscopyX.WAVE
        if Scatter is None:
            Scatter = self.ScatterX
        if Layer is None:
            Layer = self.LayerX

        from scipy import interpolate

        if (WAVEC.min() < Scatter.WAVE.min()) & (WAVEC.max() > Scatter.WAVE.min()):
            _lgr.info(f"spectral range for calculation =  {(WAVEC.min(),'-',WAVEC.max())}")
            _lgr.info(f"spectra range for optical properties =  {(Scatter.WAVE.min(),'-',Scatter.WAVE.max())}")
            raise ValueError('error calc_tau_dust :: Spectral range for calculation is outside of range in which the Aerosol properties are defined')
        
        # Calculating the opacity at each vertical layer for each dust population
        NWAVEC = len(WAVEC)
        TAUDUST = np.zeros((NWAVEC, Layer.NLAY, Scatter.NDUST))
        TAUCLSCAT = np.zeros((NWAVEC, Layer.NLAY, Scatter.NDUST))
        dTAUDUSTdq = np.zeros((NWAVEC, Layer.NLAY, Scatter.NDUST))
        dTAUCLSCATdq = np.zeros((NWAVEC, Layer.NLAY, Scatter.NDUST))

        for i in range(Scatter.NDUST):
            if i in self.AtmosphereX.DUST_RENORMALISATION.keys():
                Layer.CONT[:, i] = Layer.CONT[:, i] / Layer.CONT[:, i].sum() * 1e4 * self.AtmosphereX.DUST_RENORMALISATION[i]

            if Scatter.NWAVE > 2:
                f_kext = interpolate.interp1d(Scatter.WAVE, Scatter.KEXT[:, i], kind='cubic')
                f_ksca = interpolate.interp1d(Scatter.WAVE, Scatter.KSCA[:, i], kind='cubic')
            else:
                f_kext = interpolate.interp1d(Scatter.WAVE, Scatter.KEXT[:, i])
                f_ksca = interpolate.interp1d(Scatter.WAVE, Scatter.KSCA[:, i])

            kext = f_kext(WAVEC)
            ksca = f_ksca(WAVEC)

            # Replace invalid values using original arrays
            invalid_ksca = (ksca < 0) & (kext > 0)
            invalid_kext = (kext < 0) & (ksca > 0)
            invalid_both = (kext < ksca)

            if np.any(invalid_ksca):
                ksca[invalid_ksca] = interpolate.interp1d(Scatter.WAVE, Scatter.KSCA[:, i], fill_value="extrapolate")(WAVEC[invalid_ksca])
            if np.any(invalid_kext):
                kext[invalid_kext] = interpolate.interp1d(Scatter.WAVE, Scatter.KEXT[:, i], fill_value="extrapolate")(WAVEC[invalid_kext])
            if np.any(invalid_both):
                kext[invalid_both] = interpolate.interp1d(Scatter.WAVE, Scatter.KEXT[:, i], fill_value="extrapolate")(WAVEC[invalid_both])
                ksca[invalid_both] = interpolate.interp1d(Scatter.WAVE, Scatter.KSCA[:, i], fill_value="extrapolate")(WAVEC[invalid_both])

            # Calculating the opacity at each layer
            for j in range(Layer.NLAY):
                DUSTCOLDENS = Layer.CONT[j, i]  # particles/m2
                TAUDUST[:, j, i] = kext * 1.0e-4 * DUSTCOLDENS
                TAUCLSCAT[:, j, i] = ksca * 1.0e-4 * DUSTCOLDENS
                dTAUDUSTdq[:, j, i] = kext * 1.0e-4  # dtau/dAMOUNT (m2)
                dTAUCLSCATdq[:, j, i] = ksca * 1.0e-4  # dtau/dAMOUNT (m2)

        return TAUDUST, TAUCLSCAT, dTAUDUSTdq, dTAUCLSCATdq





###############################################################################################

    def calc_tau_rayleigh(self,IRAY=None,ISPACE=None,WAVEC=None,ID=None,ISO=None,Layer=None,MakePlot=False):
        """
        Function to calculate the Rayleigh scattering opacity in each atmospheric layer,

        Inputs
        ________

        IRAY :: Flag indicating the type of Rayleigh scattering to be applied
        ISPACE :: Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
        WAVEC :: Wavenumber (cm-1) or wavelength array (um)
        ID(NGAS) :: Radtran ID of each atmospheric gas
        ISO(NGAS) :: Radtran ID of each isotope
        Layer :: Python class defining the layering scheme to be applied in the calculations


        Optional inputs
        ________________
        
        MakePlot :: If True, a summary plot of the Rayleigh scattering optical depth is generated

        Outputs
        ________

        TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
        dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer
        """
        
       #Initialising variables
        if IRAY is None:
            IRAY = self.ScatterX.IRAY
        if ISPACE is None:
            ISPACE = WaveUnit(self.MeasurementX.ISPACE)
        if WAVEC is None:
            WAVEC = self.SpectroscopyX.WAVE
        if ID is None:
            ID = self.AtmosphereX.ID
        if ISO is None:
            ISO = self.AtmosphereX.ISO
        if Layer is None:
            Layer = self.LayerX
        
        if IRAY==RayleighScatteringMode.NOT_INCLUDED:  #No Rayleigh scattering
            TAURAY = np.zeros((len(WAVEC),Layer.NLAY))
            dTAURAY = np.zeros((len(WAVEC),Layer.NLAY))
        elif IRAY==RayleighScatteringMode.GAS_GIANT_ATM: #Gas giant atmosphere
            TAURAY,dTAURAY = calc_tau_rayleighj(ISPACE,WAVEC,Layer.TOTAM) #(NWAVE,NLAY)
        elif IRAY==RayleighScatteringMode.C02_DOMINATED_ATM:
            TAURAY,dTAURAY = calc_tau_rayleighv2(ISPACE,WAVEC,Layer.TOTAM) #(NWAVE,NLAY)
        elif IRAY==RayleighScatteringMode.JOVIAN_AIR: #Jovian air
            TAURAY,dTAURAY = calc_tau_rayleighls(ISPACE,WAVEC,ID,ISO,(Layer.PP.T/Layer.PRESS).T,Layer.TOTAM)
        else:
            raise ValueError('error in CIRSrad :: IRAY = '+str(IRAY)+' type has not been implemented yet')
            
        #Making summary plot if required
        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(7,4))
            ax1.plot(WAVEC,np.sum(TAURAY,axis=1))
            ax1.grid()
            if ISPACE==WaveUnit.Wavenumber_cm:
                vlabel = 'Wavenumber (cm$^{-1}$)'
            elif ISPACE==WaveUnit.Wavelength_um:
                vlabel = r'Wavelength ($\mu$m)'
            ax1.set_xlabel(vlabel)
            ax1.set_ylabel('Rayleigh scattering optical depth')
            ax1.set_facecolor('lightgray')
            plt.tight_layout()
            plt.show()
        return TAURAY,dTAURAY


###############################################################################################
    def calc_tau_gas(self):
        """
        Calculate the aerosol opacity in each atmospheric layer

        Inputs
        ________
        
        Measurement :: Measurement class
        Spectroscopy :: Spectroscopy class
        Layer :: Layer class 

        Outputs
        ________

        TAUGAS(NWAVE,NG,NLAY) :: Gaseous opacity in each layer for each g-ordinate (NG=1 if line-by-line)

        """

        #from NemesisPy.nemesisf import spectroscopy
        #Calculating the gaseous line opacity in each layer
        ########################################################################################################

        if self.SpectroscopyX.ILBL == SpectralCalculationMode.LINE_BY_LINE_TABLES:  #LBL-table

            TAUGAS = np.zeros((self.SpectroscopyX.NWAVE,self.SpectroscopyX.NG,self.LayerX.NLAY,self.SpectroscopyX.NGAS))  #Vertical opacity of each gas in each layer

            #Calculating the cross sections for each gas in each layer
            k = self.SpectroscopyX.calc_klbl(self.LayerX.NLAY,self.LayerX.PRESS/101325.,self.LayerX.TEMP,WAVECALC=self.SpectroscopyX.WAVE)

            for i in range(self.SpectroscopyX.NGAS):
                IGAS = np.where( (self.AtmosphereX.ID==self.SpectroscopyX.ID[i]) & (self.AtmosphereX.ISO==self.SpectroscopyX.ISO[i]) )
                IGAS = IGAS[0]

                #Calculating vertical column density in each layer
                VLOSDENS = self.LayerX.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #cm-2

                #Calculating vertical opacity for each gas in each layer
                TAUGAS[:,0,:,i] = k[:,:,i] * VLOSDENS

            #Combining the gaseous opacity in each layer
            TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)

            #Removing necessary data to save memory
            del k

        elif self.SpectroscopyX.ILBL == SpectralCalculationMode.K_TABLES:    #K-table

            #Calculating the k-coefficients for each gas in each layer
            k_gas,dkgasdT = self.SpectroscopyX.calc_kg(self.LayerX.NLAY,self.LayerX.PRESS/101325.,self.LayerX.TEMP,WAVECALC=self.SpectroscopyX.WAVE) # (NWAVE,NG,NLAY,NGAS)

            f_gas = np.zeros((self.SpectroscopyX.NGAS,self.LayerX.NLAY))
            utotl = np.zeros(self.LayerX.NLAY)
            for i in range(self.SpectroscopyX.NGAS):
                IGAS = np.where( (self.AtmosphereX.ID==self.SpectroscopyX.ID[i]) & (self.AtmosphereX.ISO==self.SpectroscopyX.ISO[i]) )
                IGAS = IGAS[0]

                #When using gradients
                f_gas[i,:] = self.LayerX.AMOUNT[:,IGAS[0]] * 1.0e-4 * 1.0e-20  #Vertical column density of the radiatively active gases in cm-2

            #Combining the k-distributions of the different gases in each layer
            k_layer,dk_layer = k_overlapg(self.SpectroscopyX.DELG,k_gas,dkgasdT,f_gas)

            #Calculating the opacity of each layer
            TAUGAS = k_layer #(NWAVE,NG,NLAY)

            #Removing necessary data to save memory
            del k_gas
            del k_layer

        else:
            raise ValueError(f'error in CIRSrad :: ILBL must be either {SpectralCalculationMode(0)} or {SpectralCalculationMode(2)}')
        return TAUGAS

    ###############################################################################################
    ###############################################################################################
    # MULTIPLE SCATTERING ROUTINES
    ###############################################################################################
    ###############################################################################################
    
    def scloud11wave(self, WAVE, Scatter, Surface, Layer, Measurement, Path, SOLAR):
        """

        Compute emergent intensity at top of multilayer cloud using the
        matrix operator algorithm.  Diffuse incident radiation is allowed
        at the bottom and single-beam incident radiation (sunlight) at
        the top. 

        If the emission angle is >90, then it is assumed that the observer is
        at the surface or bottom atmospheric layer and looking up. If the emission
        angle is <90, then it is assumed that the observer is at the top of the
        atmosphere and looking down towards the surface.
 
        Inputs
        ________

        WAVE :: Calculation wavelengths or wavenumbers
        Scatter :: Python class defining the scattering setup
        Surface :: Python class defining the surface setup
        Layer :: Python class defining the properties of each layer including the optical depths
        Measurement :: Python class defining the measurement
        Path :: Python class defining the calculation paths (i.e. viewing angles)
        SOLAR(NWAVE) :: Solar flux 

        Outputs
        ________

        SPECOUT(NWAVE) :: Modelled radiance 

        """
        
        
        from archnemesis.Multiple_Scattering_Core import scloud11wave_core

        NWAVE = Layer.TAUTOT.shape[0]
        if NWAVE!=len(WAVE):
            _lgr.info(NWAVE,len(WAVE))
            raise ValueError('error in scloud11wave :: number of calculation wavelengths is not correct')
        NG = Layer.TAUTOT.shape[1]
        SPEC = np.zeros((NWAVE, NG, Path.NPATH))

        # Scatter parameters
        MU = Scatter.MU  
        WTMU = Scatter.WTMU
        NF = Scatter.NF
        NPHI = Scatter.NPHI
        NTHETA = len(Scatter.THETA)
        IRAY = Scatter.IRAY
        IMIE = Scatter.IMIE
        NCONT = Scatter.NDUST
        _lgr.debug(f'{MU=} {WTMU=} {NF=} {NPHI=} {NTHETA=} {IRAY=} {IMIE=} {NCONT=}')

        # Path parameters
        VWAVES = WAVE
        SOL_ANGS = Path.SOL_ANG
        EMISS_ANGS = Path.EMISS_ANG
        AZI_ANGS = Path.AZI_ANG
        _lgr.debug(f'{VWAVES=} {SOL_ANGS=} {EMISS_ANGS=} {AZI_ANGS=}')

        # Surface parameters
        RADGROUND = np.zeros((NWAVE,Scatter.NMU))
        ALBEDO = np.zeros(NWAVE)
        EMISSIVITY = np.zeros(NWAVE)
        LOWBC = Surface.LOWBC

        if Surface.GASGIANT or (Surface.TSURF <= 0.0):  # No surface
            RADGROUND[:,:] = planck(Measurement.ISPACE, WAVE, Layer.TEMP[0])[:, None]
        else:
            bbsurf = planck(Measurement.ISPACE, WAVE, Surface.TSURF)
            EMISSIVITY[:] = interp1d(Surface.VEM, Surface.EMISSIVITY)(WAVE)
            for imu in range(Scatter.NMU):
                RADGROUND[:,imu] = bbsurf * EMISSIVITY

        if (not Surface.GASGIANT) and (Surface.LOWBC != LowerBoundaryCondition.THERMAL):
            BRDF_matrix = self.calc_brdf_matrix(WAVEC=WAVE, Surface=Surface, Scatter=Scatter)  #(NWAVE,NMU,NMU,NF+1)
        else:
            BRDF_matrix = np.zeros((NWAVE, Scatter.NMU, Scatter.NMU, Scatter.NF+1))
        
        _lgr.debug(f'{RADGROUND=} {ALBEDO=} {EMISSIVITY=} {LOWBC=}')

        # Layers
        BB = np.zeros(Layer.TAURAY.shape)  #Blackbody in each layer
        for ilay in range(Layer.NLAY):
            BB[:,ilay] = planck(Measurement.ISPACE, WAVE, Layer.TEMP[ilay])
        TAU = Layer.TAUTOT
        TAURAY = Layer.TAURAY
        _lgr.debug(f'{BB=} {TAU=} {TAURAY=}')

        # Calculate the fraction of each aerosol scattering
        FRAC = np.zeros((NWAVE, Layer.NLAY, NCONT))
        iiscat = np.where(Layer.TAUSCAT > 0.0)
        if iiscat[0].size > 0:
            FRAC[iiscat[0], iiscat[1], 0:Scatter.NDUST] = (
                Layer.TAUCLSCAT[iiscat[0], iiscat[1], :].T /
                Layer.TAUSCAT[iiscat[0], iiscat[1]]
            ).T
        FRAC = np.transpose(FRAC, (0, 2, 1))  #(NWAVE,NCONT,NLAY)
        _lgr.debug(f'{FRAC=}')

        # Single scattering albedo
        OMEGA = np.zeros((NWAVE, NG, Layer.NLAY))
        iin = np.where(Layer.TAUTOT > 0.0)
        if iin[0].size > 0:
            OMEGA[iin[0], iin[1], iin[2]] = (
                (Layer.TAURAY[iin[0], iin[2]] + Layer.TAUSCAT[iin[0], iin[2]]) /
                Layer.TAUTOT[iin[0], iin[1], iin[2]]
            )
        _lgr.debug(f'{OMEGA=}')
        
        # Phase function
        PHASE_ARRAY = np.zeros((Scatter.NDUST, NWAVE, 2, NTHETA))
        if Scatter.IMIE == AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:
            for i in range(Scatter.NDUST):
                PHASE_ARRAY[i, :, 0, -1] = np.interp(VWAVES,Scatter.WAVE,Scatter.F.T[i])
                PHASE_ARRAY[i, :, 0, -2] = np.interp(VWAVES,Scatter.WAVE,Scatter.G1.T[i])
                PHASE_ARRAY[i, :, 0, -3] = np.interp(VWAVES,Scatter.WAVE,Scatter.G2.T[i])
        else:
            PHASE_ARRAY[:, :, 0, :] = np.transpose(Scatter.calc_phase(Scatter.THETA, WAVE), (2, 0, 1))
            
        PHASE_ARRAY[:, :, 1, :] = np.cos(Scatter.THETA * np.pi / 180)
        _lgr.debug(f'{PHASE_ARRAY=}')
        
        # Core function call
        SPEC = scloud11wave_core(
            phasarr=PHASE_ARRAY[:, :, :, ::-1],
            radg=RADGROUND,
            sol_angs=SOL_ANGS,
            emiss_angs=EMISS_ANGS,
            solar=SOLAR,
            aphis=AZI_ANGS,
            lowbc=LOWBC,
            brdf_matrix=BRDF_matrix,
            mu1=MU,
            wt1=WTMU,
            nf=NF,
            vwaves=VWAVES,
            bnu=BB[:,:],
            taus=TAU[:,:,:],
            tauray=TAURAY[:,:],
            omegas_s=OMEGA[:,:,:],
            nphi=NPHI,
            iray=IRAY,
            imie=IMIE,
            lfrac=FRAC
        )

        SPEC = np.transpose(SPEC, (2, 1, 0))
        return SPEC

    ###############################################################################################
    def scloud11flux(self,Scatter,Surface,Layer,Measurement,SOLAR,diffuse=True):
        """
        Compute and return internal radiation fields in a scattering atmosphere
        Code uses matrix operator algorithm.  Diffuse incident radiation 
        is allowed at the bottom and single-beam incident radiation 
        (sunlight) at the top. 
 
        The layer numbers here are 1 at the top increasing to NLAY at the 
        bottom. If a Lambertian reflector is added at the bottom then the 
        number of layers is increased by 1.

        NOTE:  the angle arrays passed from the calling pgm. are assumed to be 
        in order of increasing MU.  However, the CLOUD subroutines use the 
        opposite convention, so we must reverse the order.  (The order has no 
        effect within this routine, other than in using the supplied boundary 
        conditions, but it does affect pre-/post-processing in the calling s/w.)

        Optimised for maximum speed by cutting out any code from scloud6
        which is not actually used for NIMS retrieval runs.
 
        Inputs
        ________

        Scatter :: Python class defining the scattering setup
        Surface :: Python class defining the surface setup
        Layer :: Python class defining the properties of each layer including the optical depths
        Measurement :: Python class defining the measurement
        SOLAR(NWAVE) :: Solar flux 

        Optional inputs
        _______________

        diffuse :: If False, scattering is turned off so that results is only the direct component

        Outputs
        ________

        Uplf(NWAVE,NG,NMU,NLAY,NF) :: Internal radiances in each viewing direction (downwards)
        Umif(NWAVE,NG,NMU,NLAY,NF) :: Internal radiances in each viewing direction (upwards)
        """

        from scipy.interpolate import interp1d
        #from NemesisPy import nemesisf

        ################################################################################
        #INITIALISING VARIABLES AND PERFORMING INITIAL CALCULATIONS
        ##############################################################################

        #Defining the number of scattering species
        if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
            NCONT = Scatter.NDUST + 1
        else:
            NCONT = Scatter.NDUST

        #Find correction for any quadrature errors
        xfac = np.sum(Scatter.MU*Scatter.WTMU)
        xfac = 0.5/xfac

        LTOT = Layer.NLAY     # Set internal number of layers
        LT1 = LTOT

        #In case of surface reflection, add extra dummy layer at bottom,
        #whose transmission matrix = (1-A)*Unit-Matrix. This layer must be
        #omitted from computation by doubling

        if Surface.LOWBC != LowerBoundaryCondition.THERMAL:
            LTOT = LTOT + 1

        #Reset the order of angles
        Scatter.MU = Scatter.MU[::-1]
        Scatter.WTMU = Scatter.WTMU[::-1]

        #Setting up constant matrices
        E = np.identity(Scatter.NMU)
        MM = np.zeros((Scatter.NMU,Scatter.NMU))
        MMINV = np.zeros((Scatter.NMU,Scatter.NMU))
        CC = np.zeros((Scatter.NMU,Scatter.NMU))
        CCINV = np.zeros((Scatter.NMU,Scatter.NMU))
        np.fill_diagonal(MM,[Scatter.MU])
        np.fill_diagonal(MMINV,[1./Scatter.MU])
        np.fill_diagonal(CC,[Scatter.WTMU])
        np.fill_diagonal(CCINV,[1./Scatter.WTMU])



        ################################################################################
        #CALCULATE THE ALBEDO, EMISSIVITY AND GROUND EMISSION AT THE SURFACE
        ################################################################################

        _lgr.info('scloud11flux :: Calculating surface properties')

        #Calculating the surface properties at each wavelength (emissivity, albedo and thermal emission)
        RADGROUND = np.zeros(Spectroscopy.NWAVE)
        ALBEDO = np.zeros(Spectroscopy.NWAVE)
        EMISSIVITY = np.zeros(Spectroscopy.NWAVE)

        if Surface.TSURF<=0.0:  #No surface
            RADGROUND[:] = planck(Measurement.ISPACE,Spectroscopy.WAVE,Layer.TEMP[0])
        else:
            #Calculating the blackbody at given temperature
            bbsurf = planck(Measurement.ISPACE,Spectroscopy.WAVE,Surface.TSURF)

            #Calculating the emissivity
            f = interp1d(Surface.VEM,Surface.EMISSIVITY)
            EMISSIVITY[:] = f(Spectroscopy.WAVE)

            #Calculating thermal emission from surface
            RADGROUND[:] = bbsurf * EMISSIVITY

            #Calculating ground albedo
            if Surface.GALB<0.0:
                ALBEDO[:] = 1.0 - EMISSIVITY[:]
            else:
                ALBEDO[:] = Surface.GALB



        ################################################################################
        #CALCULATING THE THERMAL EMISSION OF EACH LAYER
        ################################################################################

        _lgr.info('scloud11flux :: Calculating thermal emission of each layer')

        #Calculating the thermal emission of each atmospheric layer
        BB = np.zeros((Spectroscopy.NWAVE,Layer.NLAY))  #Blackbody in each layer
        for ilay in range(Layer.NLAY):
            BB[:,ilay] = planck(Measurement.ISPACE, Spectroscopy.WAVE, Layer.TEMP[ilay])



        ################################################################################
        #CALCULATING THE EFFECTIVE PHASE FUNCTION IN EACH LAYER
        ################################################################################

        _lgr.info('scloud11flux :: Calculating phase matrix and scattering properties of each layer')

        #Calculating the phase matrices for each aerosol population and Rayleigh scattering
        PPLPL,PPLMI = self.calc_phase_matrix(Scatter,Spectroscopy.WAVE)  #(NWAVE,NCONT,NF+1,NMU,NMU)

        #Calculating the fraction of scattering by each aerosol type and rayleigh
        FRAC = np.zeros((Spectroscopy.NWAVE,Layer.NLAY,NCONT))
        iiscat = np.where((Layer.TAUSCAT+Layer.TAURAY)>0.0)
        if(len(iiscat[0])>0):
            FRAC[iiscat[0],iiscat[1],0:Scatter.NDUST] = np.transpose(np.transpose(Layer.TAUCLSCAT[iiscat[0],iiscat[1],:],axes=[1,0]) / ((Layer.TAUSCAT[iiscat[0],iiscat[1]]+Layer.TAURAY[iiscat[0],iiscat[1]])),axes=[1,0])  #Fraction of each aerosol scattering FRAC = TAUCLSCAT/(TAUSCAT+TAURAY)
            if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
                FRAC[iiscat[0],iiscat[1],Scatter.NDUST] = Layer.TAURAY[iiscat[0],iiscat[1]] / ((Layer.TAUSCAT[iiscat[0],iiscat[1]]+Layer.TAURAY[iiscat[0],iiscat[1]])) #Fraction of Rayleigh scattering FRAC = TAURAY/(TAUSCAT+TAURAY)

        #Calculating the weighted averaged phase matrix in each layer and direction
        _lgr.info('scloud11flux :: Calculating weighted average phase matrix in each layer')
        PPLPLS = np.zeros((Spectroscopy.NWAVE,Layer.NLAY,Scatter.NF+1,Scatter.NMU,Scatter.NMU))
        PPLMIS = np.zeros((Spectroscopy.NWAVE,Layer.NLAY,Scatter.NF+1,Scatter.NMU,Scatter.NMU))

        for ilay in range(Layer.NLAY):
            PPLPLS[:,ilay,:,:,:] =  np.transpose(np.sum(np.transpose(PPLPL,axes=[2,3,4,0,1])*FRAC[:,ilay,:],axis=4),axes=[3,0,1,2])  #SUM(PPLPL*FRAC)
            PPLMIS[:,ilay,:,:,:] =  np.transpose(np.sum(np.transpose(PPLMI,axes=[2,3,4,0,1])*FRAC[:,ilay,:],axis=4),axes=[3,0,1,2])  #SUM(PPLMI*FRAC)

        #Calculating the single scattering albedo of each layer (TAURAY+TAUSCAT/TAUTOT)
        NG = Layer.TAUTOT.shape[1]
        OMEGA = np.zeros((Spectroscopy.NWAVE,NG,Layer.NLAY))
        iin = np.where(Layer.TAUTOT>0.0)
        if(len(iin[0])>0):
            OMEGA[iin[0],iin[1],iin[2]] = (Layer.TAURAY[iin[0],iin[2]]+Layer.TAUSCAT[iin[0],iin[2]]) / Layer.TAUTOT[iin[0],iin[1],iin[2]]


        if diffuse==False:
            OMEGA[:,:,:] = 0.0  #No scattering if diffuse component is turned off
        
        ################################################################################
        #CALCULATING THE REFLECTION, TRANSMISSION AND SOURCE MATRICES FOR EACH LAYER
        #################################################################################

        RL1,TL1,JL1,ISCL1 = nemesisf.mulscatter.calc_rtf_matrix(Scatter.MU,Scatter.WTMU,\
                                                    Layer.TAUTOT,OMEGA,Layer.TAURAY,BB,PPLPLS,PPLMIS)
        #(NWAVE,NG,NLAY,NF+1,NMU,NMU)
        
        
        #################################################################################
        #CALCULATING THE REFLECTION, TRANSMISSION AND SOURCE MATRICES FOR SURFACE
        #################################################################################

        JL = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NF+1,Scatter.NMU,1))  #Source function of atmosphere + surface
        RL = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NF+1,Scatter.NMU,Scatter.NMU))  #Reflection matrix of atmosphere + surface
        TL = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NF+1,Scatter.NMU,Scatter.NMU))  #Transmission matrix of atmosphere + surface 
        ISCL = np.zeros((Spectroscopy.NWAVE,NG,LTOT),dtype='int32')  #Flag indicating if the layer is scattering

        if Surface.GASGIANT==False:

            _lgr.info('scloud11flux :: Calculating the reflection, transmission and source matrices of the surface')

            JS = np.zeros((Spectroscopy.NWAVE,Scatter.NF+1,Scatter.NMU,1))  #Source function
            RS = np.zeros((Spectroscopy.NWAVE,Scatter.NF+1,Scatter.NMU,Scatter.NMU))  #Reflection matrix
            TS = np.zeros((Spectroscopy.NWAVE,Scatter.NF+1,Scatter.NMU,Scatter.NMU))  #Transmission matrix

            if Surface.LOWBC == LowerBoundaryCondition.LAMBERTIAN:  #Lambertian reflection

                IC = 0   #For the rest of the NF values, it is zero
                for j in range(Scatter.NMU):

                    JS[:,IC,j,0] = (1.0-ALBEDO[:])*RADGROUND[:]  #Source function is considered isotropic

                    for i in range(Scatter.NMU):

                        TS[:,IC,i,j] = 0.0    #Transmission at the surface is zero
                        RS[:,IC,i,j] = 2.0*ALBEDO[:]*Scatter.MU[j]*Scatter.WTMU[j]  #Sum of MU*WTMU = 0.5
                        #Make any quadrature correction
                        RS[:,IC,i,j] = RS[:,IC,i,j]*xfac

            elif Surface.LOWBC == LowerBoundaryCondition.HAPKE:  #Hapke surface

                Reflectivity = self.calc_hapke_reflectivity(Scatter,Surface,Spectroscopy.WAVE)

                #REFLECTION
                for j in range(Scatter.NMU):
                    for i in range(Scatter.NMU):
                        for kl in range(Scatter.NF+1):
                            RS[:,kl,i,j] = 2.0*Reflectivity[:,kl,i,j]*Scatter.MU[j]*Scatter.WTMU[j]  #Sum of MU*WTMU = 0.5
                            #Make any quadrature correction
                            RS[:,kl,i,j] = RS[:,kl,i,j]*xfac

                #THERMAL EMISSION
                IC = 0   #For the rest of the NF values, it is zero
                for j in range(Scatter.NMU):
                    JS[:,IC,j,0] = EMISSIVITY[:]*RADGROUND[:]  #Source function is considered isotropic



            #Adding the surface matrix to the combined atmosphere + surface system
            JL[:,:,0,:,:,:] = np.repeat(JS[:,np.newaxis,:,:,:],NG,axis=1)
            RL[:,:,0,:,:,:] = np.repeat(RS[:,np.newaxis,:,:,:],NG,axis=1)
            TL[:,:,0,:,:,:] = np.repeat(TS[:,np.newaxis,:,:,:],NG,axis=1)

            #Adding the atmosphere to the combined atmosphere + surface system
            JL[:,:,1:LTOT,:,:,:] = JL1[:,:,:,:,:,:]
            TL[:,:,1:LTOT,:,:,:] = TL1[:,:,:,:,:,:]
            RL[:,:,1:LTOT,:,:,:] = RL1[:,:,:,:,:,:]
            #ISCL[:,:,1:LTOT] = ISCL1[:,:,:]
            ISCL[:,:,:] = 1

        else:

            #Adding the atmosphere to the combined atmosphere + surface system
            JL[:,:,:,:,:,:] = JL1[:,:,:,:,:,:]
            TL[:,:,:,:,:,:] = TL1[:,:,:,:,:,:]
            RL[:,:,:,:,:,:] = RL1[:,:,:,:,:,:]
            ISCL[:,:,:] = ISCL1[:,:,:]

        ###############################################################################
        # CALCULATING THE INTERNAL RADIATION FIELDS
        ###############################################################################

        Uplf = np.zeros((Spectroscopy.NWAVE,NG,Scatter.NMU,Layer.NLAY,Scatter.NF+1))  #Internal radiances in each viewing direction (downwards)
        Umif = np.zeros((Spectroscopy.NWAVE,NG,Scatter.NMU,Layer.NLAY,Scatter.NF+1))  #Internal radiances in each viewing direction (upwards)

        _lgr.info('scloud11flux :: Calculating spectra')
        for IC in range(Scatter.NF+1):

            #***********************************************************************
            #CALCULATE UPWARD MATRICES FOR COMPOSITE OF L LAYERS FROM BASE OF CLOUD.
            #XBASE(I,J,L) IS THE X MATRIX FOR THE BOTTOM L LAYERS OF THE CLOUD.
            #AS FOR "TOP", R01 = R10 & T01 = T10 IS VALID FOR LAYER BEING ADDED ONLY.

            #i.e. XBASE(I,J,L) is the effective reflectivity, transmission and emission
            #of the bottom L layers of the atmosphere (i.e. layers LTOT-L+1 to LTOT)
            #***********************************************************************

            JBASE = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,1))  #Source function
            RBASE = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,Scatter.NMU))  #Reflection matrix
            TBASE = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,Scatter.NMU))  #Transmission matrix            

            #Filling the first value with the surface or lowest layer
            JBASE[:,:,0,:,:] = JL[:,:,0,IC,:,:]
            RBASE[:,:,0,:,:] = RL[:,:,0,IC,:,:]
            TBASE[:,:,0,:,:] = TL[:,:,0,IC,:,:]

            #Combining the adjacent layers
            for ILAY in range(LTOT-1):

                #In the Fortran version of NEMESIS the layers are defined from top to 
                #bottom while here they are from bottom to top, therefore the indexing
                #in this part of the code differs with respect to the Fortran version

                for iwave in range(Spectroscopy.NWAVE):
                    for ig in range(NG):
                        RBASE[iwave,ig,ILAY+1,:,:],TBASE[iwave,ig,ILAY+1,:,:],JBASE[iwave,ig,ILAY+1,:,:] = nemesisf.mulscatter.addp_layer(\
                            E,RL[iwave,ig,ILAY+1,IC,:,:],TL[iwave,ig,ILAY+1,IC,:,:],JL[iwave,ig,ILAY+1,IC,:,:],ISCL[iwave,ig,ILAY+1],RBASE[iwave,ig,ILAY,:,:],TBASE[iwave,ig,ILAY,:,:],JBASE[iwave,ig,ILAY,:,:])

            if IC!=0:
                JBASE[:,:,:,:,:] = 0.0

            #***********************************************************************
            #CALCULATE DOWNWARD MATRICES FOR COMPOSITE OF L LAYERS FROM TOP OF CLOUD.
            #XTOP(I,J,L) IS THE X MATRIX FOR THE TOP L LAYERS OF CLOUD.
            #NOTE THAT R21 = R12 & T21 = T12 VALID FOR THE HOMOGENEOUS LAYER BEING ADD$
            #BUT NOT FOR THE INHOMOGENEOUS RESULTING "TOP" LAYER

            #i.e. XTOP(I,J,L) is the effective reflectivity, transmission and emission
            #of the top L layers of the atmosphere (i.e. layers 1-L)

            #Specifically
            #RTOP(I,J,L) is RL0
            #TTOP(I,J,L) is T0L
            #JTOP(J,1,L) is JP0L
            #***********************************************************************
  
            JTOP = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,1))  #Source function
            RTOP = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,Scatter.NMU))  #Reflection matrix
            TTOP = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,Scatter.NMU))  #Transmission matrix            

            #Filling the first value with the surface 
            JTOP[:,:,0,:,:] = JL[:,:,LTOT-1,IC,:,:]
            RTOP[:,:,0,:,:] = RL[:,:,LTOT-1,IC,:,:]
            TTOP[:,:,0,:,:] = TL[:,:,LTOT-1,IC,:,:]

            #Combining the adjacent layers
            for ILAY in range(LTOT-1):

                #In the Fortran version of NEMESIS the layers are defined from top to 
                #bottom while here they are from bottom to top, therefore the indexing
                #in this part of the code differs with respect to the Fortran version

                for iwave in range(Spectroscopy.NWAVE):
                    for ig in range(NG):
                        RTOP[iwave,ig,ILAY+1,:,:],TTOP[iwave,ig,ILAY+1,:,:],JTOP[iwave,ig,ILAY+1,:,:] = nemesisf.mulscatter.addp_layer(\
                            E,RL[iwave,ig,LTOT-2-ILAY,IC,:,:],TL[iwave,ig,LTOT-2-ILAY,IC,:,:],JL[iwave,ig,LTOT-2-ILAY,IC,:,:],ISCL[iwave,ig,LTOT-2-ILAY],RTOP[iwave,ig,ILAY,:,:],TTOP[iwave,ig,ILAY,:,:],JTOP[iwave,ig,ILAY,:,:])

            if IC!=0:
                JTOP[:,:,:,:,:] = 0.0

            #Calculating the observing angles
            if Scatter.SOL_ANG>90.0:
                ZMU0 = np.cos((180.0 - Scatter.SOL_ANG)*np.pi/180.0)
                SOLAR[:] = 0.0
            else:
                ZMU0 = np.cos(Scatter.SOL_ANG*np.pi/180.0)


            #Finding the coefficients for interpolating the spectrum
            #at the correct angles
            ISOL = 0
            for j in range(Scatter.NMU-1):
                if((ZMU0<=Scatter.MU[j]) & (ZMU0>Scatter.MU[j+1])):
                    ISOL = j

            if ZMU0<=Scatter.MU[Scatter.NMU-1]:
                ISOL = Scatter.NMU-2

            FSOL = (Scatter.MU[ISOL]-ZMU0)/(Scatter.MU[ISOL]-Scatter.MU[ISOL+1])


            #Bottom of the atmosphere surface contribution
            UTMI = np.zeros((Spectroscopy.NWAVE,Scatter.NMU,1))
            if IC==0:
                for imu in range(Scatter.NMU):
                    UTMI[:,imu,0] = RADGROUND[:]   #Assumed to be equal in all directions
            UTMI = np.repeat(UTMI[:,np.newaxis,:,:],NG,axis=1)

            
            #Calculating the spectrum in the direction ISOL
            for IMU0 in range(ISOL,ISOL+2,1):

                #Top of the atmosphere solar contribution
                U0PL = np.zeros((Spectroscopy.NWAVE,Scatter.NMU,1))
                U0PL[:,IMU0,0] = SOLAR[:]/(2.0*np.pi*Scatter.WTMU[IMU0])
                U0PL = np.repeat(U0PL[:,np.newaxis,:,:],NG,axis=1)

                #Calculating the interior intensities for cloud (within layers)
                #UPL goes down of layer L
                #UMI goes up of layer L

                UMI = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,1))
                UPL = np.zeros((Spectroscopy.NWAVE,NG,LTOT,Scatter.NMU,1))

                UMI[:,:,0,:,:] = JBASE[:,:,0,:,:]   #Upwards intensity of surface is already calculated

                for ILAY in range(LTOT-1):

                    #Calculate I(ILAY+1)-
                    UMI[:,:,ILAY+1,:,:] = nemesisf.mulscatter.iup(\
                        E,U0PL,UTMI,\
                        RTOP[:,:,ILAY,:,:],TTOP[:,:,ILAY,:,:],JTOP[:,:,ILAY,:,:],\
                        RBASE[:,:,LTOT-2-ILAY,:,:],TBASE[:,:,LTOT-2-ILAY,:,:],JBASE[:,:,LTOT-2-ILAY,:,:])

                    #Calculate I(ILAY)+
                    UPL[:,:,ILAY,:,:] = nemesisf.mulscatter.idown(\
                        E,U0PL,UTMI,\
                        RTOP[:,:,ILAY,:,:],TTOP[:,:,ILAY,:,:],JTOP[:,:,ILAY,:,:],\
                        RBASE[:,:,LTOT-2-ILAY,:,:],TBASE[:,:,LTOT-2-ILAY,:,:],JBASE[:,:,LTOT-2-ILAY,:,:])

                #Calculating the exterior intensities (upward intensity at top of atmosphere and downward at bottom of atmosphere)
                U0MI = nemesisf.mulscatter.itop(U0PL,UTMI,RBASE[:,:,LTOT-1,:,:],TBASE[:,:,LTOT-1,:,:],JBASE[:,:,LTOT-1,:,:])
                UTPL = nemesisf.mulscatter.ibottom(U0PL,UTMI,RTOP[:,:,LT1-1,:,:],TTOP[:,:,LT1-1,:,:],JTOP[:,:,LT1-1,:,:])

                #Calculating the radiance in each viewing angle
                for IMU in range(Scatter.NMU):

                    JMU  = Scatter.NMU-1-IMU  #Using this since the MU angles were reversed at the beginning of this subroutine

                    #Uplf = np.zeros((Spectroscopy.NWAVE,NG,Scatter.NMU,Layer.NLAY,Scatter.NF+1))
                    if IMU0==ISOL:
                        Umif[:,:,JMU,0,IC] = (1.0-FSOL)*(U0MI[:,:,IMU,0])
                    else:
                        Umif[:,:,JMU,0,IC] = Umif[:,:,JMU,0,IC] + FSOL*U0MI[:,:,IMU,0]

                    for ILAY in range(LT1): #Going through each atmospheric layer
                        
                        if IMU0==ISOL:
                            if ILAY!=LT1-1:
                                Uplf[:,:,JMU,ILAY,IC] = (1.0-FSOL)*UPL[:,:,ILAY,IMU,0]
                            if ILAY!=0:
                                Umif[:,:,JMU,ILAY,IC] = (1.0-FSOL)*UMI[:,:,ILAY,IMU,0]
                        else:
                            if ILAY!=LT1-1:
                                Uplf[:,:,JMU,ILAY,IC] = Uplf[:,:,JMU,ILAY,IC] + (FSOL)*UPL[:,:,ILAY,IMU,0]
                            if ILAY!=0:
                                Umif[:,:,JMU,ILAY,IC] = Umif[:,:,JMU,ILAY,IC] + (FSOL)*UMI[:,:,ILAY,IMU,0]


                    if IMU0==ISOL:
                        Uplf[:,:,JMU,LT1-1,IC] = (1.0-FSOL)*(UTPL[:,:,IMU,0])
                    else:
                        Uplf[:,:,JMU,LT1-1,IC] = Uplf[:,:,JMU,LT1-1,IC] + (FSOL)*(UTPL[:,:,IMU,0])


                U0PL[:,:,IMU0,0] = 0.0

        #The order of the layers in Uplf and Umif goes from top to bottom.
        #In order to reconcile it with the order of the layers in the Layer class, we reverse them
        Uplfout = Uplf[:,:,:,::-1,:]
        Umifout = Umif[:,:,:,::-1,:]

        #Reset the order of angles
        Scatter.MU = Scatter.MU[::-1]
        Scatter.WTMU = Scatter.WTMU[::-1]

        return Umifout,Uplfout

    ###############################################################################################
    def streamflux(self,NLAY,NMU,MU,WTMU,Umif,Uplf):
        r"""
        Subroutine to calculate the upward and downward flux in the boundaries of each layer.

        The output of scloud11flux is the radiance observed in each viewing direction in the
        boundaries of each layer (both upwards and downwards).

        This function integrates the radiance over solar zenith angle to get the total 
        upward and downward fluxes at the boundaries of each layer.

        Inputs
        ------

        NLAY :: Number of atmospheric layers
        NMU :: Number of zenith quadrature angles
        MU(NMU) :: Zenith quadrature angles
        WTMU(NMU) :: Zenith angle quadrature weights
        Umif(NWAVE,NG,NMU,NLAY,NF+1) :: Upward intensity in the boundaries of each layer (from bottom to top)
        Uplf(NWAVE,NG,NMU,NLAY,NF+1) :: Downward intensity in the boundaries of each layer (from bottom to top)

        Outputs
        -------

        fup(NWAVE,NG,NLAY) :: Upward flux from the top of each layer
        fdown(NWAVE,NG,NLAY) :: Downward flux from the bottom of each layer

                                        /\
                                        || Flux_UP
                                        ||
                                --------------------------------
                                                Layer
                                --------------------------------
                                        ||
                                        || Flux_DOWN
                                        \/

        """

        NWAVE = Umif.shape[0]
        NG = Umif.shape[1]
        NF = Umif.shape[4]-1

        #Calculating any quadrature error
        xfac = 0.
        for i in range(NMU):
            xfac = xfac + MU[i]*WTMU[i]
        xnorm = np.pi/xfac   #XFAC should be 0.5 so this is 2pi

        
        #Integrating over zenith angle
        fdown = np.zeros((NWAVE,NG,NLAY))
        fup = np.zeros((NWAVE,NG,NLAY))
        for i in range(NMU):
            fdown[:,:,:] = fdown[:,:,:] + MU[i]*WTMU[i]*Uplf[:,:,i,:,0]
            fup[:,:,:] = fup[:,:,:] + MU[i]*WTMU[i]*Umif[:,:,i,:,0]

        fdown = fdown*xnorm
        fup = fup*xnorm

        return fup,fdown


    ###############################################################################################
    def calc_phase_matrix_v2(self,Scatter,WAVE,normalise=True):
        """

        Calculate the phase matrix at the different angles required for the multiple
        scattering calculations. These are the P++ and P+- matrices in Plass et al. (1973).
 
        Inputs
        ________

        Scatter :: Python class defining the scattering setup
        WAVE(NWAVE) :: Calculation wavelengths

        Outputs
        ________

        PPLPL(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the + direction (i.e. downward)
        PPLMI(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the - direction (i.e. upward)

        """

        #Calculating the phase function at the scattering angles
        #######################################################################

        NWAVE = len(WAVE)
        dphi = 2.0*np.pi/Scatter.NPHI

        #Defining the angles at which the phase functions must be calculated
        cpl = np.zeros(Scatter.NMU*Scatter.NMU*(Scatter.NPHI+1))
        cmi = np.zeros(Scatter.NMU*Scatter.NMU*(Scatter.NPHI+1))
        ix = 0

        for j in range(Scatter.NMU):
            for i in range(Scatter.NMU):
                sthi = np.sqrt(1.0-Scatter.MU[i]*Scatter.MU[i])   #sin(theta(i))
                sthj = np.sqrt(1.0-Scatter.MU[j]*Scatter.MU[j])   #sin(theta(j))

                for k in range(Scatter.NPHI+1):
                    phi = k*dphi
                    cpl[ix] = sthi*sthj*np.cos(phi) + Scatter.MU[i]*Scatter.MU[j]
                    cmi[ix] = sthi*sthj*np.cos(phi) - Scatter.MU[i]*Scatter.MU[j]
                    ix = ix + 1

        #Calculating the phase function at the required wavelengths and scattering angles
        cpl[np.where(cpl>1.0)]=1.0
        cmi[np.where(cmi>1.0)]=1.0
        cpl[np.where(cpl<-1.0)]=-1.0
        cmi[np.where(cmi<-1.0)]=-1.0
        apl = np.arccos(cpl) / np.pi * 180.
        ami = np.arccos(cmi) / np.pi * 180.
        
        
        ppl = Scatter.calc_phase(apl,WAVE)  #(NWAVE,NTHETA,NDUST)
        pmi = Scatter.calc_phase(ami,WAVE)  #(NWAVE,NTHETA,NDUST)

        #Normalising phase function (OLD, THE PHASE FUNCTION FROM SCATTER IS NORMALISED TO 1)
        #ppl = ppl / (4.0*np.pi)
        #pmi = pmi / (4.0*np.pi)

        if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
            ncont = Scatter.NDUST + 1
            pplr = Scatter.calc_phase_ray(apl) #(NTHETA)
            pmir = Scatter.calc_phase_ray(ami) #(NTHETA)

            #Normalising phase function (OLD, THE PHASE FUNCTION FROM SCATTER IS NORMALISED TO 1)
            #pplr = pplr / (4.0*np.pi)
            #pmir = pmir / (4.0*np.pi)
        else:
            ncont = Scatter.NDUST


        #Integrating the phase function over the azimuth direction
        #####################################################################################

        PPLPL = np.zeros((NWAVE,ncont,Scatter.NF+1,Scatter.NMU,Scatter.NMU)) #Integrated phase function coefficients in + direction (i.e. downwards)
        PPLMI = np.zeros((NWAVE,ncont,Scatter.NF+1,Scatter.NMU,Scatter.NMU)) #Integrated phase function coefficients in - direction (i.e. upwards)
        ix = 0
        for j in range(Scatter.NMU):
            for i in range(Scatter.NMU):
                for k in range(Scatter.NPHI+1):
                    phi = k*dphi
                    for kl in range(Scatter.NF+1):

                        plx = ppl[:,ix,:] * np.cos(kl*phi)
                        pmx = pmi[:,ix,:] * np.cos(kl*phi)

                        wphi = 1.0*dphi
                        if k==0:
                            wphi = 0.5*dphi
                        elif k==Scatter.NPHI:
                            wphi = 0.5*dphi

                        #_lgr.info(wphi,plx.min())

                        if kl==0:
                            wphi = wphi/(2.0*np.pi)
                        else:
                            wphi = wphi/np.pi

                        PPLPL[:,0:Scatter.NDUST,kl,i,j] = PPLPL[:,0:Scatter.NDUST,kl,i,j] + wphi*plx[:,:]
                        PPLMI[:,0:Scatter.NDUST,kl,i,j] = PPLMI[:,0:Scatter.NDUST,kl,i,j] + wphi*pmx[:,:]

                        if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
                            plrx = pplr[ix] * np.cos(kl*phi)
                            pmrx = pmir[ix] * np.cos(kl*phi)
                            PPLPL[:,Scatter.NDUST,kl,i,j] = PPLPL[:,Scatter.NDUST,kl,i,j] + wphi*plrx
                            PPLMI[:,Scatter.NDUST,kl,i,j] = PPLMI[:,Scatter.NDUST,kl,i,j] + wphi*pmrx

                    ix = ix + 1

        
        if normalise==True:
        
            #Normalising the phase matrices using the method of Hansen (1971,J.ATM.SCI., V28, 1400)
            ###############################################################################################

            #PPL,PMI ARE THE FORWARD AND BACKWARD PARTS OF THE AZIMUTHALLY-INTEGRATED
            #PHASE FUNCTION.  THE NORMALIZATION OF THE TRUE PHASE FCN. IS:
            #integral over sphere [ P(mu,mu',phi) * dO] = 1
            #WHERE dO IS THE ELEMENT OF SOLID ANGLE AND phi IS THE AZIMUTHAL ANGLE.
    
            IC = 0

            RSUM = np.zeros((NWAVE,ncont,Scatter.NMU))
            for j in range(Scatter.NMU):
                for i in range(Scatter.NMU):
                    RSUM[:,:,j] = RSUM[:,:,j] + PPLMI[:,:,IC,i,j] * Scatter.WTMU[i]*2.0*np.pi

            for icont in range(ncont):
                FC = np.ones((NWAVE,Scatter.NMU,Scatter.NMU))
                niter = 1
                converged = False
                while converged==False:
                    test = np.zeros(NWAVE)
                    TSUM = np.zeros((NWAVE,Scatter.NMU))
                    for j in range(Scatter.NMU):
                        for i in range(Scatter.NMU):
                            TSUM[:,j] = TSUM[:,j] + PPLPL[:,icont,IC,i,j]*Scatter.WTMU[i]*FC[:,i,j]*2.0*np.pi

                        testj = np.abs(  RSUM[:,icont,j] + TSUM[:,j] - 1.0 )
                        isup = np.where(testj>test)[0]
                        test[isup] = testj[isup]

                    #_lgr.info(test)
                    if test.max()<=1.0e-14:
                        converged=True
                    else:
                        for j in range(Scatter.NMU):
                            xj = (1.0-RSUM[:,icont,j])/TSUM[:,j]
                            for i in range(j+1):
                                xi = (1.0-RSUM[:,icont,i])/TSUM[:,i]
                                FC[:,i,j] = 0.5*(FC[:,i,j]*xj[:]+FC[:,j,i]*xi[:])
                                FC[:,j,i] = FC[:,i,j]

                    if niter>10000:
                        raise ValueError('error in calc_phase_matrix :: Normalisation of phase matrix did not converge')

                    niter = niter + 1

                for kl in range(Scatter.NF+1):
                    for j in range(Scatter.NMU):
                        for i in range(Scatter.NMU):
                            PPLPL[:,icont,kl,i,j] = PPLPL[:,icont,kl,i,j] * FC[:,i,j]
        

        return PPLPL,PPLMI

    ###############################################################################################
    def calc_phase_matrix(self,Scatter,WAVE):
        """

        Calculate the phase matrix at the different angles required for the multiple
        scattering calculations. These are the P++ and P+- matrices in Plass et al. (1973).
 
        Inputs
        ________

        Scatter :: Python class defining the scattering setup
        WAVE(NWAVE) :: Calculation wavelengths

        Outputs
        ________

        PPLPL(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the + direction (i.e. downward)
        PPLMI(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the - direction (i.e. upward)

        """

        #from NemesisPy import nemesisf

        #Calculating the phase function at the scattering angles
        #######################################################################

        NWAVE = len(WAVE)
        dphi = 2.0*np.pi/Scatter.NPHI

        
        #Defining the angles at which the phase functions must be calculated
        apl,ami = nemesisf.mulscatter.define_scattering_angles(nmu=Scatter.NMU,nphi=Scatter.NPHI,mu=Scatter.MU)
        
        #Calculating the phase function at the scattering angles
        ppl = Scatter.calc_phase(apl,WAVE)  #(NWAVE,NTHETA,NDUST)
        pmi = Scatter.calc_phase(ami,WAVE)  #(NWAVE,NTHETA,NDUST)

        #Normalising phase function (OLD, THE PHASE FUNCTION IN SCATTER IS ALREADY NORMALISED TO 1)
        #ppl = ppl / (4.0*np.pi)
        #pmi = pmi / (4.0*np.pi)

        #Calculating the phase function for Rayleigh scattering
        if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
            ncont = Scatter.NDUST + 1
            pplr = Scatter.calc_phase_ray(apl) #(NTHETA)
            pmir = Scatter.calc_phase_ray(ami) #(NTHETA)

            #(OLD, THE PHASE FUNCTION IN SCATTER IS ALREADY NORMALISED TO 1)
            #pplr = pplr / (4.0*np.pi)
            #pmir = pmir / (4.0*np.pi)
        else:
            ncont = Scatter.NDUST
         
        #Comparison with NEMESIS (fortran)   
        #ix = 0
        #for j in range(Scatter.NMU):
        #    for i in range(Scatter.NMU):
        #        for k in range(Scatter.NPHI+1):
        #            _lgr.info(j,i,k,np.cos(apl[ix]/180.*np.pi),ppl[0,ix,0],np.cos(ami[ix]/180.*np.pi),pmi[0,ix,0])
        #            input()
        #            ix = ix + 1
                    


        #Integrating the phase function over the azimuth direction
        #####################################################################################

        PPLPL = np.zeros((NWAVE,ncont,Scatter.NF+1,Scatter.NMU,Scatter.NMU)) #Integrated phase function coefficients in + direction (i.e. downwards)
        PPLMI = np.zeros((NWAVE,ncont,Scatter.NF+1,Scatter.NMU,Scatter.NMU)) #Integrated phase function coefficients in - direction (i.e. upwards)

        #Aerosol phase functions
        for icont in range(Scatter.NDUST):
            PPLPL[:,icont,:,:,:],PPLMI[:,icont,:,:,:] = nemesisf.mulscatter.integrate_phase_function(nwave=NWAVE,nmu=Scatter.NMU,nphi=Scatter.NPHI,nf=Scatter.NF,ppl=ppl[:,:,icont],pmi=pmi[:,:,icont])

        #To be compare with calc_pmat6 without hansen normalisation
        #for icont in range(Scatter.NDUST):
        #    for j in range(Scatter.NMU):
        #        for i in range(Scatter.NMU):
        #            _lgr.info(icont,i,j,'PTPL',PPLPL[0,icont,0,i,j],'PPLMI',PPLMI[0,icont,0,i,j])
        #    input()

        #Rayleigh phase function
        if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
            pplrx = np.repeat(pplr[np.newaxis,:],NWAVE,axis=0)
            pmirx = np.repeat(pmir[np.newaxis,:],NWAVE,axis=0)
            PPLPL[:,Scatter.NDUST,:,:,:],PPLMI[:,Scatter.NDUST,:,:,:] = nemesisf.mulscatter.integrate_phase_function(nwave=NWAVE,nmu=Scatter.NMU,nphi=Scatter.NPHI,nf=Scatter.NF,ppl=pplrx[:,:],pmi=pmirx[:,:])

        #Normalising the phase matrices using the method of Hansen (1971,J.ATM.SCI., V28, 1400)
        ###############################################################################################

        #PPL,PMI ARE THE FORWARD AND BACKWARD PARTS OF THE AZIMUTHALLY-INTEGRATED
        #PHASE FUNCTION.  THE NORMALIZATION OF THE TRUE PHASE FCN. IS:
        #integral over sphere [ P(mu,mu',phi) * dO] = 1
        #WHERE dO IS THE ELEMENT OF SOLID ANGLE AND phi IS THE AZIMUTHAL ANGLE.
        for icont in range(ncont):
            PPLPL[:,icont,:,:,:],PPLMI[:,icont,:,:,:] = nemesisf.mulscatter.normalise_phase_function(nwave=NWAVE,nmu=Scatter.NMU,nf=Scatter.NF,wtmu=Scatter.WTMU,pplpl=PPLPL[:,icont,:,:,:],pplmi=PPLMI[:,icont,:,:,:])
        
        #To be compare with calc_pmat6 after hansen normalisation
        #for icont in range(Scatter.NDUST):
        #    for j in range(Scatter.NMU):
        #        for i in range(Scatter.NMU):
        #            _lgr.info(icont,i,j,'PTPL',PPLPL[0,icont,0,i,j],'PPLMI',PPLMI[0,icont,0,i,j])
        #    input()

        return PPLPL,PPLMI

    ###############################################################################################
    def calc_layer_scatt_matrix(self,WAVE,Layer):
        """

        Calculate the effective scattering matrix (phase matrix and single scattering albedo) of
        an atmospheric layer composed of different aerosol and gaseous species (including Rayleigh scattering)
 
        Inputs
        ________

        WAVE(NWAVE) :: Calculation wavelengths
        Scatter :: Python class defining the atmospheric layer properties (including their opacities)

        Outputs
        ________

        PPLPL(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the + direction (i.e. downward)
        PPLMI(NWAVE,NDUST+1,NF,NMU,NMU) :: Phase matrix (aerosol and Rayleigh) in the - direction (i.e. upward)

        """
        
        NWAVE = len(WAVE)
        
        #Calculating the fraction of scattering by each aerosol type and rayleigh
        FRAC = np.zeros((Spectroscopy.NWAVE,Layer.NLAY,NCONT))
        iiscat = np.where((Layer.TAUSCAT+Layer.TAURAY)>0.0)
        if(len(iiscat[0])>0):
            FRAC[iiscat[0],iiscat[1],0:Scatter.NDUST] = np.transpose(np.transpose(Layer.TAUCLSCAT[iiscat[0],iiscat[1],:],axes=[1,0]) / ((Layer.TAUSCAT[iiscat[0],iiscat[1]]+Layer.TAURAY[iiscat[0],iiscat[1]])),axes=[1,0])  #Fraction of each aerosol scattering FRAC = TAUCLSCAT/(TAUSCAT+TAURAY)
            if Scatter.IRAY != RayleighScatteringMode.NOT_INCLUDED:
                FRAC[iiscat[0],iiscat[1],Scatter.NDUST] = Layer.TAURAY[iiscat[0],iiscat[1]] / ((Layer.TAUSCAT[iiscat[0],iiscat[1]]+Layer.TAURAY[iiscat[0],iiscat[1]])) #Fraction of Rayleigh scattering FRAC = TAURAY/(TAUSCAT+TAURAY)

        #Calculating the weighted averaged phase matrix in each layer and direction
        _lgr.info('scloud11flux :: Calculating weighted average phase matrix in each layer')
        PPLPLS = np.zeros((Spectroscopy.NWAVE,Layer.NLAY,Scatter.NF+1,Scatter.NMU,Scatter.NMU))
        PPLMIS = np.zeros((Spectroscopy.NWAVE,Layer.NLAY,Scatter.NF+1,Scatter.NMU,Scatter.NMU))

        for ilay in range(Layer.NLAY):
            PPLPLS[:,ilay,:,:,:] =  np.transpose(np.sum(np.transpose(PPLPL,axes=[2,3,4,0,1])*FRAC[:,ilay,:],axis=4),axes=[3,0,1,2])  #SUM(PPLPL*FRAC)
            PPLMIS[:,ilay,:,:,:] =  np.transpose(np.sum(np.transpose(PPLMI,axes=[2,3,4,0,1])*FRAC[:,ilay,:],axis=4),axes=[3,0,1,2])  #SUM(PPLMI*FRAC)

        #Calculating the single scattering albedo of each layer (TAURAY+TAUSCAT/TAUTOT)
        NG = Layer.TAUTOT.shape[1]
        OMEGA = np.zeros((Spectroscopy.NWAVE,NG,Layer.NLAY))
        iin = np.where(Layer.TAUTOT>0.0)
        if(len(iin[0])>0):
            OMEGA[iin[0],iin[1],iin[2]] = (Layer.TAURAY[iin[0],iin[2]]+Layer.TAUSCAT[iin[0],iin[2]]) / Layer.TAUTOT[iin[0],iin[1],iin[2]]

        if diffuse==False:
            OMEGA[:,:,:] = 0.0  #No scattering if diffuse component is turned off

    ###############################################################################################
    def calc_brdf_matrix(self,WAVEC=None,Scatter=None,Surface=None):
        """
        Calculate the Bidirectional Reflectance Distribution Function (BRDF) of the surface
        in the matrix form required by the multiple scattering doubling method
 
        Inputs
        ________

        Scatter :: Python class defining the scattering setup
        Surface :: Python class defining the Surface
        WAVE(NWAVE) :: Calculation wavelengths

        Outputs
        ________

        Reflectivity(NWAVE,NMU,NMU,NF+1) :: Surface BRDF matrix
        """

       #Initialising variables
        if WAVEC is None:
            WAVEC = self.SpectroscopyX.WAVE
        if Scatter is None:
            Scatter = self.ScatterX
        if Surface is None:
            Surface = self.SurfaceX

        #Calculating the bidirectional reflectance at the required angles
        #######################################################################

        NWAVE = len(WAVEC)
        dphi = 2.0*np.pi/Scatter.NPHI
        
        #Reversing the quadrature angles (that's how it is done in the doubling method)
        mu = np.zeros(Scatter.NMU)
        mu[:] = Scatter.MU[::-1]
        
        BRDF_mat = np.zeros((NWAVE,Scatter.NMU,Scatter.NMU,Scatter.NF+1))
        if Surface.LOWBC == LowerBoundaryCondition.LAMBERTIAN: #Lambertian reflection (isotropic)
            
            #We only need to calculate the BRDF in one angle since it will be the same everywhere
            BRDFx = Surface.calc_BRDF(WAVEC,[0.],[0.],[0.])[:,0] #(NWAVE)
            BRDF_mat[:,:,:,0] = BRDFx[:,None,None]

        elif Surface.LOWBC in (LowerBoundaryCondition.HAPKE,):  #Anisotropic reflection

            # Create indices for broadcasting
            j, i, k = np.meshgrid(np.arange(Scatter.NMU), np.arange(Scatter.NMU), np.arange(Scatter.NPHI+1), indexing="ij")

            # Compute angles
            EMISS_ANG = (np.arccos(mu[i]) * 180.0 / np.pi).ravel()
            SOL_ANG = (np.arccos(mu[j]) * 180.0 / np.pi).ravel()
            AZI_ANG = ((k * dphi) * 180.0 / np.pi).ravel()

            BRDF = Surface.calc_BRDF(WAVEC,SOL_ANG,EMISS_ANG,AZI_ANG) #(NWAVE,NTHETA)

            #Integrating the reflectance over the azimuth direction
            #####################################################################################
        
            ix = 0
            for j in range(Scatter.NMU):   #SOL_ANG
                for i in range(Scatter.NMU):  #EMISS_ANG
                    for k in range(Scatter.NPHI+1):  #AZI_ANG
                        phi = k*dphi
                        for ic in range(Scatter.NF+1):

                            wphi = 1.0*dphi
                            if k==0:
                                wphi = 0.5*dphi
                            elif k==Scatter.NPHI:
                                wphi = 0.5*dphi

                            #if ic==0:
                            #    wphi = wphi/(2.0*np.pi)
                            #else:
                            #    wphi = wphi/np.pi
                            wphi = wphi/(2.0*np.pi)

                            BRDF_mat[:,i,j,ic] += wphi * BRDF[:,ix] * np.cos(ic*phi)

                        ix += 1

        return BRDF_mat


    ###############################################################################################
    ###############################################################################################
    # ERROR CHECKS
    ###############################################################################################
    ###############################################################################################

    ###############################################################################################

    def check_gas_spec_atm(self):
        """
        Check whether the gases in the Spectroscopy class are present in the Atmosphere
        """
        
        for icase in range(self.SpectroscopyX.NGAS):
            gasID = self.SpectroscopyX.ID[icase]
            isoID = self.SpectroscopyX.ISO[icase]
            
            if not any(id_val == gasID and iso_val == isoID for id_val, iso_val in zip(self.AtmosphereX.ID, self.AtmosphereX.ISO)):
                from archnemesis.Data import gas_info
                known_gas_ids = ', '.join([f"({gid},{isoid}) [{gas_info[str(gid)]['name']}(iso:{isoid})]" for gid, isoid in zip(self.AtmosphereX.ID,self.AtmosphereX.ISO)])
                msg = f'Atmosphere has been defined with the following (gasID,isoID) pairs: {known_gas_ids}'
                raise ValueError(f"error in check_gas_spec_atm :: No match found for gasID={gasID} and isoID={isoID} [{gas_info[str(gasID)]['name']}(iso:{isoID})] from Spectroscopy in Atmosphere. {msg}")

    def check_wave_range_consistency(self,rel_tolerance=1.0e-6):
        """
        Check whether the wavelength range at which the different classes are listed covers well the calculation wavelengths
        """
        
        wavecalc_min,wavecalc_max = self.MeasurementX.calc_wave_range(IGEOM=None,apply_doppler=True)
        
        # Retrieve min and max wavelengths for calculations
        wavespec_min = self.SpectroscopyX.WAVE.min()
        wavespec_max = self.SpectroscopyX.WAVE.max()
        
        # Retrieve min and max wavelengths for aerosols (scatter)
        wavetau_min = self.ScatterX.WAVE.min()
        wavetau_max = self.ScatterX.WAVE.max()
        
        # Apply tolerance check
        if (wavespec_min > (1. + rel_tolerance) * wavecalc_min) or (wavespec_max < (1. - rel_tolerance) * wavecalc_max):
            raise ValueError(
                        f"Spectroscopy wavelength range [{wavespec_min}, {wavespec_max}] does not fully cover "
                        f"the calculation wavelength range [{wavecalc_min}, {wavecalc_max}] "
                        )
        
        # Apply tolerance check
        if (wavetau_min > (1. + rel_tolerance) * wavecalc_min) or (wavetau_max < (1. - rel_tolerance) * wavecalc_max):
            raise ValueError(
                        f"Aerosol wavelength range [{wavetau_min}, {wavetau_max}] does not fully cover "
                        f"the calculation wavelength range [{wavecalc_min}, {wavecalc_max}] "
                        )


#END OF FORWARD MODEL CLASS

###############################################################################################
###############################################################################################
                                 #EXTRA FUNCTIONS
###############################################################################################
###############################################################################################


###############################################################################################
#@jit(nopython=True)
def map2pro(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NLAYIN,LAYINC,DTE,DAM,DCO,INCPAR=[-1]):
    
    """
        FUNCTION NAME : map2pro()
        
        DESCRIPTION : This function maps the analytical gradients defined with respect to the Layers
                      onto the input atmospheric levels defined in Atmosphere
        
        INPUTS :
        
            dSPECIN(NWAVE,NVMR+2+NDUST,NLAYIN,NPATH) :: Rate of change of output spectrum with respect to layer
                                                         properties along the path
            NWAVE :: Number of spectral points
            NVMR :: Number of gases in reference atmosphere
            NDUST :: Number of aerosol populations in reference atmosphere
            NPRO :: Number of altitude points in reference atmosphere
            NPATH :: Number of atmospheric paths
            NLAYIN(NPATH) :: Number of layer in each of the paths
            LAYINC(NLAY,NPATH) :: Layers in each path
            DTE(NLAY,NPRO) :: Matrix relating the temperature in each layer to the temperature in the profiles
            DAM(NLAY,NPRO) :: Matrix relating the gas amounts in each layer to the gas VMR in the profiles
            DCO(NLAY,NPRO) :: Matrix relating the dust amounts in each layer to the dust abundance in the profiles
        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            dSPECOUT(NWAVE,NVMR+2+NDUST,NPRO,NPATH) :: Rate of change of output spectrum with respect to the
                                                        atmospheric profile parameters
        
        CALLING SEQUENCE:
        
            dSPECOUT = map2pro(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NLAYIN,LAYINC,DTE,DAM,DCO)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    DAMx = DAM[LAYINC,:] #NLAYIN,NPATH,NPRO
    DCOx = DCO[LAYINC,:]
    DTEx = DTE[LAYINC,:]

    dSPECOUT = np.zeros((NWAVE,NVMR+2+NDUST,NPRO,NPATH))

    if INCPAR[0]!=-1:
        NPARAM = len(INCPAR)
    else:
        NPARAM = NVMR+2+NDUST
        INCPAR = range(NPARAM)

    for ipath in range(NPATH):
        for iparam in range(NPARAM):

            if INCPAR[iparam]<=NVMR-1: #Gas gradients
                dSPECOUT1 = np.tensordot(dSPECIN[:,INCPAR[iparam],:,ipath], DAMx[:,ipath,:], axes=(1,0))
            elif INCPAR[iparam]<=NVMR: #Temperature gradients
                dSPECOUT1 = np.tensordot(dSPECIN[:,INCPAR[iparam],:,ipath], DTEx[:,ipath,:], axes=(1,0))
            elif( (INCPAR[iparam]>NVMR) & (INCPAR[iparam]<=NVMR+NDUST) ): #Dust gradient
                dSPECOUT1 = np.tensordot(dSPECIN[:,INCPAR[iparam],:,ipath], DCOx[:,ipath,:], axes=(1,0))
            elif INCPAR[iparam]==NVMR+NDUST+1: #ParaH gradient
                dSPECOUT[:,INCPAR[iparam],:,ipath] = 0.0  #Needs to be included

            dSPECOUT[:,INCPAR[iparam],:,ipath] = dSPECOUT1[:,:]

    return dSPECOUT

###############################################################################################
#@jit(nopython=True)
def map2xvec(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NX,xmap):
    
    """
        FUNCTION NAME : map2xvec()
        
        DESCRIPTION : This function maps the analytical gradients defined with respect to the Layers
                      onto the input atmospheric levels defined in Atmosphere
        
        INPUTS :
        
            dSPECIN(NWAVE,NVMR+2+NDUST,NPRO,NPATH) :: Rate of change of output spectrum with respect to profiles
            NWAVE :: Number of spectral points
            NVMR :: Number of gases in reference atmosphere
            NDUST :: Number of aerosol populations in reference atmosphere
            NPRO :: Number of altitude points in reference atmosphere
            NPATH :: Number of atmospheric paths
            NX :: Number of elements in state vector
            XMAP(NX,NVMR+2+NDUST,NPRO) :: Matrix relating the gradients in the profiles to the elemenents in state vector

        
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            dSPECOUT(NWAVE,NPATH,NX) :: Rate of change of output spectrum with respect to the elements in the state vector
        
        CALLING SEQUENCE:
        
            dSPECOUT = map2xvec(dSPECIN,NWAVE,NVMR,NDUST,NPRO,NPATH,NX,xmap)
        
        MODIFICATION HISTORY : Juan Alday (25/07/2021)
        
    """

    #Mapping the gradients to the elements in the state vector
    dSPECOUT = np.tensordot(dSPECIN, xmap, axes=([1,2],[1,2])) #NWAVE,NPATH,NX

    return dSPECOUT

###############################################################################################
@jit(nopython=True)
def add_layer(R1,T1,J1,R2,T2,J2):
    """

    Subroutine to add the diffuse reflection, transmission and reflection
    matrices for two adjacent atmospheric layers

    Note that this routine will also work if there are other dimensions
    on the left of each matrix (e.g. NWAVE, NLAY, etc.)

    Inputs
    ________

    R1(NMU,NMU) :: Diffuse reflection operator for 1st layer
    T1(NMU,NMU) :: Diffuse transmission operator for 1st layer
    J1(NMU,1) :: Diffuse source function for 1st layer
    R2(NMU,NMU) :: Diffuse reflection operator for 2nd layer
    T2(NMU,NMU) :: Diffuse transmission operator for 2nd layer
    J2(NMU,1) :: Diffuse source function for 2nd layer

    Outputs
    ________

    RANS(NMU,NMU) :: Combined diffuse reflection operator
    TANS(NMU,NMU) :: Combined diffuse transmission operator
    JANS(NMU,1) :: Combined diffuse source function

    """

    NMU = int(R1.shape[-1])
    E = np.identity(NMU)

    #BCOM = -np.matmul(R2,R1)
    BCOM = R2 @ R1
    BCOM = E + BCOM
    ACOM = np.linalg.inv(BCOM)
    BCOM = ACOM
    #CCOM = np.matmul(T1,BCOM)
    CCOM = T1 @ BCOM
    #RANS = np.matmul(CCOM,R2)
    RANS = CCOM @ R2
    #ACOM = np.matmul(RANS,T1)
    ACOM = RANS @ T1
    RANS = R1 + ACOM
    #TANS = np.matmul(CCOM,T2)
    TANS = CCOM @ T2
    #JCOM = np.matmul(R2,J1)
    JCOM = R2 @ J1
    JCOM = J2 + JCOM
    #JANS = np.matmul(CCOM,JCOM)
    JANS = CCOM @ JCOM
    JANS = J1 + JANS
    
    return RANS,TANS,JANS

###############################################################################################
#@jit(nopython=True,parallel=True)
def add_layer_jit(R1,T1,J1,R2,T2,J2):
    """

    Subroutine to add the diffuse reflection, transmission and reflection
    matrices for two adjacent atmospheric layers

    Note that this routine will also work if there are other dimensions
    on the left of each matrix (e.g. NWAVE, NLAY, etc.)

    Inputs
    ________

    R1(NMU,NMU) :: Diffuse reflection operator for 1st layer
    T1(NMU,NMU) :: Diffuse transmission operator for 1st layer
    J1(NMU,1) :: Diffuse source function for 1st layer
    R2(NMU,NMU) :: Diffuse reflection operator for 2nd layer
    T2(NMU,NMU) :: Diffuse transmission operator for 2nd layer
    J2(NMU,1) :: Diffuse source function for 2nd layer

    Outputs
    ________

    RANS(NMU,NMU) :: Combined diffuse reflection operator
    TANS(NMU,NMU) :: Combined diffuse transmission operator
    JANS(NMU,1) :: Combined diffuse source function

    """

    NMU = int(R1.shape[-1])
    E = np.identity(NMU)

    RANS = np.zeros(R1.shape,dtype='float64')
    TANS = np.zeros(T1.shape,dtype='float64')
    JANS = np.zeros(J1.shape,dtype='float64')

    ACOM = np.zeros((NMU,NMU),dtype='float64')
    BCOM = np.zeros((NMU,NMU),dtype='float64')
    CCOM = np.zeros((NMU,NMU),dtype='float64')
    JCOM = np.zeros((NMU,1),dtype='float64')

    for i in range(R1.shape[0]):
        for j in range(R1.shape[1]):

            BCOM[:,:] = -np.dot(R2[i,j,:,:],R1[i,j,:,:])
            BCOM[:,:] = E[:,:] + BCOM[:,:]
            ACOM[:,:] = np.linalg.inv(BCOM[:,:])
            BCOM[:,:] = ACOM[:,:]
            CCOM[:,:] = np.dot(T1[i,j,:,:],BCOM[:,:])
            RANS[i,j,:,:] = np.dot(CCOM,R2[i,j,:,:])
            ACOM[:,:] = np.dot(RANS[i,j,:,:],T1[i,j,:,:])
            RANS[i,j,:,:] = R1[i,j,:,:] + ACOM[:,:]
            TANS[i,j,:,:] = np.dot(CCOM[:,:],T2[i,j,:,:])
            JCOM[:,:] = np.dot(R2[i,j,:,:],J1[i,j,:,:])
            JCOM[:,:] = J2[i,j,:,:] + JCOM[:,:]
            JANS[i,j,:,:] = np.dot(CCOM[:,:],JCOM[:,:])
            JANS[i,j,:,:] = J1[i,j,:,:] + JANS[i,j,:,:]

    return RANS,TANS,JANS

###############################################################################################
@jit(nopython=True)
def double_layer(NN,R1,T1,J1):
    """

    Subroutine to double the 

    Inputs
    ________

    R1(NPOINTS,NF+1,NMU,NMU) :: Diffuse reflection operator for 1st layer
    T1(NPOINTS,NF+1,NMU,NMU) :: Diffuse transmission operator for 1st layer
    J1(NPOINTS,NF+1,NMU,NMU) :: Diffuse source function for 1st layer
    NN(NPOINTS) :: Number of times the layer needs to be doubled

    Outputs
    ________

    RANS(NMU,NMU) :: Combined diffuse reflection operator
    TANS(NMU,NMU) :: Combined diffuse transmission operator
    JANS(NMU,1) :: Combined diffuse source function

    """

    #Compute the R and T matrices for subsequent layers (doubling method)
    #**********************************************************************

    #Doing all points with equal NN simultaneously
    NNMAX = NN.max()

    NF = R1.shape[1]

    RFIN = np.zeros(R1.shape)
    TFIN = np.zeros(T1.shape)
    JFIN = np.zeros(J1.shape)

    for N in range(1,NNMAX+1,1):

        idouble = np.where(NN>=N)[0]

        #Doubling the layer
        #RNEXT,TNEXT,JNEXT = add_layer_jit(R1[idouble,:,:,:],T1[idouble,:,:,:],J1[idouble,:,:,:],R1[idouble,:,:,:],T1[idouble,:,:,:],J1[idouble,:,:,:])
        RNEXT = np.zeros(R1.shape)
        TNEXT = np.zeros(T1.shape)
        JNEXT = np.zeros(J1.shape)
        for i in range(len(idouble)):
            for ic in range(NF):
                RNEXT[idouble[i],ic,:,:],TNEXT[idouble[i],ic,:,:],JNEXT[idouble[i],ic,:,:] = add_layer(R1[idouble[i],ic,:,:],T1[idouble[i],ic,:,:],J1[idouble[i],ic,:,:],R1[idouble[i],ic,:,:],T1[idouble[i],ic,:,:],J1[idouble[i],ic,:,:])

        iseleq2 = np.where(NN[idouble]==N)[0]
        iseleq = np.where(NN==N)[0]

        if len(iseleq)>0:
            JFIN[iseleq,:,:,:] = JNEXT[iseleq,:,:,:]
            TFIN[iseleq,:,:,:] = TNEXT[iseleq,:,:,:]
            RFIN[iseleq,:,:,:] = RNEXT[iseleq,:,:,:]

        #Updating matrices for next iteration
        R1[:,:,:,:] = RNEXT[:,:,:,:]
        T1[:,:,:,:] = TNEXT[:,:,:,:]
        J1[:,:,:,:] = JNEXT[:,:,:,:]

    return RFIN,TFIN,JFIN

###############################################################################################
@jit(nopython=True,parallel=True)
def matmul(A, B):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

    return C

###############################################################################################
def calc_spectrum_location(iLOCATION,Atmosphere,Surface,Measurement,Scatter,Spectroscopy,CIA,Stellar,Variables,Layer):
    """

    Subroutine to calculate a forward model in a given location of the planet (as defined in Atmosphere and Surface)
    This function is made for being used in parallel. For normal use, please see calc_spectrum_location() 

    Inputs
    ________

    iLOCATION :: Integer indicating the location to be used in the Atmosphere and Surface classes
    Variables :: Python class defining the parameterisations and state vector
    Measurement :: Python class defining the measurements
    Atmosphere :: Python class defining the reference atmosphere
    Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
    Scatter :: Python class defining the parameters required for scattering calculations
    Stellar :: Python class defining the stellar spectrum
    Surface :: Python class defining the surface
    CIA :: Python class defining the Collision-Induced-Absorption cross-sections
    Layer :: Python class defining the layering scheme to be applied in the calculations 

    Outputs
    ________

    RANS(NMU,NMU) :: Combined diffuse reflection operator
    TANS(NMU,NMU) :: Combined diffuse transmission operator
    JANS(NMU,1) :: Combined diffuse source function

    """
    
    from copy import copy
    
    runname = 'dummy'
    FM = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    
    #Calculating a forward model for each LOCATION on the planet
    FM.MeasurementX = copy(FM.Measurement)
    #FM.Measurement = None
    FM.AtmosphereX = copy(FM.Atmosphere)
    #FM.Atmosphere = None
    FM.ScatterX = copy(FM.Scatter)
    FM.Scatter = None
    FM.StellarX = copy(FM.Stellar)
    FM.Stellar = None
    FM.SurfaceX = copy(FM.Surface)
    #FM.Surface = None
    FM.SpectroscopyX = copy(FM.Spectroscopy)
    FM.Spectroscopy = None
    FM.LayerX = copy(FM.Layer)
    FM.Layer = None
    FM.CIAX = copy(FM.CIA)
    FM.CIA = None
    flagh2p = False
        
    #Updating the forward model in all locations according to state vector
    _lgr.info('nemesisMAPfm :: calling subprofretg')
    xmap = FM.subprofretg()
    _lgr.info('nemesisMAPfm :: subprofretg is done')
        
    #Selecting only one measurement specific for the desired location
    isel = np.where((FM.MeasurementX.FLAT==FM.AtmosphereX.LATITUDE[iLOCATION]) & (FM.MeasurementX.FLON==FM.AtmosphereX.LONGITUDE[iLOCATION]))
    IGEOM = isel[0][0]
    IAV = isel[1][0]
    
    _lgr.info('nemesisMAPfm :: selecting measurement')
    FM.select_Measurement(IGEOM,IAV)
    FM.Measurement = None
        
    #Updating the required parameters based on the current geometry
    FM.ScatterX.SOL_ANG = FM.MeasurementX.SOL_ANG[0,0]
    FM.ScatterX.EMISS_ANG = FM.MeasurementX.EMISS_ANG[0,0]
    FM.ScatterX.AZI_ANG = FM.MeasurementX.AZI_ANG[0,0]
    
    #Selecting only one specific location in the Atmosphere and Surface
    _lgr.info('nemesisMAPfm :: selecting location on Atmosphere and Surface')
    FM.select_location(iLOCATION)
    FM.Atmosphere = None
    FM.Surface = None
    
    
    #Calculating the path for this particular measurement and location
    _lgr.info('nemesisMAPfm :: calculating path')
    FM.calc_path()
    
    #Calling CIRSrad to perform radiative transfer calculations
    _lgr.info('nemesisMAPfm :: calculating forward model')
    SPEC = FM.CIRSrad() #()
     
    return SPEC[:,0]

###############################################################################################
#@ray.remote
def calc_spectrum_location_parallel(iLOCATION,Atmosphere,Surface,Measurement,Scatter,Spectroscopy,CIA,Stellar,Variables,Layer):
    """

    Subroutine to calculate a forward model in a given location of the planet (as defined in Atmosphere and Surface)
    This function is made for being used in parallel. For normal use, please see calc_spectrum_location() 

    Inputs
    ________

    iLOCATION :: Integer indicating the location to be used in the Atmosphere and Surface classes
    Variables :: Python class defining the parameterisations and state vector
    Measurement :: Python class defining the measurements
    Atmosphere :: Python class defining the reference atmosphere
    Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
    Scatter :: Python class defining the parameters required for scattering calculations
    Stellar :: Python class defining the stellar spectrum
    Surface :: Python class defining the surface
    CIA :: Python class defining the Collision-Induced-Absorption cross-sections
    Layer :: Python class defining the layering scheme to be applied in the calculations 

    Outputs
    ________

    RANS(NMU,NMU) :: Combined diffuse reflection operator
    TANS(NMU,NMU) :: Combined diffuse transmission operator
    JANS(NMU,1) :: Combined diffuse source function

    """
    
    from copy import copy
    from archnemesis import ForwardModel_0
    
    runname = 'dummy'
    FM = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,Measurement=Measurement,Spectroscopy=Spectroscopy,Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    
    #Calculating a forward model for each LOCATION on the planet
    FM.MeasurementX = copy(FM.Measurement)
    #FM.Measurement = None
    FM.AtmosphereX = copy(FM.Atmosphere)
    #FM.Atmosphere = None
    FM.ScatterX = copy(FM.Scatter)
    FM.Scatter = None
    FM.StellarX = copy(FM.Stellar)
    FM.Stellar = None
    FM.SurfaceX = copy(FM.Surface)
    #FM.Surface = None
    FM.SpectroscopyX = copy(FM.Spectroscopy)
    FM.Spectroscopy = None
    FM.LayerX = copy(FM.Layer)
    FM.Layer = None
    FM.CIAX = copy(FM.CIA)
    FM.CIA = None
    flagh2p = False
    
    #Making some changes to make ray work
    FM.AtmosphereX.H = FM.AtmosphereX.H.copy()
    FM.AtmosphereX.VMR = FM.AtmosphereX.VMR.copy()
    FM.AtmosphereX.P = FM.AtmosphereX.P.copy()
        
    #Updating the forward model in all locations according to state vector
    xmap = FM.subprofretg()
        
    #Selecting only one measurement specific for the desired location
    isel = np.where((FM.MeasurementX.FLAT==FM.AtmosphereX.LATITUDE[iLOCATION]) & (FM.MeasurementX.FLON==FM.AtmosphereX.LONGITUDE[iLOCATION]))
    IGEOM = isel[0][0]
    IAV = isel[1][0]
        
    FM.select_Measurement(IGEOM,IAV)
    FM.Measurement = None
        
    #Updating the required parameters based on the current geometry
    FM.ScatterX.SOL_ANG = FM.MeasurementX.SOL_ANG[0,0]
    FM.ScatterX.EMISS_ANG = FM.MeasurementX.EMISS_ANG[0,0]
    FM.ScatterX.AZI_ANG = FM.MeasurementX.AZI_ANG[0,0]
    
    #Selecting only one specific location in the Atmosphere and Surface
    FM.select_location(iLOCATION)
    FM.Atmosphere = None
    FM.Surface = None
    
    #Calculating the path for this particular measurement and location
    FM.calc_path()
    
    #Calling CIRSrad to perform radiative transfer calculations
    SPEC = FM.CIRSrad() #()
     
    return SPEC[:,0]


###############################################################################################
###############################################################################################
                            #RAYLEIGH SCATTERING ROUTINES
###############################################################################################
###############################################################################################

@jit(nopython=True)
def calc_tau_rayleighj(ISPACE,WAVEC,TOTAM):
    """
    Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
    for Gas Giant atmospheres using data from Allen (1976) Astrophysical Quantities

    Inputs
    ________

    ISPACE :: Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
    WAVEC(NWAVE) :: Wavenumber (cm-1) or wavelength array (um)
    TOTAM(NLAY) :: Atmospheric column density in each layer (m-2)

    Outputs
    ________

    TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
    dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

    """

    AH2=13.58E-5
    BH2 = 7.52E-3
    AHe= 3.48E-5
    BHe = 2.30E-3
    fH2 = 0.864
    k = 1.37971e-23
    P0=1.01325e5
    T0=273.15
    
    NWAVE = len(WAVEC)
    NLAY = len(TOTAM)

    if ISPACE==WaveUnit.Wavenumber_cm:
        LAMBDA = 1./WAVEC * 1.0e-2  #Wavelength in metres
        x = 1.0/(LAMBDA*1.0e6)
    elif ISPACE==WaveUnit.Wavelength_um:
        LAMBDA = WAVEC * 1.0e-6 #Wavelength in metres
        x = 1.0/(LAMBDA*1.0e6)
    else:
        raise NotImplementedError(f'calc_tau_rayleigh :: Unknown value for ISPACE')

    nH2 = AH2*(1.0+BH2*x*x)
    nHe = AHe*(1.0+BHe*x*x)

    #calculate the Jupiter air's refractive index at STP (Actually n-1)
    nAir = fH2*nH2 + (1-fH2)*nHe

    #H2,He Seem pretty isotropic to me?...Hence delta = 0.
    #Penndorf (1957) quotes delta=0.0221 for H2 and 0.025 for He.
    #(From Amundsen's thesis. Amundsen assumes delta=0.02 for H2-He atmospheres
    delta = 0.0
    temp = 32*(np.pi**3.)*nAir**2.
    N0 = P0/(k*T0)

    x = N0*LAMBDA*LAMBDA
    faniso = (6.0+3.0*delta)/(6.0 - 7.0*delta)

    #Calculating the scattering cross sections in m2
    k_rayleighj = temp*faniso/(3.*(x**2)) #(NWAVE)

    #Calculating the Rayleigh opacities in each layer
    tau_ray = np.zeros((NWAVE,NLAY))
    dtau_ray = np.zeros((NWAVE,NLAY))
    for iwave in range(NWAVE):
        for ilay in range(NLAY):
            tau_ray[iwave,ilay] = k_rayleighj[iwave] * TOTAM[ilay] #(NWAVE,NLAY) 
            dtau_ray[iwave,ilay] = k_rayleighj[iwave] #dTAURAY/dTOTAM (m2)

    return tau_ray,dtau_ray

###############################################################################################

@jit(nopython=True)
def calc_tau_rayleighv(ISPACE,WAVEC,TOTAM):
    """
    Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
    for CO2-dominated atmospheres using data from Allen (1976) Astrophysical Quantities

    Inputs
    ________

    ISPACE :: Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
    WAVEC(NWAVE) :: Wavenumber (cm-1) or wavelength array (um)
    TOTAM(NLAY) :: Atmospheric column density in each layer (m-2)

    Outputs
    ________

    TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
    dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

    """

    NWAVE = len(WAVEC)
    NLAY = len(TOTAM)

    if ISPACE==WaveUnit.Wavenumber_cm:
        LAMBDA = 1./WAVEC * 1.0e-2 * 1.0e6  #Wavelength in microns
        x = 1.0/(LAMBDA*1.0e6)
    elif ISPACE == WaveUnit.Wavelength_um:
        LAMBDA = WAVEC #Wavelength in microns
    else:
        raise NotImplementedError(f'calc_tau_rayleighv :: Unknown value for ISPACE')

    C = 8.8e-28   #provided by B. Bezard

    #Calculating the scattering cross sections in m2
    k_rayleighv = C/LAMBDA**4. * 1.0e-4 #(NWAVE)
    
    #Calculating the Rayleigh opacities in each layer
    tau_ray = np.zeros((NWAVE,NLAY))
    dtau_ray = np.zeros((NWAVE,NLAY))
    for iwave in range(NWAVE):
        for ilay in range(NLAY):
            tau_ray[iwave,ilay] = k_rayleighv[iwave] * TOTAM[ilay] #(NWAVE,NLAY) 
            dtau_ray[iwave,ilay] = k_rayleighv[iwave] #dTAURAY/dTOTAM (m2)

    return tau_ray,dtau_ray

###############################################################################################

@jit(nopython=True)
def calc_tau_rayleighv2(ISPACE,WAVEC,TOTAM):
    """
    Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
    for CO2-dominated atmospheres using Ityaksov, Linnartz, Ubachs 2008, 
    Chemical Physics Letters, 462, 31-34

    Inputs
    ________

    ISPACE :: Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
    WAVEC(NWAVE) :: Wavenumber (cm-1) or wavelength array (um)
    TOTAM(NLAY) :: Atmospheric column density in each layer (m-2)

    Outputs
    ________

    TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
    dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

    """

    NWAVE = len(WAVEC)
    NLAY = len(TOTAM)

    if ISPACE==WaveUnit.Wavenumber_cm:
        LAMBDA = 1./WAVEC * 1.0e-2 * 1.0e6  #Wavelength in microns
        x = 1.0/(LAMBDA*1.0e6)
    elif ISPACE == WaveUnit.Wavelength_um:
        LAMBDA = WAVEC #Wavelength in microns
    else:
        raise NotImplementedError(f'calc_tau_rayleighv2 :: Unknown value for ISPACE')

    #dens = 1.01325d6 / (288.15 * 1.3803e-16)
    dens = 2.5475605e+19

    #wave in microns -> cm
    lam = LAMBDA*1.0e-4

    #King factor (taken from Ityaksov et al.)
    f_king = 1.14 + (25.3e-12)/(lam*lam)

    nu2 = 1./lam/lam
    term1 = 5799.3 / (16.618e9-nu2) + 120.05/(7.9609e9-nu2) + 5.3334 / (5.6306e9-nu2) + 4.3244 / (4.6020e9-nu2) + 1.218e-5 / (5.84745e6 - nu2)
    
    #refractive index
    n = 1.0 + 1.1427e3*term1

    factor1 = ( (n*n-1)/(n*n+2.0) )**2.

    k_rayleighv = (24.*np.pi**3./lam**4./dens**2.) * factor1 * f_king  #cm2
    k_rayleighv = k_rayleighv * 1.0e-4

    #Calculating the Rayleigh opacities in each layer
    tau_ray = np.zeros((NWAVE,NLAY))
    dtau_ray = np.zeros((NWAVE,NLAY))
    for iwave in range(NWAVE):
        for ilay in range(NLAY):
            tau_ray[iwave,ilay] = k_rayleighv[iwave] * TOTAM[ilay] #(NWAVE,NLAY) 
            dtau_ray[iwave,ilay] = k_rayleighv[iwave] #dTAURAY/dTOTAM (m2)

    return tau_ray,dtau_ray

###############################################################################################

@jit(nopython=True)
def calc_tau_rayleighls(ISPACE,WAVEC,ID,ISO,VMR,TOTAM):
    """
    Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
    for Jovian air using the code from Larry Sromovsky. Computes Rayleigh scattering 
    cross section per molecule considering only H2, He, CH4, and NH3 with only NH3 expressed
    as a volume mixing ratio

    Inputs
    ________

    ISPACE :: Flag indicating the spectral units (0) Wavenumber in cm-1 (1) Wavelegnth (um)
    WAVEC(NWAVE) :: Wavenumber (cm-1) or wavelength array (um)
    ID(NGAS) :: Radtran ID of each atmospheric gas
    ISO(NGAS) :: Radtran ID of each isotope
    VMR(NLAY,NGAS) :: Volume mixing ratio of each gas in each atmospheric layer
    TOTAM(NLAY) :: Atmospheric column density in each layer (m-2)

    Outputs
    ________

    TAURAY(NWAVE,NLAY) :: Rayleigh scattering opacity in each layer
    dTAURAY(NWAVE,NLAY) :: Rate of change of Rayleigh scattering opacity in each layer

    """
    
    #Calculating the fractions of He and CH4 wrt to H2
    NVMR = VMR.shape[1]
    NLAY = VMR.shape[0]
    NWAVE = len(WAVEC)

    #Finding the location of H2, He, CH4 and NH3 in the atmosphere    
    ih2 = -1
    inh3 = -1
    ihe = -1
    ich4 = -1
    
    fh2 = np.zeros(NLAY)
    fhe = np.zeros(NLAY)
    fch4 = np.zeros(NLAY)
    fnh3 = np.zeros(NLAY)
    for j in range(NVMR):
        
        if ID[j]==39:  #H2
            if((ISO[j]==0) or (ISO[j]==1)):
                ih2 = j
                fh2[:] = VMR[:,ih2]
        elif ID[j]==40:  #He
            if((ISO[j]==0) or (ISO[j]==1)):
                ihe = j
                fhe[:] = VMR[:,ihe]
        elif ID[j]==6:  #CH4
            if((ISO[j]==0) or (ISO[j]==1)):
                ich4 = j
                fch4[:] = VMR[:,ich4]
        elif ID[j]==11:  #NH3
            if((ISO[j]==0) or (ISO[j]==1)):
                inh3 = j
                fnh3[:] = VMR[:,inh3]
                
    
    fheh2 = np.zeros(NLAY)
    fch4h2 = np.zeros(NLAY)
    inot = np.where(fh2>0.0)
    fheh2[inot] = fhe[inot]/fh2[inot]
    fch4h2[inot] = fch4[inot]/fh2[inot]
        
    #Calculating the relative amounts of H2,CH4,He and NH3 (with the assumption that the sum of these gases provide VMR=1)
    comp = np.zeros((NLAY,4))
    comp[:,0] = (1.0 - fnh3)/(1.0+fheh2+fch4h2)   #H2
    comp[:,1] = fheh2 * comp[:,0]                 #He
    comp[:,2] = fch4h2 * comp[:,0]                #CH4
    comp[:,3] = fnh3[:]                           #NH3
    
    #loschpm3 is molecules per cubic micron at STP
    loschpm3=2.687e19*1.0e-12
    
    if ISPACE==WaveUnit.Wavenumber_cm:
        wl = 1./WAVEC * 1.0e-2 * 1.0e6  #Wavelength in microns
    elif ISPACE == WaveUnit.Wavelength_um:
        wl = WAVEC #Wavelength in microns
    else:
        raise NotImplementedError(f'calc_tau_rayleighls :: Unknown value for ISPACE')
    
    
    #refractive index equation coefficients from Allen, Astrophys. Quant., p 87 (1964)
    #where n-1=A(1+B/wl^2), where wl is wavelength
    #and n is the refractive index at STP (0C, 1 Atm=1.01325bar)
    
    #used NH3 value as a guess for CH4 which is not listed
    #depol. factors from Penndorf, J. Opt. Soc. of Amer., 47, 176-182 (1957)
    #used Parthasarathy (1951) values from Table II.
    #used CO2 value as a guess for CH4 which is not listed
    
    A = np.array((13.58e-5, 3.48e-5, 37.0e-5, 37.0e-5))  #H2,He,CH4,NH3
    B = np.array((7.52e-3,  2.3e-3, 12.0e-3, 12.0e-3))
    D = np.array((0.0221,   0.025,    .0922, .0922))
    
    #Compute summation over molecule-dependent scattering properties
    #Cross section formula also given in van de Hulst (1957)
    #xc1=0.
    #sumwt=0.
    xc1 = np.zeros((NLAY,NWAVE))
    sumwt = np.zeros(NLAY)
    for j in range(4):
        nr = 1.0 + A[j]*(1.0+B[j]/wl**2.)  #(NWAVE)
        for ilay in range(NLAY):
            xc1[ilay,:] = xc1[ilay,:] + (nr**2.0 - 1.0)**2.0*comp[ilay,j]*(6.0+3.0*D[j])/(6.0-7.0*D[j])
        sumwt[:] = sumwt[:] + comp[:,j]

    fact=8.0*(np.pi**3.0)/(3.0*(wl**4.0)*(loschpm3**2.0))   #(NWAVE)

    #average cross section in m^2 per molecule 
    k_rayleighls=np.transpose(fact*1e-8*xc1)/sumwt * 1.0e-4 #(NWAVE,NLAY)
    
    #Calculating the Rayleigh opacities in each layer
    tau_ray = np.zeros((NWAVE,NLAY))
    dtau_ray = np.zeros((NWAVE,NLAY))

    tau_ray[:,:] = k_rayleighls[:,:] * TOTAM  #(NWAVE,NLAY) 
    dtau_ray[:,:] = k_rayleighls[:,:]               #dTAURAY/dTOTAM (m2)
                
    return tau_ray, dtau_ray


###############################################################################################
###############################################################################################
                                    #INTERPOLATIONS
###############################################################################################
###############################################################################################

@jit(nopython=True)
def bilinear_xy(Q00, Q10, Q01, Q11, x1, x2, y1, y2, x, y):
    fxy1 = ((x2 - x + 1e-30) / (x2 - x1 + 2e-30)) * Q00 + ((x - x1 + 1e-30) / (x2 - x1 + 2e-30)) * Q10
    fxy2 = ((x2 - x + 1e-30) / (x2 - x1 + 2e-30)) * Q01 + ((x - x1 + 1e-30) / (x2 - x1 + 2e-30)) * Q11
    return ((y2 - y + 1e-30) / (y2 - y1 + 2e-30)) * fxy1 + ((y - y1 + 1e-30) / (y2 - y1 + 2e-30)) * fxy2

@jit(nopython=True)
def trilinear_interpolation(grid, x_values, y_values, z_values, x_array, y_array, z_array):
    """
    Performs trilinear interpolation on a 3D grid for arrays of x, y, and z coordinates.
    Points outside the grid are assigned a value of 0.

    :param grid: 3D array of grid values.
    :param x_values: 1D array of x-axis values in the grid.
    :param y_values: 1D array of y-axis values in the grid.
    :param z_values: 1D array of z-axis values in the grid.
    :param x_array: 1D array of x-coordinates for interpolation.
    :param y_array: 1D array of y-coordinates for interpolation.
    :param z_array: 1D array of z-coordinates for interpolation.
    :return: 3D array of interpolated values.
    """
    result = np.zeros((z_array.size, x_array.size))

    for k in range(z_array.size):
        for i in range(x_array.size):
            x = x_array[i]
            if grid.shape[0] == 1:
                x = 0
            y = y_array[i]
            z = z_array[k]

            if x < x_values[0] or x > x_values[-1] \
            or y < y_values[0] or y > y_values[-1] \
            or z < z_values[0] or z > z_values[-1]:
                result[k, i] = 0
            else:
                ix = np.searchsorted(x_values, x) - 1
                iy = np.searchsorted(y_values, y) - 1
                iz = np.searchsorted(z_values, z) - 1

                ix = max(min(ix, grid.shape[0] - 2), 0)
                if ix == 0 and grid.shape[0] == 1:
                    ix = -1

                iy = max(min(iy, grid.shape[1] - 2), 0)
                iz = max(min(iz, grid.shape[2] - 2), 0)

                x1, x2 = x_values[ix], x_values[ix + 1]
                y1, y2 = y_values[iy], y_values[iy + 1]
                z1, z2 = z_values[iz], z_values[iz + 1]

                Q000, Q100, Q010, Q110 = grid[ix, iy, iz], grid[ix+1, iy, iz], grid[ix, iy+1, iz], grid[ix+1, iy+1, iz]
                Q001, Q101, Q011, Q111 = grid[ix, iy, iz+1], grid[ix+1, iy, iz+1], grid[ix, iy+1, iz+1], grid[ix+1, iy+1, iz+1]

                fz1 = bilinear_xy(Q000, Q100, Q010, Q110, x1, x2, y1, y2, x, y)
                fz2 = bilinear_xy(Q001, Q101, Q011, Q111, x1, x2, y1, y2, x, y)

                result[k, i] = ((z2 - z + 1e-30) / (z2 - z1 + 2e-30)) * fz1 + ((z - z1 + 1e-30) / (z2 - z1 + 2e-30)) * fz2
    return result


###############################################################################################
###############################################################################################
                                 #K-COEFFICIENT OVERLAP
###############################################################################################
###############################################################################################

@jit(nopython=True)
def k_overlapg(del_g,k_w_g_l_gas,dkdT_w_g_l_gas,amount_layer):
    """
    Combine k distributions of multiple gases given their number densities.

    Parameters
    ----------
    k_w_g_l_gas(NGAS,NG) : ndarray
        K-distributions of the different gases.
        Each row contains a k-distribution defined at NG g-ordinates.
        Unit: cm^2 (per particle)
    amount(NGAS) : ndarray
        Absorber amount of each gas,
        i.e. amount = VMR x layer absorber per area
        Unit: (no. of partiicles) cm^-2
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    tau_g(NG) : ndarray
        Opatical path from mixing k-distribution weighted by absorber amounts.
        Unit: dimensionless
    """
    NWAVE, NG, NLAYER, NGAS = k_w_g_l_gas.shape
    tau_w_g_l = np.zeros((NWAVE, NG, NLAYER))
    dk_w_g_l_param = np.zeros((NWAVE, NG, NLAYER,NGAS+1))
    
    if NGAS == 1:
        tau_w_g_l = k_w_g_l_gas[:,:,:,0]*amount_layer[None,None,0,:]
        dk_w_g_l_param[:,:,:,0] = k_w_g_l_gas[:,:,:,0]
        dk_w_g_l_param[:,:,:,1] = dkdT_w_g_l_gas[:,:,:,0]*amount_layer[None,None,0,:]
        
        return tau_w_g_l,dk_w_g_l_param
    
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            amount = amount_layer[:,ilayer]
            k_g_gas = k_w_g_l_gas[iwave,:,ilayer,:]
            dkdT_g_param = dkdT_w_g_l_gas[iwave,:,ilayer,:]
            
            random_weight = np.zeros(NG*NG)
            random_tau = np.zeros(NG*NG)
            random_grad = np.zeros((NG*NG,NGAS+1))
            
            cutoff = 0
            
            tau_g = np.zeros(NG)
            dk_g_param = np.zeros((NG,NGAS+1))
            
            for igas in range(NGAS-1):
                # first pair of gases
                if igas == 0:
                    # if opacity due to first gas is negligible
                    if k_g_gas[:,igas][-1] * amount[igas] <= cutoff:
                        tau_g = k_g_gas[:,igas+1] * amount[igas+1]
                        dk_g_param[:,igas+1] = k_g_gas[:,igas+1]
                        dk_g_param[:,igas+2] = dkdT_g_param[:,igas+1] * amount[igas+1]
                        
                        
                    # if opacity due to second gas is negligible
                    elif k_g_gas[:,igas+1][-1] * amount[igas+1] <= cutoff:
                        tau_g = k_g_gas[:,igas] * amount[igas]
                        dk_g_param[:,igas] = k_g_gas[:,igas]
                        dk_g_param[:,igas+2] = dkdT_g_param[:,igas] * amount[igas]                       
                        
                        
                    # else resort-rebin with random overlap approximation
                    else:
                        iloop = 0
                        for ig in range(NG):
                            for jg in range(NG):
                                random_weight[iloop] = del_g[ig] * del_g[jg]
                                random_tau[iloop] = k_g_gas[ig,igas] * amount[igas] \
                                    + k_g_gas[jg,igas+1] * amount[igas+1]
                                random_grad[iloop,igas] = k_g_gas[ig,igas]
                                random_grad[iloop,igas+1] = k_g_gas[jg,igas+1]
                                random_grad[iloop,igas+2] = dkdT_g_param[ig,igas]*amount[igas]+\
                                                            dkdT_g_param[jg,igas+1]*amount[igas+1]
                                iloop = iloop + 1
                                
                                
                        tau_g,dk_g_param = rankg(random_weight,random_tau,del_g,random_grad,igas+3)
                # subsequent gases, add amount*k to previous summed k
                
                else:
                    # if opacity due to next gas is negligible
                    if k_g_gas[:,igas+1][-1] * amount[igas+1] <= cutoff:
                        dk_g_param[:,igas+2] = dk_g_param[:,igas+1]
                        dk_g_param[:,igas+1] *= 0
                    # if opacity due to previous gases is negligible
                    elif tau_g[-1] <= cutoff:
                        tau_g = k_g_gas[:,igas+1] * amount[igas+1]
                        dk_g_param[:,igas+1] = k_g_gas[:,igas+1]
                        dk_g_param[:,igas+2] = dkdT_g_param[:,igas+1] * amount[igas+1]
                    # else resort-rebin with random overlap approximation
                    else:
                        iloop = 0
                        for ig in range(NG):
                            for jg in range(NG):
                                random_weight[iloop] = del_g[ig] * del_g[jg]
                                random_tau[iloop] = tau_g[ig] + k_g_gas[jg,igas+1] * amount[igas+1]
                                
                                random_grad[iloop,:igas+1] = dk_g_param[ig,:igas+1]
                                random_grad[iloop,igas+1] = k_g_gas[jg,igas+1]
                                random_grad[iloop,igas+2] = dk_g_param[ig,igas+1]+\
                                                            dkdT_g_param[jg,igas+1]*amount[igas+1]
                                
                                
                                iloop = iloop + 1
                        tau_g,dk_g_param = rankg(random_weight,random_tau,del_g,random_grad,igas+3)
            tau_w_g_l[iwave,:,ilayer] = tau_g
            dk_w_g_l_param[iwave,:,ilayer,:] = dk_g_param
                        
    return tau_w_g_l, dk_w_g_l_param

@jit(nopython=True)
def rankg(weight, cont, del_g, grad, n):
    """
    Combine the randomly overlapped k distributions of two gases into a single
    k distribution.

    Parameters
    ----------
    weight(NG) : ndarray
        Weights of points in the random k-dist
    cont(NG) : ndarray
        Random k-coeffs in the k-dist.
    del_g(NG) : ndarray
        Required weights of final k-dist.
    grad(NG, NPARAM) : ndarray

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-dist.
        Unit: cm^2 (per particle)
    """
    ng = len(del_g)
    nloop = len(weight.flatten())
    nparam = grad.shape[1]
    # sum delta gs to get cumulative g ordinate
    g_ord = np.zeros(ng+1)
    g_ord[1:] = np.cumsum(del_g)
    g_ord[ng] = 1
    
    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    ico = np.argsort(cont)
    cont = cont[ico]
    weight = weight[ico] # sort weights accordingly
    grad = grad[ico,:]
    gdist = np.cumsum(weight)
    k_g = np.zeros(ng)
    dkdq = np.zeros((ng,nparam))
    ig = 0
    sum1 = 0.0
    cont_weight = cont * weight
    grad_weight = grad * weight[:,None]
    for iloop in range(nloop):
        if gdist[iloop] < g_ord[ig+1] and ig < ng:
            k_g[ig] = k_g[ig] + cont_weight[iloop]
            dkdq[ig,:n] += grad_weight[iloop,:n]
            
            sum1 = sum1 + weight[iloop]
        else:
            frac = (g_ord[ig+1] - gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
            k_g[ig] = k_g[ig] + frac*cont_weight[iloop]
            dkdq[ig,:n] += frac * grad_weight[iloop,:n]
                
            sum1 = sum1 + frac * weight[iloop]
            k_g[ig] = k_g[ig]/sum1
            dkdq[ig,:n] = dkdq[ig,:n]/sum1
                
            ig = ig + 1
            if ig < ng:
                sum1 = (1.0-frac)*weight[iloop]
                k_g[ig] = (1.0-frac)*cont_weight[iloop]
                dkdq[ig,:n] = (1.0-frac)* grad_weight[iloop,:n]
                    
    if ig == ng-1:
        k_g[ig] = k_g[ig]/sum1
        dkdq[ig,:n] = dkdq[ig,:n]/sum1
    return k_g, dkdq


@jit(nopython=True)
def k_overlap(del_g,k_w_g_l_gas,amount_layer):
    """
    Combine k distributions of multiple gases given their number densities.

    Parameters
    ----------
    k_w_g_l_gas(NGAS,NG) : ndarray
        K-distributions of the different gases.
        Each row contains a k-distribution defined at NG g-ordinates.
        Unit: cm^2 (per particle)
    amount(NGAS) : ndarray
        Absorber amount of each gas,
        i.e. amount = VMR x layer absorber per area
        Unit: (no. of partiicles) cm^-2
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    tau_g(NG) : ndarray
        Opatical path from mixing k-distribution weighted by absorber amounts.
        Unit: dimensionless
    """
    NWAVE, NG, NLAYER, NGAS = k_w_g_l_gas.shape
    tau_w_g_l = np.zeros((NWAVE, NG, NLAYER))
    if NGAS == 1:
        tau_w_g_l = k_w_g_l_gas[:,:,:,0]*amount_layer[None,None,0,:]
        return tau_w_g_l
    
    for iwave in range(NWAVE):
        for ilayer in range(NLAYER):
            amount = amount_layer[:,ilayer]
            k_g_gas = k_w_g_l_gas[iwave,:,ilayer,:]
            
            random_weight = np.zeros(NG*NG)
            random_tau = np.zeros(NG*NG)
            cutoff = 0
            
            tau_g = np.zeros(NG)
            
            for igas in range(NGAS-1):
                # first pair of gases
                if igas == 0:
                    # if opacity due to first gas is negligible
                    if k_g_gas[:,igas][-1] * amount[igas] <= cutoff:
                        tau_g = k_g_gas[:,igas+1] * amount[igas+1]
                        
                    # if opacity due to second gas is negligible
                    elif k_g_gas[:,igas+1][-1] * amount[igas+1] <= cutoff:
                        tau_g = k_g_gas[:,igas] * amount[igas]
                        
                    # else resort-rebin with random overlap approximation
                    else:
                        iloop = 0
                        for ig in range(NG):
                            for jg in range(NG):
                                random_weight[iloop] = del_g[ig] * del_g[jg]
                                random_tau[iloop] = k_g_gas[ig,igas] * amount[igas] \
                                    + k_g_gas[jg,igas+1] * amount[igas+1]
                                iloop = iloop + 1
                                
                                
                        tau_g = rank(random_weight,random_tau,del_g)
                # subsequent gases, add amount*k to previous summed k
                
                else:
                    # if opacity due to next gas is negligible
                    if k_g_gas[:,igas+1][-1] * amount[igas+1] <= cutoff:
                        pass
                    # if opacity due to previous gases is negligible
                    elif tau_g[-1] <= cutoff:
                        tau_g = k_g_gas[:,igas+1] * amount[igas+1]
                    # else resort-rebin with random overlap approximation
                    else:
                        iloop = 0
                        for ig in range(NG):
                            for jg in range(NG):
                                random_weight[iloop] = del_g[ig] * del_g[jg]
                                random_tau[iloop] = tau_g[ig] + k_g_gas[jg,igas+1] * amount[igas+1]
                                
                                iloop = iloop + 1
                        tau_g = rank(random_weight,random_tau,del_g)
            tau_w_g_l[iwave,:,ilayer] = tau_g
                        
    return tau_w_g_l

@jit(nopython=True)
def rank(weight, cont, del_g):
    """
    Combine the randomly overlapped k distributions of two gases into a single
    k distribution.

    Parameters
    ----------
    weight(NG) : ndarray
        Weights of points in the random k-dist
    cont(NG) : ndarray
        Random k-coeffs in the k-dist.
    del_g(NG) : ndarray
        Required weights of final k-dist.

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-dist.
        Unit: cm^2 (per particle)
    """
    ng = len(del_g)
    nloop = len(weight.flatten())
    # sum delta gs to get cumulative g ordinate
    g_ord = np.zeros(ng+1)
    g_ord[1:] = np.cumsum(del_g)
    g_ord[ng] = 1
    
    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    ico = np.argsort(cont)
    cont = cont[ico]
    weight = weight[ico] # sort weights accordingly
    gdist = np.cumsum(weight)
    k_g = np.zeros(ng)
    ig = 0
    sum1 = 0.0
    cont_weight = cont * weight
    for iloop in range(nloop):
        if gdist[iloop] < g_ord[ig+1] and ig < ng:
            k_g[ig] = k_g[ig] + cont_weight[iloop]
            sum1 = sum1 + weight[iloop]
        else:
            frac = (g_ord[ig+1] - gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
            k_g[ig] = k_g[ig] + frac*cont_weight[iloop]
                
            sum1 = sum1 + frac * weight[iloop]
            k_g[ig] = k_g[ig]/sum1
                
            ig = ig + 1
            if ig < ng:
                sum1 = (1.0-frac)*weight[iloop]
                k_g[ig] = (1.0-frac)*cont_weight[iloop]
                    
    if ig == ng-1:
        k_g[ig] = k_g[ig]/sum1
    return k_g


###############################################################################################
###############################################################################################
                                    #THERMAL EMISSION
###############################################################################################
###############################################################################################

###############################################################################################
@jit(nopython=True)
def planck(ispace,wave,temp):


    """
    FUNCTION NAME : planck()

    DESCRIPTION : Function to calculate the blackbody radiation given by the Planck function

    INPUTS : 

        ispace :: Flag indicating the spectral units
                  (0) Wavenumber (cm-1)
                  (1) Wavelength (um)
        wave(nwave) :: Wavelength or wavenumber array
        temp(ntemp) :: Temperature of the blackbody (K)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    bb(nwave,ntemp) :: Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
 
    CALLING SEQUENCE:

	    bb = planck(ispace,wave,temp)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    c1 = 1.1911e-12
    c2 = 1.439 
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    return bb

###############################################################################################
@jit(nopython=True)
def planckg(ispace,wave,temp):


    """
    FUNCTION NAME : planckg()

    DESCRIPTION : Function to calculate the blackbody radiation given by the Planck function
                    as well as its derivative with respect to temperature

    INPUTS : 

        ispace :: Flag indicating the spectral units
                  (0) Wavenumber (cm-1)
                  (1) Wavelength (um)
        wave(nwave) :: Wavelength or wavenumber array
        temp :: Temperature of the blackbody (K)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    bb(nwave) :: Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
        dBdT(nwave) :: Temperature gradient (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)/K
 
    CALLING SEQUENCE:

	    bb,dBdT = planckg(ispace,wave,temp)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """

    c1 = 1.1911e-12
    c2 = 1.439 
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
        ap = c1 * c2 * (y**4.)/temp**2.
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
        ap = c1 * c2 * (y**6.) / 1.0e4 / temp**2.

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    tmpp = c2 * y / temp
    bp = (np.exp(tmp) - 1.)**2.
    tp = np.exp(tmpp) * ap
    dBdT = tp/bp

    return bb,dBdT


###############################################################################################
@jit(nopython=True)
def calc_thermal_emission_spectrum(ISPACE,WAVE,TAUTOT_PATH,TEMP,PRESS,TSURF,EMISSIVITY,SOLFLUX,REFLECTANCE,SOL_ANG,EMISS_ANG):


    """
    FUNCTION NAME : thermal_emission()

    DESCRIPTION : Function to calculate the spectrum considering only thermal emission from 
                  the surface and atmosphere (no scattering and no solar component)

    INPUTS : 

        ISPACE :: Flag indicating the spectral units (0 - Wavenumber in cm-1 ; 1 - Wavelength in um)
        WAVE(NWAVE) :: Wavenumber of wavelength array
        TAUTOT_PATH(NWAVE,NG,NLAYIN) :: Total optical depth along the line-of-sight in each layer and wavelength
        TEMP(NLAYIN) :: Temperature of each layer along the path (K)
        PRESS(NLAYIN) :: Pressure of each layer along the path (Pa)
        TSURF :: Surface temperature (K) - If TSURF<0, then the planet is considered not to have surface
        EMISSIVITY(NWAVE) :: Emissivity of the surface
        SOLFLUX(NWAVE) :: Solar flux at the top of the atmosphere (W cm-2 um-1 or W cm-2 (cm-1)-1)
        REFLECTANCE(NWAVE) :: Surface reflectance
        SOL_ANG :: Solar zenith angle
        EMISS_ANG :: Emission angle

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    SPECOUT(NWAVE,NG) :: Spectrum in W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1
 
    CALLING SEQUENCE:

	    SPECOUT = calc_thermal_emission_spectrum(ISPACE,WAVE,TAUTOT_PATH,TEMP,PRESS,TSURF,EMISSIVITY)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """
    
    #Getting relevant array sizes
    NWAVE = TAUTOT_PATH.shape[0]
    NG = TAUTOT_PATH.shape[1]
    NLAYIN = TAUTOT_PATH.shape[2]
    
    SPECOUT = np.zeros((NWAVE,NG))  #Output spectrum

    for iwave in range(NWAVE):
        for ig in range(NG):
            
            #Initialising values
            taud = 0.
            trold = 1.
            specg = 0.
            
            #Calculating the atmospheric contribution
            #Looping through each layer along the path
            for j in range(NLAYIN):

                taud += TAUTOT_PATH[iwave,ig,j]
                tr = np.exp(-taud)
                bb = planck(ISPACE,WAVE[iwave],TEMP[j])
                specg += (trold-tr)*bb
                trold = tr

            #Calculating surface contribution
            p1 = PRESS[int(NLAYIN/2)-1]
            p2 = PRESS[int(NLAYIN-1)]

            if p2>p1:  #If not limb path, we add the surface contribution

                if TSURF<=0.0: #No surface contribution, getting temperature from bottom of atm
                    radground = planck(ISPACE,WAVE[iwave],TEMP[NLAYIN-1])
                else:
                    bbsurf = planck(ISPACE,WAVE[iwave],TSURF)
                    radground = bbsurf * EMISSIVITY[iwave]

                specg += trold * radground
                
            #Adding the surface reflection in case of downward-looking spectra (assumes plane-parallel atmosphere)
            if EMISS_ANG < 90. and SOL_ANG < 90.:
                refl = REFLECTANCE[iwave]
                solar = SOLFLUX[iwave]
                mu = np.cos(EMISS_ANG/180.*np.pi)
                mu0 = np.cos(SOL_ANG/180.*np.pi)
                specg += trold*np.exp(-taud*mu/mu0)*solar*refl

            SPECOUT[iwave,ig] = specg
            
    return SPECOUT

###############################################################################################
@jit(nopython=True)
def calc_thermal_emission_spectrumg(ISPACE,WAVE,TAUTOT_PATH,dTAUTOT_PATH,NVMR,TEMP,PRESS,TSURF,EMISSIVITY):


    """
    FUNCTION NAME : thermal_emission()

    DESCRIPTION : Function to calculate the spectrum considering only thermal emission from 
                  the surface and atmosphere (no scattering and no solar component)

    INPUTS : 

        ISPACE :: Flag indicating the spectral units (0 - Wavenumber in cm-1 ; 1 - Wavelength in um)
        WAVE(NWAVE) :: Wavenumber of wavelength array
        TAUTOT_PATH(NWAVE,NG,NLAYIN) :: Total optical depth along the line-of-sight in each layer and wavelength
        dTAUTOT_PATH(NWAVE,NG,NVMR+2+NDUST,NLAYIN) :: Derivative of TAUTOT_PATH wrt each of the atmospheric parameters (gases+temperature+dust)
        NVMR :: Number of gases in the atmosphere
        TEMP(NLAYIN) :: Temperature of each layer along the path (K)
        PRESS(NLAYIN) :: Pressure of each layer along the path (Pa)
        TSURF :: Surface temperature (K) - If TSURF<0, then the planet is considered not to have surface
        EMISSIVITY(NWAVE) :: Emissivity of the surface

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    SPECOUT(NWAVE,NG) :: Spectrum in W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1
        dSPECOUT(NWAVE,NG,NVMR+2+NDUST,NLAYIN) :: Gradient of the spectrum wrt each of the atmospheric parameters in each layer
        dTSURF(NWAVE,NG) :: Gradient of the spectrum wrt the surface temperature
 
    CALLING SEQUENCE:

	    SPECOUT,dSPECOUT,dTSURF = calc_thermal_emission_spectrumg(ISPACE,WAVE,TAUTOT_PATH,dTAUTOT_PATH,NVMR,TEMP,PRESS,TSURF,EMISSIVITY)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """
    
    #Getting relevant array sizes
    NWAVE = TAUTOT_PATH.shape[0]
    NG = TAUTOT_PATH.shape[1]
    NLAYIN = TAUTOT_PATH.shape[2]
    NPAR = dTAUTOT_PATH.shape[2]
    
    SPECOUT = np.zeros((NWAVE,NG))  #Output spectrum
    dSPECOUT = np.zeros((NWAVE,NG,NPAR,NLAYIN))  #Gradient with respect to each atmospheric parameter in each layer
    dTSURF = np.zeros((NWAVE,NG)) #Gradient with respect to the surface temperature
    

    for iwave in range(NWAVE):
        for ig in range(NG):
            
            #Initialising values
            tlayer = 0.
            taud = 0.
            trold = 1.
            specg = 0.
            
            dtolddq = np.zeros((NPAR,NLAYIN))
            dtrdq = np.zeros((NPAR,NLAYIN))
            dspecg = np.zeros((NPAR,NLAYIN))
            
            #Calculating atmospheric contribution
            #Looping through the layers along the path
            for j in range(NLAYIN):
                
                taud += TAUTOT_PATH[iwave,ig,j]
                tlayer = np.exp(-TAUTOT_PATH[iwave,ig,j])
                tr = trold * tlayer

                #Calculating the spectrum
                bb,dBdT = planckg(ISPACE,WAVE[iwave],TEMP[j])
                specg += (trold-tr)*bb
                
                #Setting up the gradients
                for k in range(NPAR):
                    
                    j1 = 0
                    while j1<j:
                        dtrdq[k,j1] = dtolddq[k,j1] * tlayer
                        dspecg[k,j1] += (dtolddq[k,j1]-dtrdq[k,j1])*bb
                        j1 += 1

                    tmp = dTAUTOT_PATH[iwave,ig,k,j1]
                    dtrdq[k,j1] = -tmp * tlayer * trold
                    dspecg[k,j1] += (dtolddq[k,j1]-dtrdq[k,j1])*bb
                    
                    if k==NVMR:  #This is the index of the gradient with respect to the temperature
                        dspecg[k,j] += (trold-tr)*dBdT

                #Saving arrays for next iteration
                trold = tr
                j1 = 0
                while j1<j:
                    dtolddq[:,j1] = dtrdq[:,j1]
                    j1 += 1
                dtolddq[:,j1] = dtrdq[:,j1]

            #Calculating surface contribution
            p1 = PRESS[int(NLAYIN/2)-1]
            p2 = PRESS[int(NLAYIN-1)]

            tempgtsurf = 0.
            if p2>p1:  #If not limb path, we add the surface contribution

                if TSURF<=0.0: #No surface contribution, getting temperature from bottom of atm
                    radground,dradgrounddT = planckg(ISPACE,WAVE[iwave],TEMP[NLAYIN-1])
                else:
                    bbsurf,dbsurfdT = planckg(ISPACE,WAVE[iwave],TSURF)
                    
                    radground = bbsurf * EMISSIVITY[iwave]
                    dradgrounddT = dbsurfdT * EMISSIVITY[iwave]

                specg += trold*radground
                tempgtsurf = trold * dradgrounddT

                for j in range(NLAYIN):
                    for k in range(NPAR):
                        dspecg[k,j] += radground * dtolddq[k,j]

            SPECOUT[iwave,ig] = specg
            dSPECOUT[iwave,ig,:,:] = dspecg[:,:]
            dTSURF[iwave,ig] = tempgtsurf
            
    return SPECOUT,dSPECOUT,dTSURF


###############################################################################################
#@jit(nopython=True)
def calc_singlescatt_plane_spectrum(ISPACE,WAVE,TAUTOT_PATH,TEMP,OMEGA,PHASE,TSURF,EMISSIVITY,BRDF,SOLFLUX,SOL_ANG,EMISS_ANG):


    """
    FUNCTION NAME : thermal_emission()

    DESCRIPTION : Function to calculate the spectrum considering only thermal emission from 
                  the surface and atmosphere (no scattering and no solar component)

    INPUTS : 

        ISPACE :: Flag indicating the spectral units (0 - Wavenumber in cm-1 ; 1 - Wavelength in um)
        WAVE(NWAVE) :: Wavenumber of wavelength array
        TAUTOT_PATH(NWAVE,NG,NLAYIN) :: Total optical depth along the line-of-sight in each layer and wavelength
        TEMP(NLAYIN) :: Temperature of each layer along the path (K)
        PRESS(NLAYIN) :: Pressure of each layer along the path (Pa)
        OMEGA(NWAVE,NG,NLAYIN) :: Single scattering albedo of each layer along the path
        PHASE(NWAVE,NLAYIN) :: Average phase function of each layer along the path
        TSURF :: Surface temperature (K) - If TSURF<0, then the planet is considered not to have surface
        EMISSIVITY(NWAVE) :: Emissivity of the surface
        BRDF(NWAVE) :: Bidirectional reflectance distributon function at the required geometry
        SOLFLUX(NWAVE) :: Solar flux at the top of the atmosphere (W cm-2 um-1 or W cm-2 (cm-1)-1)
        SOL_ANG :: Incident angle (degrees)
        EMISS_ANG :: Emission angle (degrees)

    OPTIONAL INPUTS:  none

    OUTPUTS : 

	    SPECOUT(NWAVE,NG) :: Spectrum in W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1
 
    CALLING SEQUENCE:

	    SPECOUT = calc_singlescatt_plane_spectrum(ISPACE,WAVE,TAUTOT_PATH,TEMP,PRESS,OMEGA,PHASE,TSURF,EMISSIVITY,BRDF,SOLFLUX,SOL_ANG,EMISS_ANG)
 
    MODIFICATION HISTORY : Juan Alday (29/07/2021)

    """
    
    #Getting relevant array sizes
    NWAVE = TAUTOT_PATH.shape[0]
    NG = TAUTOT_PATH.shape[1]
    NLAYIN = TAUTOT_PATH.shape[2]
    
    #Calculating angles
    mu = np.cos(EMISS_ANG/180.*np.pi)
    mu0 = np.cos(SOL_ANG/180.*np.pi)
    ssfac = mu0/(mu0+mu)
    
    SPECOUT = np.zeros((NWAVE,NG))  #Output spectrum

    for iwave in range(NWAVE):
        for ig in range(NG):
            
            #Initialising values
            taud = 0.
            trold = 1.
            specg = 0.
            
            #Calculating the atmospheric contribution
            #Looping through each layer along the path
            for j in range(NLAYIN):
                
                omega_lay = OMEGA[iwave,ig,j]
                phase_lay = PHASE[iwave,j]
                taud += TAUTOT_PATH[iwave,ig,j]
                tr = np.exp(-taud)
                
                #Scattering contribution
                specg += (trold-tr)*ssfac*omega_lay*phase_lay*SOLFLUX[iwave]/(4.*np.pi) 
                
                #Thermal emission contribution
                bb = planck(ISPACE,WAVE[iwave],TEMP[j])
                specg += (trold-tr)*bb
                
                trold = tr

            #Calculating surface contribution
            if TSURF<=0.0: #No surface contribution, getting temperature from bottom of atm
                radground = planck(ISPACE,WAVE[iwave],TEMP[NLAYIN-1])
            else:
                bbsurf = planck(ISPACE,WAVE[iwave],TSURF)
                radground = bbsurf * EMISSIVITY[iwave]

            specg += trold * radground
                
            #Calculating reflectance from the ground
            specg += trold*SOLFLUX[iwave]*mu0*BRDF[iwave]
            
            SPECOUT[iwave,ig] = specg
            
    return SPECOUT