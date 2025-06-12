from archnemesis import *
from archnemesis.enums import (
    WaveUnit,
    ScatteringCalculationMode,
    RayleighScatteringMode,  
    AerosolPhaseFunctionCalculationMode
)
import numpy as np
import matplotlib.pyplot as plt
import os

from archnemesis.helpers import h5py_helper

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

Scattering Class. Includes the absorption and scattering properties of aerosol particles.

JIT-optimised Rayleigh Scattering routines
"""

class Scatter_0:

    def __init__(self, 
        ISPACE=WaveUnit.Wavenumber_cm,
        ISCAT=ScatteringCalculationMode.MULTIPLE_SCATTERING,
        IRAY=RayleighScatteringMode.NOT_INCLUDED,
        IMIE=AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN,
        NMU=5, NF=2, NPHI=101, NDUST=1,
        SOL_ANG=0.0, EMISS_ANG=0.0, AZI_ANG=0.0,
        NTHETA=None, THETA=None):

        """
        Inputs
        ------
        @param ISPACE: int,
            Flag indicating the spectral units
            (0) Wavenumber (cm-1)
            (1) Wavelength (um)
        @param ISCAT: int,
            Flag indicating the type of scattering calculation that must be performed 
            (0) Thermal emission calculation (no scattering)
            (1) Multiple scattering
            (2) Internal scattered radiation field is calculated first (required for limb-scattering calculations)
            (3) Single scattering in plane-parallel atmosphere
            (4) Single scattering in spherical atmosphere
        @param IRAY int,
            Flag indicating the type of Rayleigh scattering to include in the calculations:
            (0) Rayleigh scattering optical depth not included
            (1) Rayleigh optical depths for gas giant atmosphere
            (2) Rayleigh optical depth suitable for CO2-dominated atmosphere
            (3) Rayleigh optical depth suitable for a N2-O2 atmosphere
            (>3) Rayleigh optical depth suitable for Jovian air (adapted from Larry Sromovsky) 
        @param IMIE int,
            Flag indicating how the aerosol phase function needs to be computed (only relevant for ISCAT>0):
            (0) Phase function is computed from the associated Henyey-Greenstein parameters stored in G1,G2
            (1) Phase function is computed from the Mie-Theory parameters stored in PHASE
            (3) Phase function is computed from the Legendre Polynomials stored in WLPOL
        @param NDUST: int,
            Number of aerosol populations included in the atmosphere
        @param NMU: int,
            Number of zenith ordinates to perform the scattering calculations                
        @param NF: int,
            Number of Fourier components to perform the scattering calculations in the azimuth direction
        @param NPHI: int,
            Number of azimuth ordinates to perform the scattering calculations using Fourier analysis
        @param SOL_ANG: float,
            Observation solar angle (degrees)
        @param EMISS_ANG: float,
            Observation emission angle (degrees)
        @param AZI_ANG: float,
            Observation azimuth angle (degrees)
            
        Attributes
        ----------
        @attribute NWAVE: int,
            Number of wavelengths used to define its spectral properties 
        @attribute NTHETA: int,
            Number of angles used to define the scattering phase function of the aerosols 
        @attribute WAVE: 1D array,
            Wavelengths at which the spectral properties of the aerosols are defined      
        @attribute KEXT: 2D array,
            Extinction cross section of each of the aerosol populations at each wavelength (cm2)
        @attribute SGLALB: 2D array,
            Single scattering albedo of each of the aerosol populations at each wavelength
        @attribute KABS: 2D array,
            Absorption cross section of each of the aerosol populations at each wavelength (cm2)
        @attribute KSCA: 2D array,
            Scattering cross section of each of the aerosol populations at each wavelength (cm2)
        @attribute PHASE: 3D array,
            Scattering phase function of each of the aerosol populations at each wavelength
        @attribute F: 2D array,
            Parameter defining the relative contribution of G1 and G2 of the double Henyey-Greenstein phase function
            See Irvine (1965)
        @attribute G1: 2D array,
            Parameter defining the first assymetry factor of the double Henyey-Greenstein phase function
            See Irvine (1965)
        @attribute G2: 2D array,
            Parameter defining the second assymetry factor of the double Henyey-Greenstein phase function
            See Irvine (1965)
        @attribute WLPOL: 2D array
            Weights of the Legendre polynomials used to model the phase function
        @attribute MU: 1D array,
            Cosine of the zenith angles corresponding to the Gauss-Lobatto quadrature points
        @attribute WTMU: 1D array,
            Quadrature weights of the Gauss-Lobatto quadrature points
        @attribute ALPHA: real,
            Scattering angle (degrees) computed from the observing angles

        Methods
        -------
        Scatter_0.assess()
        Scatter_0.write_hdf5()
        Scatter_0.read_hdf5()
        Scatter_0.initialise_arrays()
        Scatter_0.calc_GAUSS_LOBATTO()
        Scatter_0.read_xsc()
        Scatter_0.write_xsc()
        Scatter_0.read_hgphase()
        Scatter_0.wrtite_hgphase()
        Scatter_0.calc_hgphase()
        Scatter_0.interp_phase()
        Scatter_0.calc_phase()
        Scatter_0.calc_phase_ray()
        Scatter_0.read_phase()
        Scatter_0.write_phase()
        Scatter_0.read_lpphase()
        Scatter_0.write_lpphase()
        Scatter_0.calc_lpphase()
        Scatter_0.check_phase_norm()
        Scatter_0.check_fourier_decomposition()
        Scatter_0.read_refind()
        Scatter_0.read_refind_file()
        Scatter_0.makephase()
        
        
        Scatter_0.miescat()
        Scatter_0.initialise_arrays()
        """

        #Input parameters
        self.NMU = NMU
        self.NF = NF
        self.NPHI = NPHI
        #self.ISPACE = WaveUnit(ISPACE) if not isinstance(ISPACE, WaveUnit) else ISPACE
        #self.ISCAT = ScatteringCalculationMode(ISCAT)
        #self.IRAY = RayleighScatteringMode(IRAY)
        #self.IMIE = AerosolPhaseFunctionCalculationMode(IMIE)
        self.SOL_ANG = SOL_ANG
        self.EMISS_ANG = EMISS_ANG
        self.AZI_ANG = AZI_ANG
        self.NDUST = NDUST
        
        # Input the following profiles using the edit_ methods.
        self.NWAVE = None
        self.WAVE = None #np.zeros(NWAVE)
        self.KEXT = None #np.zeros(NWAVE,NDUST)
        self.KABS = None #np.zeros(NWAVE,NDUST)
        self.KSCA = None #np.zeros(NWAVE,NDUST)
        self.SGLALB = None #np.zeros(NWAVE,NDUST)
        self.PHASE = None #np.zeros(NWAVE,NTHETA,NDUST)

        self.MU = None # np.zeros(NMU)
        self.WTMU = None # np.zeros(NMU)

        # Fortran defaults
        if THETA is None:
            self.THETA = np.array([
                0, 1, 2, 3, 4, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 50, 60, 70, 80,
                90, 100, 110, 120, 130, 140, 145, 150, 155, 160, 162.5, 165, 167.5, 170, 172.5,
                175, 176, 177, 178, 179, 180
            ])
            self.NTHETA = 41
        else:
            self.THETA = THETA
            self.NTHETA = NTHETA
        
        
        #Henyey-Greenstein phase function parameters
        self.G1 = None  #np.zeros(NWAVE,NDUST)
        self.G2 = None #np.zeros(NWAVE,NDUST)
        self.F = None #np.zeros(NWAVE,NDUST)

        #Legendre polynomials phase function parameters
        self.NLPOL = None #int
        self.WLPOL = None #np.zeros(NWAVE,NLPOL,NDUST)

        #Refractive index of a given aerosol population
        self.NWAVER = None 
        self.WAVER = None #np.zeros(NWAVER)
        self.REFIND_REAL = None #np.zeros(NWAVER)
        self.REFIND_IM = None #np.zeros(NWAVER)

        self.calc_GAUSS_LOBATTO()
        
        # private attributes 
        self._ispace = None
        self._iscat = None
        self._iray = None
        self._imie = None
        
        # set property values
        self.ISPACE = ISPACE
        self.ISCAT = ISCAT
        self.IRAY = IRAY
        self.IMIE = IMIE
    
    @property
    def ISPACE(self):
        return self._ispace
    
    @ISPACE.setter
    def ISPACE(self, value):
        self._ispace = WaveUnit(value)
    
    @property
    def ISCAT(self):
        return self._iscat
    
    @ISCAT.setter
    def ISCAT(self, value):
        self._iscat = ScatteringCalculationMode(value)
    
    @property
    def IRAY(self):
        return self._iray
    
    @IRAY.setter
    def IRAY(self, value):
        self._iray = RayleighScatteringMode(value)
    
    @property
    def IMIE(self):
        return self._imie
    
    @IMIE.setter
    def IMIE(self, value):
        self._imie = AerosolPhaseFunctionCalculationMode(value)
    
    

    ###########################################################################################################################

    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert isinstance(self.ISPACE, WaveUnit), 'ISPACE must be WaveUnit enum'
        assert self.ISPACE in (WaveUnit.Wavenumber_cm, WaveUnit.Wavelength_um), \
            'ISPACE must be Wavenumber_cm or Wavelength_um'
        assert isinstance(self.ISCAT, ScatteringCalculationMode), \
            'ISCAT must be ScatteringCalculationMode enum'
        assert isinstance(self.IRAY, RayleighScatteringMode), \
            'IRAY must be RayleighScatteringMode enum'
        assert isinstance(self.IMIE, AerosolPhaseFunctionCalculationMode), \
            'IMIE must be AerosolPhaseFunctionCalculationMode enum'

        assert np.issubdtype(type(self.NMU), np.integer) == True , \
            'NMU must be int'
        assert self.NMU >= 0 , \
            'NMU must be >=0'
        
        assert np.issubdtype(type(self.NF), np.integer) == True , \
            'NF must be int'
        assert self.NF >= 0 , \
            'NF must be >=0'

        assert np.issubdtype(type(self.NPHI), np.integer) == True , \
            'NPHI must be int'
        assert self.NPHI >= 0 , \
            'NPHI must be >=0'

        assert np.issubdtype(type(self.NDUST), np.integer) == True , \
            'NDUST must be int'
        assert self.NDUST >= 0 , \
            'NDUST must be >=0'
        
        assert np.issubdtype(type(self.NWAVE), np.integer) == True , \
            'NWAVE must be int'
        assert self.NWAVE >= 2 , \
            'NWAVE must be >=2'
        
        if self.NDUST>0:  #There are aerosols in the atmosphere

            assert self.KEXT.shape == (self.NWAVE,self.NDUST) , \
                'KEXT must have size (NWAVE,NDUST)'
            
            assert self.SGLALB.shape == (self.NWAVE,self.NDUST) , \
                'SGLALB must have size (NWAVE,NDUST)'
            
            if self.ISCAT>0:  #Scattering is turned on

                if self.IMIE == AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:  #Henyey-Greenstein phase function

                    assert self.G1.shape == (self.NWAVE,self.NDUST) , \
                        'G1 must have size (NWAVE,NDUST)'
                    
                    assert self.G2.shape == (self.NWAVE,self.NDUST) , \
                        'G2 must have size (NWAVE,NDUST)'

                    assert self.F.shape == (self.NWAVE,self.NDUST) , \
                        'F must have size (NWAVE,NDUST)'
                    
                elif self.IMIE == AerosolPhaseFunctionCalculationMode.MIE_THEORY:  #Explicit phase function 

                    assert np.issubdtype(type(self.NTHETA), np.integer) == True , \
                        'NTHETA must be int'
                    
                    assert self.PHASE.shape == (self.NWAVE,self.NTHETA,self.NDUST) , \
                        'PHASE must have size (NWAVE,NTHETA,NDUST)'
                    
                elif self.IMIE == AerosolPhaseFunctionCalculationMode.LEGENDRE_POLYNOMIALS:  #Phase function from Legrende polynomials 

                    assert np.issubdtype(type(self.NLPOL), np.integer) == True , \
                        'NLPOL must be int'
                    
                    assert self.WLPOL.shape == (self.NWAVE,self.NLPOL,self.NDUST) , \
                        'WLPOL must have size (NWAVE,NLPOL,NDUST)'

    ###########################################################################################################################

    def write_hdf5(self,runname):
        """
        Write the scattering properties into an HDF5 file
        """

        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess()

        f = h5py.File(runname+'.h5','a')
        #Checking if Scatter already exists
        if ('/Scatter' in f)==True:
            del f['Scatter']   #Deleting the Scatter information that was previously written in the file

        grp = f.create_group("Scatter")

        #Writing the spectral units
        dset = grp.create_dataset('ISPACE',data=int(self.ISPACE))
        dset.attrs['title'] = "Spectral units"
        if self.ISPACE==WaveUnit.Wavenumber_cm:
            dset.attrs['units'] = 'Wavenumber / cm-1'
        elif self.ISPACE==WaveUnit.Wavelength_um:
            dset.attrs['units'] = 'Wavelength / um'

        #Writing the scattering calculation type
        dset = grp.create_dataset('ISCAT', data=int(self.ISCAT))
        dset.attrs['title'] = "Scattering calculation type"
        dset.attrs['type'] = self.ISCAT.name.replace('_', ' ').title()

        #Writing the Rayleigh scattering type
        dset = grp.create_dataset('IRAY',data=int(self.IRAY))
        dset.attrs['title'] = "Rayleigh scattering type"
        dset.attrs['type'] = self.IRAY.name.replace('_', ' ').title()

        #Writing the aerosol scattering type
        dset = grp.create_dataset('IMIE',data=int(self.IMIE))
        dset.attrs['title'] = "Aerosol scattering phase function type"
        dset.attrs['type'] = self.IMIE.name.replace('_', ' ').title()
        
        #Writing some of the scattering calculation parameters
        dset = grp.create_dataset('NMU',data=self.NMU)
        dset.attrs['title'] = "Number of polar angles for multiple scattering calculation"

        dset = grp.create_dataset('NF',data=self.NF)
        dset.attrs['title'] = "Number of Fourier components for azimuth decomposition"

        dset = grp.create_dataset('NPHI',data=self.NPHI)
        dset.attrs['title'] = "Number of azimuth angles for multiple scattering calculation"

        dset = grp.create_dataset('NDUST',data=self.NDUST)
        dset.attrs['title'] = "Number of aerosol populations in atmosphere"

        dset = grp.create_dataset('NWAVE',data=self.NWAVE)
        dset.attrs['title'] = "Number of spectral points"

        if self.NDUST>0:  #There are aerosols in the atmosphere

            dset = grp.create_dataset('WAVE',data=self.WAVE)
            dset.attrs['title'] = "Spectral array"
            if self.ISPACE==WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um'

            dset = grp.create_dataset('KEXT',data=self.KEXT)
            dset.attrs['title'] = "Extinction coefficient"
            dset.attrs['units'] = "cm2"
            dset.attrs['note'] = "KEXT can be normalised to a certain wavelength if the aerosol profiles are also normalised"
            
            dset = grp.create_dataset('SGLALB',data=self.SGLALB)
            dset.attrs['title'] = "Single scattering albedo"
            dset.attrs['units'] = ""

            if self.ISCAT>0:

                if self.IMIE == AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:  #H-G phase function

                    dset = grp.create_dataset('G1',data=self.G1)
                    dset.attrs['title'] = "Assymmetry parameter of first Henyey-Greenstein function"
                    dset.attrs['units'] = ""

                    dset = grp.create_dataset('G2',data=self.G2)
                    dset.attrs['title'] = "Assymmetry parameter of second Henyey-Greenstein function"
                    dset.attrs['units'] = ""

                    dset = grp.create_dataset('F',data=self.F)
                    dset.attrs['title'] = "Relative contribution from first Henyey-Greenstein function (from 0 to 1)"
                    dset.attrs['units'] = ""

                elif self.IMIE == AerosolPhaseFunctionCalculationMode.MIE_THEORY:  #Explicit phase function

                    dset = grp.create_dataset('NTHETA',data=self.NTHETA)
                    dset.attrs['title'] = "Number of angles to define phase function"

                    dset = grp.create_dataset('THETA',data=self.THETA)
                    dset.attrs['title'] = "Angles to define phase function"
                    dset.attrs['units'] = "degrees"

                    dset = grp.create_dataset('PHASE',data=self.PHASE)
                    dset.attrs['title'] = "Phase function of each aerosol population"
                    dset.attrs['units'] = ""

                elif self.IMIE == AerosolPhaseFunctionCalculationMode.LEGENDRE_POLYNOMIALS:  #Phase function from Legendre polynomials

                    dset = grp.create_dataset('NLPOL',data=self.NLPOL)
                    dset.attrs['title'] = "Number of Legendre coefficients to define phase function"

                    dset = grp.create_dataset('WLPOL',data=self.WLPOL)
                    dset.attrs['title'] = "Weights of the Legendre coefficients to define phase function"
                    dset.attrs['units'] = ""

        f.close()

    ###########################################################################################################################

    def read_hdf5(self,runname):
        """
        Read the Scatter properties from an HDF5 file
        """

        import h5py

        f = h5py.File(runname+'.h5','r')

        #Checking if Surface exists
        e = "/Scatter" in f
        if e==False:
            raise ValueError('error :: Scatter is not defined in HDF5 file')
        else:

            self.NDUST = h5py_helper.retrieve_data(f, 'Scatter/NDUST', np.int32)
            self.ISPACE = h5py_helper.retrieve_data(f, 'Scatter/ISPACE', lambda x:  WaveUnit(np.int32(x)))
            self.ISCAT = h5py_helper.retrieve_data(f, 'Scatter/ISCAT', lambda x:  ScatteringCalculationMode(np.int32(x)))
            self.IRAY = h5py_helper.retrieve_data(f, 'Scatter/IRAY', lambda x:  RayleighScatteringMode(np.int32(x)))
            self.IMIE = h5py_helper.retrieve_data(f, 'Scatter/IMIE', lambda x:  AerosolPhaseFunctionCalculationMode(np.int32(x)))
            self.NMU = h5py_helper.retrieve_data(f, 'Scatter/NMU', np.int32)
            self.NF = h5py_helper.retrieve_data(f, 'Scatter/NF', np.int32)
            self.NPHI = h5py_helper.retrieve_data(f, 'Scatter/NPHI', np.int32)
            self.NWAVE = h5py_helper.retrieve_data(f, 'Scatter/NWAVE', np.int32)
            
            if self.NDUST>0:

                self.WAVE = h5py_helper.retrieve_data(f, 'Scatter/WAVE', np.array)
                self.KEXT = h5py_helper.retrieve_data(f, 'Scatter/KEXT', np.array)
                self.SGLALB = h5py_helper.retrieve_data(f, 'Scatter/SGLALB', np.array)
                self.KSCA = self.SGLALB * self.KEXT
                self.KABS = self.KEXT - self.KSCA

                if self.ISCAT>0:

                    if self.IMIE == AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:  #H-G phase function
                        self.G1 = h5py_helper.retrieve_data(f, 'Scatter/G1', np.array)
                        self.G2 = h5py_helper.retrieve_data(f, 'Scatter/G2', np.array)
                        self.F = h5py_helper.retrieve_data(f, 'Scatter/F', np.array)
                    elif self.IMIE == AerosolPhaseFunctionCalculationMode.MIE_THEORY:  #Explicit phase function
                        self.NTHETA = h5py_helper.retrieve_data(f, 'Scatter/NTHETA', np.int32)
                        self.THETA = h5py_helper.retrieve_data(f, 'Scatter/THETA', np.array)
                        self.PHASE = h5py_helper.retrieve_data(f, 'Scatter/PHASE', np.array)
                    elif self.IMIE == AerosolPhaseFunctionCalculationMode.LEGENDRE_POLYNOMIALS:  #Phase function from Legendre polynomials
                        self.NLPOL = h5py_helper.retrieve_data(f, 'Scatter/NLPOL', np.int32)
                        self.WLPOL = h5py_helper.retrieve_data(f, 'Scatter/WLPOL', np.array)

        self.calc_GAUSS_LOBATTO()

        f.close()
        
    ###########################################################################################################################

    def initialise_arrays(self,NDUST,NWAVE,NTHETA,NLPOL=100):
        """
        Initialise arrays for storing the scattering properties of the aerosols
        """

        self.NDUST = NDUST
        self.NWAVE = NWAVE
        self.NTHETA = NTHETA
        self.WAVE = np.zeros(self.NWAVE)
        self.KEXT = np.zeros((self.NWAVE,self.NDUST))
        self.KSCA = np.zeros((self.NWAVE,self.NDUST))
        self.KABS = np.zeros((self.NWAVE,self.NDUST))
        self.SGLALB = np.zeros((self.NWAVE,self.NDUST))
        
        if self.IMIE == AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:
            self.G1 = np.zeros((self.NWAVE,self.NDUST))
            self.G2 = np.zeros((self.NWAVE,self.NDUST))
            self.F = np.zeros((self.NWAVE,self.NDUST))
        elif self.IMIE == AerosolPhaseFunctionCalculationMode.MIE_THEORY:
            self.PHASE = np.zeros((self.NWAVE,self.NTHETA,self.NDUST))
        elif self.IMIE == AerosolPhaseFunctionCalculationMode.LEGENDRE_POLYNOMIALS:
            self.NLPOL = NLPOL
            self.WLPOL = np.zeros((self.NWAVE,self.NLPOL,self.NDUST))
            
    ###########################################################################################################################

    def calc_GAUSS_LOBATTO(self):
        """
        Calculate the Gauss-Lobatto quadrature points and weights.
        """

        nzen = 2*self.NMU    #The gauss_lobatto function calculates both positive and negative angles, and Nemesis just uses the posiive
        x,w = gauss_lobatto(nzen,n_digits=12)
        self.MU = np.array(x[self.NMU:nzen],dtype='float64')
        self.WTMU = np.array(w[self.NMU:nzen],dtype='float64')
        
    ###########################################################################################################################

    def read_xsc(self,runname,MakePlot=False):
        """
        Read the aerosol properties from the .xsc file
        """

        from archnemesis.Files import file_lines

        #reading number of lines in file
        nlines = file_lines(runname+'.xsc')
        nwave = int((nlines-1)/ 2)

        #Reading file
        f = open(runname+'.xsc','r')
    
        s = f.readline().split()
        naero = int(s[0])
    
        wave = np.zeros([nwave])
        ext_coeff = np.zeros([nwave,naero])
        sglalb = np.zeros([nwave,naero])
        for i in range(nwave):
            s = f.readline().split()
            wave[i] = float(s[0])
            for j in range(naero):
                ext_coeff[i,j] = float(s[j+1])
            s = f.readline().split()
            for j in range(naero):
                sglalb[i,j] = float(s[j])

        f.close()

        self.NDUST = naero
        self.NWAVE = nwave
        self.WAVE = wave
        self.KEXT = ext_coeff
        self.SGLALB = sglalb
        self.KSCA = self.SGLALB * self.KEXT
        self.KABS = self.KEXT - self.KSCA

        if MakePlot==True:

            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6))

            for i in range(self.NDUST):

                ax1.plot(self.WAVE,self.KEXT[:,i],label='Dust population '+str(i+1))
                ax2.plot(self.WAVE,self.SGLALB[:,i])

            ax1.legend()
            ax1.grid()
            ax2.grid()
            ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax2.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel('k$_{ext}$ (cm$^2$)')
            ax2.set_ylabel('Single scattering albedo')

            plt.tight_layout()
            plt.show()

    ###########################################################################################################################

    def write_xsc(self,runname,MakePlot=False):
        """
        Write the aerosol scattering and absorving properties to the .xsc file
        """

        f = open(runname+'.xsc','w')
        f.write('%i \n' % (self.NDUST))

        for i in range(self.NWAVE):
            str1 = str('{0:7.6f}'.format(self.WAVE[i]))
            str2 = ''
            for j in range(self.NDUST):
                str1 = str1+'\t'+str('{0:7.6e}'.format(self.KEXT[i,j]))
                str2 = str2+'\t'+str('{0:7.6f}'.format(self.SGLALB[i,j]))

            f.write(str1+'\n')
            f.write(str2+'\n')

        f.close()
        
    ###########################################################################################################################
    
    def read_hgphase(self,NDUST=None):
        """
        Read the Henyey-Greenstein phase function parameters stored in the hgphaseN.dat files
        """
       
        from archnemesis.Files import file_lines
       
        if NDUST!=None:
            self.NDUST = NDUST

        #Getting the number of wave points
        nwave = file_lines('hgphase1.dat')
        self.NWAVE = nwave

        wave = np.zeros(nwave)
        g1 = np.zeros((self.NWAVE,self.NDUST))
        g2 = np.zeros((self.NWAVE,self.NDUST))
        fr = np.zeros((self.NWAVE,self.NDUST))

        for IDUST in range(self.NDUST):

            filename = 'hgphase'+str(IDUST+1)+'.dat'

            f = open(filename,'r')
            for j in range(self.NWAVE):
                s = f.readline().split()
                wave[j] = float(s[0])
                fr[j,IDUST] = float(s[1])
                g1[j,IDUST] = float(s[2])
                g2[j,IDUST] = float(s[3])
            f.close()

        self.WAVE = wave
        self.G1 = np.array(g1,dtype='float64')
        self.G2 =  np.array(g2,dtype='float64')
        self.F =  np.array(fr,dtype='float64')
        
    ###########################################################################################################################

    def write_hgphase(self):
        """
        Write the Henyey-Greenstein phase function parameters into the hgphaseN.dat files
        """

        for IDUST in range(self.NDUST):

            filename = 'hgphase'+str(IDUST+1)+'.dat'

            f = open(filename,'w')
            for j in range(self.NWAVE):

                f.write('%10.7f \t %10.7f \t %10.7f \t %10.7f \n' % (self.WAVE[j],self.F[j,IDUST],self.G1[j,IDUST],self.G2[j,IDUST]))

            f.close()
            
    ###########################################################################################################################

    def calc_hgphase(self,Theta):
        
        """
        Calculate the phase function at Theta angles given the double Henyey-Greenstein parameters
        @param Theta: 1D array or real scalar
            Scattering angle (degrees)
        """

        if np.isscalar(Theta)==True:
            ntheta = 1
            Thetax = [Theta]
        else:
            Thetax = Theta

        #Re-arranging the size of Thetax to be (NTHETA,NWAVE,NDUST)
        Thetax = np.repeat(Thetax[:,np.newaxis],self.NWAVE,axis=1)
        Thetax = np.repeat(Thetax[:,:,np.newaxis],self.NDUST,axis=2)

        t1 = (1.-self.G1**2.)/(1. - 2.*self.G1*np.cos(Thetax/180.*np.pi) + self.G1**2.)**1.5
        t2 = (1.-self.G2**2.)/(1. - 2.*self.G2*np.cos(Thetax/180.*np.pi) + self.G2**2.)**1.5
        
        phase = self.F * t1 + (1.0 - self.F) * t2
        
        #The formula as is for now is normalised such that the integral over 4pi steradians is 4pi
        #In NEMESIS we need the phase function to be normalised to 1.
        phase = phase / (4.0*np.pi)
        
        phase = np.transpose(phase,axes=[1,0,2])
        
        return phase
    
    ###########################################################################################################################

    def interp_phase(self,Theta):
        """
        Interpolate the phase function at Theta angles fiven the phase function in the Scatter class

        Input
        ______

        @param Theta: 1D array
            Scattering angle (degrees)


        Output
        _______

        @param phase(NWAVE,NTHETA,NDUST) : 3D array
            Phase function interpolated at the correct Theta angles

        """

        from scipy.interpolate import interp1d

        s = interp1d(self.THETA,self.PHASE,axis=1)
        phase = s(Theta)

        return phase
    
    ###########################################################################################################################

    def calc_phase(self,Theta,Wave):
        """
        Calculate the phase function of each aerosol type at a given  scattering angle Theta and a given set of Wavelengths/Wavenumbers
        If IMIE=0 in the Scatter class, then the phase function is calculated using the Henyey-Greenstein parameters.
        If IMIE=1 in the Scatter class, then the phase function is interpolated from the values stored in the PHASE array
        If IMIE=2 in the Scatter class, then the phase function is calculated using Legendre Polynomials

        Input
        ______

        @param Theta: real or 1D array
            Scattering angle (degrees)
        @param Wave: 1D array
            Wavelengths (um) or wavenumbers (cm-1) ; It must be the same units as given by the ISPACE

        Outputs
        ________

        @param phase(NWAVE,NTHETA,NDUST) : 3D array
            Phase function at each wavelength, angle and for each aerosol type

        """

        from scipy.interpolate import interp1d

        nwave = len(Wave)

        #Check if the wavelengths are within the required spectral range
        rel_tolerance = 1.0e-6
        
        if (self.WAVE.min() > (1. + rel_tolerance) * Wave.min()) or (self.WAVE.max() < (1. - rel_tolerance) * Wave.max()):
            raise ValueError(
                        f"Aerosol wavelength range [{self.WAVE.min()}, {self.WAVE.max()}] does not fully cover "
                        f"the calculation wavelength range [{Wave.min()}, {Wave.max()}] "
                        )


        if np.isscalar(Theta)==True:
            Thetax = [Theta]
        else:
            Thetax = Theta

        ntheta = len(Thetax)

        phase2 = np.zeros((nwave,ntheta,self.NDUST))
        
        if self.IMIE == AerosolPhaseFunctionCalculationMode.HENYEY_GREENSTEIN:
            
            #Calculating the phase function at the wavelengths defined in the Scatter class
            phase1 = self.calc_hgphase(Thetax)

        elif self.IMIE == AerosolPhaseFunctionCalculationMode.MIE_THEORY:

            #Interpolate the phase function to the Scattering angle at the wavelengths defined in the Scatter class
            phase1 = self.interp_phase(Thetax)

        elif self.IMIE == AerosolPhaseFunctionCalculationMode.LEGENDRE_POLYNOMIALS:

            #Calculating the phase function at the wavelengths defined in the Scatter class
            #using the Legendre polynomials
            phase1 = self.calc_lpphase(Thetax)

        else:
            raise ValueError('error :: IMIE value not valid in Scatter class')


        #Interpolating the phase function to the wavelengths defined in Wave
        s = interp1d(self.WAVE,phase1,axis=0,bounds_error=False,fill_value='extrapolate')
        phase = s(Wave)

        return phase
    
    ###########################################################################################################################

    def calc_phase_ray(self,Theta):
        """
        Calculate the phase function of Rayleigh scattering at a given scattering angle (Dipole scattering)

        Input
        ______

        @param Theta: real or 1D array
            Scattering angle (degrees)

        Outputs
        ________

        @param phase(NTHETA) : 1D array
            Phase function at each angle

        """

        phase = 0.75 * ( 1.0 + np.cos(Theta/180.*np.pi) * np.cos(Theta/180.*np.pi) )
        
        #The formula as is for now is normalised such that the integral over 4pi steradians is 4pi
        #In NEMESIS we need the phase function to be normalised to 1.
        phase = phase / (4.0*np.pi)

        return phase
    
    ###########################################################################################################################

    def read_phase(self,NDUST=None):
        """
        Read a file with the format of the PHASE*.DAT using the format required by NEMESIS

        Optional inputs
        ----------------

        @NDUST: int
            If included, then several files from 1 to NDUST will be read with the name format being PHASE*.DAT
        """

        if NDUST!=None:
            self.NDUST = NDUST

        mwave = 5000
        mtheta = 361
        kext = np.zeros((mwave,self.NDUST))
        sglalb = np.zeros((mwave,self.NDUST))
        phase = np.zeros((mwave,mtheta,self.NDUST))

        for IDUST in range(self.NDUST):

            filename = 'PHASE'+str(IDUST+1)+'.DAT'         

            f = open(filename,'r')
            s = f.read()[0:1000].split()
            f.close()
            #Getting the spectral unit
            if s[0]=='wavenumber':
                self.ISPACE = WaveUnit.Wavenumber_cm
            elif s[0]=='wavelength':
                self.ISPACE = WaveUnit.Wavelength_um

            #Calculating the wave array
            vmin = float(s[1])
            vmax = float(s[2])
            delv = float(s[3])
            nwave = int(s[4])
            nphase = int(s[5])
            wave = np.linspace(vmin,vmax,nwave)

            #Reading the rest of the information
            f = open(filename,'r')
            s = f.read()[1000:].split()
            f.close()
            i0 = 0
            #Reading the phase angle
            theta = np.zeros(nphase)
            for i in range(nphase):
                theta[i] = s[i0]
                i0 = i0 + 1

            #Reading the data
            wave1 = np.zeros(nwave)
            kext1 = np.zeros(nwave)
            sglalb1 = np.zeros(nwave)
            phase1 = np.zeros((nwave,nphase))
            for i in range(nwave):

                wave1[i]=s[i0]
                i0 = i0 + 1
                kext1[i] = float(s[i0])
                i0 = i0 + 1
                sglalb1[i] = float(s[i0])
                i0 = i0 + 1

                for j in range(nphase):
                    phase1[i,j] = float(s[i0])
                    i0 = i0 + 1

            kext[0:nwave,IDUST] = kext1[:]
            sglalb[0:nwave,IDUST] = sglalb1[:]
            phase[0:nwave,0:nphase,IDUST] = phase1[:,:]
            
        #Filling the parameters in the class based on the information in the files
        self.NWAVE = nwave
        self.NTHETA = nphase
        self.WAVE = wave1
        self.THETA = theta
        self.KEXT = np.zeros((self.NWAVE,self.NDUST))
        self.KSCA = np.zeros((self.NWAVE,self.NDUST))
        self.KABS = np.zeros((self.NWAVE,self.NDUST))
        self.SGLALB = np.zeros((self.NWAVE,self.NDUST))
        self.PHASE = np.zeros((self.NWAVE,self.NTHETA,self.NDUST)) 

        self.KEXT[:,:] = kext[0:self.NWAVE,0:self.NDUST]
        self.SGLALB[:,:] = sglalb[0:self.NWAVE,0:self.NDUST]
        self.KSCA[:,:] = self.KEXT[:,:] * self.SGLALB[:,:]
        self.KABS[:,:] = self.KEXT[:,:] - self.KSCA[:,:]
        self.PHASE[:,:,:] = phase[0:self.NWAVE,0:self.NTHETA,0:self.NDUST]
        
    ###########################################################################################################################

    def write_phase(self,IDUST):
        """
        Write a file with the format of the PHASE*.DAT using the format required by NEMESIS

        Inputs
        ----------------

        @IDUST: int
            Aerosol population whose properties will be written in the PHASE.DAT file
        """

        f = open('PHASE'+str(IDUST+1)+'.DAT','w')
        
        #First buffer
        if self.ISPACE==WaveUnit.Wavenumber_cm:
            wavetype='wavenumber'
        elif self.ISPACE==WaveUnit.Wavelength_um:
            wavetype='wavelength'

        str1 = "{:<512}".format(' %s  %8.2f  %8.2f  %8.4f  %4i  %4i' % (wavetype,self.WAVE.min(),self.WAVE.max(),self.WAVE[1]-self.WAVE[0],self.NWAVE,self.NTHETA))
 
        #Second buffer
        comment = 'Mie scattering  - Particle size distribution not known'
        str2 = "{:<512}".format(' %s' % (comment)  )

        #Third buffer
        strxx = ''
        for i in range(self.NTHETA):
            strx = ' %8.3f' % (self.THETA[i])
            strxx = strxx+strx

        str3 = "{:<512}".format(strxx)
        if len(str3)>512:
            raise ValueError('error writing PHASEN.DAT file :: File format does not support so many scattering angles (NTHETA)')

        #Fourth buffer
        str4 = ''
        for i in range(self.NWAVE):
            strxx = ''
            strx1 = ' %8.6f %12.5e %12.5e' % (self.WAVE[i],self.KEXT[i,IDUST],self.SGLALB[i,IDUST])
            strx2 = ''
            for j in range(self.NTHETA):
                strx2 = strx2+' %10.4f' % (self.PHASE[i,j,IDUST])
            strxx = "{:<512}".format(strx1+strx2)
            if len(strxx)>512:
                raise ValueError('error while writing PHASEN.DAT :: File format does not support so many scattering angles (NTHETA)')
            str4=str4+strxx

        f.write(str1+str2+str3+str4)
        f.close()
        
    ###########################################################################################################################

    def read_lpphase(self,NDUST=None):
        """
        Read the weights of the Legendre polynomials used to model the phase function (stored in the lpphaseN.dat files)
        These files are assumed to be pickle files with the correct format
        """

        import pickle

        if NDUST!=None:
            self.NDUST = NDUST

        #Reading the first file to read dimensions of the data
        filen = open('lpphase1.dat','rb')
        wave = pickle.load(filen)
        wlegpol = pickle.load(filen)

        self.NWAVE = len(wave)
        self.NLPOL = wlegpol.shape[1]

        wlpol = np.zeros((self.NWAVE,self.NLPOL,self.NDUST))
        for IDUST in range(self.NDUST):
            filen = open('lpphase'+str(IDUST+1)+'.dat','rb')
            wave = pickle.load(filen)
            wlegpol = pickle.load(filen)            
            wlpol[:,:,IDUST] = wlegpol[:,:]

        self.WAVE = wave
        self.WLPOL = wlpol
        
    ###########################################################################################################################

    def write_lpphase(self):
        """
        Writing the coefficients of the Legendre polynomials into a pickle file
        """
        
        if not hasattr(self,"WPOL"):
            _lgr.warning(f'Not writing lpphase.dat (Solar Occultation) file as self.WPOL attribute is not present')
            return

        import pickle

        #Saving the Legendre polynomials as a pickle file for each particle size
        for i in range(self.NDUST):
            runname = 'lpphase'+str(i+1)+'.dat'
            filehandler = open(runname,"wb")
            pickle.dump(self.WAVE,filehandler,pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.WLPOL[:,:,i],filehandler,pickle.HIGHEST_PROTOCOL)
            filehandler.close()
            
    ###########################################################################################################################

    def calc_lpphase(self,Theta):
        """
        Calculate the phase function at Theta angles given the weights of the Legendre polynomials
        
        Inputs
        ______
        
        Theta(ntheta) :: Scattering angle (degrees)
        
        Outputs
        _______
        
        phase(nwave,ntheta,ndust) :: Phase function evaluated for each wavelength, angle and aerosol population
        """

        if np.isscalar(Theta)==True:
            ntheta = 1
            Thetax = np.array([Theta])
        else:
            Thetax = Theta

        ntheta = len(Thetax)
        phase = np.zeros((self.NWAVE,ntheta,self.NDUST))

        cosTheta = np.cos(Thetax / 180.0 * np.pi)
        
        for IL in range(self.NLPOL):
            P_n = legendre_p(IL, cosTheta)
            for IDUST in range(self.NDUST):
                for IWAVE in range(self.NWAVE):
                    phase[IWAVE, :, IDUST] += P_n * self.WLPOL[IWAVE, IL, IDUST]
                    
        return phase
    
    ###########################################################################################################################
    
    def check_phase_norm(self):
        """
        Function to quickly check whether the phase function is correctly normalised to 1 
        """
        
        #Defining angles
        Theta = np.linspace(0.,180.,721)
        
        #Calculating phase function
        phase = self.calc_phase(Theta,self.WAVE)
        
        #Integrating phase function
        total = 2 * np.pi * np.trapz(phase, -np.cos(Theta/180.*np.pi),axis=1)
        
        _lgr.info('Normalisation of phase function should be 1.0')
        _lgr.info(f'Minimum integral of phase function is  {(total.min())}')
        _lgr.info(f'Maximum integral of phase function is  {(total.max())}')
        
    ###########################################################################################################################
    
    def check_fourier_decomposition(self,idust=0):
        """
        Function to quickly check whether the phase function is accurately described 
        with the specified number of Fourier components (NF)
        """
        
        from archnemesis.Multiple_Scattering_Core import check_fourier_decomposition
        
        #Defining angles
        Ntheta = 361
        Theta = np.linspace(0.,180.,Ntheta)
        
        phase = self.calc_phase(Theta, self.WAVE)  #(NWAVE,NTHETA,NDUST)
        
        iwave = 0 #Generally the lowerst wavelength will suffer more
        phase_reconstructed = check_fourier_decomposition(Theta,phase[iwave,:,idust],self.NF,self.NPHI)
        
        fig,ax1 = plt.subplots(1,1,figsize=(9,3))
        
        ax1.plot(Theta,phase[iwave,:,idust],label='Original')
        ax1.plot(Theta,phase_reconstructed,label='Reconstructed')
        ax1.set_facecolor('lightgray')
        ax1.set_yscale('log')
        
        ax1.grid()
        ax1.set_xlabel(r'Scattering angle ($^\circ$)')
        ax1.set_ylabel('Phase function')
        
    ###########################################################################################################################
    
    def read_refind(self,aeroID):
        """
        Read a file of the refractive index from the NEMESIS aerosol database 

        Inputs
        ________

        aeroID :: ID of the aerosol type

        Outputs
        ________

        WAVER(NWAVER) :: Wavelength (um) array
        REFIND_REAL(NWAVER) :: Real part of the refractive index
        REFIND_IM(NWAVER) :: Imaginary part of the refractive index
        """

        from archnemesis import aerosol_info

        wave_aero = aerosol_info[str(aeroID)]["wave"]
        refind_real_aero1 = aerosol_info[str(aeroID)]["refind_real"]
        refind_im_aero1 = aerosol_info[str(aeroID)]["refind_im"]

        self.NWAVER = len(wave_aero)
        self.WAVER = wave_aero
        self.REFIND_REAL = refind_real_aero1
        self.REFIND_IM = refind_im_aero1
        
    ###########################################################################################################################

    def read_refind_file(self,filename,MakePlot=False):
        """
        Read a file of the refractive index using the format required by NEMESIS

        @filename: str
            Name of the file where the data is stored

        Outputs
        ________

        @ISPACE: int
            Flag indicating whether the refractive index is expressed in Wavenumber (0) of Wavelength (1)
        @NWAVER: int
            Number of spectral point
        @WAVER: 1D array
            Wavenumber (cm-1) / Wavelength (um) array
        @REFIND_REAL: 1D array
            Real part of the refractive index
        @REFIND_IM: 1D array
            Imaginary part of the refractive index
        """
        
        from archnemesis.Files import file_lines

        nlines = file_lines(filename)

        #Reading buffer
        ibuff = 0
        with open(filename,'r') as fsol:
            for curline in fsol:
                if curline.startswith("#"):
                    ibuff = ibuff + 1
                else:
                    break

        #Reading buffer
        f = open(filename,'r')
        for i in range(ibuff):
            s = f.readline().split()

        #Reading file
        s = f.readline().split()
        nwave = int(s[0])
        s = f.readline().split()
        ispace = int(s[0])

        wave = np.zeros(nwave)
        refind_real = np.zeros(nwave)
        refind_im = np.zeros(nwave)
        for i in range(nwave):
            s = f.readline().split()
            wave[i] = float(s[0])
            refind_real[i] = float(s[1])
            refind_im[i] = float(s[2])

        f.close()

        return ispace,nwave,wave,refind_real,refind_im
    
    ###########################################################################################################################

    def makephase(self, idust, iscat, pars, rs=None):
        """
        Function to calculate the extinction coefficient, single scattering albedo and phase functions
        for different aerosol populations using Mie Theory.

        Inputs
        ________

        @param idust: int
            Integer indicating to which aerosol population this calculation corresponds to (from 0 to NDUST-1)
            
        @param iscat: int
            Flag indicating the particle size distribution
                1 :: Standard gamma distribution - n(r) = r ** alpha * exp(-r/(a*b))
                2 :: Log-normal distribution - n(r) = sqrt(2*pi) / (r*sigma) * exp(-0.5 * ((log(r) - log(mu)) / sigma)**2)
                3 :: MCS Modified gamma distribution - n(r) = r * a * exp(-b * r**c)
                4 :: Single particle size - r0 in microns
                5 :: Isotropic phase function
                6 :: Double Henyey-Greenstein scattering phase function
                7 :: Dipole scattering
                
        @param pars: int
            Particle size distribution parameters
                iscat == 1 :: pars(0) == a
                              pars(1) == b
                              pars(2) == alpha
                iscat == 2 :: pars(0) == mu
                              pars(1) == sigma
                iscat == 3 :: pars(0) == a
                              pars(1) == b
                              pars(2) == c
                iscat == 4 :: pars(0) == r0
                iscat == 5 :: pars(0) == empty or dummy value
                iscat == 6 :: pars(0) == f
                              pars(1) == g1
                              pars(2) == g2
                iscat == 7 :: pars(0) == r0
             

        Optional inputs
        ________________
        
        rs(3) :: Limits for particle size distribution and step size (in microns)
                 If None, they are automatically calculated and the integration
                 is terminated when N * QS < 10**-6 * Max

        Outputs
        ________

        Updated KEXT,KABS,KSCA in class.
        If IMIE=0 :: (Henyey Greenstein) Updated G1,G2 and F in the class
        If IMIE=1 :: (Explicit phase function) Updated PHASE in the class
        If IMIE=2 :: (Legendre polynomials) Updated WLPOL in the class
        """
        
        from numpy.polynomial import legendre as L
        
        #Interpolating the refractive indices to the calculation wavelengths
        if self.ISPACE==WaveUnit.Wavenumber_cm:
            wavel = 1./self.WAVE * 1.0e4
        elif self.ISPACE==WaveUnit.Wavelength_um:
            wavel = self.WAVE
            
        iord = np.argsort(wavel)
        wavel = wavel[iord]
        
        #Interpolating the refractive index to the relevant wavelengths
        if( (iscat!=5) & (iscat!=6) ): #These are the options that use Mie theory and therefore require a refractive index
            
            refind_r = np.interp(wavel,self.WAVER,self.REFIND_REAL)
            refind_im = np.interp(wavel,self.WAVER,self.REFIND_IM)
        
            refindx = np.zeros((len(wavel),2))
            refindx[:,0] = refind_r
            refindx[:,1] = refind_im
            
        else:
            
            refindx = np.zeros((len(wavel),2))
        
        #Defining the particle size distribution sampling
        if rs is None:
            rs = np.zeros(3)
            rs[0] = 0.015 * np.min(wavel)
            rs[1] = 0.
            rs[2] = rs[0]
        else:
            rs = np.array(rs)
            
        #If single particle size
        if iscat == 4:
            rs = np.zeros(3)
            rs[:] = pars[0]
            
        #For theta we only keep the values below 90, then we update the array with the new theta
        ihi = np.where(self.THETA>90.)
        ilo = np.where(self.THETA<=90.)
        theta = np.zeros(self.NTHETA)
        theta[ilo] = self.THETA[ilo]
        theta[ihi] = 180. - self.THETA[ihi]
        theta = np.unique(theta)
        theta = np.sort(theta)
        
        pardist = np.zeros(3)
        pardist[0:len(pars)] = pars[:]
        xscat, xext, thetax, phas = makephase(wavel, iscat, pardist, rs, refindx, theta)
        
        #Normalising the phase function to 1
        phas /= 4. * np.pi
        phas[iord,:] = phas[:,:]
        
        #Checking that the normalisation is correct (should be 1)    
        total = 2 * np.pi * np.trapz(phas, -np.cos(self.THETA/180.*np.pi),axis=1)
        _lgr.info('Normalisation of phase function should be 1.0')
        _lgr.info(f'Minimum integral of phase function is {total.min()}')
        _lgr.info(f'Maximum integral of phase function is {total.max()}')
        
        #Updating the parameters in the class
        #########################################################################################
        
        #Updating the parameters in the class
        self.KEXT[iord,idust] = xext[:]
        self.KSCA[iord,idust] = xscat[:]
        self.KABS[iord,idust] = xext[:] - xscat[:]
        self.SGLALB[iord,idust] = xscat[:]/xext[:]
        
        #Depending of IMIE we may want to define the phase function in different formats
        if self.IMIE==0: #Phase function defined as double Henyey-Greenstein function
        
            #Fitting a double HG function to the phase function
            #subfithgm requires the phase function normalised at 4pi
            f, g1, g2, rms = subfithgm(thetax, phas*4.*np.pi)
            
            self.F[:,idust] = f[:]
            self.G1[:,idust] = g1[:]
            self.G2[:,idust] = g2[:]

        elif self.IMIE==1: #Phase function defined explicitly
            
            #Finding the values of the phase function at our desired THETA
            phas2 = np.zeros((self.NWAVE,self.NTHETA))
            for it in range(self.NTHETA):
                iss = np.where(thetax==self.THETA[it])[0][0]
                phas2[:,it] = phas[:,iss]
            
            self.PHASE[:,:,idust] = phas2[:,:]
            
        elif self.IMIE==2: #Phase function defined with the weights of Legendre polynomials
            
            for iwave in range(self.NWAVE):
                self.WLPOL[iwave,:,idust], stats = L.legfit(np.cos(thetax/180.*np.pi),phas[iwave,:],self.NLPOL-1,full=True)
                



##############################################################################################################
##############################################################################################################
#                                        MIE SCATTERING CALCULATIONS
##############################################################################################################
##############################################################################################################

from numba import njit

@njit(fastmath=True, error_model='numpy')
def dmie(x, rfr, rfi, thetd, jx):
    '''
    Subroutine for computing the parameters of the electromagnetic radiation
    scattered by a sphere.
    
    Inputs
    ------
    
    x :: Size parameter of the scattering sphere (2*pi*r/wavel)
    rfr :: Real part of the refractive index of material of the sphere
    rfi :: Imaginary part of the refractive index of material of the sphere
    thetd(ntheta) :: Values of the scattering angle in degrees. Note that 
                     output for direction 180-theta is automatically returned 
                     so thetd must be lower than 90
    
    Outputs
    --------
    
    qext :: Efficiency factor for extinction
    qscat :: Efficiency factor for scattering
    ctbrqs :: Average cos(theta)*qscat
    eltrmx(4,ntheta,2) :: Three dimensional array representing the four 
                          elements of the transformation matrix defined
                          by Van de Hulst.
                          
                          (1,J,1) M2 for theta, i.e., S2S2*
                          (2,J,1) M1 for theta, i.e., S1S1*
                          (3,J,1) S21 for theta
                          (4,J,1) D21 for theta
                          (1,J,2) M2 for 180-theta
                          (2,J,2) M1 for 180-theta
                          (3,J,2) S21 for 180-theta
                          (4,J,2) D21 for 180theta
                              
    '''
    
    jx = len(thetd)
    
    ncap = 30000
    acap = np.zeros(ncap, dtype=np.complex128)
    eltrmx = np.zeros((4, jx, 2))
    pi = np.zeros((3, jx))
    tau = np.zeros((3, jx))
    cstht = np.zeros(jx)
    si2tht = np.zeros(jx)
    t = np.zeros(5)
    taa = np.zeros(2)
    tab = np.zeros(2)
    tb = np.zeros(2)
    tc = np.zeros(2)
    td = np.zeros(2)
    te = np.zeros(2)
    rf = complex(rfr, -rfi)
    rrf = 1.0 / rf
    rx = 1.0 / x
    rrfx = rrf * rx
    t[0] = x * x * (rfr * rfr + rfi * rfi)
    t[0] = np.sqrt(t[0])
    nmx1 = int(1.1 * t[0])
    if not nmx1 < ncap - 1:
        print('LIMIT FOR ACAP IS NOT ENOUGH')
        qext = -1
        return
    nmx2 = int(t[0])
    if not nmx1 > 150:
        nmx1 = 150
        nmx2 = 135
    acap[nmx1] = complex(0)
    for n in range(1, nmx1 + 1):
        nn = nmx1 - n + 1
        acap[nn] = (nn + 1) * rrfx - 1.0 / ((nn + 1) * rrfx + acap[nn + 1])
    for i in range(1, len(acap)):
        acap[i - 1] = acap[i]
    for j in range(jx):
        if thetd[j] < 0.0:
            thetd[j] = abs(thetd[j])
        if thetd[j] == 0.0:
            cstht[j] = 1.0
            si2tht[j] = 0.0
        if thetd[j] > 0.0:
            if thetd[j] < 90.0:
                t[0] = np.pi * thetd[j] / 180.0
                cstht[j] = np.cos(t[0])
                si2tht[j] = 1.0 - cstht[j] * cstht[j]
            elif thetd[j] == 90.0:
                cstht[j] = 0.0
                si2tht[j] = 1.0
            else:
                print('THE VALUE OF THE SCATTERING ANGLE IS GREATER THAN 90.0')
                return
    for j in range(jx):
        pi[0, j] = 0.0
        pi[1, j] = 1.0
        tau[0, j] = 0.0
        tau[1, j] = cstht[j]
    t[0] = np.cos(x)
    t[1] = np.sin(x)
    wm1 = complex(t[0], -t[1])
    wfn1 = complex(t[1], t[0])
    wfn2 = rx * wfn1 - wm1
    tc1 = acap[0] * rrf + rx
    tc2 = acap[0] * rf + rx
    taa[0], taa[1] = (wfn1.real, wfn1.imag)
    tab[0], tab[1] = (wfn2.real, wfn2.imag)
    fna = (tc1 * tab[0] - taa[0]) / (tc1 * wfn2 - wfn1)
    fnb = (tc2 * tab[0] - taa[0]) / (tc2 * wfn2 - wfn1)
    fnap = fna
    fnbp = fnb
    t[0] = 1.5
    tb[0], tb[1] = (fna.real, fna.imag)
    tc[0], tc[1] = (fnb.real, fnb.imag)
    tb *= t[0]
    tc *= t[0]
    for j in range(jx):
        eltrmx[0, j, 0] = tb[0] * pi[1, j] + tc[0] * tau[1, j]
        eltrmx[1, j, 0] = tb[1] * pi[1, j] + tc[1] * tau[1, j]
        eltrmx[2, j, 0] = tc[0] * pi[1, j] + tb[0] * tau[1, j]
        eltrmx[3, j, 0] = tc[1] * pi[1, j] + tb[1] * tau[1, j]
        eltrmx[0, j, 1] = tb[0] * pi[1, j] - tc[0] * tau[1, j]
        eltrmx[1, j, 1] = tb[1] * pi[1, j] - tc[1] * tau[1, j]
        eltrmx[2, j, 1] = tc[0] * pi[1, j] - tb[0] * tau[1, j]
        eltrmx[3, j, 1] = tc[1] * pi[1, j] - tb[1] * tau[1, j]
    qext = 2.0 * (tb[0] + tc[0])
    qscat = (tb[0] ** 2 + tb[1] ** 2 + tc[0] ** 2 + tc[1] ** 2) / 0.75
    ctbrqs = 0.0
    n = 2
    while True:
        t[0] = 2 * n - 1
        t[1] = n - 1
        t[2] = 2 * n + 1
        for j in range(jx):
            pi[2, j] = (t[0] * pi[1, j] * cstht[j] - n * pi[0, j]) / t[1]
            tau[2, j] = cstht[j] * (pi[2, j] - pi[0, j]) - t[0] * si2tht[j] * pi[1, j] + tau[0, j]
        wm1 = wfn1
        wfn1 = wfn2
        wfn2 = t[0] * rx * wfn1 - wm1
        taa[0], taa[1] = (wfn1.real, wfn1.imag)
        tab[0], tab[1] = (wfn2.real, wfn2.imag)
        tc1 = acap[n - 1] * rrf + n * rx
        tc2 = acap[n - 1] * rf + n * rx
        fna = (tc1 * tab[0] - taa[0]) / (tc1 * wfn2 - wfn1)
        fnb = (tc2 * tab[0] - taa[0]) / (tc2 * wfn2 - wfn1)
        t[4] = n
        t[3] = t[0] / (t[4] * t[1])
        t[1] = t[1] * (t[4] + 1.0) / t[4]
        tb[0], tb[1] = (fna.real, fna.imag)
        tc[0], tc[1] = (fnb.real, fnb.imag)
        td[0], td[1] = (fnap.real, fnap.imag)
        te[0], te[1] = (fnbp.real, fnbp.imag)
        ctbrqs += t[1] * (td[0] * tb[0] + td[1] * tb[1] + te[0] * tc[0] + te[1] * tc[1]) + t[3] * (td[0] * te[0] + td[1] * te[1])
        qext += t[2] * (tb[0] + tc[0])
        t[3] = tb[0] ** 2 + tc[0] ** 2 + tb[1] ** 2 + tc[1] ** 2
        qscat += t[2] * t[3]
        t[1] = n * (n + 1)
        t[0] = t[2] / t[1]
        k = n // 2 * 2
        for j in range(jx):
            eltrmx[0, j, 0] += t[0] * (tb[0] * pi[2, j] + tc[0] * tau[2, j])
            eltrmx[1, j, 0] += t[0] * (tb[1] * pi[2, j] + tc[1] * tau[2, j])
            eltrmx[2, j, 0] += t[0] * (tc[0] * pi[2, j] + tb[0] * tau[2, j])
            eltrmx[3, j, 0] += t[0] * (tc[1] * pi[2, j] + tb[1] * tau[2, j])
            if k == n:
                eltrmx[0, j, 1] += t[0] * (-tb[0] * pi[2, j] + tc[0] * tau[2, j])
                eltrmx[1, j, 1] += t[0] * (-tb[1] * pi[2, j] + tc[1] * tau[2, j])
                eltrmx[2, j, 1] += t[0] * (-tc[0] * pi[2, j] + tb[0] * tau[2, j])
                eltrmx[3, j, 1] += t[0] * (-tc[1] * pi[2, j] + tb[1] * tau[2, j])
            else:
                eltrmx[0, j, 1] += t[0] * (tb[0] * pi[2, j] - tc[0] * tau[2, j])
                eltrmx[1, j, 1] += t[0] * (tb[1] * pi[2, j] - tc[1] * tau[2, j])
                eltrmx[2, j, 1] += t[0] * (tc[0] * pi[2, j] - tb[0] * tau[2, j])
                eltrmx[3, j, 1] += t[0] * (tc[1] * pi[2, j] - tb[1] * tau[2, j])
        if t[3] < 1e-14:
            break
        n += 1
        for j in range(jx):
            pi[0, j] = pi[1, j]
            pi[1, j] = pi[2, j]
            tau[0, j] = tau[1, j]
            tau[1, j] = tau[2, j]
        fnap = fna
        fnbp = fnb
        if n <= nmx2:
            continue
        else:
            qext = -1
            print('test')
            return
    for j in range(jx):
        for k in range(2):
            t[:4] = eltrmx[:, j, k]
            eltrmx[0, j, k] = t[2] ** 2 + t[3] ** 2
            eltrmx[1, j, k] = t[0] ** 2 + t[1] ** 2
            eltrmx[2, j, k] = t[0] * t[2] + t[1] * t[3]
            eltrmx[3, j, k] = t[1] * t[2] - t[3] * t[0]
    t[0] = 2.0 * rx * rx
    qext *= t[0]
    qscat *= t[0]
    ctbrqs = 2.0 * ctbrqs * t[0]
    return (qext, qscat, ctbrqs, eltrmx)

@njit(fastmath=True, error_model='numpy')
def miescat(xlam, iscat, dsize, rs, refindx, theta):
    '''
    Calculates the phase function, scattering, and extinction
    coefficients at a a given wavelength for a specified size 
    distribution using Mie theory.
    
    Inputs
    ________
    
    xlam :: Wavelength in microns
    iscat :: Type of particle size distribution
                    iscat = 1 : Mie scattering, standard gamma distribution
                    iscat = 2 : Mie scattering, log-normal distribution
                    iscat = 3 : Mie scattering, MCS Modified gamma disitribution
                    iscat = 4 : Mie scattering, single particle size
    dsize(3) :: Parameters describing the particle size distribution
                    iscat = 1 : (0:3)=A,B,ALPHA
                    iscat = 2 : (0:2)=R0,RSIG
                    iscat = 3 : (0:3)=A,B,C
                    iscat = 4 : (0)=R
    rs(3) :: Size integration limits (0:2) and step size (2)
             If rs[1]<rs[0] then integration is terminated when 
             N * QS < 10**-6 * Max
    refindx(2) :: Real and imaginary part of the refractive index
    theta(ntheta) :: Scattering angle (degrees) from 0 to 90, angles >90 degrees
                     will be automatically returned as 180-theta

    
    Outputs
    ________
    
    xscat :: Mean scattering cross section (cm2)
    xext :: Mean extinction cross section (cm2)
    thetax(nphas) :: Array of scattering angles from 0 to 180 degrees
    phas(nphas) :: Phase function (normalised to X)
    '''
    
    ntheta = len(theta)
    
    #Checking whether theta goes from 0 to 90
    for j in range(ntheta):
        if theta[j] < 0 or theta[j] > 90:
            print(' error :: ANGLE <0 OR > 90')
            return
    
    #Checking whether theta=90 exists to calculate size of phase function
    ilim = np.where(theta==90.)[0]
    if len(ilim)==1:
        nphas = int(2 * ntheta - 1)
    else:
        nphas = int(2 * ntheta)

    #Constructing new theta from 0 to 180
    thetax = np.zeros(nphas)
    for i in range(ntheta): #First part from 0 to 90
        thetax[i] = theta[i]
    for i in range(ntheta,nphas): #Second part from 90 to 180
        thetax[i] = 180.0 - thetax[nphas - i - 1]
    
    #Initialising some variables
    pi = np.pi
    pi2 = 1.0 / np.sqrt(2.0 * pi)
    func = np.zeros((4, 2 * ntheta))
    phas0 = np.zeros(2 * ntheta)
    phas = np.zeros(nphas)
    idump = 0
    numflag = False
    
    thetd = np.zeros_like(theta)
    for j in range(ntheta):
        thetd[j] = theta[j]

        
    #Defining size integration parameters
    ##########################################################################

    #Defininf the number of dropsizes that must be calculated
    r1 = rs[0]
    delr = rs[2]
    if rs[1] < rs[0]:
        inr = 1000000001
        cont0 = False
    else:
        inr = 1 + int((rs[1] - rs[0]) / rs[2])
        if inr > 1 and inr % 2 != 0:
            inr += 1
        cont0 = True
        
    nqmax = 0.
    
    #Compute the peaks of the size distribution
    ##########################################################################
    
    if(not cont0):
        rmax = 0.
        if dsize[1] != 0:
            aa = dsize[0]
            bb = dsize[1]
            alpha = 0.0
            cc = 0.0
            if iscat == 0:   #continuous
                rmax = rs[1]
            if iscat == 1:   #Standard gamma distribution
                alpha = dsize[2]
                rmax = alpha * aa * bb
            elif iscat == 2: #Log-normal distribution
                rmax = np.exp(np.log(aa) - bb ** 2)
            elif iscat == 3: #MCS Modified gamma distribution
                cc = dsize[2]
                rmax = (aa / (bb * cc)) ** (1.0 / cc)
                    
    #if iscat == 0: # continuous
    #    r_grid = np.exp(np.linspace(np.log(r1),np.log(rs[1]),len(dsize)))
    #    r_dist_grid = dsize[0]
    #    r_dist = np.interp(np.arange(r1,rs[1]+delr,delr),r_grid,r_dist_grid)
        

    #Calculating of phase function for each dropsize
    #################################################################
    
    kscat = 0.0
    area = 0.0
    volume = 0.0
    kext = 0.0
    anorm = 0.0
    rfr = refindx[0]
    rfi = refindx[1]
    for m in range(inr):
        
        rr = r1 + m * delr
        csratio = -1.0
        if csratio == -1.0:
            #Use homogeneous sphere scattering model unless otherwise specified
            xx = 2.0 * pi * rr / xlam
            qext, qscat, cq, eltrmx = dmie(xx, rfr, rfi, thetd, ntheta)
        else:
            #Use coated sphere scattering model if Maltmieser explicitly specified
            #not currently in use
            print('error :: Mie theory for coated spheres has not yet been implemented')
            return
            
        #Note if there has been an overflow in the Mie calculation.
        if qext < 0.0:
            numflag = True
        
        for j in range(1, ntheta + 1):
            for i in range(4):
                func[i, j - 1] = eltrmx[i, j - 1, 0] if qext >= 0.0 else -999.9
        for j in range(ntheta + 1, nphas + 1):
            for i in range(4):
                func[i, j - 1] = eltrmx[i, nphas - j, 1] if qext >= 0.0 else -999.9
                
        #Compute particle size distributions and check for size cut-off
        #########################################################################
                
        anr = 0.0
        cont = cont0

        if dsize[1] != 0:
            aa = dsize[0]
            bb = dsize[1]
            alpha = 0.0
            cc = 0.0
            if iscat == 1: #Standard gamma distribution
                alpha = dsize[2]
                anr1 = rr ** alpha * np.exp(-rr / (aa * bb))
            elif iscat == 2: #Log-normal distribution
                anr1 = 1. / (rr * bb * np.sqrt(2 * pi)) * np.exp( - (np.log(rr)-np.log(aa))**2. / (2.*bb**2.))
            elif iscat == 3: #MCS Modified standard gamma distribution
                cc = dsize[2]
                anr1 = rr ** aa * np.exp(-bb * rr ** cc)
        else:
            anr1 = 1.

        anr += anr1
        nqmax = max(nqmax, anr1 * qscat)
        if not cont:
            if rr < rmax or anr1 * qscat > 1e-06 * nqmax:
                cont = True


        #Integration over drop size by Simpson's rule
        ######################################################################
        
        #Initialise for integration
        if m % 2 == 0:
            vv = 2.0 * delr / 3.0
        else:
            vv = 4.0 * delr / 3.0
        if m == 0 or m == inr - 1:
            vv = delr / 3.0
        
        if qext >= 0:
            for j in range(1, nphas + 1):
                phas0[j - 1] += 0.5 * anr * vv * (func[0, j - 1] + func[1, j - 1])
            kscat += pi * rr * rr * qscat * anr * vv
            kext += pi * rr * rr * qext * anr * vv
            anorm += anr * vv
            area += pi * rr * rr * anr * vv
            volume += 4.0 * pi * rr * rr * rr * anr * vv / 3.0
            if idump == 1:
                print(rr, anr, pi * rr * rr, 1.3333 * np.pi * rr ** 3)
                
        #Normalise integrations. Cross sections are returned as cm2. Volume returned as cm-3
        if anorm > 0.0:
            xscat = float(kscat / anorm * 1e-08)
            xext = float(kext / anorm * 1e-08)
            area *= 1e-08 / anorm
            volume *= 1e-12 / anorm
        else:
            xscat = 0.0
            xext = 0.0
            kscat = 1.0
        for j in range(1, nphas + 1):
            phas[j - 1] = xlam * xlam * float(phas0[j - 1] / (np.pi * kscat))
        if idump == 1:
            print('Volume (cm3) = ', volume)
            print('area (cm2) = ', area)
        if not cont0 and (not cont):

            return (xscat, xext, thetax, phas)
        
        m+=1
        
    return (xscat, xext, thetax, phas)

@njit(fastmath=True, error_model='numpy')
def makephase(wavel, iscat, dsize, rs, refindx, theta):
    '''
    Calculates the phase function, scattering, and extinction
    coefficients at a a given wavelength for a specified size 
    distribution using Mie theory.
    
    Inputs
    ________
    
    wavel(nwave) :: Wavelength array in microns
    iscat :: Type of particle size distribution
                    iscat = 0 : Continuous
                    iscat = 1 : Mie scattering, standard gamma distribution
                    iscat = 2 : Mie scattering, log-normal distribution
                    iscat = 3 : Mie scattering, MCS Modified gamma disitribution
                    iscat = 4 : Mie scattering, single particle size
                    iscat = 5 : Isotropic scattering
                    iscat = 6 : Henyey-Greenstein scattering
                    iscat = 7 : Dipole scattering
    dsize(3) :: Parameters describing the particle size distribution
                    iscat = 1 : (0:3)=A,B,ALPHA
                    iscat = 2 : (0:2)=R0,RSIG
                    iscat = 3 : (0:3)=A,B,C
                    iscat = 4 : (0)=R0 - Particle size in microns
                    iscat = 5 : None
                    iscat = 6 : (0:3)=F,G1,G2
                    iscat = 7 : (0)=R0 - Particle size in microns
    rs(3) :: Size integration limits (0:2) and step size (2)
             If rs[1]<rs[0] then integration is terminated when 
             N * QS < 10**-6 * Max
    refindx(nwave,2) :: Real and imaginary part of the refractive index
    theta(ntheta) :: Scattering angle (degrees) from 0 to 90, angles >90 degrees
                     will be automatically returned as 180-theta

    
    Outputs
    ________
    
    xscat(nwave) :: Mean scattering cross section (cm2)
    xext(nwave) :: Mean extinction cross section (cm2)
    thetax(nphas) :: Array of scattering angles from 0 to 180 degrees
    phas(nwave,nphas) :: Phase function (normalised to X)
    '''
    
    nwave = len(wavel)
    ntheta = len(theta)
    
    #Checking whether theta goes from 0 to 90
    for j in range(ntheta):
        if theta[j] < 0 or theta[j] > 90:
            print(' error :: ANGLE <0 OR > 90')
            return
    
    #Checking whether theta=90 exists to calculate size of phase function
    ilim = np.where(theta==90.)[0]
    if len(ilim)==1:
        nphas = int(2 * ntheta - 1)
    else:
        nphas = int(2 * ntheta)
        
    #Constructing new theta from 0 to 180
    thetax = np.zeros(nphas)
    for i in range(ntheta): #First part from 0 to 90
        thetax[i] = theta[i]
    for i in range(ntheta,nphas): #Second part from 90 to 180
        thetax[i] = 180.0 - thetax[nphas - i - 1]
    
    #Computing the optical properties
    ###########################################################################
    
    xscat = np.zeros(nwave)
    xext = np.zeros(nwave)
    phas = np.zeros((nwave,nphas))
    
    #Mie Theory cases
    if( (iscat==1) or (iscat==2) or (iscat==3) or (iscat==4) ):
        
        for iwave in range(nwave):
            xscat[iwave], xext[iwave], thetax, phas[iwave,:] = miescat(wavel[iwave], iscat, dsize, rs, refindx[iwave,:], theta)
       
    #Isotropic scattering (note that the cross sections in this case are not defined)
    elif iscat==5:
        phas[:,:] = 1.
            
    #Henyey-Greenstein scattering (note that the cross sections in this case are not defined)
    elif iscat==6:
        f = dsize[0]
        g1 = dsize[1]
        g2 = dsize[2]
        
        for i in range(nphas):
            calpha = np.cos(thetax[i]/180.*np.pi)
            phas[:,i] = henyey(calpha, f, g1, g2)
        
    #Dipole scattering
    elif iscat==7:
        
        rr = dsize[0]
        
        for iwave in range(nwave):
            nc = complex(refindx[iwave,0], -refindx[iwave,1])
            x = 2 * np.pi * rr /wavel[iwave]
            
            qsca = (8./3.)*(x**4) * np.abs((nc**2-1)/(nc**2 + 2))
            qabs = - 4 * x * ((nc**2 - 1)/(nc**2 + 2)).imag
            qext = qsca + qabs
            omega = qsca/qext
            gsec = np.pi * (rr * 1.e-4)**2.
            xext[iwave] = qext * gsec
            xscat[iwave] = xext[iwave] * omega
            
            
        for i in range(nphas):
            calpha = np.cos(thetax[i]/180.*np.pi)
            phas[:,i] = 0.75 * (1.0 + calpha*calpha)
        
            
    return xscat, xext, thetax, phas

@njit(fastmath=True)
def subfithgm(theta, phase):
    '''
    Fits a combined Henyey-Grrenstein function to an ASCII phase function
    file using Levenburg-Marquardt technique
    
    Inputs
    ________
    
    theta(ntheta) :: Angles at which the phase function is defined (degrees)
    phase(nwave,ntheta) :: Phase function normalised to 4*pi

    
    Outputs
    ________
    
    f(nwave) :: Parameter defining the relative contribution of g1 and g2 to the double Henyey-Greenstein function
    g1(nwave) :: Parameter defining the first assymetry factor of the double Henyey-Greenstein phase function
    g2(nwave) :: Parameter defining the first assymetry factor of the double Henyey-Greenstein phase function
    rms(nwave) :: proxy of residuals of the fit
    '''
    nphase = phase.shape[1]
    nwave = phase.shape[0]
    
    f = np.zeros(nwave)
    g1 = np.zeros(nwave)
    g2 = np.zeros(nwave)
    rms = np.zeros(nwave)
    for iwave in range(nwave):
    
        mx = 3
        my = 100
        x = np.array([0.5, 0.5, -0.5])
        alamda = -1
        nover = 1000
        nc = 0
        ochisq = 0.0
        lphase = np.log(phase[iwave,:])
        for itemp in range(1, nover + 1):
            if alamda < 0:
                alpha, beta, chisq = mrqcofl(nphase, theta, phase[iwave,:], x)
                ochisq = chisq
                alamda = 1000.0
            alamda, chisq = mrqminl(nphase, theta, lphase, x, alamda, alpha, beta, chisq, ochisq)
            if chisq == ochisq:
                nc += 1
                break
            else:
                ochisq = chisq
                nc = 0
        f[iwave] = x[0]
        g1[iwave] = x[1]
        g2[iwave] = x[2]
        rms[iwave] = np.sqrt(chisq)
        
    return (f, g1, g2, rms)

@njit(fastmath=True)
def mrqminl(nphase, theta, phase, x, alamda, alpha, beta, chisq, ochisq):
    '''
    Fit phase function with henyey-greenstein parameters in log space
    '''
    max_thet = 100
    mx = 3
    my = 100
    covar = np.zeros((mx, mx))
    da = np.copy(beta)[:, None]
    for j in range(mx):
        for k in range(mx):
            covar[j, k] = alpha[j, k]
        covar[j, j] = alpha[j, j] * (1.0 + alamda)
    covar = np.ascontiguousarray(np.linalg.inv(covar))
    da = np.dot(covar, np.ascontiguousarray(da))
    if alamda == 0.0:
        return
    xt = x + da[:, 0]
    for i in range(3):
        if i == 0:
            xt[i] = min(max(xt[i], 1e-06), 0.999999)
        elif i == 1:
            xt[i] = min(max(xt[i], 0.0), 0.98)
        elif i == 2:
            xt[i] = min(max(xt[i], -0.98), -0.1)
    covar, da, chisq = mrqcofl(nphase, theta, phase, xt)
    if chisq <= ochisq:
        alamda *= 0.9
        ochisq = chisq
        alpha[:, :] = covar
        beta[:] = da
        x[:] = xt
    else:
        alamda *= 1.5
        chisq = ochisq
        if alamda > 1e+36:
            alamda = 1e+36
    return (alamda, chisq)

@njit(fastmath=True)
def mrqcofl(nphase, theta, phase, x):
    '''
    Fit phase function with henyey-greenstein parameters in log space
    '''
    mx = 3
    MY = 100
    alpha = np.zeros((mx, mx))
    beta = np.zeros(mx)
    chisq = 0.0
    cphase, kk = subhgphas(nphase, theta, x)
    kk = kk / cphase[:, np.newaxis]
    cphase = np.log(cphase)
    for i in range(nphase):
        dy = phase[i] - cphase[i]
        for j in range(mx):
            wt = kk[i, j]
            for k in range(j + 1):
                alpha[j, k] += wt * kk[i, k]
            beta[j] += dy * wt
        chisq += dy * dy
    for j in range(1, mx):
        for k in range(j):
            alpha[k, j] = alpha[j, k]
    return (alpha, beta, chisq)

@njit(fastmath=True, error_model='numpy')
def subhgphas(nphase, theta, x):
    '''
    Fit phase function with henyey-greenstein parameters in log space
    '''
    pi = np.pi
    cphase = np.zeros(nphase)
    kk = np.zeros((nphase, 3))
    alpha = np.zeros(nphase)
    tphase = np.zeros(nphase)
    xt = np.zeros(3)
    f = x[0]
    g1 = x[1]
    g2 = x[2]
    alpha = np.cos(theta * pi / 180.0)
    cphase = henyey(alpha, f, g1, g2)
    xt[:] = x[:]
    for j in range(3):
        dx = 0.01
        xt[j] = x[j] + dx
        if j == 0:
            if xt[j] > 0.99:
                xt[j] = x[j] - dx
        elif j == 1:
            if xt[j] > 0.98:
                xt[j] = x[j] - dx
        dx = xt[j] - x[j]
        f = xt[0]
        g1 = xt[1]
        g2 = xt[2]
        for i in range(nphase):
            tphase[i] = henyey(alpha[i], f, g1, g2)
        for i in range(nphase):
            kk[i, j] = (tphase[i] - cphase[i]) / dx
        xt[j] = x[j]
    return (cphase, kk)

@njit(fastmath=True)
def henyey(alpha, f, g1, g2):
    '''
    Fit phase function with henyey-greenstein parameters in log space
    '''
    x1 = (1.0 - g1 * g1) / ((1.0 + g1 * g1 - 2 * g1 * alpha) ** 1.5)
    x2 = (1.0 - g2 * g2) / ((1.0 + g2 * g2 - 2 * g2 * alpha) ** 1.5)
    y = f * x1 + (1.0 - f) * x2
    return y

@njit(fastmath=True)
def kk_new_sub(vi, k, vm, nm):
    '''
    Calculates real part of refractive index, given a wavelength grid, 
    imaginary refractive index on this grid, and the real refractive
    index at a reference wavelength.
    '''
    
    npoints = len(vi)
    va = np.zeros(npoints)
    ka = np.zeros(npoints)
    na = np.zeros(npoints)
    # Reverse order logic
    irev = False
    if vi[0] > vi[-1]:
        va = vi[::-1]
        ka = k[::-1]
        irev = True
    else:
        va = vi
        ka = k

    # Linear interpolation function (verint) to find km at vm
    km = np.interp(vm, va, ka)

    # Integration loop
    for i in range(npoints):
        v = va[i]
        y = np.zeros(npoints)
        for j in range(npoints):
            alpha = va[j]**2 - v**2
            beta = va[j]**2 - vm**2
            if alpha != 0 and beta != 0:
                d1 = ka[j]*va[j] - ka[i]*va[i]
                d2 = ka[j]*va[j] - km*vm
                y[j] = d1/alpha - d2/beta

        # Summation
        sum_ = 0.0
        for l in range(npoints - 1):
            dv = va[l + 1] - va[l]
            sum_ += 0.5 * (y[l] + y[l + 1]) * dv
        na[i] = nm - (2. / np.pi) * sum_

    # Prepare output based on reverse logic
    n = na[::-1] if irev else na

    return n
##############################################################################################################
##############################################################################################################
#                                          OTHER USEFUL CALCULATIONS
##############################################################################################################
##############################################################################################################

###############################################################################################

def legendre_p(l, x):
    '''
    Calculate the Legendre polynomials
    '''
    if l == 0:
        return np.ones_like(x)
    elif l == 1:
        return x
    else:
        P0 = np.ones_like(x)
        P1 = x
        for n in range(2, l + 1):
            Pn = ((2 * n - 1) * x * P1 - (n - 1) * P0) / n
            P0 = P1
            P1 = Pn
        return P1

def gauss_lobatto(n, n_digits=None):
    """
    Compute the Gauss-Lobatto quadrature points and weights of order n.

    Parameters
    ----------
    n : int
        The order of the quadrature (number of nodes).
    n_digits : int, optional
        If given, round the points and weights to this many decimal places.

    Returns
    -------
    x : ndarray
        The array of quadrature nodes of length n.
    w : ndarray
        The array of corresponding weights of length n.
    """
    
    from numpy.polynomial.legendre import Legendre
    
    if n < 2:
        raise ValueError("Gauss-Lobatto requires n >= 2.")

    # Build the (n-1)-th Legendre polynomial in the standard (orthonormal) form:
    #   P_{n-1}(x) = Legendre.basis(n-1)
    Pn_1 = Legendre.basis(n-1)

    # Differentiate to get P'_{n-1}(x)
    dPn_1 = Pn_1.deriv()

    # Roots of the derivative give the interior Lobatto nodes
    interior_roots = dPn_1.roots()

    # Evaluate P_{n-1}(x) at those interior nodes
    Pn_1_vals = Pn_1(interior_roots)

    # Compute weights for the interior nodes
    w_interior = 2.0 / (n * (n - 1) * (Pn_1_vals**2))

    # Append the boundary nodes: -1 and +1
    x = np.concatenate(([-1.0], interior_roots, [1.0]))
    # Boundary weights
    w_boundary = 2.0 / (n * (n - 1))
    w = np.concatenate(([w_boundary], w_interior, [w_boundary]))

    # Sort them in ascending order (not strictly necessary, but often desired)
    # Because dPn_1.roots() is usually ascending, the below is optional:
    order = np.argsort(x)
    x, w = x[order], w[order]

    # If requested, round to n_digits decimal places
    if n_digits is not None:
        x = np.round(x, n_digits)
        w = np.round(w, n_digits)

    return x, w
