#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# Measurement_0.py - Object to represent the measurement parameters and data.
#
# Copyright (C) 2025 Juan Alday, Joseph Penn, Patrick Irwin,
# Jack Dobinson, Jon Mason, Jingxuan Yang
#
# This file is part of archNEMESIS.
#
# archNEMESIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.




import os.path


import numpy as np
import scipy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib as matplotlib
from numba import jit

from archnemesis.helpers import h5py_helper
from archnemesis.enums import InstrumentLineshape, WaveUnit, SpectraUnit
from archnemesis import gauss_lobatto


import logging
_lgr = logging.getLogger(__name__)


#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

###############################################################################################

"""
Created on Tue Mar 29 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Measurement_0:

    """Measurement class.

    This class includes all information required to model the specification of the measurement, such as the geometry of the observation
    or the instrument characteristics. 

    Attributes
    ----------
    runname : str
        Name of the Nemesis run
    NGEOM : int       
        Number of observing geometries
    FWHM : float
        Full-width at half-maximum of the instrument
    ISHAPE : int
        Instrument lineshape (only used if FWHM>0)
            (0) Square lineshape
            (1) Triangular
            (2) Gaussian
            (3) Hamming
            (4) Hanning
    ISPACE : int
        Spectral units
            (0) Wavenumber (cm-1)
            (1) Wavelength (um)
    IFORM : int
        Units of the spectra
            (0) Radiance - W cm-2 sr-1 (cm-1)-1 if ISPACE=0 ---- W cm-2 sr-1 μm-1 if ISPACE=1
            (1) F_planet/F_star - Dimensionsless
            (2) A_planet/A_star - 100.0 * A_planet/A_star (dimensionsless)
            (3) Integrated spectral power of planet - W (cm-1)-1 if ISPACE=0 ---- W um-1 if ISPACE=1
            (4) Atmospheric transmission multiplied by solar flux
            (5) Normalised radiance to a given wavelength (VNORM)
            (6) Integrated radiance over filter function - W cm-2 sr-1 if ISPACE=0 ---- W cm-2 sr-1 if ISPACE=1
    LATITUDE : float
        Planetocentric latitude at centre of the field of view
    LONGITUDE : float
        Planetocentric longitude at centre of the field of view
    V_DOPPLER : float
        Doppler velocity between the observed body and the observer (km/s)
        It is considered positive if source body is moving towards observer, and negative if it is moving away
    NCONV : 1D array, int (NGEOM)
        Number of convolution spectral points in each spectrum
    NAV : 1D array, int (NGEOM)
        For each geometry, number of individual geometries need to be calculated and averaged to reconstruct the field of view
    VCONV : 2D array, float (NCONV,NGEOM)
        Convolution spectral points (wavelengths/wavenumbers) in each spectrum
    WOFF : float
        Wavenumber/Wavelength offset to add to measured spectrum
    MEAS : 2D array, float (NCONV,NGEOM)
        Measured spectrum for each geometry
    ERRMEAS : 2D array, float (NCONV,NGEOM)
        Noise in the measured spectrum for each geometry
    FLAT : 2D array, float (NGEOM,AV)
        Latitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
    FLON : 2D array, float (NGEOM,NAV)
        Longitude of each averaging point needed to reconstruct the FOV (when NAV > 1)
    SOL_ANG : 2D array, float (NGEOM,NAV)
        Solar indicent angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
    EMISS_ANG : 2D array, float (NGEOM,NAV)
        Emission angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
    AZI_ANG : 2D array, float (NGEOM,NAV)
        Azimuth angle of each averaging point needed to reconstruct the FOV (when NAV > 1)
    TANHE : 2D array, float (NGEOM,NAV)
        Tangent height of each averaging point needed to reconstruct the FOV (when NAV > 1)
        (For limb or solar occultation observations)
    WGEOM : 2D array, float (NGEOM,NAV)
        Weights of each point for the averaging of the FOV (when NAV > 1)
    NWAVE : int
        Number of calculation wavelengths required to model the convolution wavelengths
    WAVE : 1D array (NWAVE)
        Calculation wavenumbers for one particular geometry
    NFIL : 1D array, int (NCONV)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        NFIL represents the number of spectral points to defined the ILS for each convolution wavenumber.
    VFIL : 2D array, int (NFIL,NCONV)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        VFIL represents the calculation wavenumbers at which the ILS is defined for each each convolution wavenumber.
    AFIL : 2D array, int (NFIL,NCONV)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        AFIL represents the value of the ILS at each VFIL for each convolution wavenumber.
    NY : int
        Number of points in the Measurement vector (sum of all NCONV)
    Y : 1D array, float (NY)
        Measurement vector (concatenation of all spectra in the class)
    SE : 2D array, float (NY,NY)
        Measurement uncertainty covariance matrix (assumed to be diagonal)
    SPECMOD : 2D array, float (NCONV,NGEOM)
        Modelled spectrum for each geometry
    VNORM : float 
        If IFORM=5, then VNORM defines the wavelength at which the spectra must be normalised
    VFMERR : 1D array, float (NVFMERR)
        Wavenumber/wavelength at which the forward modelling error is defined
    FMERR : 1D array, float (NVFMERR)
        Forward modelling error at each VFMERR
    NORDERS_AOTF : int
        Number of orders to consider to reconstruct AOTF filter function (if required)
    VCONV_AOTF : 2D array, float (NCONV,NGEOM,NORDERS_AOTF)
        Convolution wavelengths/wavenumbers for each order (if required)
    TRANS_AOTF : 2D array, float (NCONV,NGEOM,NORDERS_AOTF)
        Transmission of the AOTF filter for each diffraction order (if required)
    NFIL_AOTF : 2D array, int (NCONV,NORDERS_AOTF)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        NFIL represents the number of spectral points to defined the ILS for each convolution wavenumber.
        In this case, we use a different ILS for each diffraction order for the reconstruction of the AOTF filter function.
    VFIL_AOTF : 3D array, int (NFIL,NCONV,NORDERS_AOTF)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        VFIL represents the calculation wavenumbers at which the ILS is defined for each each convolution wavenumber.
        In this case, we use a different ILS for each diffraction order for the reconstruction of the AOTF filter function.
    AFIL_AOTF : 3D array, int (NFIL,NCONV,NORDERS_AOTF)
        If FWHM<0.0, the ILS is expected to be defined separately for each convolution wavenumber.
        AFIL represents the value of the ILS at each VFIL for each convolution wavenumber.
        In this case, we use a different ILS for each diffraction order for the reconstruction of the AOTF filter function.

    Methods
    ----------
    
    Measurement_0.assess()
    Measurement_0.summary_info()
    
    Measurement_0.write_hdf5()
    Measurement_0.read_hdf5()
    Measurement_0.read_spx()
    Measurement_0.read_spx_SO()
    Measurement_0.write_spx()
    Measurement_0.write_spx_SO()
    Measurement_0.read_sha()
    Measurement_0.write_sha()
    Measurement_0.read_fil()
    Measurement_0.write_fil()
    
    Measurement_0.edit_VCONV()
    Measurement_0.edit_MEAS()
    Measurement_0.edit_ERRMEAS()
    Measurement_0.edit_SPECMOD()
    Measurement_0.edit_FLAT()
    Measurement_0.edit_FLON()
    Measurement_0.edit_SOL_ANG()
    Measurement_0.edit_EMISS_ANG()
    Measurement_0.edit_AZI_ANG()
    Measurement_0.edit_TANHE()
    Measurement_0.edit_WGEOM()
    
    Measurement_0.add_fmerr()
    Measurement_0.calc_MeasurementVector()
    
    Measurement_0.remove_geometry()
    Measurement_0.select_geometry()
    Measurement_0.select_geometries()
    Measurement_0.select_TANHE_SO()
    Measurement_0.crop_wave()
    
    Measurement_0.calc_wave_range()
    
    Measurement_0.lblconv()
    Measurement_0.lblconvg()
    Measurement_0.conv()
    Measurement_0.cong()
    
    Measurement_0.calc_doppler_shift()
    Measurement_0.invert_doppler_shift()
    Measurement_0.correct_doppler_shift()

    Measurement_0.calc_avepoints_exoplanet()
    Measurement_0.calc_geometry_primary_transit()
    
    Measurement_0.plot_ils()
    Measurement_0.plot_filters()
    Measurement_0.plot_SO()
    Measurement_0.plot_nadir()
    Measurement_0.plot_disc_averaging()
    
    """

    def __init__(
            self, 
            runname='', 
            NGEOM=1, 
            FWHM=0.0, 
            ISHAPE=InstrumentLineshape.Gaussian, 
            IFORM=SpectraUnit.Radiance, 
            ISPACE=WaveUnit.Wavenumber_cm, 
            LATITUDE=0.0, 
            LONGITUDE=0.0, 
            V_DOPPLER=0.0, 
            NCONV=np.array([1],dtype="int32"), 
            NAV=np.array([1],dtype="int32"),
    ):

        #Input parameters
        self.runname = runname
        self.NGEOM = NGEOM
        self.FWHM = FWHM
        #self.ISPACE = ISPACE
        #self.ISHAPE = ISHAPE
        #self.IFORM = IFORM
        self.LATITUDE = LATITUDE        
        self.LONGITUDE = LONGITUDE
        self.V_DOPPLER = V_DOPPLER
        self.NAV = NAV       #np.zeros(NGEOM)
        self.NCONV = NCONV   #np.zeros(NGEOM)
        self.WOFF = 0.0
        self.VNORM = None    
        
        # Input the following profiles using the edit_ methods.
        self.VCONV = None # np.zeros(NCONV,NGEOM)
        self.MEAS =  None # np.zeros(NCONV,NGEOM)
        self.ERRMEAS = None # np.zeros(NCONV,NGEOM)
        self.FLAT = None # np.zeros(NGEOM,NAV)
        self.FLON = None # np.zeros(NGEOM,NAV)
        self.SOL_ANG = None # np.zeros(NGEOM,NAV)
        self.EMISS_ANG = None # np.zeros(NGEOM,NAV)
        self.AZI_ANG = None # np.zeros(NGEOM,NAV)
        self.TANHE = None # np.zeros(NGEOM,NAV)
        self.WGEOM = None # np.zeros(NGEOM,NAV)
        self.NY = None #np.sum(NCONV)
        self.Y = None #np.zeros(NY)
        self.SE = None #np.zeros(NY,NY)

        self.SPECMOD = None #np.zeros(NCONV,NGEOM)

        self.NFIL = None  #np.zeros(NCONV)
        self.VFIL = None  #np.zeros(NFIL,NCONV)
        self.AFIL = None  #np.zeros(NFIL,NCONV)
        
        self.VFMERR = None  # np.zeros(NVFMERR)
        self.AFMERR = None  # np.zeros(NVFMERR)
        
        self.SUBOBS_LAT = 0.0 # sub observer latitude, optional
        self.SUBOBS_LON = 0.0 # sub observer longintude, optional
        
        self.NORDERS_AOTF = None #Number of orders to consider to reconstruct AOTF filter function (if required)
        self.VCONV_AOTF = None #np.zeros(NCONV,NORDERS_AOTF) #Convolution wavelengths/wavenumbers for each order (if required)
        self.TRANS_AOTF = None #np.zeros(NCONV,NORDERS_AOTF) #Weights of each order to reconstruct AOTF filter function (if required)
        self.NFIL_AOTF = None  #np.zeros(NCONV,NORDERS_AOTF)
        self.VFIL_AOTF = None  #np.zeros(NFIL_AOTF,NCONV,NORDERS_AOTF)
        self.AFIL_AOTF = None  #np.zeros(NFIL_AOTF,NCONV,NORDERS_AOTF)

        # private attributes
        self._ishape = None
        self._ispace = None
        self._iform = None
        
        # properties
        self.ISHAPE = ISHAPE
        self.ISPACE = ISPACE
        self.IFORM = IFORM
    
    @property
    def ISHAPE(self) -> InstrumentLineshape:
        return self._ishape
    
    @ISHAPE.setter
    def ISHAPE(self, value):
        self._ishape = InstrumentLineshape(value)
    
    @property
    def ISPACE(self) -> WaveUnit:
        return self._ispace
    
    @ISPACE.setter
    def ISPACE(self, value):
        self._ispace = WaveUnit(value)
    
    @property
    def IFORM(self) -> SpectraUnit:
        return self._iform
    
    @IFORM.setter
    def IFORM(self, value):
        self._iform = SpectraUnit(value)

    #################################################################################################################

    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert isinstance(self.NGEOM, (int, np.integer)), 'NGEOM must be int'
        assert self.NGEOM > 0, 'NGEOM must be >0'
        
        assert isinstance(self.IFORM, (int, np.integer, SpectraUnit)), 'IFORM must be int'
        assert self.IFORM >= SpectraUnit.Radiance, 'IFORM must be >=0 and <=6'
        assert self.IFORM <= SpectraUnit.Integrated_radiance, 'IFORM must be >=0 and <=6'
            
        if self.IFORM == SpectraUnit.Normalised_radiance:
            assert isinstance(self.VNORM, float), 'VNORM must be float if IFORM=5'
            for i in range(self.NGEOM):
                assert self.VNORM >= self.VCONV[0:self.NCONV[i]].min(), 'VNORM must be >= min(VCONV)'
                assert self.VNORM <= self.VCONV[0:self.NCONV[i]].max(), 'VNORM must be <= max(VCONV)'

        if self.IFORM == SpectraUnit.Integrated_radiance:
            assert self.FWHM < 0.0, 'FWHM must be <0 if IFORM=6 (Integrated radiance over filter function)'

        assert isinstance(self.ISPACE, (int, np.integer, WaveUnit)), 'ISPACE must be int'
        assert self.ISPACE >= WaveUnit.Wavenumber_cm, 'ISPACE must be >=0 and <=1'
        assert self.ISPACE <= WaveUnit.Wavelength_um, 'ISPACE must be >=0 and <=1'
        
        assert isinstance(self.FWHM, float), 'FWHM must be float'
        assert isinstance(self.V_DOPPLER, float), 'V_DOPPLER must be float'
        assert len(self.NCONV) == self.NGEOM, 'NCONV must have size (NGEOM)'
        assert self.VCONV.shape == (self.NCONV.max(),self.NGEOM), 'VCONV must have size (NCONV,NGEOM)'
        assert self.MEAS.shape == (self.NCONV.max(),self.NGEOM), 'MEAS must have size (NCONV,NGEOM)'
        assert self.ERRMEAS.shape == (self.NCONV.max(),self.NGEOM), 'ERRMEAS must have size (NCONV,NGEOM)'
        assert len(self.NAV) == self.NGEOM, 'NAV must have size (NGEOM)'
        assert self.FLAT.shape == (self.NGEOM,self.NAV.max()), 'FLAT must have size (NGEOM,NAV)'
        assert self.FLON.shape == (self.NGEOM,self.NAV.max()), 'FLON must have size (NGEOM,NAV)'
        assert self.WGEOM.shape == (self.NGEOM,self.NAV.max()), 'WGEOM must have size (NGEOM,NAV)'
        assert self.EMISS_ANG.shape == (self.NGEOM,self.NAV.max()), 'EMISS_ANG must have size (NGEOM,NAV)'

        #Checking if there are any limb-viewing geometries
        if self.EMISS_ANG.min()<0.0:
            assert self.TANHE.shape == (self.NGEOM,self.NAV.max()), 'TANHE must have size (NGEOM,NAV)'
            
        #Checking if there are any nadir-viewing / upward looking geometries
        if self.EMISS_ANG.max() >= 0.0:
            assert self.SOL_ANG.shape == (self.NGEOM,self.NAV.max()), 'SOL_ANG must have size (NGEOM,NAV)'
            assert self.AZI_ANG.shape == (self.NGEOM,self.NAV.max()), 'AZI_ANG must have size (NGEOM,NAV)'

        if self.FWHM > 0.0: #Analytical instrument lineshape
            assert isinstance(self.ISHAPE, (int, np.integer, InstrumentLineshape)), 'ISHAPE must be int'
        elif self.FWHM == 0.0: # Lineshape baked into K-tables
            pass
        elif self.FWHM < 0.0: # lineshape described by files
            if self.NORDERS_AOTF is None:
                assert self.NFIL is not None, "NFIL must be defined for FWHM < 0"
                assert self.VFIL is not None, "VFIL must be defined for FWHM < 0"
                assert self.AFIL is not None, "AFIL must be defined for FWHM < 0"
            else:
                assert self.NFIL_AOTF is not None, "NFIL_AOTF must be defined for FWHM < 0"
                assert self.VFIL_AOTF is not None, "VFIL_AOTF must be defined for FWHM < 0"
                assert self.AFIL_AOTF is not None, "AFIL_AOTF must be defined for FWHM < 0"

        #Checking that the wavelengths in the Forward modelling error encompass the spectrum
        if self.VFMERR is not None:
            assert self.FMERR is not None, "AFMERR must be defined if VFMERR is defined"
            assert len(self.VFMERR) == len(self.FMERR), "VFMERR and AFMERR must have the same length"
            for i in range(self.NGEOM):
                assert self.VFMERR.min() <= self.VCONV[0:self.NCONV[i]].min(), 'VFMERR must be <= min(VCONV)'
                assert self.VFMERR.max() >= self.VCONV[0:self.NCONV[i]].max(), 'VFMERR must be >= max(VCONV)'

        #Checking if the AOTF parameters are included and correctly defined
        if self.NORDERS_AOTF is not None:
            assert self.NORDERS_AOTF > 0, 'NORDERS_AOTF must be >0'
            assert self.VCONV_AOTF.shape == (self.NCONV.max(),self.NGEOM,self.NORDERS_AOTF), 'VCONV_AOTF must have size (NCONV,NORDERS_AOTF)'
            assert self.TRANS_AOTF.shape == (self.NCONV.max(),self.NGEOM,self.NORDERS_AOTF), 'TRANS_AOTF must have size (NCONV,NORDERS_AOTF)'


    #################################################################################################################
            
    def summary_info(self):
        """
        Subroutine to print summary of information about the class
        """      

        #Defining spectral resolution
        if self.FWHM > 0.0:
            _lgr.info(f"Spectral resolution of the measurement (FWHM) ::  {(self.FWHM)}")
        elif self.FWHM < 0.0:
            if self.IFORM == SpectraUnit.Integrated_radiance:
                _lgr.info('Filter functions defined in .fil file')
            else:
                _lgr.info('Instrument line shape defined in .fil file')
        else:
            _lgr.info('Spectral resolution of the measurement is account for in the k-tables')

        wavelength_unit = 'um'
        wavenumber_unit = 'cm^-1'
        to_wavelength = (lambda x: x) if self.ISPACE==WaveUnit.Wavelength_um else (lambda x: 1E4/x)
        to_wavenumber = (lambda x: x) if self.ISPACE==WaveUnit.Wavenumber_cm else (lambda x: 1E4/x)
        wavenumber_str = lambda x: str(to_wavenumber(x))+' '+wavenumber_unit
        wavelength_str = lambda x: str(to_wavelength(x))+' '+wavelength_unit

        #Defining geometries
        _lgr.info(f"Field-of-view centered at ::  {('Latitude',self.LATITUDE,'- Longitude',self.LONGITUDE)}")
        _lgr.info(f"There are  {(self.NGEOM,'geometries in the measurement vector')}")
        for i in range(self.NGEOM):
            _lgr.info('')
            _lgr.info('GEOMETRY '+str(i+1))
            _lgr.info('Minimum wavelength/wavenumber :: '
                +wavelength_str(self.VCONV[0,i])
                +'/'
                +wavenumber_str(self.VCONV[0,i])
                +' - Maximum wavelength/wavenumber :: '
                +wavelength_str(self.VCONV[self.NCONV[i]-1,i])
                +'/'
                +wavenumber_str(self.VCONV[self.NCONV[i]-1,i])
            )
            if self.NAV[i]>1:
                _lgr.info(self.NAV[i],' averaging points')
                for j in range(self.NAV[i]):
                
                    if self.EMISS_ANG[i,j]<0.0:
                        if isinstance(self.TANHE,np.ndarray)==True:
                            _lgr.info(f"Averaging point {(j+1,' - Weighting factor ',self.WGEOM[i,j])}")
                            _lgr.info(f"Limb-viewing or solar occultation measurement. Latitude ::  {(self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.TANHE[i,j])}")
                        else:
                            _lgr.info(f"Averaging point {(j+1,' - Weighting factor ',self.WGEOM[i,j])}")
                            _lgr.info(f"Limb-viewing or solar occultation measurement. Latitude ::  {(self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.SOL_ANG[i,j])}")
                    
                    else:
                        _lgr.info(f"Averaging point {(j+1,' - Weighting factor ',self.WGEOM[i,j])}")
                        _lgr.info(f"Nadir-viewing geometry. Latitude ::  {(self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Emission angle :: ',self.EMISS_ANG[i,j],' - Solar Zenith Angle :: ',self.SOL_ANG[i,j],' - Azimuth angle :: ',self.AZI_ANG[i,j])}")

            else:
                j = 0
                if self.EMISS_ANG[i,j]<0.0:
                    if isinstance(self.TANHE,np.ndarray)==True:
                        _lgr.info(f"Limb-viewing or solar occultation measurement. Latitude ::  {(self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.TANHE[i,j])}")
                    else:
                        _lgr.info(f"Limb-viewing or solar occultation measurement. Latitude ::  {(self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Tangent height :: ',self.SOL_ANG[i,j])}")
                else:
                    _lgr.info(f"Nadir-viewing geometry. Latitude ::  {(self.FLAT[i,j],' - Longitude :: ',self.FLON[i,j],' - Emission angle :: ',self.EMISS_ANG[i,j],' - Solar Zenith Angle :: ',self.SOL_ANG[i,j],' - Azimuth angle :: ',self.AZI_ANG[i,j])}")

            
    #################################################################################################################
            
    def write_hdf5(self,runname):
        """
        Write the Measurement parameters into an HDF5 file
        """

        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess()

        with h5py.File(runname+'.h5','a') as f:
            #Checking if Atmosphere already exists
            if ('/Measurement' in f)==True:
                del f['Measurement']   #Deleting the Measurement information that was previously written in the file

            grp = f.create_group("Measurement")
            

            #Writing the latitude/longitude at the centre of FOV
            dset = h5py_helper.store_data(grp, 'LATITUDE', self.LATITUDE)
            dset.attrs['title'] = "Latitude at centre of FOV"
            dset.attrs['units'] = 'degrees'

            dset = h5py_helper.store_data(grp, 'LONGITUDE', self.LONGITUDE)
            dset.attrs['title'] = "Longitude at centre of FOV"
            dset.attrs['units'] = 'degrees'
            
            # Write optional sub observer lat/lon so we can plot the measurement easily
            dset = h5py_helper.store_data(grp, 'SUBOBS_LAT', self.SUBOBS_LAT)
            dset.attrs['title'] = "Latitude at point directly below the observer (optional)"
            dset.attrs['units'] = 'degrees'

            dset = h5py_helper.store_data(grp, 'SUBOBS_LON', self.SUBOBS_LON)
            dset.attrs['title'] = "Longitude at point directly below the observer (optional)"
            dset.attrs['units'] = 'degrees'
            
            #Writing the Doppler velocity
            dset = h5py_helper.store_data(grp, 'V_DOPPLER', self.V_DOPPLER)
            dset.attrs['title'] = "Doppler velocity between the observed body and the observer"
            dset.attrs['units'] = 'km s-1'

            #Writing the spectral units
            dset = h5py_helper.store_data(grp, 'ISPACE', int(self.ISPACE))
            dset.attrs['title'] = "Spectral units"
            if self.ISPACE==WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um'

            #Writing the measurement units
            dset = h5py_helper.store_data(grp, 'IFORM', int(self.IFORM))
            dset.attrs['title'] = "Measurement units"
            
            if self.ISPACE==WaveUnit.Wavenumber_cm:  #Wavenumber space
                if self.IFORM==SpectraUnit.Radiance:
                    lunit = 'Radiance / W cm-2 sr-1 (cm-1)-1'
                elif self.IFORM==SpectraUnit.FluxRatio:
                    lunit = 'Secondary transit depth (Fplanet/Fstar) / Dimensionless'
                elif self.IFORM==SpectraUnit.TransitDepth:
                    lunit = 'Primary transit depth (100*Aplanet/Astar) / Dimensionless'
                elif self.IFORM==SpectraUnit.Integrated_spectral_power:
                    lunit = 'Integrated spectral power of planet / W (cm-1)-1'
                elif self.IFORM==SpectraUnit.Atmospheric_transmission:
                    lunit = 'Atmospheric transmission multiplied by solar flux / W cm-2 (cm-1)-1'
                elif self.IFORM==SpectraUnit.Normalised_radiance:
                    lunit = 'Spectra normalised to VNORM'
                elif self.IFORM==SpectraUnit.Integrated_radiance:
                    lunit = 'Integrated radiance over filter function / W cm-2 sr-1'

            elif self.ISPACE==WaveUnit.Wavelength_um:  #Wavelength space
                if self.IFORM==SpectraUnit.Radiance:
                    lunit = 'Radiance / W cm-2 sr-1 um-1'
                elif self.IFORM==SpectraUnit.FluxRatio:
                    lunit = 'Secondary transit depth (Fplanet/Fstar) / Dimensionless'
                elif self.IFORM==SpectraUnit.TransitDepth:
                    lunit = 'Primary transit depth (100*Aplanet/Astar) / Dimensionless'
                elif self.IFORM==SpectraUnit.Integrated_spectral_power:
                    lunit = 'Integrated spectral power of planet / W um-1'
                elif self.IFORM==SpectraUnit.Atmospheric_transmission:
                    lunit = 'Atmospheric transmission multiplied by solar flux / W cm-2 um-1'
                elif self.IFORM==SpectraUnit.Normalised_radiance:
                    lunit = 'Spectra normalised to VNORM'
                elif self.IFORM==SpectraUnit.Integrated_radiance:
                    lunit = 'Integrated radiance over filter function / W cm-2 sr-1'
                    
            dset.attrs['units'] = lunit
            
            if self.IFORM==SpectraUnit.Normalised_radiance:
                dset = h5py_helper.store_data(grp, 'VNORM', self.VNORM)
                if self.ISPACE==WaveUnit.Wavenumber_cm:
                    dset.attrs['title'] = "Wavenumber for normalisation"
                    dset.attrs['units'] = 'cm-1'
                elif self.ISPACE==WaveUnit.Wavelength_um:
                    dset.attrs['title'] = "Wavelength for normalisation"
                    dset.attrs['units'] = 'um'

            #Writing the number of geometries
            dset = h5py_helper.store_data(grp, 'NGEOM', self.NGEOM)
            dset.attrs['title'] = "Number of measurement geometries"

            #Defining the averaging points required to reconstruct the field of view 
            dset = h5py_helper.store_data(grp, 'NAV', self.NAV)
            dset.attrs['title'] = "Number of averaging points needed to reconstruct the field-of-view"

            dset = h5py_helper.store_data(grp, 'FLAT', self.FLAT)
            dset.attrs['title'] = "Latitude of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = "Degrees"
    
            dset = h5py_helper.store_data(grp, 'FLON', self.FLON)
            dset.attrs['title'] = "Longitude of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = "Degrees"

            dset = h5py_helper.store_data(grp, 'WGEOM', self.WGEOM)
            dset.attrs['title'] = "Weight of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = ""

            dset = h5py_helper.store_data(grp, 'EMISS_ANG', self.EMISS_ANG)
            dset.attrs['title'] = "Emission angle of each averaging point needed to reconstruct the field-of-view"
            dset.attrs['unit'] = "Degrees"

            #Checking if there are any limb-viewing geometries
            if np.nanmin(self.EMISS_ANG)<0.0:

                dset = h5py_helper.store_data(grp, 'TANHE', self.TANHE)
                dset.attrs['title'] = "Tangent height of each averaging point needed to reconstruct the field-of-view"
                dset.attrs['unit'] = "km"

            #Checking if there are any nadir-viewing / upward looking geometries
            if np.nanmax(self.EMISS_ANG) >= 0.0:

                dset = h5py_helper.store_data(grp, 'SOL_ANG', self.SOL_ANG)
                dset.attrs['title'] = "Solar zenith angle of each averaging point needed to reconstruct the field-of-view"
                dset.attrs['unit'] = "Degrees"

                dset = h5py_helper.store_data(grp, 'AZI_ANG', self.AZI_ANG)
                dset.attrs['title'] = "Azimuth angle of each averaging point needed to reconstruct the field-of-view"
                dset.attrs['unit'] = "Degrees"

            dset = h5py_helper.store_data(grp, 'NCONV', self.NCONV)
            dset.attrs['title'] = "Number of spectral bins in each geometry"

            dset = h5py_helper.store_data(grp, 'WOFF', self.WOFF)
            dset.attrs['title'] = "Wavelength/Wavenumber offset to add to each measurement"


            dset = h5py_helper.store_data(grp, 'VCONV', self.VCONV)
            dset.attrs['title'] = "Spectral bins"
            if self.ISPACE==WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE==WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um'

            dset = h5py_helper.store_data(grp, 'MEAS', self.MEAS)
            dset.attrs['title'] = "Measured spectrum in each geometry"
            dset.attrs['units'] = lunit

            dset = h5py_helper.store_data(grp, 'ERRMEAS', self.ERRMEAS)
            dset.attrs['title'] = "Uncertainty in the measured spectrum in each geometry"
            dset.attrs['units'] = lunit

            if self.FWHM > 0.0:
                dset = h5py_helper.store_data(grp, 'ISHAPE', int(self.ISHAPE))
                dset.attrs['title'] = "Instrument lineshape"
                if self.ISHAPE==InstrumentLineshape.Square:
                    lils = 'Square function'
                elif self.ISHAPE==InstrumentLineshape.Triangular:
                    lils = 'Triangular function'
                elif self.ISHAPE==InstrumentLineshape.Gaussian:
                    lils = 'Gaussian function'
                elif self.ISHAPE==InstrumentLineshape.Hamming:
                    lils = 'Hamming function'
                elif self.ISHAPE==InstrumentLineshape.Hanning:
                    lils = 'Hanning function'
                dset.attrs['type'] = lils

            dset = h5py_helper.store_data(grp, 'FWHM', self.FWHM)
            dset.attrs['title'] = "FWHM of instrument lineshape"
            if self.FWHM > 0.0:
                if self.ISPACE==WaveUnit.Wavenumber_cm:
                    dset.attrs['units'] = 'Wavenumber / cm-1'
                elif self.ISPACE==WaveUnit.Wavelength_um:
                    dset.attrs['units'] = 'Wavelength / um'
                dset.attrs['type'] = 'Analytical lineshape ('+lils+')'
            elif self.FWHM==0.0:
                dset.attrs['type'] = 'Convolution already performed in k-tables'
            elif self.FWHM < 0.0:
                dset.attrs['type'] = 'Explicit definition of instrument lineshape in each spectral bin'

            if self.FWHM < 0.0:
                dset = h5py_helper.store_data(grp, 'NFIL', self.NFIL)
                dset.attrs['title'] = "Number of points required to define the ILS in each spectral bin"

                if self.ISPACE==WaveUnit.Wavenumber_cm:
                    dset = h5py_helper.store_data(grp, 'VFIL', self.VFIL)
                    dset.attrs['title'] = "Wavenumber of the points required to define the ILS in each spectral bin"
                    dset.attrs['unit'] = "Wavenumber / cm-1"
                elif self.ISPACE==WaveUnit.Wavelength_um:
                    dset = h5py_helper.store_data(grp, 'VFIL', self.VFIL)
                    dset.attrs['title'] = "Wavelength of the points required to define the ILS in each spectral bin"
                    dset.attrs['unit'] = "Wavelength / um"

                dset = h5py_helper.store_data(grp, 'AFIL', self.AFIL)
                dset.attrs['title'] = "ILS in each spectral bin"
                dset.attrs['unit'] = ""

            if self.VFMERR is not None:
                
                if self.ISPACE==WaveUnit.Wavenumber_cm:
                    dset = h5py_helper.store_data(grp, 'VFMERR', self.VFMERR)
                    dset.attrs['title'] = "Wavenumber of the points at which the forward modelling error is defined"
                    dset.attrs['unit'] = "Wavenumber / cm-1"
                elif self.ISPACE==WaveUnit.Wavelength_um:
                    dset = h5py_helper.store_data(grp, 'VFMERR', self.VFMERR)
                    dset.attrs['title'] = "Wavelength of the points at which the forward modelling error is defined"
                    dset.attrs['unit'] = "Wavelength / um"

                dset = h5py_helper.store_data(grp, 'FMERR', self.FMERR)
                dset.attrs['title'] = "Forward modelling error"
                dset.attrs['unit'] = lunit

            if self.NORDERS_AOTF is not None:
                
                dset = h5py_helper.store_data(grp, 'NORDERS_AOTF', self.NORDERS_AOTF)
                dset.attrs['title'] = "Number of diffraction orders to combine on measurement"

                if self.ISPACE==WaveUnit.Wavenumber_cm:
                    
                    dset = h5py_helper.store_data(grp, 'VCONV_AOTF', self.VCONV_AOTF)
                    dset.attrs['title'] = "Convolution wavenumbers for each of the diffraction orders"
                    dset.attrs['unit'] = "Wavenumber / cm-1"
                    
                    dset = h5py_helper.store_data(grp, 'TRANS_AOTF', self.TRANS_AOTF)
                    dset.attrs['title'] = "Transmission of the AOTF for each of the diffraction orders"

                    if self.FWHM < 0.0:
                        
                        dset = h5py_helper.store_data(grp, 'NFIL_AOTF', self.NFIL_AOTF)
                        dset.attrs['title'] = "Number of points required to define the ILS in each spectral bin (for each diffraction order)"

                        dset = h5py_helper.store_data(grp, 'VFIL_AOTF', self.VFIL_AOTF)
                        dset.attrs['title'] = "Wavenumber of the points required to define the ILS in each spectral bin (for each diffraction order)"
                        dset.attrs['unit'] = "Wavenumber / cm-1"

                        dset = h5py_helper.store_data(grp, 'AFIL_AOTF', self.AFIL_AOTF)
                        dset.attrs['title'] = "ILS in each spectral bin"
                        dset.attrs['unit'] = ""
                        
                if self.ISPACE==WaveUnit.Wavelength_um:
                    
                    dset = h5py_helper.store_data(grp, 'VCONV_AOTF', self.VCONV_AOTF)
                    dset.attrs['title'] = "Convolution wavelengths for each of the diffraction orders"
                    dset.attrs['unit'] = "Wavelength / um"
                    
                    dset = h5py_helper.store_data(grp, 'TRANS_AOTF', self.TRANS_AOTF)
                    dset.attrs['title'] = "Transmission of the AOTF for each of the diffraction orders"

                    if self.FWHM < 0.0:
                        
                        dset = h5py_helper.store_data(grp, 'NFIL_AOTF', self.NFIL_AOTF)
                        dset.attrs['title'] = "Number of points required to define the ILS in each spectral bin (for each diffraction order)"

                        dset = h5py_helper.store_data(grp, 'VFIL_AOTF', self.VFIL_AOTF)
                        dset.attrs['title'] = "Wavelength of the points required to define the ILS in each spectral bin (for each diffraction order)"
                        dset.attrs['unit'] = "Wavelength / um"

                        dset = h5py_helper.store_data(grp, 'AFIL_AOTF', self.AFIL_AOTF)
                        dset.attrs['title'] = "ILS in each spectral bin"
                        dset.attrs['unit'] = ""

    #################################################################################################################

    def read_hdf5(self,runname,calc_MeasurementVector=True):
        """
        Read the Measurement properties from an HDF5 file
        """

        import h5py

        with h5py.File(runname+'.h5','r') as f:

            #Checking if Measurement exists
            e = "/Measurement" in f
            if e==False:
                raise ValueError('error :: Measurement is not defined in HDF5 file')
            else:

                self.NGEOM = h5py_helper.retrieve_data(f, 'Measurement/NGEOM', np.int32)
                self.ISPACE = h5py_helper.retrieve_data(f, 'Measurement/ISPACE', lambda x:  WaveUnit(np.int32(x)))
                self.IFORM = h5py_helper.retrieve_data(f, 'Measurement/IFORM', lambda x:  SpectraUnit(np.int32(x)))
                self.LATITUDE = h5py_helper.retrieve_data(f, 'Measurement/LATITUDE', np.float64)
                self.LONGITUDE = h5py_helper.retrieve_data(f, 'Measurement/LONGITUDE', np.float64)
                self.SUBOBS_LAT = h5py_helper.retrieve_data(f, 'Measurement/SUBOBS_LAT', np.float64)
                self.SUBOBS_LON = h5py_helper.retrieve_data(f, 'Measurement/SUBOBS_LON', np.float64)
                self.NAV = h5py_helper.retrieve_data(f, 'Measurement/NAV', np.array)
                self.FLAT = h5py_helper.retrieve_data(f, 'Measurement/FLAT', np.array)
                self.FLON = h5py_helper.retrieve_data(f, 'Measurement/FLON', np.array)
                self.WGEOM = h5py_helper.retrieve_data(f, 'Measurement/WGEOM', np.array)
                self.EMISS_ANG = h5py_helper.retrieve_data(f, 'Measurement/EMISS_ANG', np.array)
                
                if self.IFORM==SpectraUnit.Normalised_radiance:
                    if 'Measurement/VNORM' in f:
                        self.VNORM = h5py_helper.retrieve_data(f, 'Measurement/VNORM', np.float64)
                
                #Reading Doppler shift if exists
                if 'Measurement/V_DOPPLER' in f:
                    self.V_DOPPLER = h5py_helper.retrieve_data(f, 'Measurement/V_DOPPLER', np.float64)

                #Checking if there are any limb-viewing geometries
                if np.nanmin(self.EMISS_ANG)<0.0:
                    self.TANHE = h5py_helper.retrieve_data(f, 'Measurement/TANHE', np.array)

                #Checking if there are any nadir-viewing / upward looking geometries
                if np.nanmax(self.EMISS_ANG) >= 0.0:
                    self.SOL_ANG = h5py_helper.retrieve_data(f, 'Measurement/SOL_ANG', np.array)
                    self.AZI_ANG = h5py_helper.retrieve_data(f, 'Measurement/AZI_ANG', np.array)


                self.NCONV = h5py_helper.retrieve_data(f, 'Measurement/NCONV', np.array)
                self.WOFF = h5py_helper.retrieve_data(f, 'Measurement/WOFF', np.float64)
                self.VCONV = h5py_helper.retrieve_data(f, 'Measurement/VCONV', np.array) + self.WOFF
                self.MEAS = h5py_helper.retrieve_data(f, 'Measurement/MEAS', np.array)
                self.ERRMEAS = h5py_helper.retrieve_data(f, 'Measurement/ERRMEAS', np.array)

                self.FWHM = h5py_helper.retrieve_data(f, 'Measurement/FWHM', np.float64)
                if self.FWHM > 0.0:
                    self.ISHAPE = h5py_helper.retrieve_data(f, 'Measurement/ISHAPE', lambda x:  InstrumentLineshape(np.int32(x)))
                elif self.FWHM < 0.0:
                    self.NFIL = h5py_helper.retrieve_data(f, 'Measurement/NFIL', np.array)
                    self.VFIL = h5py_helper.retrieve_data(f, 'Measurement/VFIL', np.array)
                    self.AFIL = h5py_helper.retrieve_data(f, 'Measurement/AFIL', np.array)

                if 'Measurement/VFMERR' in f:
                    self.VFMERR = h5py_helper.retrieve_data(f, 'Measurement/VFMERR', np.array)
                    self.FMERR = h5py_helper.retrieve_data(f, 'Measurement/FMERR', np.array)

                if 'Measurement/NORDERS_AOTF' in f:
                    self.NORDERS_AOTF = h5py_helper.retrieve_data(f, 'Measurement/NORDERS_AOTF', np.int32)
                    self.VCONV_AOTF = h5py_helper.retrieve_data(f, 'Measurement/VCONV_AOTF', np.array)
                    self.TRANS_AOTF = h5py_helper.retrieve_data(f, 'Measurement/TRANS_AOTF', np.array)
                    if self.FWHM < 0.0:
                        self.NFIL_AOTF = h5py_helper.retrieve_data(f, 'Measurement/NFIL_AOTF', np.array)
                        self.VFIL_AOTF = h5py_helper.retrieve_data(f, 'Measurement/VFIL_AOTF', np.array)
                        self.AFIL_AOTF = h5py_helper.retrieve_data(f, 'Measurement/AFIL_AOTF', np.array)

        self.assess()
        
        self.build_ils() 
        if calc_MeasurementVector==True:
            self.calc_MeasurementVector()
            self.add_fmerr()
                         
    #################################################################################################################
            
    def read_spx(self):
    
        """
        Read the .spx file and fill the attributes and parameters of the Measurement class.
        """

        #Opening file
        f = open(self.runname+'.spx','r')

        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=4,dtype='float')
        inst_fwhm = float(tmp[0])
        xlat = float(tmp[1])
        xlon = float(tmp[2])
        ngeom = int(tmp[3])

        #Defining variables
        navmax = 100
        nconvmax = 15000
        nconv = np.zeros([ngeom],dtype='int')
        nav = np.zeros([ngeom],dtype='int')
        flattmp = np.zeros([ngeom,navmax])
        flontmp = np.zeros([ngeom,navmax])
        sol_angtmp = np.zeros([ngeom,navmax])
        emiss_angtmp = np.zeros([ngeom,navmax])
        azi_angtmp = np.zeros([ngeom,navmax])
        wgeomtmp = np.zeros([ngeom,navmax])
        wavetmp = np.zeros([nconvmax,ngeom])
        meastmp = np.zeros([nconvmax,ngeom])
        errmeastmp = np.zeros([nconvmax,ngeom])
        for i in range(ngeom):
            nconv[i] = int(f.readline().strip())
            nav[i] = int(f.readline().strip())
            for j in range(nav[i]):
                tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
                flattmp[i,j] = float(tmp[0])
                flontmp[i,j] = float(tmp[1])
                sol_angtmp[i,j] = float(tmp[2])
                emiss_angtmp[i,j] = float(tmp[3])
                azi_angtmp[i,j] = float(tmp[4])
                wgeomtmp[i,j] = float(tmp[5])
            for iconv in range(nconv[i]):
                tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
                wavetmp[iconv,i] = float(tmp[0])
                meastmp[iconv,i] = float(tmp[1])
                errmeastmp[iconv,i] = float(tmp[2])

        #Making final arrays for the measured spectra
        nconvmax2 = max(nconv)
        navmax2 = max(nav)
        wave = np.zeros([nconvmax2,ngeom])
        meas = np.zeros([nconvmax2,ngeom])
        errmeas = np.zeros([nconvmax2,ngeom])
        flat = np.zeros([ngeom,navmax2])
        flon = np.zeros([ngeom,navmax2])
        sol_ang = np.zeros([ngeom,navmax2])
        emiss_ang = np.zeros([ngeom,navmax2])
        azi_ang = np.zeros([ngeom,navmax2])
        wgeom = np.zeros([ngeom,navmax2])
        for i in range(ngeom):
            wave[0:nconv[i],i] = wavetmp[0:nconv[i],i] + self.WOFF
            meas[0:nconv[i],i] = meastmp[0:nconv[i],i]
            errmeas[0:nconv[i],i] = errmeastmp[0:nconv[i],i]  
            flat[i,0:nav[i]] = flattmp[i,0:nav[i]]
            flon[i,0:nav[i]] = flontmp[i,0:nav[i]]
            sol_ang[i,0:nav[i]] = sol_angtmp[i,0:nav[i]]
            emiss_ang[i,0:nav[i]] = emiss_angtmp[i,0:nav[i]]
            azi_ang[i,0:nav[i]] = azi_angtmp[i,0:nav[i]]
            wgeom[i,0:nav[i]] = wgeomtmp[i,0:nav[i]]
        self.FWHM = inst_fwhm
        self.LATITUDE = xlat
        self.LONGITUDE = xlon
        self.NGEOM = ngeom
        self.NCONV = nconv
        self.NAV = nav
        self.edit_VCONV(wave)
        self.edit_MEAS(meas)
        self.edit_ERRMEAS(errmeas)
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_WGEOM(wgeom)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_SOL_ANG(sol_ang)
        if self.EMISS_ANG.min()<0.0:
            self.edit_TANHE(sol_ang)
        self.edit_AZI_ANG(azi_ang)

        self.calc_MeasurementVector()
            
    #################################################################################################################
            
    def read_spx_SO(self):
        """
        Read the .spx file and fill the attributes and parameters of the Measurement class.
        This routine is specific for solar occultation and limb observations
        """

        #Opening file
        f = open(self.runname+'.spx','r')
    
        #Reading first line
        tmp = np.fromfile(f,sep=' ',count=4,dtype='float')
        inst_fwhm = float(tmp[0])
        xlat = float(tmp[1])
        xlon = float(tmp[2])
        ngeom = int(tmp[3])
    
        #Defining variables
        nav = 1 #it needs to be generalized to read more than one NAV per observation geometry
        nconv = np.zeros([ngeom],dtype='int')
        flat = np.zeros([ngeom,nav])
        flon = np.zeros([ngeom,nav])
        tanhe = np.zeros([ngeom,nav])
        wgeom = np.zeros([ngeom,nav])
        nconvmax = 20000
        emiss_ang = np.zeros((ngeom,nav))
        wavetmp = np.zeros([nconvmax,ngeom])
        meastmp = np.zeros([nconvmax,ngeom])
        errmeastmp = np.zeros([nconvmax,ngeom])
        for i in range(ngeom):
            nconv[i] = int(f.readline().strip())
            for j in range(nav):
                _ = int(f.readline().strip()) # navsel
                tmp = np.fromfile(f,sep=' ',count=6,dtype='float')
                flat[i,j] = float(tmp[0])
                flon[i,j] = float(tmp[1])
                tanhe[i,j] = float(tmp[2])
                emiss_ang[i,j] = float(tmp[3])
                wgeom[i,j] = float(tmp[5])
            for iconv in range(nconv[i]):
                tmp = np.fromfile(f,sep=' ',count=3,dtype='float')
                wavetmp[iconv,i] = float(tmp[0])
                meastmp[iconv,i] = float(tmp[1])
                errmeastmp[iconv,i] = float(tmp[2])


        #Making final arrays for the measured spectra
        nconvmax2 = max(nconv)
        wave = np.zeros([nconvmax2,ngeom])
        meas = np.zeros([nconvmax2,ngeom])
        errmeas = np.zeros([nconvmax2,ngeom])
        for i in range(ngeom):
            wave[0:nconv[i],:] = wavetmp[0:nconv[i],:] + self.WOFF
            meas[0:nconv[i],:] = meastmp[0:nconv[i],:]
            errmeas[0:nconv[i],:] = errmeastmp[0:nconv[i],:]

        self.NGEOM = ngeom
        self.FWHM = inst_fwhm
        self.LATITUDE = xlat
        self.LONGITUDE = xlon
        self.NCONV = nconv
        self.NAV = np.ones(ngeom,dtype='int32')

        self.edit_WGEOM(wgeom)
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_VCONV(wave)
        self.edit_MEAS(meas)
        self.edit_ERRMEAS(errmeas)
        self.edit_TANHE(tanhe)
        self.edit_EMISS_ANG(emiss_ang)

        self.calc_MeasurementVector()

    #################################################################################################################

    def write_spx(self):
    
        """
        Write the .spx file based on the information on the Measurement class
        """

        fspx = open(self.runname+'.spx','w')
        fspx.write('%7.5f \t %7.5f \t %7.5f \t %i \n' % (self.FWHM,self.LATITUDE,self.LONGITUDE,self.NGEOM))

        for i in range(self.NGEOM):
            fspx.write('\t %i \n' % (self.NCONV[i]))
            fspx.write('\t %i \n' % (self.NAV[i]))
            for j in range(self.NAV[i]):
                fspx.write('\t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t \n' % (self.FLAT[i,j],self.FLON[i,j],self.SOL_ANG[i,j],self.EMISS_ANG[i,j],self.AZI_ANG[i,j],self.WGEOM[i,j]))
            for k in range(self.NCONV[i]):
                fspx.write('\t %10.5f \t %20.7e \t %20.7e \n' % (self.VCONV[k,i],self.MEAS[k,i],self.ERRMEAS[k,i]))

        fspx.close()

    #################################################################################################################

    def write_spx_SO(self):
    
        """
        Write the .spx file for a solar occultation measurement
        """

        if self.TANHE is None:
            _lgr.warning('Not writing *.spx (Solar Occultation) file as self.TANHE is NONE')
            return

        fspx = open(self.runname+'.spx','w')
        fspx.write('%7.5f \t %7.5f \t %7.5f \t %i \n' % (self.FWHM,self.LATITUDE,self.LONGITUDE,self.NGEOM))

        for i in range(self.NGEOM):
            fspx.write('\t %i \n' % (self.NCONV[i]))
            fspx.write('\t %i \n' % (1))
            dummy1 = -1.0
            dummy2 = 180.0
            dummy3 = 1.0
            fspx.write('\t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t %7.4f \t \n' % (self.LATITUDE,self.LONGITUDE,self.TANHE[i,0],dummy1,dummy2,dummy3))
            for k in range(self.NCONV[i]):
                fspx.write('\t %10.5f \t %20.7f \t %20.7f \n' % (self.VCONV[k,i],self.MEAS[k,i],self.ERRMEAS[k,i]))

        fspx.close()

    #################################################################################################################

    def read_sha(self):
        """
        Read the .sha file to see what the Instrument Lineshape.
        This file is only read if FWHM>0.
        """

        #Opening file
        with open(self.runname+'.sha','r') as f:
            s = f.readline().split()
        lineshape = int(s[0])
        self.ISHAPE = InstrumentLineshape(lineshape)
        self.build_ils()

    #################################################################################################################

    def write_sha(self):
    
        """
        Write the .sha file to define the shape of the Instrument function
        (Only valid if FWHM>0.0)
        """

        if self.FWHM < 0.0:
            raise ValueError('error in write_sha() :: The .sha file is only used if FWHM>0')

        with open(self.runname+'.sha','w') as f:
            f.write("%i \n" %  (int(self.ISHAPE)))

    #################################################################################################################

    def read_fil(self,MakePlot=False):
    
        """
        Read the .fil file to see what the Instrument Lineshape for each convolution wavenumber 
        This file is only read if FWHM<0.
        """

        #Opening file
        with open(self.runname+'.fil','r') as f:
    
            #Reading first and second lines
            nconv = int(np.fromfile(f,sep=' ',count=1,dtype='int'))
            wave = np.zeros([nconv],dtype='d')
            nfil = np.zeros([nconv],dtype='int')
            nfilmax = 100000
            vfil1 = np.zeros([nfilmax,nconv],dtype='d')
            afil1 = np.zeros([nfilmax,nconv],dtype='d')
            for i in range(nconv):
                wave[i] = np.fromfile(f,sep=' ',count=1,dtype='d')
                nfil[i] = np.fromfile(f,sep=' ',count=1,dtype='int')
                for j in range(nfil[i]):
                    tmp = np.fromfile(f,sep=' ',count=2,dtype='d')
                    vfil1[j,i] = tmp[0]
                    afil1[j,i] = tmp[1]

        nfil1 = nfil.max()
        vfil = np.zeros([nfil1,nconv],dtype='d')
        afil = np.zeros([nfil1,nconv],dtype='d')
        for i in range(nconv):
            vfil[0:nfil[i],i] = vfil1[0:nfil[i],i]
            afil[0:nfil[i],i] = afil1[0:nfil[i],i]
    
        if self.NCONV[0]!=nconv:
            raise ValueError('error :: Number of convolution wavelengths in .fil and .spx files must be the same')

        self.NFIL = nfil
        self.VFIL = vfil
        self.AFIL = afil

        if MakePlot==True:
            fsize = 11
            axis_font = {'size':str(fsize)}
            fig, ([ax1,ax2,ax3]) = plt.subplots(1,3,figsize=(12,4))
        
            ix = 0  #First wavenumber
            ax1.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax1.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax1.set_ylabel(r'f($\nu$)',**axis_font)
            ax1.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax1.ticklabel_format(useOffset=False)
            ax1.grid()
        
            ix = int(nconv/2)-1  #Centre wavenumber
            ax2.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax2.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax2.set_ylabel(r'f($\nu$)',**axis_font)
            ax2.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax2.ticklabel_format(useOffset=False)
            ax2.grid()
        
            ix = nconv-1  #Last wavenumber
            ax3.plot(vfil[0:nfil[ix],ix],afil[0:nfil[ix],ix],linewidth=2.)
            ax3.set_xlabel(r'Wavenumber $\nu$ (cm$^{-1}$)',**axis_font)
            ax3.set_ylabel(r'f($\nu$)',**axis_font)
            ax3.set_xlim([vfil[0:nfil[ix],ix].min(),vfil[0:nfil[ix],ix].max()])
            ax3.ticklabel_format(useOffset=False)
            ax3.grid()
        
            plt.tight_layout()
            plt.show()

    #################################################################################################################

    def write_fil(self,IGEOM=0):
    
        """
        Write the .fil file to see what the Instrument Lineshape for each convolution wavenumber 
        (Only valid if FWHM<0.0)
        """
        
        if self.NFIL is None:
            _lgr.warning('Not writing *.fil file as self.NFIL is NONE')
            return

        with open(self.runname+'.fil','w') as f:
            f.write("%i \n" %  (self.NCONV[IGEOM]))

            #Running for each spectral point
            for i in range(self.NCONV[IGEOM]):
                f.write("%10.7f\n" % self.VCONV[i,IGEOM])

                f.write("%i \n" %  (self.NFIL[i]))
                for j in range(self.NFIL[i]):
                    f.write("%10.10f %10.10e\n" % (self.VFIL[j,i], self.AFIL[j,i]) )

    #################################################################################################################
            
    def edit_VCONV(self, VCONV_array):
        """
        Edit the convolution wavelengths/wavenumbers array in each geometry

        Parameters
        ----------
        VCONV_array : 2D array, float (NCONV,NGEOM)
            Convolution wavelengths/wavenumbers of the spectrum in each geometry

        """
        VCONV_array = np.array(VCONV_array)
        try:
            assert VCONV_array.shape == (self.NCONV.max(), self.NGEOM),\
                'VCONV should be NCONV by NGEOM.'
        except:
            assert VCONV_array.shape == (self.NCONV[0]) and self.NGEOM==1,\
                'VCONV should be NCONV.'

        self.VCONV = VCONV_array

    #################################################################################################################

    def edit_MEAS(self, MEAS_array):
        """
        Edit the measured spectrum in each geometry in each geometry

        Parameters
        ----------
        MEAS_array : 2D array, float (NCONV,NGEOM)
            Measured spectrum in each geometry

        """
        MEAS_array = np.array(MEAS_array)
        try:
            assert MEAS_array.shape == (self.NCONV.max(), self.NGEOM),\
                'MEAS should be NCONV by NGEOM.'
        except:
            assert MEAS_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'MEAS should be NCONV.'

        self.MEAS = MEAS_array

    #################################################################################################################

    def edit_ERRMEAS(self, ERRMEAS_array):
        """
        Edit the measured uncertainty of the spectrum in each geometry

        Parameters
        ----------
        ERRMEAS_array : 2D array, float (NCONV,NGEOM)
            Measured uncertainty of the spectrum in each geometry
        """
        ERRMEAS_array = np.array(ERRMEAS_array)
        try:
            assert ERRMEAS_array.shape == (self.NCONV.max(), self.NGEOM),\
                'ERRMEAS should be NCONV by NGEOM.'
        except:
            assert ERRMEAS_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'ERRMEAS should be NCONV.'

        self.ERRMEAS = ERRMEAS_array

    #################################################################################################################

    def edit_SPECMOD(self, SPECMOD_array):
        """
        Edit the modelled spectrum in each geometry in each geometry

        Parameters
        ----------
        SPECMOD_array : 2D array, float (NCONV,NGEOM)
            Modelled spectrum in each geometry

        """
        SPECMOD_array = np.array(SPECMOD_array)
        try:
            assert SPECMOD_array.shape == (self.NCONV.max(), self.NGEOM),\
                'SPECMOD should be NCONV by NGEOM.'
        except:
            assert SPECMOD_array.shape == (self.NCONV,) and self.NGEOM==1,\
                'SPECMOD should be NCONV.'

        self.SPECMOD = SPECMOD_array

    #################################################################################################################

    def edit_FLAT(self, FLAT_array):
        """
        Edit the latitude of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        FLAT_array : 2D array, float (NAV,NGEOM)
            Latitude of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        FLAT_array = np.array(FLAT_array)
        try:
            assert FLAT_array.shape == (self.NGEOM, self.NAV.max()),\
                'FLAT should be NGEOM by NAV.'
        except:
            assert FLAT_array.shape == (self.NGEOM,self.NAV) and self.NGEOM==1,\
                'FLAT should be NAV.'

        self.FLAT = FLAT_array

    #################################################################################################################

    def edit_FLON(self, FLON_array):
        """
        Edit the longitude of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        FLON_array : 2D array, float (NAV,NGEOM)
            Longitude of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        FLON_array = np.array(FLON_array)

        assert FLON_array.shape == (self.NGEOM, self.NAV.max()),\
            'FLON should be NGEOM by NAV.'

        self.FLON = FLON_array

    #################################################################################################################

    def edit_SOL_ANG(self, SOL_ANG_array):
        """
        Edit the solar zenith angle of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        SOL_ANG_array : 2D array, float (NAV,NGEOM)
            Solar zenith angle of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        SOL_ANG_array = np.array(SOL_ANG_array)
        
        assert SOL_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'SOL_ANG should be NGEOM by NAV.'

        self.SOL_ANG = SOL_ANG_array

    #################################################################################################################

    def edit_EMISS_ANG(self, EMISS_ANG_array):
        """
        Edit the emission angle of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        EMISS_ANG_array : 2D array, float (NAV,NGEOM)
            Azimuth angle of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        EMISS_ANG_array = np.array(EMISS_ANG_array)
        
        assert EMISS_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'EMISS_ANG should be NGEOM by NAV.'

        self.EMISS_ANG = EMISS_ANG_array

    #################################################################################################################

    def edit_AZI_ANG(self, AZI_ANG_array):
        """
        Edit the azimuth angle of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        AZI_ANG_array : 2D array, float (NAV,NGEOM)
            Azimuth angle of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """

        AZI_ANG_array = np.array(AZI_ANG_array)
        
        assert AZI_ANG_array.shape == (self.NGEOM, self.NAV.max()),\
            'AZI_ANG should be NGEOM by NAV.'

        self.AZI_ANG = AZI_ANG_array

    #################################################################################################################

    def edit_TANHE(self, TANHE_array):
        """
        Edit the tangent height of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        TANHE_array : 2D array, float (NAV,NGEOM)
            Tangent height of each averaging point needed
            to reconstruct the FOV (when NAV > 1)
        """
        TANHE_array = np.array(TANHE_array)
        
        assert TANHE_array.shape == (self.NGEOM, self.NAV.max()),\
            'TANHE should be NGEOM by NAV.'

        self.TANHE = TANHE_array

    #################################################################################################################

    def edit_WGEOM(self, WGEOM_array):
        """
        Edit the weight of each sub-geometry needed to reconstruct the FOV (when NAV > 1)

        Parameters 
        ----------
        WGEOM_array : 2D array, float (NAV,NGEOM)
            Weight of each averaging point needed
            to reconstruct the FOV (when NAV > 1)

        """
        WGEOM_array = np.array(WGEOM_array)
        
        assert WGEOM_array.shape == (self.NGEOM, self.NAV.max()),\
            'WGEOM should be NGEOM by NAV.'

        self.WGEOM = WGEOM_array

    #################################################################################################################

    def add_fmerr(self):
        """
        Add a forward modelling error to the measurement error covariance matrix
        """

        if self.VFMERR is None or self.FMERR is None:
            return 
        
        verr = self.VFMERR
        fmerr = self.FMERR

        if self.SE is None:
            raise ValueError('error in add_fmerr :: Measurement error covariance matrix has not been calculated yet')

        fmerrx = np.zeros(self.NY)
        ix = 0
        for i in range(self.NGEOM):

            fmerrx[ix:ix+self.NCONV[i]] = np.interp(self.VCONV[0:self.NCONV[i],i],verr,fmerr,left=fmerr[0],right=fmerr[-1])
            ix = ix + self.NCONV[i]

        self.SE += np.diag(fmerrx**2)

    #################################################################################################################

    def calc_MeasurementVector(self):
        """
        Calculate the measurement vector based on the other parameters
        defined in this class
        """

        self.NY = np.sum(self.NCONV)
        y1 = np.zeros(self.NY)
        se1 = np.zeros(self.NY)
        ix = 0
        for i in range(self.NGEOM):
            y1[ix:ix+self.NCONV[i]] = self.MEAS[0:self.NCONV[i],i]
            se1[ix:ix+self.NCONV[i]] = self.ERRMEAS[0:self.NCONV[i],i]
            ix = ix + self.NCONV[i]

        self.Y = y1
        se = np.zeros([self.NY,self.NY])
        for i in range(self.NY):
            se[i,i] = se1[i]**2.

        self.SE = se

    #################################################################################################################

    def remove_geometry(self,IGEOM):
        """
        Remove one spectrum (i.e., one geometry) from the Measurement class

        Parameters
        ----------
        IGEOM : int
            Integer indicating the geometry to be erased (from 0 to NGEOM-1)

        """

        if IGEOM>self.NGEOM-1:
            raise ValueError('error in remove_geometry :: IGEOM must be between 0 and NGEOM')

        self.NGEOM = self.NGEOM - 1
        self.NCONV = np.delete(self.NCONV,IGEOM,axis=0)

        self.VCONV = np.delete(self.VCONV,IGEOM,axis=1)
        self.MEAS = np.delete(self.MEAS,IGEOM,axis=1)
        self.ERRMEAS = np.delete(self.ERRMEAS,IGEOM,axis=1)
        self.FLAT = np.delete(self.FLAT,IGEOM,axis=0)
        self.FLON = np.delete(self.FLON,IGEOM,axis=0)

        if isinstance(self.TANHE,np.ndarray)==True:
            self.TANHE = np.delete(self.TANHE,IGEOM,axis=0)
        if isinstance(self.SOL_ANG,np.ndarray)==True:
            self.SOL_ANG = np.delete(self.SOL_ANG,IGEOM,axis=0)
        if isinstance(self.EMISS_ANG,np.ndarray)==True:
            self.EMISS_ANG = np.delete(self.EMISS_ANG,IGEOM,axis=0)
        if isinstance(self.AZI_ANG,np.ndarray)==True:
            self.AZI_ANG = np.delete(self.AZI_ANG,IGEOM,axis=0)
        if isinstance(self.WGEOM,np.ndarray)==True:
            self.WGEOM = np.delete(self.WGEOM,IGEOM,axis=0)
            
        self.assess()
        
    #################################################################################################################
        
    def select_geometry(self,IGEOM):
        """
        Select only one spectrum (i.e., one geometry) from the Measurement class
        and delete the rest of them

        Parameters
        ----------
        IGEOM : int
            Integer indicating the geometry to be selected (from 0 to NGEOM-1)

        """

        if IGEOM>self.NGEOM-1:
            raise ValueError('error in select_geometry :: IGEOM must be between 0 and NGEOM')

        self.NGEOM = 1
        NCONV = np.zeros(self.NGEOM,dtype='int32')
        NCONV[0] = self.NCONV[IGEOM]
        self.NCONV = NCONV
        
        VCONV = np.zeros((NCONV.max(),1))
        MEAS = np.zeros((NCONV.max(),1))
        ERRMEAS = np.zeros((NCONV.max(),1))
        VCONV[:,0] = self.VCONV[0:NCONV[0],IGEOM]
        MEAS[:,0] = self.MEAS[0:NCONV[0],IGEOM]
        ERRMEAS[:,0] = self.ERRMEAS[0:NCONV[0],IGEOM]
        
        self.edit_VCONV(VCONV)
        self.edit_MEAS(MEAS)
        self.edit_ERRMEAS(ERRMEAS)
        
        NAV = np.zeros(self.NGEOM,dtype='int32')
        NAV[0] = self.NAV[IGEOM]
        self.NAV = NAV
        
        FLAT = np.zeros((self.NGEOM,self.NAV.max()))
        FLON = np.zeros((self.NGEOM,self.NAV.max()))
        WGEOM = np.zeros((self.NGEOM,self.NAV.max()))
        FLAT[0,:] = self.FLAT[IGEOM,0:NAV[0]]
        FLON[0,:] = self.FLON[IGEOM,0:NAV[0]]
        WGEOM[0,:] = self.WGEOM[IGEOM,0:NAV[0]]
        self.edit_FLAT(FLAT)
        self.edit_FLON(FLON)
        self.edit_WGEOM(WGEOM)
        
        if isinstance(self.TANHE,np.ndarray)==True:
            TANHE = np.zeros((self.NGEOM,self.NAV.max()))
            TANHE[0,:] = self.TANHE[IGEOM,0:NAV[0]]
            self.edit_TANHE(TANHE)
            
        if isinstance(self.EMISS_ANG,np.ndarray)==True:
            EMISS_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            EMISS_ANG[0,:] = self.EMISS_ANG[IGEOM,0:NAV[0]]
            self.edit_EMISS_ANG(EMISS_ANG)
            
        if isinstance(self.SOL_ANG,np.ndarray)==True:
            SOL_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            SOL_ANG[0,:] = self.SOL_ANG[IGEOM,0:NAV[0]]
            self.edit_SOL_ANG(SOL_ANG)
            
        if isinstance(self.AZI_ANG,np.ndarray)==True:
            AZI_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            AZI_ANG[0,:] = self.AZI_ANG[IGEOM,0:NAV[0]]
            self.edit_AZI_ANG(AZI_ANG)
            
        self.assess()
          
    #################################################################################################################
          
    def select_geometries(self,IGEOM):
        """
        Select only some spectra (i.e., some geometries) from the Measurement class
        and delete the rest of them

        Parameters
        ----------
        IGEOM : 1D array
            Array of integers indicating the geometry to be selected (from 0 to NGEOM-1)

        """

        NGEOMsel = len(IGEOM)
        if np.max(IGEOM)>self.NGEOM-1:
            raise ValueError('error in select_geometries :: IGEOM must be between 0 and NGEOM')
        if np.min(IGEOM)<0:
            raise ValueError('error in select_geometries :: IGEOM must be between 0 and NGEOM')

        self.NGEOM = NGEOMsel
        NCONV = np.zeros(self.NGEOM,dtype='int32')
        NCONV[:] = self.NCONV[IGEOM]
        self.NCONV = NCONV
        
        VCONV = np.zeros((NCONV.max(),self.NGEOM))
        MEAS = np.zeros((NCONV.max(),self.NGEOM))
        ERRMEAS = np.zeros((NCONV.max(),self.NGEOM))
        for i in range(self.NGEOM):
            VCONV[0:NCONV[i],i] = self.VCONV[0:NCONV[i],IGEOM[i]]
            MEAS[0:NCONV[i],i] = self.MEAS[0:NCONV[i],IGEOM[i]]
            ERRMEAS[0:NCONV[i],i] = self.ERRMEAS[0:NCONV[i],IGEOM[i]]
        
        self.edit_VCONV(VCONV)
        self.edit_MEAS(MEAS)
        self.edit_ERRMEAS(ERRMEAS)
        
        NAV = np.zeros(self.NGEOM,dtype='int32')
        NAV[:] = self.NAV[IGEOM]
        self.NAV = NAV
        
        FLAT = np.zeros((self.NGEOM,self.NAV.max()))
        FLON = np.zeros((self.NGEOM,self.NAV.max()))
        WGEOM = np.zeros((self.NGEOM,self.NAV.max()))
        for i in range(self.NGEOM):
            FLAT[i,0:NAV[i]] = self.FLAT[IGEOM[i],0:NAV[i]]
            FLON[i,0:NAV[i]] = self.FLON[IGEOM[i],0:NAV[i]]
            WGEOM[i,0:NAV[i]] = self.WGEOM[IGEOM[i],0:NAV[i]]
        self.edit_FLAT(FLAT)
        self.edit_FLON(FLON)
        self.edit_WGEOM(WGEOM)
        
        if isinstance(self.TANHE,np.ndarray)==True:
            TANHE = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                TANHE[i,0:NAV[i]] = self.TANHE[IGEOM[i],0:NAV[i]]
            self.edit_TANHE(TANHE)
            
        if isinstance(self.EMISS_ANG,np.ndarray)==True:
            EMISS_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                EMISS_ANG[i,0:NAV[i]] = self.EMISS_ANG[IGEOM[i],0:NAV[i]]
            self.edit_EMISS_ANG(EMISS_ANG)
            
        if isinstance(self.SOL_ANG,np.ndarray)==True:
            SOL_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                SOL_ANG[i,0:NAV[i]] = self.SOL_ANG[IGEOM[i],0:NAV[i]]
            self.edit_SOL_ANG(SOL_ANG)
            
        if isinstance(self.AZI_ANG,np.ndarray)==True:
            AZI_ANG = np.zeros((self.NGEOM,self.NAV.max()))
            for i in range(self.NGEOM):
                AZI_ANG[i,0:NAV[i]] = self.AZI_ANG[IGEOM[i],0:NAV[i]]
            self.edit_AZI_ANG(AZI_ANG)
            
        self.assess()
        
    #################################################################################################################
        
    def select_TANHE_SO(self,TANHE_min,TANHE_max):
    
        """
        Based on the information of the Measurement class, update it based on selected tangent heights
        (Applicable to Solar Occultation measurements)
        """

        #Selecting the tangent heights
        iTANHE = np.where( (self.TANHE[:,0]>=TANHE_min) & (self.TANHE[:,0]<=TANHE_max) )[0]

        #Defining arrays
        ngeom = len(iTANHE)
        nav = 1 #it needs to be generalized to read more than one NAV per observation geometry
        nconv = np.zeros([ngeom],dtype='int')
        flat = np.zeros([ngeom,nav])
        flon = np.zeros([ngeom,nav])
        tanhe = np.zeros([ngeom,nav])
        wgeom = np.zeros([ngeom,nav])
        emiss_ang = np.zeros((ngeom,nav))
        wavetmp = np.zeros([self.NCONV.max(),ngeom])
        meastmp = np.zeros([self.NCONV.max(),ngeom])
        errmeastmp = np.zeros([self.NCONV.max(),ngeom])

        #Filling arrays
        nconv[:] = self.NCONV[iTANHE]
        flat[:,:] = self.FLAT[iTANHE,:]
        flon[:,:] = self.FLON[iTANHE,:]
        tanhe[:,:] = self.TANHE[iTANHE,:]
        wgeom[:,:] = self.WGEOM[iTANHE,:]
        emiss_ang[:,:] = self.EMISS_ANG[iTANHE,:]
        wavetmp[:,:] = self.VCONV[:,iTANHE]
        meastmp[:,:] = self.MEAS[:,iTANHE]
        errmeastmp[:,:] = self.ERRMEAS[:,iTANHE]

        #Updating class
        self.NGEOM = ngeom
        self.NCONV = nconv
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_TANHE(tanhe)
        self.edit_WGEOM(wgeom)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_VCONV(wavetmp)
        self.edit_MEAS(meastmp)
        self.edit_ERRMEAS(errmeastmp)

    #################################################################################################################
        
    def calc_avepoints_exoplanet(self,nmu=5,phase=0.,igeom=0):
        """
        Calculate the averaging points for exoplanet disc-averaged measurements following Irwin et al. (2020)

        The calculation of the averaging points is valid for other planets. The exoplanet tag relates to the
        assumption that the planet is tidally-locked to the star, which is used to calculate the latitude, longitude
        and stellar angles for each averaging point. 

        This code has been adapted from nemesispy

        Parameters
        ----------

        nmu : int, optional
            Number of averaging points in zenith direction 
        phase : float, optional
            Orbital phase in degrees (0 =parimary transit and 180 = secondary eclipse)
        """

        #Initialising arrays
        assert nmu >=2, "Need at least 2 quadrature rings"
        phase = phase%360
        dtr = np.pi/180 # degree to radians conversion factor
        delR = 1./nmu

        # set up the output arrays
        nsample = 1000 # large array size to hold calculations
        tablat = np.zeros(nsample) # latitudes
        tablon = np.zeros(nsample) # longitudeds
        tabzen = np.zeros(nsample) # zenith angle in quadrature scheme
        tabsol = np.zeros(nsample) # solar zenith angle
        tabazi = np.zeros(nsample) # solar azimuth angle (scattering phase angle?)
        tabwt = np.zeros(nsample)  # weight of each sample

        #Calculating mu and weights based on Gauss-Lobatto quadrature
        mu,wtmu = gauss_lobatto(2*nmu)
        mu = mu[nmu::]
        wtmu = wtmu[nmu::]

        # trace out the day/night terminator
        z_term = np.linspace(-1,1,201) # pick out some z coordinates for the terminator
        if 0<= phase <= 180:
            # terminator is to the left of z axis
            theta_term = 2*np.pi - np.arccos(z_term)
        else:
            # terminator in the right side of the disc
            theta_term = np.arccos(z_term)
        x_term = np.sin(theta_term) * np.around(np.cos(phase*dtr),14) # x coords of terminator
        r_term = np.sqrt(x_term**2+z_term**2) # radial coords of terminator
        rmin = min(r_term) # least radius (is on x axis)

        # define FOV averaging points
        isample = 0
        for imu in range(0, nmu): # quadrature rings
            r_quad = np.sqrt(1.-mu[imu]**2) # quadrature radius (from small to large)
            half_circum = np.pi*r_quad # half the circumference

            # see if the quadrature ring intersects the terminator
            # if so, find the intersection point and place a sample point there
            if r_quad > rmin: # quadrature ring intersects the terminator
                ikeep = np.where(r_term<=r_quad)
                ikeep = ikeep[0] # index of the points on the terminator with radius > r_quad
                i_intersect = np.array([ikeep[0], ikeep[-1]]) # index of two intersectionns
                x_intersect = x_term[i_intersect] # x coordinates of intersection
                z_intersect = z_term[i_intersect] # z coordinates of intersection

                # take the intersection in the upper hemisphere
                if z_intersect[1] > 0:
                    alpha_intersect = arctan(x_intersect[1],z_intersect[1])/dtr
                else:
                    alpha_intersect = arctan(x_intersect[0],z_intersect[0])/dtr

                # place the sample points on the quadrature rings on either side of the intersection
                nalpha1 = int(0.5+half_circum*(alpha_intersect/180.0)/delR) # round up; separation ~ R/nmu
                nalpha2 = int(0.5+half_circum*((180.-alpha_intersect)/180.0)/delR)

                # at least 1 point either side of the intersection
                if(nalpha1 < 2):
                    nalpha1=2
                if(nalpha2 < 2):
                    nalpha2=2

                # set the alphas of the sample points on current quadrature ring
                nalpha = nalpha1+nalpha2-1 # intersection point double counted
                alpha1 = alpha_intersect/(nalpha1-1) * np.arange(nalpha1)
                alpha2 = alpha_intersect+(180.-alpha_intersect)/(nalpha2-1) * np.arange(nalpha2)
                alpha2 = alpha2[1:(nalpha2)] # intersect was counted twice
                alpha_sample_list = np.concatenate((alpha1,alpha2))

            else: # quadrature ring does not intersect terminator
                if(half_circum > 0.0):
                    nalpha = int(0.5+half_circum/delR)
                    alpha_sample_list = 180*np.arange(nalpha)/(nalpha-1)
                else:
                    nalpha=1

            if(nalpha > 1): # more than one sample on the quadrature ring

                for ialpha in np.arange(0,nalpha):

                    alpha_sample = alpha_sample_list[ialpha]

                    thetasol_sample, azi_sample, lat_sample, lon_sample \
                        = generate_angles_exoplanet(phase,r_quad,alpha_sample)

                    # trapezium rule weights
                    if (ialpha == 0):
                        wt_trap = (alpha_sample_list[ialpha+1]-alpha_sample_list[ialpha])/2.0
                    elif (ialpha == nalpha-1):
                        wt_trap = (alpha_sample_list[ialpha]-alpha_sample_list[ialpha-1])/2.0
                    else:
                        wt_trap = (alpha_sample_list[ialpha+1]-alpha_sample_list[ialpha-1])/2.0

                    wt_azi= wt_trap/180. # sample azimuthal weight


                    tablat[isample] = lat_sample # sample lattitude
                    tablon[isample] = lon_sample # sample longitude
                    tabzen[isample] = np.arccos(mu[imu])/dtr # sample emission zenith angle
                    tabsol[isample] = thetasol_sample/dtr # sample stellar zenith angle
                    tabazi[isample] = azi_sample/dtr # sample stellar azimuth angle
                    tabwt[isample] = 2*mu[imu]*wtmu[imu]*wt_azi # sample weight
                    isample = isample+1

            else:
                alpha_sample = 0.
                thetasol_sample,azi_sample, lat_sample,lon_sample \
                    = generate_angles_exoplanet(phase,r_quad,alpha_sample)
                if(tabzen[isample] == 0.0):
                    azi_sample = 180.
                tablat[isample] = lat_sample
                tablon[isample] = lon_sample
                tabzen[isample] = np.arccos(mu[imu])/dtr
                tabsol[isample] = thetasol_sample/dtr
                tabazi[isample] = azi_sample
                tabwt[isample] = 2*mu[imu]*wtmu[imu]
                isample = isample+1

        nav = isample
        wav = np.zeros((6,isample))
        sum=0.
        for i in np.arange(0,isample):
            wav[0,i]=tablat[i]              # 0th array is lattitude
            wav[1,i]=tablon[i]%360          # 1st array is longitude
            wav[2,i]=tabsol[i]              # 2nd array is stellar zenith angle
            wav[3,i]=tabzen[i]              # 3rd array is emission zenith angle
            wav[4,i]=tabazi[i]              # 4th array is stellar azimuth angle
            wav[5,i]=tabwt[i]               # 5th array is weight
            sum = sum+tabwt[i]

        #sum should be very close to 1.0, but normalising just in case
        for i in range(isample):            # normalise weights so they add up to 1
            wav[5,i]=wav[5,i]/sum

        #Updating class


        if nav>self.NAV.max():
            #we need to resize the arrays
            flat = np.zeros((self.NGEOM,nav))
            flon = np.zeros((self.NGEOM,nav))
            tanhe = np.zeros((self.NGEOM,nav))
            wgeom = np.zeros((self.NGEOM,nav))
            emiss_ang = np.zeros((self.NGEOM,nav))
            sol_ang = np.zeros((self.NGEOM,nav))
            azi_ang = np.zeros((self.NGEOM,nav))
            for i in range(self.NGEOM):
                if self.FLAT is not None:
                    flat[i,0:self.NAV[i]] = self.FLAT[i,0:self.NAV[i]]
                if self.FLON is not None:
                    flon[i,0:self.NAV[i]] = self.FLON[i,0:self.NAV[i]]
                if self.TANHE is not None:
                    tanhe[i,0:self.NAV[i]] = self.TANHE[i,0:self.NAV[i]]
                if self.WGEOM is not None:
                    wgeom[i,0:self.NAV[i]] = self.WGEOM[i,0:self.NAV[i]]
                if self.EMISS_ANG is not None:
                    emiss_ang[i,0:self.NAV[i]] = self.EMISS_ANG[i,0:self.NAV[i]]
                if self.SOL_ANG is not None:
                    sol_ang[i,0:self.NAV[i]] = self.SOL_ANG[i,0:self.NAV[i]]
                if self.AZI_ANG is not None:
                    azi_ang[i,0:self.NAV[i]] = self.AZI_ANG[i,0:self.NAV[i]]
        else:

            flat = self.FLAT.copy()
            flon = self.FLON.copy()
            if self.TANHE is not None:
                tanhe = self.TANHE.copy()
            wgeom = self.WGEOM.copy()
            emiss_ang = self.EMISS_ANG.copy()
            sol_ang = self.SOL_ANG.copy()
            azi_ang = self.AZI_ANG.copy()

        NAV = np.zeros(self.NGEOM,dtype='int32')
        NAV[:] = self.NAV[:]
        NAV[igeom] = nav
        self.NAV = NAV
        flat[igeom,0:nav] = wav[0,0:nav]
        flon[igeom,0:nav] = wav[1,0:nav]
        sol_ang[igeom,0:nav] = wav[2,0:nav]
        emiss_ang[igeom,0:nav] = wav[3,0:nav]
        azi_ang[igeom,0:nav] = wav[4,0:nav]
        wgeom[igeom,0:nav] = wav[5,0:nav]

        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_SOL_ANG(sol_ang)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_AZI_ANG(azi_ang)
        self.edit_WGEOM(wgeom)

    #################################################################################################################
        
    def calc_geometry_primary_transit(self):
        """
        Initialise the geometry array for a primary transit observation

        The initialisation here is very simple since all the angles are fixed for a primary transit
        """
        
        if self.NGEOM!=1:
            raise ValueError('error in calc_geometry_primary_transit :: NGEOM must be equal to 1')
        
        NAV = np.ones(self.NGEOM,dtype='int32')
        
        flat = np.zeros((self.NGEOM,NAV.max()))
        flon = np.zeros((self.NGEOM,NAV.max()))
        tanhe = np.zeros((self.NGEOM,NAV.max()))
        wgeom = np.zeros((self.NGEOM,NAV.max()))
        emiss_ang = np.zeros((self.NGEOM,NAV.max()))
        sol_ang = np.zeros((self.NGEOM,NAV.max()))
        azi_ang = np.zeros((self.NGEOM,NAV.max()))

        flat[:,:] = self.LATITUDE
        flon[:,:] = self.LONGITUDE
        tanhe[:,:] = 0.0
        wgeom[:,:] = 1.0
        emiss_ang[:,:] = -90.0  #Limb viewing
        sol_ang[:,:] = 0.0      #Star at the limb
        azi_ang[:,:] = 0.0      #Arbitrary since sun at limb

        self.NAV = NAV
        self.edit_FLAT(flat)
        self.edit_FLON(flon)
        self.edit_TANHE(tanhe)
        self.edit_WGEOM(wgeom)
        self.edit_EMISS_ANG(emiss_ang)
        self.edit_SOL_ANG(sol_ang)
        self.edit_AZI_ANG(azi_ang)


    #################################################################################################################
        
    def crop_wave(self,wavemin,wavemax,iconv=None):
    
        """
        Based on the information of the Measurement class, update it based on selected minimum and maximum wavelengths
        """

        if iconv is None:
            #Selecting the tangent heights
            iconv = np.where( (self.VCONV[:,0]>=wavemin) & (self.VCONV[:,0]<=wavemax) )[0]

        if len(iconv)<=0:
            raise ValueError('error in crop_wave :: there are no wavelengths within the specified spectral range')

        #Defining arrays
        nconvx = len(iconv)
        vconvx = self.VCONV[iconv,:]
        measx = self.MEAS[iconv,:]
        errmeasx = self.ERRMEAS[iconv,:]
                
        #Updating class
        self.NCONV = np.zeros(self.NGEOM,dtype='int32') + nconvx
        self.edit_VCONV(vconvx)
        self.edit_MEAS(measx)
        self.edit_ERRMEAS(errmeasx)
        
        if self.FWHM<0.0:
            nfilx = self.NFIL[iconv]
            vfilx = self.VFIL[0:nfilx.max(),iconv]
            afilx = self.AFIL[0:nfilx.max(),iconv]
            
            self.NFIL = nfilx
            self.VFIL = vfilx
            self.AFIL = afilx

        self.assess()

    #################################################################################################################
        
    def build_ils(self,IGEOM=0):
        """
        Subroutine to calculate the instrument lineshape kernel

        Parameters
        ----------
        FWHM : float
            Full-width at half-maximum of the instrument lineshape (if negative the the lineshape is explicitly defined)
        ISHAPE : int
            Flag defining the instrument lineshape (in case it FWHM>0)
        """
        
        if self.FWHM>0.0:

            #Calculating the limits defining the ILS
            if self.ISHAPE==InstrumentLineshape.Square:   #Square lineshape
                dv = 0.5*self.FWHM
            elif self.ISHAPE==InstrumentLineshape.Triangular: #Triangular
                dv = self.FWHM
            elif self.ISHAPE==InstrumentLineshape.Gaussian: #Gaussian
                dv = 3.* 0.5 * self.FWHM / np.sqrt(np.log(2.0))
            else: 
                raise ValueError('error in build_ils :: ishape not included yet in function')
                
            #Defining the x-array for the definition of the ILS
            vwave = np.linspace(-dv,dv,101)
            nwave = len(vwave)
            ils = np.zeros(nwave)
            
            #Defining the ILS
            if self.ISHAPE==InstrumentLineshape.Square:   #Square lineshape
                ils[:] = 1.0    
            elif self.ISHAPE==InstrumentLineshape.Triangular: #Triangular
                ils[:] = 1.0 - np.abs(vwave)/self.FWHM
            elif self.ISHAPE==InstrumentLineshape.Gaussian: #Gaussian
                sig = 0.5 * self.FWHM / np.sqrt(np.log(2.0))
                ils[:] = np.exp(-((vwave)/sig)**2)
                
            # Normalize kernel area to 1
            ils_sum = ils.sum()
            if ils_sum > 0:
                ils /= ils_sum
                
            #Construct the NFIL,VFIL,AFIL arrays in the class
            nfil = np.zeros(self.NCONV[IGEOM],dtype='int32') + nwave
            vfil = np.zeros((nwave,self.NCONV[IGEOM]))
            afil = np.zeros((nwave,self.NCONV[IGEOM]))
            for i in range(self.NCONV[IGEOM]):
                vfil[:,i] = self.VCONV[i,IGEOM] + vwave
                afil[:,i] = ils[:]
                
            self.NFIL = nfil
            self.VFIL = vfil
            self.AFIL = afil
    
        elif self.FWHM<0.0:
            dv = 0.0 #This is not used in this case since the lineshape is already defined
            
    #################################################################################################################

    def calc_wave_range(self,IGEOM=None,apply_doppler=True):
        """
        Subroutine to calculate which 'calculation' wavelength range needed to 
        calculate the spectrum at the required 'convolution wavelengths'.

        Parameters
        ----------
        IGEOM : int, optional
            Integer defining a specific geometry within the Measurement class
        apply_doppler : log, optional
            Flag indicating whether the Doppler shift must be accounted for
        """

        if self.FWHM>0.0:

            if self.ISHAPE==InstrumentLineshape.Square:
                dv = 0.5*self.FWHM
            elif self.ISHAPE==InstrumentLineshape.Triangular:
                dv = self.FWHM
            elif self.ISHAPE==InstrumentLineshape.Gaussian:
                dv = 3.* 0.5 * self.FWHM / np.sqrt(np.log(2.0))
            else:
                dv = 3.*self.FWHM

            if IGEOM is not None:
                wavemin = self.VCONV[0,IGEOM] - dv
                wavemax = self.VCONV[self.NCONV[IGEOM]-1,IGEOM] + dv
            else:
                #Finding the minimum and maximum wave for all geometries
                wavemin = 1.0e10
                wavemax = 0.
                for igeom in range(self.NGEOM):
                    if ((self.VCONV[0,igeom] - dv)<wavemin):
                        wavemin = self.VCONV[0,igeom] - dv
                    if ((self.VCONV[self.NCONV[igeom]-1,igeom] + dv)>wavemax):
                        wavemax = self.VCONV[self.NCONV[igeom]-1,igeom] + dv
                        
        elif self.FWHM<0.0:

            wavemin = 1.0e10
            wavemax = 0.0
            for i in range(self.NCONV[0]):  #In this case all geometries are assumed to have the same spectral array
                vminx = self.VFIL[0,i]
                vmaxx = self.VFIL[self.NFIL[i]-1,i]
                if vminx<wavemin:
                    wavemin = vminx
                if vmaxx>wavemax:
                    wavemax= vmaxx

        elif self.FWHM==0.0:
        
            if IGEOM is not None:
                wavemin = self.VCONV[0,IGEOM]
                wavemax = self.VCONV[self.NCONV[IGEOM]-1,IGEOM]
            else:
                #Finding the minimum and maximum wave for all geometries
                wavemin = 1.0e10
                wavemax = 0.
                for igeom in range(self.NGEOM):
                    if ((self.VCONV[0,igeom])<wavemin):
                        wavemin = self.VCONV[0,igeom]
                    if ((self.VCONV[self.NCONV[igeom]-1,igeom])>wavemax):
                        wavemax = self.VCONV[self.NCONV[igeom]-1,igeom]

        #Correcting the wavelengths for Doppler shift
        if apply_doppler is True:
            if self.V_DOPPLER!=0.0:
                _lgr.info(f"nemesis :: Correcting for Doppler shift of  {(self.V_DOPPLER,'km/s')}")        
            wavemin = self.invert_doppler_shift(wavemin)
            wavemax = self.invert_doppler_shift(wavemax)
        
        #Sorting the wavenumbers if the ILS is flipped
        if wavemin>=wavemax:
            raise ValueError('error in wavesetc :: the spectral points defining the instrument lineshape must be increasing')

        return wavemin,wavemax

    #################################################################################################################

    def lblconv(self,Wave,ModSpec,IGEOM='All'):
        """
        Subroutine to convolve the Modelled spectrum with the Instrument Line Shape 

        Parameters
        ----------
        Wave : 1D or 2D array (NWAVE)
            Calculation wavelengths or wavenumbers
        ModSpec : 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum

        Other Parameters
        ----------------
        IGEOM : int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)

        Returns
        -------
        SPECONV : 1D or 2D array (NCONV,NGEOM)
            Convolved spectrum with the instrument lineshape
        """
        
        #Accounting for the Doppler shift that was previously introduced
        wavecorr = self.correct_doppler_shift(Wave)

        if self.FWHM>0.0:    #Convolution with ISHAPE
            if IGEOM=='All':
                IG = 0
                if ModSpec.ndim!=2:
                    raise ValueError('error in lblconv :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                SPECONV = lblconv_ngeom(len(Wave),wavecorr,ModSpec,self.NCONV[IG],self.VCONV[:,IG],self.ISHAPE,self.FWHM)
            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in lblconv :: ModSpec must have 1 dimensions (NWAVE)')
                IG = IGEOM
                SPECONV = lblconv(len(Wave),wavecorr,ModSpec,self.NCONV[IG],self.VCONV[:,IG],self.ISHAPE,self.FWHM)
            
        elif self.FWHM<0.0:  #Convolution with VFIL,AFIL
            if IGEOM=='All':
                if ModSpec.ndim!=2:
                    raise ValueError('error in lblconv :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                IG = 0
                SPECONV = lblconv_fil_ngeom(len(Wave),wavecorr,ModSpec,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)
            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in lblconv :: ModSpec must have 1 dimensions (NWAVE)')
                IG = IGEOM
                SPECONV = lblconv_fil(len(Wave),wavecorr,ModSpec,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)

        elif self.FWHM==0.0:  #No convolution
            if IGEOM=='All':
                if ModSpec.ndim!=2:
                    raise ValueError('error in lblconv :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                SPECONV = np.zeros(self.VCONV.shape)
                for IG in range(self.NGEOM):
                    SPECONV[:,IG] = np.interp(self.VCONV[:,IG],wavecorr,ModSpec[:,IG])
            else:
                IG = IGEOM
                SPECONV = np.interp(self.VCONV[:,IG],wavecorr,ModSpec)

        return SPECONV

    #################################################################################################################

    def lblconvg(self,Wave,ModSpec,ModGrad,IGEOM='All'):
    
        """
        Subroutine to convolve the Modelled spectrum and the gradients with the Instrument Line Shape 

        Parameters
        ----------
        Wave : 1D array (NWAVE)
            Calculation wavelengths or wavenumbers
        ModSpec : 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum
        ModGrad: 2D or 3D array (NWAVE,NGEOM,NX)
            Modelled gradients

        Other Parameters
        ----------------
        IGEOM : int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)

        Returns
        -------
        SPECONV : 1D or 2D array (NCONV,NGEOM)
            Convolved spectrum with the instrument lineshape
        dSPECONV : 2D or 3D array (NCONV,NGEOM,NX)
            Convolved gradients with the instrument lineshape
        """

        #Accounting for the Doppler shift that was previously introduced
        NWAVE = len(Wave)
        wavecorr = self.correct_doppler_shift(Wave)

        if self.FWHM>0.0:   #Convolution with ISHAPE

            if IGEOM=='All':
                IG = 0
                if ModSpec.ndim!=2:
                    raise ValueError('error in lblconvg :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                if ModGrad.ndim!=3:
                    raise ValueError('error in lblconvg :: ModGrad must have 3 dimensions (NWAVE,NGEOM,NX)')
                SPECONV,dSPECONV = lblconvg_ngeom(NWAVE,wavecorr,ModSpec,ModGrad,self.NCONV[IG],self.VCONV[:,IG],self.ISHAPE,self.FWHM)
            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in lblconvg :: ModSpec must have 1 dimensions (NWAVE)')
                if ModGrad.ndim!=2:
                    raise ValueError('error in lblconvg :: ModGrad must have 2 dimensions (NWAVE,NX)')
                IG = IGEOM
                SPECONV,dSPECONV = lblconvg(NWAVE,wavecorr,ModSpec,ModGrad,self.NCONV[IG],self.VCONV[:,IG],self.ISHAPE,self.FWHM)
            
        elif self.FWHM<0.0:  #Convolution with VFIL, AFIL

            if IGEOM=='All':
                if ModSpec.ndim!=2:
                    raise ValueError('error in lblconvg :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                if ModGrad.ndim!=3:
                    raise ValueError('error in lblconvg :: ModGrad must have 3 dimensions (NWAVE,NGEOM,NX)')
                IG = 0
                SPECONV,dSPECONV = lblconvg_fil_ngeom(NWAVE,wavecorr,ModSpec,ModGrad,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)

            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in lblconvg :: ModSpec must have 1 dimensions (NWAVE)')
                if ModGrad.ndim!=2:
                    raise ValueError('error in lblconvg :: ModGrad must have 2 dimensions (NWAVE,NX)')
                IG = IGEOM
                SPECONV,dSPECONV = lblconvg_fil(NWAVE,wavecorr,ModSpec,ModGrad,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)

        elif self.FWHM==0.0:
            
            if IGEOM=='All':
                if ModSpec.ndim!=2:
                    raise ValueError('error in lblconvg :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                if ModGrad.ndim!=3:
                    raise ValueError('error in lblconvg :: ModGrad must have 3 dimensions (NWAVE,NGEOM,NX)')
                SPECONV = ModSpec
                dSPECONV = ModGrad
                
            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in lblconvg :: ModSpec must have 1 dimensions (NWAVE)')
                if ModGrad.ndim!=2:
                    raise ValueError('error in lblconvg :: ModGrad must have 2 dimensions (NWAVE,NX)')

                SPECONV = ModSpec[:,IGEOM]
                dSPECONV = ModGrad[:,IGEOM,:]

        return SPECONV,dSPECONV

    #################################################################################################################
    
    def conv(self,Wave,ModSpec,IGEOM='All',FWHMEXIST=''):
    
        """
        Subroutine to convolve the Modelled spectrum with the Instrument Line Shape 

        Parameters
        ----------
        Wave : 1D array (NWAVE)
            Calculation wavelengths or wavenumbers
        ModSpec : 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum

        Other Parameters
        ----------------
        IGEOM : int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)
        FWHMEXIST : int
            If not '', then FWHMEXIST indicates that the .fwhm exists (that includes the variation of FWHM for each wave) and
            FWHMEXIST is expected to be the name of the Nemesis run

        Returns
        -------
        SPECONV : 1D or 2D array (NCONV,NGEOM)
            Convolved spectrum with the instrument lineshape
        """


        nstep = 20
        NWAVE = len(Wave)

        if IGEOM=='All':

            #It is assumed all geometries cover the same spectral range
            IG = 0 
            yout = np.zeros((self.NCONV[IG],self.NGEOM))
            ynor = np.zeros((self.NCONV[IG],self.NGEOM))

            if self.FWHM>0.0:

                raise ValueError('error in conv :: IGEOM=All with FWHM>0 has not yet been implemented')

            elif self.FWHM==0.0:
                
                #Channel Integrator mode where the k-tables have been previously
                #tabulated INCLUDING the filter profile. In which case all we
                #need do is just transfer the outputs
                s = sp.interpolate.interp1d(Wave,ModSpec,axis=0)
                yout[:,:] = s(self.VCONV[0:self.NCONV[IG],IG])

            elif self.FWHM<0.0:

                raise ValueError('error in conv :: IGEOM=All with FWHM<0 has not yet been implemented')

        else:

            yout = np.zeros(self.NCONV[IGEOM])
            ynor = np.zeros(self.NCONV[IGEOM])

            if self.FWHM>0.0:

                nwave1 = NWAVE
                wave1 = np.zeros(nwave1+2)
                y1 = np.zeros(nwave1+2)
                wave1[1:nwave1+1] = Wave
                y1[1:nwave1+1] = ModSpec[0:NWAVE]

                #Extrapolating the last wavenumber
                iup = 0
                if(self.VCONV[self.NCONV[IGEOM],IGEOM]>(Wave.max()-self.FWHM/2.)):
                    nwave1 = nwave1 +1
                    wave1[nwave1-1] = self.VCONV[self.NCONV[IGEOM],IGEOM] + self.FWHM
                    frac = (ModSpec[NWAVE-1]-ModSpec[NWAVE-2])/(Wave[NWAVE-1]-Wave[NWAVE-2])
                    y1[nwave1-1] = ModSpec[NWAVE-1] + frac * (wave1[nwave1-1]-Wave[NWAVE-1])
                    iup=1

                #Extrapolating the first wavenumber
                idown = 0
                if(self.VCONV[0,IGEOM]<(Wave.min()+self.FWHM/2.)):
                    nwave1 = nwave1 + 1
                    wave1[0] = self.VCONV[0,IGEOM] - self.FWHM
                    frac = (ModSpec[1] - ModSpec[2])/(Wave[1]-Wave[0])
                    y1[0] = ModSpec[0] + frac * (wave1[0] - Wave[0])
                    idown = 1

                #Re-shaping the spectrum
                nwave = self.NWAVE + iup + idown
                wave = wave1[1-idown:nwave-(1-iup)]
                y = y1[1-idown:nwave-(1-iup)]

                #Checking if .fwh file exists (indicating that FWHM varies with wavelength)
                ifwhm = 0
                if os.path.exists(FWHMEXIST+'.fwh')==True:

                    #Reading file
                    f = open(FWHMEXIST+'.fwh')
                    s = f.readline().split()
                    nfwhm = int(s[0])
                    vfwhm = np.zeros(nfwhm)
                    xfwhm = np.zeros(nfwhm)
                    for ifwhm in range(nfwhm):
                        s = f.readline().split()
                        vfwhm[ifwhm] = float(s[0])
                        xfwhm[ifwhm] = float(s[1])
                    f.close()

                    ffwhm = sp.interpolate.interp1d(vfwhm,xfwhm)
                    ifwhm==1

                fy = sp.interpolate.CubicSpline(wave,y)
                for ICONV in range(self.NCONV[IGEOM]):
                    
                    if ifwhm==1:
                        yfwhm = ffwhm(self.VCONV[ICONV,IGEOM])
                    else:
                        yfwhm = self.FWHM

                    x1 = self.VCONV[ICONV,IGEOM] - yfwhm/2.
                    x2 = self.VCONV[ICONV,IGEOM] + yfwhm/2.
                    delx = (x2-x1)/(nstep-1)
                    xi = np.linspace(x1,x2,nstep)
                    yi = fy(xi)
                    
                    yold=None
                    for j in range(nstep):
                        if j==0:
                            sum1 = 0.0 
                        else:
                            sum1 = sum1 + (yi[j] - yold) * delx/2.
                        yold = yi[j]

                    yout[ICONV] = sum1 / yfwhm

            elif self.FWHM==0.0:

                #Channel Integrator mode where the k-tables have been previously
                #tabulated INCLUDING the filter profile. In which case all we
                #need do is just transfer the outputs
                s = scipy.interpolate.interp1d(Wave,ModSpec)
                yout[:] = s(self.VCONV[0:self.NCONV[IGEOM],IGEOM])

            elif self.FWHM<0.0:

                #Channel Integrator Mode: Slightly more advanced than previous

                #In this case the filter function for each convolution wave is defined in the .fil file
                #This file has been previously read and its variables are stored in NFIL,VFIL,AFIL

                for ICONV in range(self.NCONV[IGEOM]):

                    v1 = self.VFIL[0,ICONV]
                    v2 = self.VFIL[self.NFIL[ICONV]-1,ICONV]
                    #Find relevant points in tabulated files
                    iwavelox = np.where( (Wave<v1) )
                    iwavelox = iwavelox[0]
                    iwavehix = np.where( (Wave>v2) )
                    iwavehix = iwavehix[0]
                    inwave = np.linspace(iwavelox[len(iwavelox)-1],iwavehix[0],iwavehix[0]-iwavelox[len(iwavelox)-1]+1,dtype='int32')
                    
                    np1 = len(inwave)
                    xp = np.zeros([self.NFIL[ICONV]])
                    yp = np.zeros([self.NFIL[ICONV]])
                    xp[:] = self.VFIL[0:self.NFIL[ICONV],ICONV]
                    yp[:] = self.AFIL[0:self.NFIL[ICONV],ICONV]


                    for i in range(np1):
                        #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                        f1 = np.interp(Wave[inwave[i]],xp,yp)
                        if f1>0.0:
                            yout[ICONV] = yout[ICONV] + f1*ModSpec[inwave[i]]
                            ynor[ICONV] = ynor[ICONV] + f1

                    yout[ICONV] = yout[ICONV]/ynor[ICONV]
                
        return yout

    #################################################################################################################

    def convg(self,Wave,ModSpec,ModGrad,IGEOM='All',FWHMEXIST=''):
    
        """
        Subroutine to convolve the Modelled spectrum and the gradients with the Instrument Line Shape 

        Parameters
        ----------
        ModSpec : 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum
        ModGrad: 2D or 3D array (NWAVE,NGEOM,NX)
            Modelled gradients
        
        Other Parameters
        ----------------
        IGEOM : int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)
        FWHMEXIST : int
            If not '', then FWHMEXIST indicates that the .fwhm exists (that includes the variation of FWHM for each wave) and
            FWHMEXIST is expected to be the name of the Nemesis run

        Returns
        -------
        SPECONV : 1D or 2D array (NCONV,NGEOM)
            Convolved spectrum with the instrument lineshape
        dSPECONV : 2D or 3D array (NCONV,NGEOM,NX)
            Convolved gradients with the instrument lineshape
        """

        import os.path
        from scipy import interpolate

        nstep = 20
        NWAVE = len(Wave)

        if IGEOM=='All':

            #It is assumed all geometries cover the same spectral range
            IG = 0 
            NX = len(ModGrad[0,0,:])
            yout = np.zeros((self.NCONV[IG],self.NGEOM))
            ynor = np.zeros((self.NCONV[IG],self.NGEOM))
            gradout = np.zeros((self.NCONV[IG],self.NGEOM,NX))
            gradnorm = np.zeros((self.NCONV[IG],self.NGEOM,NX))

            if self.FWHM>0.0:

                raise ValueError('error in convg :: IGEOM=All with FWHM>0 has not yet been implemented')

            elif self.FWHM==0.0:

                #Channel Integrator mode where the k-tables have been previously
                #tabulated INCLUDING the filter profile. In which case all we
                #need do is just transfer the outputs
                s = scipy.interpolate.interp1d(Wave,ModSpec,axis=0)
                yout[:,:] = s(self.VCONV[0:self.NCONV[IG],IG])
                
                s = scipy.interpolate.interp1d(Wave,ModGrad,axis=0)
                gradout[:,:,:] = s(self.VCONV[0:self.NCONV[IG],IG])

            elif self.FWHM<0.0:

                raise ValueError('error in convg :: IGEOM=All with FWHM<0 has not yet been implemented')
            

        else:

            yout = np.zeros(self.NCONV[IGEOM])
            ynor = np.zeros(self.NCONV[IGEOM])
            NX = len(ModGrad[0,:])
            gradout = np.zeros((self.NCONV[IGEOM],NX))
            gradnorm = np.zeros((self.NCONV[IGEOM],NX))

            if self.FWHM>0.0:

                nwave1 = NWAVE
                wave1 = np.zeros(nwave1+2)
                y1 = np.zeros(nwave1+2)
                grad1 = np.zeros((nwave1+2,NX))
                wave1[1:nwave1+1] = Wave
                y1[1:nwave1+1] = ModSpec[0:NWAVE]
                grad1[1:nwave1+1,:] = ModGrad[0:NWAVE,:]

                #Extrapolating the last wavenumber
                iup = 0
                if(self.VCONV[self.NCONV[IGEOM],IGEOM]>(Wave.max()-self.FWHM/2.)):
                    nwave1 = nwave1 +1
                    wave1[nwave1-1] = self.VCONV[self.NCONV[IGEOM],IGEOM] + self.FWHM
                    frac = (ModSpec[NWAVE-1]-ModSpec[NWAVE-2])/(Wave[NWAVE-1]-Wave[NWAVE-2])
                    y1[nwave1-1] = ModSpec[NWAVE-1] + frac * (wave1[nwave1-1]-Wave[NWAVE-1])
                    grad1[nwave1-1,:] = ModGrad[NWAVE-1,:] + frac * (wave1[nwave1-1]-Wave[NWAVE-1])
                    iup=1

                #Extrapolating the first wavenumber
                idown = 0
                if(self.VCONV[0,IGEOM]<(Wave.min()+self.FWHM/2.)):
                    nwave1 = nwave1 + 1
                    wave1[0] = self.VCONV[0,IGEOM] - self.FWHM
                    frac = (ModSpec[1] - ModSpec[2])/(Wave[1]-Wave[0])
                    y1[0] = ModSpec[0] + frac * (wave1[0] - Wave[0])
                    grad1[0,:] = ModGrad[0,:] + frac * (wave1[0] - Wave[0])
                    idown = 1

                #Re-shaping the spectrum
                nwave = nwave1 + iup + idown
                wave = np.zeros(nwave)
                y = np.zeros(nwave)
                grad = np.zeros((nwave,NX))
                if((idown==1) & (iup==1)):
                    wave[:] = wave1[:]
                    y[:] = y1[:]
                    grad[:,:] = grad1[:,:]
                elif((idown==1) & (iup==0)):
                    wave[0:nwave] = wave1[0:nwave1-1]
                    y[0:nwave] = y1[0:nwave1-1]
                    grad[0:nwave,:] = grad1[0:nwave1-1,:]
                elif((idown==0) & (iup==1)):
                    wave[0:nwave] = wave1[1:nwave1]
                    y[0:nwave] = y1[1:nwave1]
                    grad[0:nwave,:] = grad1[1:nwave1,:]
                else:
                    wave[0:nwave] = wave1[1:nwave1-1]
                    y[0:nwave] = y1[1:nwave1-1]
                    grad[0:nwave,:] = grad1[1:nwave1-1,:]

                #Checking if .fwh file exists (indicating that FWHM varies with wavelength)
                ifwhm = 0
                if os.path.exists(FWHMEXIST+'.fwh')==True:

                    #Reading file
                    f = open(FWHMEXIST+'.fwh')
                    s = f.readline().split()
                    nfwhm = int(s[0])
                    vfwhm = np.zeros(nfwhm)
                    xfwhm = np.zeros(nfwhm)
                    for ifwhm in range(nfwhm):
                        s = f.readline().split()
                        vfwhm[ifwhm] = float(s[0])
                        xfwhm[ifwhm] = float(s[1])
                    f.close()

                    ffwhm = interpolate.interp1d(vfwhm,xfwhm)
                    ifwhm==1

                fy = interpolate.CubicSpline(wave,y)
                fpy = []
                for IX in range(NX):
                    fpy1 = interpolate.CubicSpline(wave,grad[:,IX])
                    fpy.append(fpy1)

                _lgr.info(fpy)
                _lgr.info('error in convg :: This part of the programme has not been tested yet')
                raise ValueError()
                
                for ICONV in range(self.NCONV[IGEOM]):
                    
                    if ifwhm==1:
                        yfwhm = ffwhm(self.VCONV[ICONV,IGEOM])
                    else:
                        yfwhm = self.FWHM

                    x1 = self.VCONV[ICONV,IGEOM] - yfwhm/2.
                    x2 = self.VCONV[ICONV,IGEOM] + yfwhm/2.
                    delx = (x2-x1)/(nstep-1)
                    xi = np.linspace(x1,x2,nstep)
                    yi = fy(xi)
                    
                    yold=None
                    for j in range(nstep):
                        if j==0:
                            sum1 = 0.0 
                        else:
                            sum1 = sum1 + (yi[j] - yold) * delx/2.
                        yold = yi[j]

                    yout[ICONV] = sum1 / yfwhm

            elif self.FWHM==0.0:

                #Channel Integrator mode where the k-tables have been previously
                #tabulated INCLUDING the filter profile. In which case all we
                #need do is just transfer the outputs
                s = scipy.interpolate.interp1d(Wave,ModSpec)
                yout[:] = s(self.VCONV[0:self.NCONV[IGEOM],IGEOM])
                
                s = scipy.interpolate.interp1d(Wave,ModGrad,axis=0)
                gradout[:,:] = s(self.VCONV[0:self.NCONV[IGEOM],IGEOM])

            elif self.FWHM<0.0:

                #Channel Integrator Mode: Slightly more advanced than previous

                #In this case the filter function for each convolution wave is defined in the .fil file
                #This file has been previously read and its variables are stored in NFIL,VFIL,AFIL

                for ICONV in range(self.NCONV[IGEOM]):

                    v1 = self.VFIL[0,ICONV]
                    v2 = self.VFIL[self.NFIL[ICONV]-1,ICONV]
                    #Find relevant points in tabulated files
                    iwavelox = np.where( (Wave<v1) )
                    iwavelox = iwavelox[0]
                    iwavehix = np.where( (Wave>v2) )
                    iwavehix = iwavehix[0]
                    inwave = np.linspace(iwavelox[len(iwavelox)-1],iwavehix[0],iwavehix[0]-iwavelox[len(iwavelox)-1]+1,dtype='int32')
                    
                    np1 = len(inwave)
                    xp = np.zeros([self.NFIL[ICONV]])
                    yp = np.zeros([self.NFIL[ICONV]])
                    xp[:] = self.VFIL[0:self.NFIL[ICONV],ICONV]
                    yp[:] = self.AFIL[0:self.NFIL[ICONV],ICONV]


                    for i in range(np1):
                        #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                        f1 = np.interp(Wave[inwave[i]],xp,yp)
                        if f1>0.0:
                            yout[ICONV] = yout[ICONV] + f1*ModSpec[inwave[i]]
                            ynor[ICONV] = ynor[ICONV] + f1
                            gradout[ICONV,:] = gradout[ICONV,:] + f1*ModGrad[inwave[i],:]
                            gradnorm[ICONV,:] = gradnorm[ICONV,:] + f1

                    yout[ICONV] = yout[ICONV]/ynor[ICONV]
                    gradout[ICONV,:] = gradout[ICONV,:]/gradnorm[ICONV,:]
                
        return yout,gradout
        
    #################################################################################################################

    def integrate_filter(self,Wave,ModSpec,IGEOM='All'):
        """
        Subroutine to integrate the Modelled spectrum over the filter profile

        Parameters
        ----------
        Wave : 1D or 2D array (NWAVE)
            Calculation wavelengths or wavenumbers
        ModSpec : 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum

        Other Parameters
        ----------------
        IGEOM : int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)

        Returns
        -------
        SPECONV : 1D or 2D array (NCONV,NGEOM)
            Integrated spectrum over the filter profile
        """
        
        #Accounting for the Doppler shift that was previously introduced
        wavecorr = self.correct_doppler_shift(Wave)

        if self.FWHM>=0.0:
            raise ValueError('error in integrate_filter :: FWHM must be < 0 for filter integration')
        else:    
            
            if IGEOM=='All':
                if ModSpec.ndim!=2:
                    raise ValueError('error in integrate_filter :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                IG = 0
                SPECONV = integrate_filter_ngeom(len(Wave),wavecorr,ModSpec,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)
            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in integrate_filter :: ModSpec must have 1 dimensions (NWAVE)')
                IG = IGEOM
                SPECONV = integrate_filter(len(Wave),wavecorr,ModSpec,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)

        return SPECONV
        
        
    #################################################################################################################

    def integrate_filterg(self,Wave,ModSpec,ModGrad,IGEOM='All'):
    
        """
        Subroutine to integrate the Modelled spectrum and the gradients over the filter function 

        Parameters
        ----------
        Wave : 1D array (NWAVE)
            Calculation wavelengths or wavenumbers
        ModSpec : 1D or 2D array (NWAVE,NGEOM)
            Modelled spectrum
        ModGrad: 2D or 3D array (NWAVE,NGEOM,NX)
            Modelled gradients

        Other Parameters
        ----------------
        IGEOM : int
            If All, it is assumed all geometries cover exactly the same spetral range and ModSpec is expected to be (NWAVE,NGEOM)
            If not, IGEOM should be an integer indicating the geometry it corresponds to in the Measurement class (or .spx file)

        Returns
        -------
        SPECONV : 1D or 2D array (NCONV,NGEOM)
            Integrated spectrum over the filter profile
        dSPECONV : 2D or 3D array (NCONV,NGEOM,NX)
            Integrated gradients over the filter profile
        """

        #Accounting for the Doppler shift that was previously introduced
        NWAVE = len(Wave)
        wavecorr = self.correct_doppler_shift(Wave)

        if self.FWHM>=0.0:   
            raise ValueError('error in integrate_filterg :: FWHM must be < 0 for filter integration')

        else:
            if IGEOM=='All':
                if ModSpec.ndim!=2:
                    raise ValueError('error in integrate_filterg :: ModSpec must have 2 dimensions (NWAVE,NGEOM)')
                if ModGrad.ndim!=3:
                    raise ValueError('error in integrate_filterg :: ModGrad must have 3 dimensions (NWAVE,NGEOM,NX)')
                IG = 0
                SPECONV,dSPECONV = integrate_filterg_ngeom(NWAVE,wavecorr,ModSpec,ModGrad,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)

            else:
                if ModSpec.ndim!=1:
                    raise ValueError('error in integrate_filterg :: ModSpec must have 1 dimensions (NWAVE)')
                if ModGrad.ndim!=2:
                    raise ValueError('error in integrate_filterg :: ModGrad must have 2 dimensions (NWAVE,NX)')
                IG = IGEOM
                SPECONV,dSPECONV = integrate_filterg(NWAVE,wavecorr,ModSpec,ModGrad,self.NCONV[IG],self.VCONV[:,IG],self.NFIL,self.VFIL,self.AFIL)

        return SPECONV,dSPECONV
        
    #################################################################################################################

    def calc_doppler_shift(self,wave):
        """
        Subroutine to calculate the Doppler shift in wavenumber or wavelength units based on
        the Doppler velocity between the observed body and the observer. The formula is:
            
            shift = lambda - lambda_0 = lambda_0 * v / c
        
        V_DOPPLER is defined as positive if moving towards the observer and negative if moving away
        
        This function returns Delta_Wave, where:
            Delta_Wave = Wave * v / c
        """
        
        c = 299792458.0   #Speed of light (m/s)
        
        if self.ISPACE==WaveUnit.Wavenumber_cm:
            #Wavenumber (cm-1)
            wave_shift = self.V_DOPPLER*1.0e3 / c * wave
        elif self.ISPACE==WaveUnit.Wavelength_um:
            #Wavelength (um)
            wave_shift = -self.V_DOPPLER*1.0e3 / c * wave
        
        return wave_shift
    
    #################################################################################################################
    
    def invert_doppler_shift(self,wave):
        """
        Subroutine to calculate the Doppler shift in wavenumber or wavelength units based on
        the Doppler velocity between the observed body and the observer.
        
        Knowing the observed wavelength lambda, we want to calculate the non-shifted wavelength lambda_0.
        
         lambda - lambda_0 = lambda_0 * v /c -----> lambda_0 = lambda / (1 + v/c)
        
        V_DOPPLER is defined as positive if moving towards the observer and negative if moving away
        
        This function returns WAVE_0 (in walengths or wavenumbers, depending on ISPACE)
        """
        
        c = 299792458.0   #Speed of light (m/s)
        
        if self.ISPACE==WaveUnit.Wavenumber_cm:
            #Wavenumber (cm-1)
            wave_0 = wave / (1.0-self.V_DOPPLER*1.0e3 / c)
        elif self.ISPACE==WaveUnit.Wavelength_um:
            #Wavelength (um)
            wave_0 = wave / (1.0+self.V_DOPPLER*1.0e3 / c)
        
        return wave_0
    
    #################################################################################################################
    
    def correct_doppler_shift(self,wave_0):
        """
        Subroutine to calculate the Doppler shift in wavenumber or wavelength units based on
        the Doppler velocity between the observed body and the observer.
        
        Knowing the non-shifted wavelength lambda_0, we want to calculate the shifted wavelength lambda.
        
         lambda - lambda_0 = lambda_0 * v /c -----> lambda = lambda_0 * (1 + v/c)
        
        V_DOPPLER is defined as positive if moving towards the observer and negative if moving away
        
        This function returns WAVE_0 (in walengths or wavenumbers, depending on ISPACE)
        """
        
        c = 299792458.0   #Speed of light (m/s)
        
        if self.ISPACE==WaveUnit.Wavenumber_cm:
            #Wavenumber (cm-1)
            wave = wave_0 * (1.0-self.V_DOPPLER*1.0e3 / c)
        elif self.ISPACE==WaveUnit.Wavelength_um:
            #Wavelength (um)
            wave = wave_0 * (1.0+self.V_DOPPLER*1.0e3 / c)
        
        return wave
    
    #################################################################################################################
    
    def plot_ils(self):
        """
        Subroutine to make a summary plot of the instrument lineshape
        """

        fig,ax1 = plt.subplots(1,1,figsize=(10,4))

        if self.ISPACE==WaveUnit.Wavenumber_cm:
            xlabel=r'Wavenumber (cm$^{-1}$)'
            xsymbol = r'$\nu$'
            xunit = r'cm$^{-1}$'
        elif self.ISPACE==WaveUnit.Wavelength_um:
            xlabel=r'Wavelength ($\mu$m)'
            xsymbol = r'$\lambda$'
            xunit = r'$\mu$m'

        iconv = 0
        ax1.plot(self.VFIL[0:self.NFIL[iconv],iconv]-self.VCONV[iconv,0],self.AFIL[0:self.NFIL[iconv],iconv],label=xsymbol+' = '+str(np.round(self.VCONV[iconv,0],1))+' '+xunit)
        
        iconv = int(self.NCONV[0] / 2) - 1
        ax1.plot(self.VFIL[0:self.NFIL[iconv],iconv]-self.VCONV[iconv,0],self.AFIL[0:self.NFIL[iconv],iconv],label=xsymbol+' = '+str(np.round(self.VCONV[iconv,0],1))+' '+xunit)
        
        iconv = self.NCONV[0] - 1
        ax1.plot(self.VFIL[0:self.NFIL[iconv],iconv]-self.VCONV[iconv,0],self.AFIL[0:self.NFIL[iconv],iconv],label=xsymbol+' = '+str(np.round(self.VCONV[iconv,0],1))+' '+xunit)
                
        
        ax1.set_facecolor('lightgray')
        ax1.grid()
        ax1.legend()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Instrument lineshape')
        plt.tight_layout()
        plt.show()
        
    #################################################################################################################
    
    def plot_filters(self):
        """
        Subroutine to make a summary plot of the filter functions
        """

        fig,ax1 = plt.subplots(1,1,figsize=(10,4))

        if self.ISPACE==WaveUnit.Wavenumber_cm:
            xlabel=r'Wavenumber (cm$^{-1}$)'
            xsymbol = r'$\nu$'
            xunit = r'cm$^{-1}$'
        elif self.ISPACE==WaveUnit.Wavelength_um:
            xlabel=r'Wavelength ($\mu$m)'
            xsymbol = r'$\lambda$'
            xunit = r'$\mu$m'

        for iconv in range(self.NCONV[0]):
            ax1.plot(self.VFIL[0:self.NFIL[iconv],iconv],self.AFIL[0:self.NFIL[iconv],iconv],label=xsymbol+' = '+str(np.round(self.VCONV[iconv,0],1))+' '+xunit)

        ax1.set_facecolor('lightgray')
        ax1.grid()
        ax1.legend()
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Filter function')
        plt.tight_layout()
        plt.show()
    
    #################################################################################################################
    
    def plot_SO(self):
        """
        Subroutine to make a summary plot of a solar occultation observation
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig,ax1 = plt.subplots(1,1,figsize=(10,4))

        colormap = 'nipy_spectral'
        cmap = matplotlib.cm.get_cmap(colormap,100)
        cmin = self.TANHE[:,0].min()
        cmax = self.TANHE[:,0].max()

        for igeom in range(self.NGEOM):
            
            color = (self.TANHE[igeom,0]-cmin)/(cmax-cmin)
            
            #ax1.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom],c=s_m.to_rgba([self.TANHE[igeom,0]]))
            ax1.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom],color=cmap(color))

        if np.mean(self.VCONV)>30.:
            ax1.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        else:
            ax1.set_xlabel(r'Wavelength ($\mu$m)')
        ax1.set_ylabel('Transmission')
        ax1.set_title('Latitude = '+str(np.round(self.LATITUDE,1))+' - Longitude = '+str(np.round(self.LONGITUDE,1)))
        ax1.set_facecolor('lightgray')
        ax1.grid()
        
        # Create a ScalarMappable object
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])  # Set dummy array to create the colorbar

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar2 = plt.colorbar(sm,cax=cax,orientation='vertical')
        cbar2.set_label('Altitude (km)')
        
        # Update colorbar ticks based on TANHE values
        n = 10
        cbar_ticksx = np.linspace(0, 1, num=n)  # Adjust the number of ticks as needed
        cbar_ticks = np.linspace(cmin, cmax, num=n)  # Adjust the number of ticks as needed
        cbar2.set_ticks(cbar_ticksx)
        cbar2.set_ticklabels([f"{tick:.2f}" for tick in cbar_ticks])  # Adjust the formatting as needed

        
        plt.tight_layout()
        plt.show()
        
    #################################################################################################################

    def plot_nadir(self,subobs_lat=None,subobs_lon=None):
        """
        Subroutine to make a summary plot of a nadir-viewing observation

        Other Parameters
        ----------
        subobs_lat : float, optional
            Sub-observer latitude (degrees)
        subobs_lon : float, optional
            Sub-observer longitude (degrees)

        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        subobs_lat = self.SUBOBS_LAT if subobs_lat is None else subobs_lat
        subobs_lon = self.SUBOBS_LON if subobs_lon is None else subobs_lon


        #Making a figure for each geometry
        for igeom in range(self.NGEOM):

            plt.figure(figsize=(12,7))

            #Plotting the geometry
            ax1 = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=1)
            map = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
            
            map.drawparallels(np.linspace(-90, 90, 13)) # lats
            map.drawmeridians(np.linspace(-180, 180, 13)) #  lons

            if self.NAV[igeom]>1:
                im = map.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.WGEOM[igeom,:])

                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("bottom", size="5%", pad=0.15)
                cbar2 = plt.colorbar(im,cax=cax,orientation='horizontal')
                cbar2.set_label('Weight')
            else:
                im = map.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True)

            ax1.set_title('Geometry '+str(igeom+1))

            #Plotting the spectra in linear scale
            ax2 = plt.subplot2grid((2,3),(0,1),rowspan=1,colspan=2)
            ax2.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax2.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])
            ax2.set_title('Spectra linear scale')
            ax2.grid()

            #Plotting the spectra in log scale
            ax3 = plt.subplot2grid((2,3),(1,1),rowspan=1,colspan=2,sharex=ax2)
            ax3.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax3.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])
            ax3.set_title('Spectra log scale')
            ax3.set_yscale('log')

            if np.mean(self.VCONV)>30.:
                ax3.set_xlabel(r'Wavenumber (cm$^{-1}$)')
                ax3.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
                ax2.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
            else:
                ax3.set_xlabel(r'Wavelength ($\mu$m)')
                ax3.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
                ax2.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')

            ax3.grid()

            plt.tight_layout()
        plt.show()
        
    #################################################################################################################

    def plot_disc_averaging(self,subobs_lat=None,subobs_lon=None, colormap='cividis'):
        """
        Subroutine to make a summary plot of a disc averaging observation 

        Parameters
        ----------
        subobs_lat : float, optional
            Sub-observer latitude (degrees)
        subobs_lon : float, optional
            Sub-observer longitude (degrees)

        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #Making a figure for each geometry
        for igeom in range(self.NGEOM):

            plt.figure(figsize=(15,7))

            #Plotting the geometry
            ax1 = plt.subplot2grid((2,4),(0,0),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map1 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map1 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)


            map1.drawparallels(np.linspace(-90, 90, 13)) # lats
            map1.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im1 = map1.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.WGEOM[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar1 = plt.colorbar(im1,cax=cax,orientation='horizontal')
            cbar1.set_label('Weight')




            ax2 = plt.subplot2grid((2,4),(0,1),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map2 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map2 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            map2.drawparallels(np.linspace(-90, 90, 13)) # lats
            map2.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im2 = map2.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.EMISS_ANG[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im2,cax=cax,orientation='horizontal')
            cbar2.set_label('Emission angle')






            ax3 = plt.subplot2grid((2,4),(1,0),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map3 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map3 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            map3.drawparallels(np.linspace(-90, 90, 13)) # lats
            map3.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im3 = map3.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.SOL_ANG[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im3,cax=cax,orientation='horizontal')
            cbar2.set_label('Solar Zenith angle')






            ax4 = plt.subplot2grid((2,4),(1,1),rowspan=1,colspan=1)
            if((subobs_lat!=None) & (subobs_lon!=None)):
                map4 = Basemap(projection='ortho', resolution=None,
                    lat_0=subobs_lat, lon_0=subobs_lon)
            else:
                map4 = Basemap(projection='ortho', resolution=None,
                    lat_0=self.LATITUDE, lon_0=self.LONGITUDE)

            
            map4.drawparallels(np.linspace(-90, 90, 13)) # lats
            map4.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im4 = map4.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.AZI_ANG[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im4,cax=cax,orientation='horizontal')
            cbar2.set_label('Azimuth angle')
            
            #Plotting the spectra in linear scale
            ax5 = plt.subplot2grid((2,4),(0,2),rowspan=1,colspan=2)
            ax5.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax5.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])

            ax5.grid()

            #Plotting the spectra in log scale
            ax6 = plt.subplot2grid((2,4),(1,2),rowspan=1,colspan=2,sharex=ax5)
            ax6.fill_between(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]-self.ERRMEAS[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom]+self.ERRMEAS[0:self.NCONV[igeom],igeom],alpha=0.3)
            ax6.plot(self.VCONV[0:self.NCONV[igeom],igeom],self.MEAS[0:self.NCONV[igeom],igeom])
            ax6.set_yscale('log')

            if np.mean(self.VCONV)>30.:
                ax6.set_xlabel(r'Wavenumber (cm$^{-1}$)')
                ax6.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
                ax5.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
            else:
                ax6.set_xlabel(r'Wavelength ($\mu$m)')
                ax6.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
                ax5.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')

            ax6.grid()
            ax5.set_title('Geometry '+str(igeom+1))

            plt.tight_layout()
        plt.show()
        
    #################################################################################################################

    def plot_fov(self, colormap='turbo'):
        """
        Subroutine to make a summary plot of the field-of-view geometry for a disc-averaging observation

        """

        from mpl_toolkits.basemap import Basemap
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        #Making a figure for each geometry
        for igeom in range(self.NGEOM):

            plt.figure(figsize=(12,3))

            #Plotting the geometry
            ax1 = plt.subplot2grid((1,4),(0,0),rowspan=1,colspan=1)

            if self.EMISS_ANG[igeom,:].min()>0.0:
                _lgr.warning('there is no EMISS_ANG equal to zero, sub-observer point cannot be determined automatically, using central LATITUDE and LONGITUDE')

                subobs_lat = self.LATITUDE
                subobs_lon = self.LONGITUDE

            else:

                isubobs = np.where(self.EMISS_ANG[igeom,:]==np.min(self.EMISS_ANG[igeom,:]))[0][0]
                subobs_lat = self.FLAT[igeom,isubobs]
                subobs_lon = self.FLON[igeom,isubobs]

            map1 = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
            map1.drawparallels(np.linspace(-90, 90, 13)) # lats
            map1.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im1 = map1.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.WGEOM[igeom,:],cmap=colormap)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar1 = plt.colorbar(im1,cax=cax,orientation='horizontal')
            cbar1.set_label('Weight')




            ax2 = plt.subplot2grid((1,4),(0,1),rowspan=1,colspan=1)
            map2 = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
            map2.drawparallels(np.linspace(-90, 90, 13)) # lats
            map2.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im2 = map2.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.EMISS_ANG[igeom,:],cmap=colormap,vmin=0.0,vmax=90.0)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im2,cax=cax,orientation='horizontal')
            cbar2.set_label('Emission angle')






            ax3 = plt.subplot2grid((1,4),(0,2),rowspan=1,colspan=1)
            map3 = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
            map3.drawparallels(np.linspace(-90, 90, 13)) # lats
            map3.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im3 = map3.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.SOL_ANG[igeom,:],cmap=colormap,vmin=0.0,vmax=90.0)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            ax3.set_facecolor('lightgray')
            divider = make_axes_locatable(ax3)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im3,cax=cax,orientation='horizontal')
            cbar2.set_label('Solar Zenith angle')
            


            ax4 = plt.subplot2grid((1,4),(0,3),rowspan=1,colspan=1)
            map4 = Basemap(projection='ortho', resolution=None,
                lat_0=subobs_lat, lon_0=subobs_lon)
            map4.drawparallels(np.linspace(-90, 90, 13)) # lats
            map4.drawmeridians(np.linspace(-180, 180, 13)) # lons
            im4 = map4.scatter(self.FLON[igeom,:],self.FLAT[igeom,:],latlon=True,c=self.AZI_ANG[igeom,:],cmap=colormap,vmin=0.0,vmax=180.0)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("bottom", size="5%", pad=0.15)
            cbar2 = plt.colorbar(im4,cax=cax,orientation='horizontal')
            cbar2.set_label('Azimuth angle')

            plt.tight_layout()
        plt.show()


    #################################################################################################################
     
#################################################################################################################
#################################################################################################################
#                                             EXTRA FUNCTIONS
#################################################################################################################
#################################################################################################################


#################################################################################################################
#################################################################################################################
#                                             CONVOLUTIONS
#################################################################################################################
#################################################################################################################

###############################################################################################
@jit(nopython=True)
def lblconv(nwave,vwave,y,nconv,vconv,ishape,fwhm):

    """
        FUNCTION NAME : lblconv()
        
        DESCRIPTION : Convolve the modelled spectrum with a given instrument line shape.
                      In this case, the line shape is defined by ISHAPE and FWHM.
                      Only valid if FWHM>0
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            ishape :: Instrument lineshape (only used if FWHM>0)
                        (0) Square lineshape
                        (1) Triangular
                        (2) Gaussian
                        (3) Hamming
                        (4) Hanning
            fwhm :: Full width at half maximum of the ILS

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Convolved spectrum

        CALLING SEQUENCE:
        
            yout = lblconv(fwhm,ishape,nwave,vwave,y,nconv,vconv)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    yout = np.zeros(nconv)
    ynor = np.zeros(nconv)

    #Set total width of Hamming/Hanning function window in terms of
    #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
    nfw = 3.

    for j in range(nconv):
        yfwhm = fwhm
        vcen = vconv[j]
        if ishape==InstrumentLineshape.Square:
            v1 = vcen-0.5*yfwhm
            v2 = v1 + yfwhm
        elif ishape==InstrumentLineshape.Triangular:
            v1 = vcen-yfwhm
            v2 = vcen+yfwhm
        elif ishape==InstrumentLineshape.Gaussian:
            sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
            v1 = vcen - 3.*sig
            v2 = vcen + 3.*sig
        else:
            v1 = vcen - nfw*yfwhm
            v2 = vcen + nfw*yfwhm


        #Find relevant points in tabulated files
        inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
        inwave = inwave1[0]

        np1 = len(inwave)
        for i in range(np1):
            f1=0.0
            if ishape==InstrumentLineshape.Square:
                #Square instrument lineshape
                f1=1.0
            elif ishape==InstrumentLineshape.Triangular:
                #Triangular instrument shape
                f1=1.0 - abs(vwave[inwave[i]] - vcen)/yfwhm
            elif ishape==InstrumentLineshape.Gaussian:
                #Gaussian instrument shape
                f1 = np.exp(-((vwave[inwave[i]]-vcen)/sig)**2.0)
            else:
                pass

            if f1>0.0:
                yout[j] = yout[j] + f1*y[inwave[i]]
                ynor[j] = ynor[j] + f1

        yout[j] = yout[j]/ynor[j]

    return yout

###############################################################################################
@jit(nopython=True)
def lblconv_ngeom(nwave,vwave,y,nconv,vconv,ishape,fwhm):

    """
        FUNCTION NAME : lblconv()
        
        DESCRIPTION : Convolve the modelled spectra (NGEOM spectra) with a given instrument line shape.
                      In this case, the line shape is defined by ISHAPE and FWHM.
                      Only valid if FWHM>0
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave,ngeom) :: Modelled spectrum
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            ishape :: Instrument lineshape (only used if FWHM>0)
                        (0) Square lineshape
                        (1) Triangular
                        (2) Gaussian
                        (3) Hamming
                        (4) Hanning
            fwhm :: Full width at half maximum of the ILS

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Convolved spectrum

        CALLING SEQUENCE:
        
            yout = lblconv(fwhm,ishape,nwave,vwave,y,nconv,vconv)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """
    if y.ndim==2:

        #It is assumed all geometries cover the same spectral range
        #nconv1 = y.shape[0]
        ngeom = y.shape[1]

        yout = np.zeros((nconv,ngeom))
        ynor = np.zeros((nconv,ngeom))

        if fwhm>0.0:
            #Set total width of Hamming/Hanning function window in terms of
            #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
            nfw = 3.
            for j in range(nconv):
                yfwhm = fwhm
                vcen = vconv[j]
                if ishape==InstrumentLineshape.Square:
                    v1 = vcen-0.5*yfwhm
                    v2 = v1 + yfwhm
                elif ishape==InstrumentLineshape.Triangular:
                    v1 = vcen-yfwhm
                    v2 = vcen+yfwhm
                elif ishape==InstrumentLineshape.Gaussian:
                    sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                    v1 = vcen - 3.*sig
                    v2 = vcen + 3.*sig
                else:
                    v1 = vcen - nfw*yfwhm
                    v2 = vcen + nfw*yfwhm

                #Find relevant points in tabulated files
                inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
                inwave = inwave1[0]

                np1 = len(inwave)
                for i in range(np1):
                    f1=0.0
                    if ishape==InstrumentLineshape.Square:
                        #Square instrument lineshape
                        f1=1.0
                    elif ishape==InstrumentLineshape.Triangular:
                        #Triangular instrument shape
                        f1=1.0 - abs(vwave[inwave[i]] - vcen)/yfwhm
                    elif ishape==InstrumentLineshape.Gaussian:
                        #Gaussian instrument shape
                        f1 = np.exp(-((vwave[inwave[i]]-vcen)/sig)**2.0)

                    if f1>0.0:
                        yout[j,:] = yout[j,:] + f1*y[inwave[i],:]
                        ynor[j,:] = ynor[j,:] + f1

                yout[j,:] = yout[j,:]/ynor[j,:]

    return yout

###############################################################################################
@jit(nopython=True)
def lblconv_fil(nwave,vwave,y,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : lblconv_fil()
        
        DESCRIPTION : Convolve the modelled spectrum with a given instrument line shape.
                      In this case, the line shape is defined by NFIL,VFIL and AFIL from the
                      .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum 
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the Instrument Lineshape
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the Instrument 
                                Lineshape in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the Instrument Lineshape in each convolution wavenumber

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Convolved spectrum

        CALLING SEQUENCE:
        
            yout = lblconv_fil(nwave,vwave,y,nconv,vconv,nfil,vfil,afil)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    if y.ndim==1:

        yout = np.zeros((nconv))
        ynor = np.zeros((nconv))

        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            xp = np.zeros((nfil[j]))
            yp = np.zeros((nfil[j]))
            xp[:] = vfil[0:nfil[j],j]
            yp[:] = afil[0:nfil[j],j]
            for i in range(np1):
                #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                f1 = np.interp(vwave[inwave[i]],xp,yp)
                if f1>0.0:
                    yout[j] = yout[j] + f1*y[inwave[i]]
                    ynor[j] = ynor[j] + f1

            yout[j] = yout[j]/ynor[j]

    return yout

###############################################################################################
@jit(nopython=True)
def lblconv_fil_ngeom(nwave,vwave,y,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : lblconv_fil()
        
        DESCRIPTION : Convolve the modelled spectra (NGEOM spectra) with a given instrument line shape.
                      In this case, the line shape is defined by NFIL,VFIL and AFIL from the
                      .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave,ngeom) :: Modelled spectrum 
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the Instrument Lineshape
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the Instrument 
                                Lineshape in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the Instrument Lineshape in each convolution wavenumber

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Convolved spectrum

        CALLING SEQUENCE:
        
            yout = lblconv_fil(nwave,vwave,y,nconv,vconv,nfil,vfil,afil)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    if y.ndim==2:

        #It is assumed all geometries cover the same spectral range
        #nconv1 = y.shape[0]
        ngeom = y.shape[1]

        yout = np.zeros((nconv,ngeom))
        ynor = np.zeros((nconv,ngeom))

        #Line shape for each convolution number in each case is read from .fil file
        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            xp = np.zeros((nfil[j]))
            yp = np.zeros((nfil[j]))
            xp[:] = vfil[0:nfil[j],j]
            yp[:] = afil[0:nfil[j],j]

            for i in range(np1):
                #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                f1 = np.interp(vwave[inwave[i]],xp,yp)
                if f1>0.0:
                    yout[j,:] = yout[j,:] + f1*y[inwave[i],:]
                    ynor[j,:] = ynor[j,:] + f1

            yout[j,:] = yout[j,:]/ynor[j,:]

    return yout

###############################################################################################
@jit(nopython=True)
def lblconvg_ngeom(nwave,vwave,y,dydx,nconv,vconv,ishape,fwhm):

    """
        FUNCTION NAME : lblconvg_ngeom()
        
        DESCRIPTION : Convolve the modelled spectra (NGEOM spectra) and gradients with a given instrument line shape.
                      In this case, the line shape is defined by ISHAPE and FWHM.
                      Only valid if FWHM>0
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave,ngeom) :: Modelled spectrum
            dydx(nwave,ngeom,nx) :: Modelled gradients with respect to each element of the state vector
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            ishape :: Instrument lineshape (only used if FWHM>0)
                        (0) Square lineshape
                        (1) Triangular
                        (2) Gaussian
                        (3) Hamming
                        (4) Hanning
            fwhm :: Full width at half maximum of the ILS

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv,ngeom) :: Convolved spectrum
            dyoutdx(nconv,ngeom,nx) :: Convolved gradients

        CALLING SEQUENCE:
        
            yout,dyoutdx = lblconvg_ngeom(nwave,vwave,y,dydx,nconv,vconv,ishape,fwhm)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #If all geometries are included in the array
    if ( (y.ndim==2) & (dydx.ndim==3)):

        #It is assumed all geometries cover the same spectral range
        nx = dydx.shape[2]
        ngeom = dydx.shape[1]

        yout = np.zeros((nconv,ngeom))
        ynor = np.zeros((nconv,ngeom))
        gradout = np.zeros((nconv,ngeom,nx))
        gradnorm = np.zeros((nconv,ngeom,nx))
    
        #Set total width of Hamming/Hanning function window in terms of
        #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
        nfw = 3.
        for j in range(nconv):
            yfwhm = fwhm
            vcen = vconv[j]
            if ishape==InstrumentLineshape.Square:
                v1 = vcen-0.5*yfwhm
                v2 = v1 + yfwhm
            elif ishape==InstrumentLineshape.Triangular:
                v1 = vcen-yfwhm
                v2 = vcen+yfwhm
            elif ishape==InstrumentLineshape.Gaussian:
                sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                v1 = vcen - 3.*sig
                v2 = vcen + 3.*sig
            else:
                v1 = vcen - nfw*yfwhm
                v2 = vcen + nfw*yfwhm

            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            for i in range(np1):
                f1=0.0
                if ishape==InstrumentLineshape.Square:
                    #Square instrument lineshape
                    f1=1.0
                elif ishape==InstrumentLineshape.Triangular:
                    #Triangular instrument shape
                    f1=1.0 - abs(vwave[inwave[i]] - vcen)/yfwhm
                elif ishape==InstrumentLineshape.Gaussian:
                    #Gaussian instrument shape
                    f1 = np.exp(-((vwave[inwave[i]]-vcen)/sig)**2.0)

                if f1>0.0:
                    yout[j,:] = yout[j,:] + f1*y[inwave[i],:]
                    ynor[j,:] = ynor[j,:] + f1
                    gradout[j,:,:] = gradout[j,:,:] + f1*dydx[inwave[i],:,:]
                    gradnorm[j,:,:] = gradnorm[j,:,:] + f1

            yout[j,:] = yout[j,:]/ynor[j,:]
            gradout[j,:,:] = gradout[j,:,:]/gradnorm[j,:,:]

    return yout,gradout

###############################################################################################
@jit(nopython=True)
def lblconvg(nwave,vwave,y,dydx,nconv,vconv,ishape,fwhm):

    """
        FUNCTION NAME : lblconvg()
        
        DESCRIPTION : Convolve the modelled spectrum and gradients with a given instrument line shape.
                      In this case, the line shape is defined by ISHAPE and FWHM.
                      Only valid if FWHM>0
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum
            dydx(nwave,nx) :: Modelled gradients with respect to each element of the state vector
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            ishape :: Instrument lineshape (only used if FWHM>0)
                        (0) Square lineshape
                        (1) Triangular
                        (2) Gaussian
                        (3) Hamming
                        (4) Hanning
            fwhm :: Full width at half maximum of the ILS

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv,ngeom) :: Convolved spectrum
            dyoutdx(nconv,ngeom,nx) :: Convolved gradients

        CALLING SEQUENCE:
        
            yout,dyoutdx = lblconvg(nwave,vwave,y,dydx,nconv,vconv,ishape,fwhm)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #If only one geometry needs to be convolved
    if ( (y.ndim==1) & (dydx.ndim==2)):
    
        #It is assumed all geometries cover the same spectral range
        nx = dydx.shape[1]

        yout = np.zeros((nconv))
        ynor = np.zeros((nconv))
        gradout = np.zeros((nconv,nx))
        gradnorm = np.zeros((nconv,nx))

        #Set total width of Hamming/Hanning function window in terms of
        #numbers of FWHMs for ISHAPE=3 and ISHAPE=4
        nfw = 3.
        for j in range(nconv):
            yfwhm = fwhm
            vcen = vconv[j]
            if ishape==InstrumentLineshape.Square:
                v1 = vcen-0.5*yfwhm
                v2 = v1 + yfwhm
            elif ishape==InstrumentLineshape.Triangular:
                v1 = vcen-yfwhm
                v2 = vcen+yfwhm
            elif ishape==InstrumentLineshape.Gaussian:
                sig = 0.5*yfwhm/np.sqrt( np.log(2.0)  )
                v1 = vcen - 3.*sig
                v2 = vcen + 3.*sig
            else:
                v1 = vcen - nfw*yfwhm
                v2 = vcen + nfw*yfwhm

            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            for i in range(np1):
                f1=0.0
                if ishape==InstrumentLineshape.Square:
                    #Square instrument lineshape
                    f1=1.0
                elif ishape==InstrumentLineshape.Triangular:
                    #Triangular instrument shape
                    f1=1.0 - abs(vwave[inwave[i]] - vcen)/yfwhm
                elif ishape==InstrumentLineshape.Gaussian:
                    #Gaussian instrument shape
                    f1 = np.exp(-((vwave[inwave[i]]-vcen)/sig)**2.0)

                if f1>0.0:
                    yout[j] = yout[j] + f1*y[inwave[i]]
                    ynor[j] = ynor[j] + f1
                    gradout[j,:] = gradout[j,:] + f1*dydx[inwave[i],:]
                    gradnorm[j,:] = gradnorm[j,:] + f1

            yout[j] = yout[j]/ynor[j]
            gradout[j,:] = gradout[j,:]/gradnorm[j,:]

    return yout,gradout

###############################################################################################
@jit(nopython=True)
def lblconvg_fil_ngeom(nwave,vwave,y,dydx,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : lblconvg_ngeom()
        
        DESCRIPTION : Convolve the modelled spectra (NGEOM spectra) and gradients with a given instrument line shape.
                      In this case, the line shape is defined by NFIL,VFIL and AFIL from the
                      .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave,ngeom) :: Modelled spectrum
            dydx(nwave,ngeom,nx) :: Modelled gradients with respect to each element of the state vector
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the Instrument Lineshape
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the Instrument 
                                Lineshape in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the Instrument Lineshape in each convolution wavenumber


        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv,ngeom) :: Convolved spectrum
            dyoutdx(nconv,ngeom,nx) :: Convolved gradients

        CALLING SEQUENCE:
        
            yout,dyoutdx = lblconvg_fil_ngeom(nwave,vwave,y,dydx,nconv,vconv,ishape,fwhm)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #If all geometries are included in the array
    if ( (y.ndim==2) & (dydx.ndim==3)):

        #It is assumed all geometries cover the same spectral range
        nx = dydx.shape[2]
        ngeom = dydx.shape[1]

        yout = np.zeros((nconv,ngeom))
        ynor = np.zeros((nconv,ngeom))
        gradout = np.zeros((nconv,ngeom,nx))
        gradnorm = np.zeros((nconv,ngeom,nx))

        #Line shape for each convolution number in each case is read from .fil file
        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            xp = np.zeros((nfil[j]))
            yp = np.zeros((nfil[j]))
            xp[:] = vfil[0:nfil[j],j]
            yp[:] = afil[0:nfil[j],j]

            for i in range(np1):
                #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                f1 = np.interp(vwave[inwave[i]],xp,yp)
                if f1>0.0:
                    yout[j,:] = yout[j,:] + f1*y[inwave[i],:]
                    ynor[j,:] = ynor[j,:] + f1
                    gradout[j,:,:] = gradout[j,:,:] + f1*dydx[inwave[i],:,:]
                    gradnorm[j,:,:] = gradnorm[j,:,:] + f1

            yout[j,:] = yout[j,:]/ynor[j,:]
            gradout[j,:,:] = gradout[j,:,:]/gradnorm[j,:,:]

    return yout,gradout

###############################################################################################
@jit(nopython=True)
def lblconvg_fil(nwave,vwave,y,dydx,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : lblconvg_fil()
        
        DESCRIPTION : Convolve the modelled spectrum and gradients with a given instrument line shape.
                      In this case, the line shape is defined by ISHAPE and FWHM.
                      Only valid if FWHM>0
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum
            dydx(nwave,nx) :: Modelled gradients with respect to each element of the state vector
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the Instrument Lineshape
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the Instrument 
                                Lineshape in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the Instrument Lineshape in each convolution wavenumber


        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv,ngeom) :: Convolved spectrum
            dyoutdx(nconv,ngeom,nx) :: Convolved gradients

        CALLING SEQUENCE:
        
            yout,dyoutdx = lblconvg_fil(nwave,vwave,y,dydx,nconv,vconv,ishape,fwhm)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #If only one geometry needs to be convolved
    if ( (y.ndim==1) & (dydx.ndim==2)):
    
        #It is assumed all geometries cover the same spectral range
        nx = dydx.shape[1]

        yout = np.zeros((nconv))
        ynor = np.zeros((nconv))
        gradout = np.zeros((nconv,nx))
        gradnorm = np.zeros((nconv,nx))

        #Line shape for each convolution number in each case is read from .fil file
        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave1 = np.where( (vwave>=v1) & (vwave<=v2) )
            inwave = inwave1[0]

            np1 = len(inwave)
            xp = np.zeros((nfil[j]))
            yp = np.zeros((nfil[j]))
            xp[:] = vfil[0:nfil[j],j]
            yp[:] = afil[0:nfil[j],j]

            for i in range(np1):
                #Interpolating (linear) for finding the lineshape at the calculation wavenumbers
                f1 = np.interp(vwave[inwave[i]],xp,yp)
                if f1>0.0:
                    yout[j] = yout[j] + f1*y[inwave[i]]
                    ynor[j] = ynor[j] + f1
                    gradout[j,:] = gradout[j,:] + f1*dydx[inwave[i],:]
                    gradnorm[j,:] = gradnorm[j,:] + f1

            yout[j] = yout[j]/ynor[j]
            gradout[j,:] = gradout[j,:]/gradnorm[j,:]

    return yout,gradout


#################################################################################################################
#################################################################################################################
#                                             FILTER INTEGRATIONS
#################################################################################################################
#################################################################################################################


###############################################################################################
@jit(nopython=True)
def integrate_filter(nwave,vwave,y,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : integrate_filter()
        
        DESCRIPTION : Integrate the modelled spectrum with the filter function
                      In this case, the filter function is defined by NFIL,VFIL and AFIL from the .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum 
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the filter function
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the filter function in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the filter function in each convolution wavenumber
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Integrated spectrum

        CALLING SEQUENCE:
        
            yout = integrate_filter(nwave,vwave,y,nconv,vconv,nfil,vfil,afil)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    if y.ndim==1:

        yout = np.zeros((nconv))

        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave = np.where( (vwave>=v1) & (vwave<=v2) )[0]
            np1 = len(inwave)
            
            #Interpolating the filter function at the calculation wavenumbers
            afil_interp = np.interp(vwave[inwave],vfil[0:nfil[j],j],afil[0:nfil[j],j])
            
            #Integrating the product of the spectrum and the filter function
            yout[j] = np.trapz(y[inwave] * afil_interp, vwave[inwave])
            
    return yout

###############################################################################################
@jit(nopython=True)
def integrate_filter_ngeom(nwave,vwave,y,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : integrate_filter_ngeom()
        
        DESCRIPTION : Integrate the modelled spectra (NGEOM spectra) with the filter function
                      In this case, the filter function is defined by NFIL,VFIL and AFIL from the .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave,ngeom) :: Modelled spectrum 
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the filter function
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the filter function in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the filter function in each convolution wavenumber
        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Integrated spectrum

        CALLING SEQUENCE:
        
            yout = integrate_filter_ngeom(nwave,vwave,y,nconv,vconv,nfil,vfil,afil)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    if y.ndim==2:

        #It is assumed all geometries cover the same spectral range
        #nconv1 = y.shape[0]
        ngeom = y.shape[1]

        yout = np.zeros((nconv,ngeom))

        #Filter function for each convolution number in each case is read from .fil file
        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave = np.where( (vwave>=v1) & (vwave<=v2) )[0]
            
            #Interpolating the filter function at the calculation wavenumbers
            afil_interp = np.interp(vwave[inwave],vfil[0:nfil[j],j],afil[0:nfil[j],j])
            
            #Integrating the product of the spectrum and the filter function
            yout[j,:] = np.trapz(y[inwave,:] * afil_interp[:, None], vwave[inwave], axis=0)

    return yout

###############################################################################################
@jit(nopython=True)
def integrate_filterg(nwave,vwave,y,dydx,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : integrate_filterg()
        
        DESCRIPTION : Integrate the modelled spectrum and gradients over a filter function
                      In this case, the filter function is defined by NFIL,VFIL and AFIL from the .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave) :: Modelled spectrum
            dydx(nwave,nx) :: Modelled gradients with respect to each element of the state vector
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the filter function
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the filter function in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the filter function in each convolution wavenumber

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv) :: Integrated spectrum
            dyoutdx(nconv,nx) :: Integrated gradients

        CALLING SEQUENCE:
        
            yout,dyoutdx = integrate_filterg(nwave,vwave,y,dydx,nconv,vconv,nfil,vfil,afil)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #If only one geometry needs to be convolved
    if ( (y.ndim==1) & (dydx.ndim==2)):
    
        #It is assumed all geometries cover the same spectral range
        nx = dydx.shape[1]

        yout = np.zeros(nconv)
        gradout = np.zeros((nconv,nx))

        #Filter function for each convolution number in each case is read from .fil file
        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave = np.where( (vwave>=v1) & (vwave<=v2) )[0]

            #Interpolating the filter function at the calculation wavenumbers
            afil_interp = np.interp(vwave[inwave],vfil[0:nfil[j],j],afil[0:nfil[j],j])
            
            #Integrating the product of the spectrum and the filter function
            yout[j] = np.trapz(y[inwave] * afil_interp, vwave[inwave])
            for k in range(nx):
                gradout[j,k] = np.trapz(dydx[inwave,k] * afil_interp, vwave[inwave])

    return yout,gradout

###############################################################################################
@jit(nopython=True)
def integrate_filterg_ngeom(nwave,vwave,y,dydx,nconv,vconv,nfil,vfil,afil):

    """
        FUNCTION NAME : integrate_filterg_ngeom()
        
        DESCRIPTION : Integrate the modelled spectra (NGEOM spectra) and gradients over a filter function
                      In this case, the filter function is defined by NFIL,VFIL and AFIL from the .fil file
        
        INPUTS :
            nwave :: Number of calculation wavenumbers
            vwave(nwave) :: Calculation wavenumbers
            y(nwave,ngeom) :: Modelled spectra
            dydx(nwave,ngeom,nx) :: Modelled gradients with respect to each element of the state vector
            nconv :: Number of convolution wavenumbers
            vconv(nconv) :: Convolution wavenumbers
            nfil(nconv) :: Number of wavenumbers required to define the filter function
                            in each convolution wavenumber
            vfil(nfil,nconv) :: Wavenumbers required to define the filter function in each convolution wavenumber
            afil(nfil,nconv) :: Function defining the filter function in each convolution wavenumber

        OPTIONAL INPUTS: none
        
        OUTPUTS :
        
            yout(nconv,ngeom) :: Integrated spectra
            dyoutdx(nconv,ngeom,nx) :: Integrated gradients

        CALLING SEQUENCE:
        
            yout,dyoutdx = integrate_filterg_ngeom(nwave,vwave,y,dydx,nconv,vconv,nfil,vfil,afil)
        
        MODIFICATION HISTORY : Juan Alday (29/04/2021)
        
    """

    #If only one geometry needs to be convolved
    if ( (y.ndim==2) & (dydx.ndim==3)):
    
        #It is assumed all geometries cover the same spectral range
        nx = dydx.shape[2]
        ngeom = dydx.shape[1]

        yout = np.zeros((nconv,ngeom))
        gradout = np.zeros((nconv,ngeom,nx))

        #Filter function for each convolution number in each case is read from .fil file
        for j in range(nconv):
            v1 = vfil[0,j]
            v2 = vfil[nfil[j]-1,j]
            #Find relevant points in tabulated files
            inwave = np.where( (vwave>=v1) & (vwave<=v2) )[0]

            #Interpolating the filter function at the calculation wavenumbers
            afil_interp = np.interp(vwave[inwave],vfil[0:nfil[j],j],afil[0:nfil[j],j])
            
            #Integrating the product of the spectrum and the filter function
            yout[j,:] = np.trapz(y[inwave,:] * afil_interp[:, None], vwave[inwave], axis=0)
            
            for k in range(nx):
                gradout[j,:,k] = np.trapz(dydx[inwave,:,k] * afil_interp[:, None], vwave[inwave], axis=0)

    return yout,gradout


#################################################################################################################
#################################################################################################################
#                                            GEOMETRY CALCULATIONS - EXOPLANETS
#################################################################################################################
#################################################################################################################


#Generate sample locations and corresponding weights on a planetary disc
#for calculating the disc-averaged radiance of a transitting planet
#at variable orbital phases.

#Solar zenith and azimuth angles of the sample points
#are also generated, but not currently used asreflected sunlight is not included.

#Assumptions and conventions:

#    1. The visible planetary disc is a circle (with dimensionless radius 1).

#    2. The orbital phase is 0 at primary transit and 180 at secondary eclipse.
#    Orbital phase increases in the direction of orbital motion.

#    3. The planet is tidally locked, i.e. it's in synchronous rotation
#    and always present the same hemisphere to the star. Therefore, We are
#    observing the planet's orbit edge on.

#    4. The planetocentric longitude is defined such that the antistellar point
#    is on the 0E meridian, and the substellar point is on the 180E meridian.
#    Longitude increases in the direction of planet's self-rotation.

#    5. The north pole of the planet is defined such that when viewed directly
#    above the north pole the planet is rotating anticlockwise.

#Coordinate system (origin is at the centre of the target):

#    x-axis points from observer's 9 o'clock to observer's 3 o'clock
#    y-axis points from observer to the target
#    z-axis points from south pole to north pole
#    r is the radial distance
#    theta is measured clockwiseky from the z-axis
#    phi is measured anticlockwise from x-axis

#    x = r * sin(theta) * cos(phi) = rho * cos(alpha)
#    y = r * sin(theta) * sin(phi) = rho * sin(alpha)
#    z = r * cos(theta)

#    rho: projected distance from a point on the disc to the centre of the disc
#    alpha: argument of a point on the projected disc, measured anticlockwise from x axis

#################################################################################################

def arctan(x,y):
    """
    Calculate the argument of the point (x,y) in the range [0,2pi).

    Parameters
    ----------
        x : real
            x-coordinate of the point (length of the adjacent side)
        y : real
            y-coordinate of the point (length of the opposite side)
    Returns
    -------
        ang : real
            Argument of (x,y) in radians
    """
    if(x == 0.0):
        if (y == 0.0) : ang = 0.0 # (x,y) is the origin, ill-defined
        elif (y > 0.0) : ang = 0.5*np.pi # (x,y) is on positive y-axis
        else : ang = 1.5*np.pi  # (x,y) is on negative y-axis
    else:
        ang=np.arctan(y/x)
        if (y > 0.0) :
            if (x > 0.0) : ang = ang # (x,y) is in 1st quadrant
            else : ang = ang+np.pi # (x,y) is in 2nd quadrant
        elif (y == 0.0) :
            if (x > 0.0) : ang = 0 # (x,y) is on positive x-axis
            else : ang = np.pi # (x,y) is on negative x-axis
        else:
            if (x > 0.0) : ang = ang+2*np.pi # (x,y) is in 4th quadrant
            else : ang = ang+np.pi # (x,y) is in 3rd quadrant
    return ang

#################################################################################################

def rotatey(v, phi):
    """
    Rotate a 3D vector v anticlockwisely about the y-axis by angle phi

    Parameters
    ----------
        v : ndarray
            A real 3D vector to rotate
        phi : real
            Angle to rotate the vector v by (radians)

    Returns
    -------
        v_new : ndarray
            Rotated 3D vector
    """
    a = np.zeros((3,3)) # construct the rotation matrix
    a[0,0] = np.cos(phi)
    a[0,2] = np.sin(phi)
    a[1,1] = 1.
    a[2,0] = -np.sin(phi)
    a[2,2] = np.cos(phi)
    # v_new = np.matmul(a,v) # unsupported NumPy function
    v_new = np.zeros(3)
    for i in range(3):
        for j in range(3):
            v_new[i] += a[i,j] * v[j]
    return v_new

#################################################################################################

def rotatez(v, phi):
    """
    Rotate a 3D vector v anticlockwisely about the z-axis by angle phi

    Parameters
    ----------
        v : ndarray
            A real 3D vector to rotate
        phi : real
            Angle to rotate the vector by (radians)

    Returns
    -------
        v_new : ndarray
            Rotated 3D vector
    """
    a = np.zeros((3,3))
    a[0,0] = np.cos(phi)
    a[0,1] = -np.sin(phi)
    a[1,0] = np.sin(phi)
    a[1,1] = np.cos(phi)
    a[2,2] = 1
    # v_new = np.matmul(a,v) # unsupported NumPy function
    v_new = np.zeros(3)
    for i in range(3):
        for j in range(3):
            v_new[i] += a[i,j] *v[j]
    return v_new

#################################################################################################

def generate_angles_exoplanet(phase,rho,alpha):
    """
    Finds the stellar zenith angle, stellar azimuth angle, lattitude and longitude
    of a chosen point on the visible disc of a planet under observation. The planet
    is assumed to be tidally locked, and is observed on an edgy-on orbit.

    Refer to the begining of the trig.py file for geomety and convections.

    Parameters
    ----------
    phase : real
        Orbital phase in degrees. 0 at  parimary transit and 180 at secondary eclipse.
        Range: [0,360)
    rho	: real
        Fractional radius of the point on disc, must be between 0 and 1 inclusive.
        Range: [0,1]
    alpha : real
        Argument of the point on visible disc (degrees), measured
        anticlockwise from 3 o'clock.
        Range: [0,360)

    Returns
    -------
    zen : real
        Computed solar zenith angle (radians), which is the angle between the local
        normal and the stellar direction vector.
    azi	: real
        Computed solar azimuth angle (radians). Uses convention that
        forward scatter = 0. !!! need to define.
    lat	: real
        Planetocentric latitude of the point (degrees).
    lon	: real
        Planetocentric longitude of the point (degrees).
    """
    phase = np.mod(phase,360)
    dtr = np.pi/180. # degree to radiance conversion factor
    assert rho <=1, "Fractional radius should be less or equal to 1"

    # get stellar direction vector in Cartesian coordinates
    # ie unit vector in direction of star
    theta_star = np.pi/2. # star lies in planet's equitorial plane
    phi_star = 90.0 + phase # when phase angle is 0 the star lies on the y-axis
    x_star = np.sin(theta_star)*np.cos(phi_star*dtr)
    y_star = np.sin(theta_star)*np.sin(phi_star*dtr)
    z_star = np.cos(theta_star)
    v_star = np.array([x_star,y_star,z_star])

    # get Cartesian coordinates of input point
    # calculate point position vector using spherical polars (r=1,theta,phi)
    theta_point = np.arccos(rho*np.sin(alpha*dtr)) # ie planetary zenith angle of poiny
    if np.sin(theta_point) != 0.0:
        cos_phi = rho*np.cos(alpha*dtr)/abs(np.sin(theta_point))
        phi_point = (-np.arccos(cos_phi))%(2*np.pi) # azimuth angle of point (on our side)
    else:
        phi_point = 0.0 # sin(theta_point) = 0 at north polt
    x_point = np.sin(theta_point)*np.cos(phi_point)
    y_point = np.sin(theta_point)*np.sin(phi_point)
    z_point = np.cos(theta_point)
    v_point = np.array([x_point,y_point,z_point])

    # calculate angle between solar position vector and local normal
    # i.e. zen solar zenith angle
    inner_product = np.sum(v_star*v_point)
    zen = np.arccos(inner_product)
    zen = np.around(zen, 10)

    # calculate latitude and longitude of the spot
    # (sub-stellar point = 180E, anti-stellar point = 0E, longtitudes in the direction of self-rotation)
    lat = np.around(90.-theta_point*180/np.pi, 10) # southern hemisphere has negative lattitude
    lon = (phi_point/dtr - (phi_star+180))%360 # substellar point is 180E

    # calculate emission viewing angle direction vector (-y axis) (Observer direction vecto)
    x_observer = 0.
    y_observer = -1.0
    z_observer = 0.0
    v_observer = np.array([x_observer,y_observer,z_observer])

    ### calculate azimuth angle
    # Rotate frame clockwise by phi_point about z (v_point is now x-axis)
    v_star_1=rotatez(v_star,-phi_point)
    v_point_1=rotatez(v_point,-phi_point)
    v_observer_1=rotatez(v_observer,-phi_point)

    # Rotate frame clockwise by theta_point about y (v_point is now z-axis )
    v1B=rotatey(v_star_1,-theta_point)
    v2B=rotatey(v_point_1,-theta_point)
    v3B=rotatey(v_observer_1,-theta_point)

    # thetsolB=np.arccos(v1B[2])
    # thetobsB=np.arccos(v3B[2])
    phisolB=arctan(v1B[0], v1B[1])
    phiobsB=arctan(v3B[0], v3B[1])

    azi = abs(phiobsB-phisolB)
    if(azi > np.pi):
        azi=2*np.pi-azi

    # Ensure azi meets convention where azi=0 means forward-scattering
    azi = np.pi-azi

    return zen, azi, lat, lon

