#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# Spectroscopy_0.py - Subroutines to store the absorption cross sections.
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



#from archnemesis import *
from archnemesis.enums import (
    WaveUnit,
    #SpectraUnit,
    SpectralCalculationMode,
    SpectroscopicLineProfile,
    #Gas
)
import numpy as np
import scipy
#import matplotlib.pyplot as plt
import os
import os.path
import archnemesis as ans
from numba import jit, njit
from archnemesis.database.datatypes.wave_range import WaveRange

from archnemesis.helpers import h5py_helper, path_redirect
import matplotlib.pyplot as plt


import logging
_lgr = logging.getLogger(__name__)

###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

State Vector Class.
"""

class Spectroscopy_0:

    def __init__(
            self, 
            RUNNAME: str = '', 
            ILBL: SpectralCalculationMode = SpectralCalculationMode.LINE_BY_LINE_TABLES, 
            NGAS: int = 2, 
            ONLINE: bool = False,
            VREL: float = 25.0,
        ):

        """
        Inputs
        ------
        @param ISPACE: int,
            Flag indicating the units of the spectral coordinate:
            (0) Wavenumber (cm-1) 
            (1) Wavelength (um)
        @param ILBL: SpectralCalculationMode enum,
            Flag indicating if the calculations are performed using:
            - K_TABLES (0) - Pre-tabulated correlated-k tables
            - LINE_BY_LINE_RUNTIME (1) - Calculate line-by-line during runtime  
            - LINE_BY_LINE_TABLES (2) - Pre-tabulated line-by-line tables
        @param ONLINE: bool,
            Flag indicating whether the look-up tables must be read and stored on memory (False), 
            or they are read online when calling calc_klbl or calc_k (True)
        @param NGAS: int,
            Number of active gases to include in the atmosphere
        @param ID: 1D array,
            Gas ID for each active gas (using Gas enum)
        @param ISO: 1D array,
            Isotope ID for each gas, default 0 for all isotopes in terrestrial relative abundance
        @param LOCATION: 1D array,
            List of strings indicating where the .lta or .kta tables are stored for each of the gases
        @param IPROC: 1D array,
            If ILBL=1, then IPROC indicates the line profile to use for each gas:
                (0 - VOIGT) Voigt profile
                (1 - SUBLORENTZ_CO2_BROADENING) Assume Sub-Lorentzian line broadening (CO2) 
                (2 - VANVLECK_WEISSKOPF) VanVleck-Weisskopf lineshape (far-IR) 
                (4 - LORENTZ) Lorentz profile
                (12 - DOPPLER) Doppler profile
        @param VREL: real,
            If ILBL=1, then VREL indicates the spectral window to use around each line for the line-by-line calculation (cm-1)
        @param NWAVE: int,
            Number of wavelengths included in the K-tables or LBL-tables
        @param WAVE: 1D array,
            Wavelengths at which the K-tables or LBL-tables are defined
        @param NP: int,
            Number of pressure levels at which the K-tables or LBL-tables were computed
        @param NT: int,
            Number of temperature levels at which the K-tables or LBL-tables were computed
        @param PRESS: 1D array
            Pressure levels at which the K-tables or LBL-tables were computed (Pa)
        @param TEMP: 1D array
            Temperature levels at which the K-tables or LBL-tables were computed (K)
        @param NG: int,
            Number of g-ordinates included in the k-tables (NG=1 for line-by-line)
        @param G_ORD: 1D array,
            G-ordinates
        @param DELG: 1D array,
            Intervals of g-ordinates
        @param FWHM: real,
            Full-width at half maximum (only in case of K-tables)
        


        Methods
        -------
        Spectroscopy_0.edit_WAVE()
        Spectroscopy_0.edit_K()
        Spectroscopy_0.write_hdf5()
        Spectroscopy_0.read_hdf5()
        Spectroscopy_0.read_lls()
        Spectroscopy_0.read_kls()
        Spectroscopy_0.read_header()
        Spectroscopy_0.read_tables()
        Spectroscopy_0.write_table_hdf5()
        Spectroscopy_0.calc_klbl()
        Spectroscopy_0.calc_k()
        Spectroscopy_0.calc_klblg()
        Spectroscopy_0.calc_kg()
        """


        # Input parameters with validation
        self.runname = RUNNAME
        #self.ILBL = SpectralCalculationMode(ILBL) if not isinstance(ILBL, SpectralCalculationMode) else ILBL
        self.NGAS = NGAS
        self.ONLINE = ONLINE
        self.VREL = VREL

        # Attributes with proper typing
        #self.ISPACE: Optional[WaveUnit] = None
        self.ID: None | np.ndarray = None  # Array of Gas enum values (NGAS)
        self.ISO = None       #(NGAS)
        self._locations = path_redirect.PathRedirectList() #(NGAS)
        self.IPROC = None     #(NGAS)
        self.NWAVE = None     
        self.WAVE = None      #(NWAVE)
        self.NP = None
        self.NT = None
        self.PRESS = None     #(NP)
        self.TEMP = None      #(NT)
        self.NG = None
        self.G_ORD = None     #(NG)
        self.DELG = None      #(NG)
        self.FWHM = None
        
        self.K = None #(NWAVE,NG,NP,NT,NGAS)
        
        # private attributes
        self._ilbl = None
        self._ispace = None
        self._iproc = None
        self._locations_initialised = False
        
        # set property values
        self.ILBL = ILBL
        self.ISPACE = WaveUnit.Wavenumber_cm  # Default value

    @property
    def LOCATION(self) -> list[str]:
        # NOTE: paths are stored as strings so we should be able to use `str.startswith(...)` to match them up.
        if not self._locations_initialised:
            return None
        
        return self._locations


    @LOCATION.setter
    def LOCATION(self, value) -> None:
        if value is None:
            self._locations_initialised = False
        else:
            self._locations._raw_paths = [x for x in value]
            self._locations_initialised = True


    @property
    def RUNNAME(self):
        return self.runname
    
    @property
    def ILBL(self) -> SpectralCalculationMode:
        return self._ilbl
    
    @ILBL.setter
    def ILBL(self, value):
        self._ilbl = SpectralCalculationMode(value)
    
    @property
    def IPROC(self) -> list[SpectroscopicLineProfile]:
        return self._iproc

    @IPROC.setter
    def IPROC(self, value):
        if value is None:
            self._iproc = None
        else:
            self._iproc = [SpectroscopicLineProfile(v) for v in value]

    @property
    def ISPACE(self) -> WaveUnit:
        return self._ispace
    
    @ISPACE.setter
    def ISPACE(self, value):
        self._ispace = WaveUnit(value)
    
    
    

    ######################################################################################################

    def assess(self):
        """
        Subroutine to assess whether the variables of the Spectroscopy class are correct
        """   
        # Checking common parameters
        assert isinstance(self.ILBL, SpectralCalculationMode), \
            'ILBL must be SpectralCalculationMode enum'

        if self.ISPACE is not None:
            assert isinstance(self.ISPACE, WaveUnit), \
                'ISPACE must be WaveUnit enum'
            assert self.ISPACE in (WaveUnit.Wavenumber_cm, WaveUnit.Wavelength_um), \
                'ISPACE must be Wavenumber_cm or Wavelength_um'

        assert np.issubdtype(type(self.NGAS), np.integer) == True , \
            'NGAS must be int'
        assert self.NGAS >= 0 , \
            'NGAS must be >=0'

        if self.NGAS>0:
            assert len(self.LOCATION) == self.NGAS , \
                'LOCATION must have size (NGAS)'

        if self.ILBL == 1:

            assert self.IPROC is not None , \
                'IPROC must be set when ILBL=1'
            assert len(self.IPROC) == self.NGAS , \
                'IPROC must have size (NGAS)'
            for i in range(self.NGAS):
                assert isinstance(self.IPROC[i], int), \
                    f'IPROC[{i}] must be an integer'
                assert isinstance(self.IPROC[i], SpectroscopicLineProfile), \
                    f'IPROC[{i}] must be SpectroscopicLineProfile enum'

    ######################################################################################################
    def summary_info(self):
        """
        Subroutine to print summary of information about the class
        """    

        from archnemesis.Data import gas_info

        if self.ILBL==SpectralCalculationMode.K_TABLES:
            _lgr.info(f'Calculation type ILBL ::  {(self.ILBL," (k-distribution)")}')
            _lgr.info(f'Number of radiatively-active gaseous species ::  {(self.NGAS)}')
            gasname = ['']*self.NGAS
            for i in range(self.NGAS):
                gasname1 = gas_info[str(self.ID[i])]['name']
                if self.ISO[i]!=0:
                    gasname1 = gasname1+' ('+str(self.ISO[i])+')'
                gasname[i] = gasname1
            _lgr.info(f'Gaseous species ::  {(gasname)}')

            _lgr.info(f'Number of g-ordinates ::  {(self.NG)}')

            _lgr.info(f'Number of spectral points ::  {(self.NWAVE)}')
            _lgr.info(f"Wavelength range ::  {(self.WAVE.min(),'-',self.WAVE.max())}")
            _lgr.info(f'Step size ::  {(self.WAVE[1]-self.WAVE[0])}')

            _lgr.info(f'Spectral resolution of the k-tables (FWHM) ::  {(self.FWHM)}')

            _lgr.info(f'Number of temperature levels ::  {(self.NT)}')
            _lgr.info(f"Temperature range ::  {(self.TEMP.min(),'-',self.TEMP.max())}")

            _lgr.info(f'Number of pressure levels ::  {(self.NP)}')
            _lgr.info(f"Pressure range ::  {(self.PRESS.min(),'-',self.PRESS.max())}")

        elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:
            _lgr.info(f'Calculation type ILBL ::  {(self.ILBL," (line-by-line)")}')
            _lgr.info(f'Number of radiatively-active gaseous species ::  {(self.NGAS)}')
            gasname = ['']*self.NGAS
            for i in range(self.NGAS):
                gasname1 = gas_info[str(self.ID[i])]['name']
                if self.ISO[i]!=0:
                    gasname1 = gasname1+' ('+str(self.ISO[i])+')'
                gasname[i] = gasname1
            _lgr.info(f'Gaseous species ::  {(gasname)}')

            _lgr.info(f'Number of spectral points ::  {(self.NWAVE)}')
            _lgr.info(f"Wavelength range ::  {(self.WAVE.min(),'-',self.WAVE.max())}")
            _lgr.info(f'Step size ::  {(self.WAVE[1]-self.WAVE[0])}')

            _lgr.info(f'Number of temperature levels ::  {(self.NT)}')
            _lgr.info(f"Temperature range ::  {(self.TEMP.min(),'-',self.TEMP.max())}")

            _lgr.info(f'Number of pressure levels ::  {(self.NP)}')
            _lgr.info(f"Pressure range ::  {(self.PRESS.min(),'-',self.PRESS.max())}")

        elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_RUNTIME:
            _lgr.info(f'Calculation type ILBL ::  {(self.ILBL," (line-by-line runtime)")}')
            _lgr.info(f'Number of radiatively-active gaseous species ::  {(self.NGAS)}')
            gasname = ['']*self.NGAS
            for i in range(self.NGAS):
                gasname1 = gas_info[str(self.ID[i])]['name']
                if self.ISO[i]!=0:
                    gasname1 = gasname1+' ('+str(self.ISO[i])+')'
                gasname[i] = gasname1
            _lgr.info(f'Gaseous species ::  {(gasname)}')

            _lgr.info(f'Number of spectral points ::  {(self.NWAVE)}')
            _lgr.info(f"Wavelength range ::  {(self.WAVE.min(),'-',self.WAVE.max())}")
            _lgr.info(f'Step size ::  {(self.WAVE[1]-self.WAVE[0])}')

            lineshape = ['']*self.NGAS
            for i in range(self.NGAS):
                if self.IPROC[i]==0:
                    lineshape1 = 'VOIGT'
                elif self.IPROC[i]==1:
                    lineshape1 = 'SUBLORENTZ_CO2_BROADENING'
                elif self.IPROC[i]==2:
                    lineshape1 = 'VANVLECK_WEISSKOPF'
                elif self.IPROC[i]==4:
                    lineshape1 = 'LORENTZ'
                elif self.IPROC[i]==12:
                    lineshape1 = 'DOPPLER'
                else:
                    lineshape1 = 'UNKNOWN'
                lineshape[i] = lineshape1

            _lgr.info(f'Line shapes for each gas ::  {(lineshape)}')


    ######################################################################################################
    def edit_WAVE(self, array):
        """
        Edit the wavenumbers (ISPACE=0) or wavelengths (ISPACE=1)
        @param array: 1D array (NWAVE)
        """
        WAVE_array = np.array(array)

        assert len(WAVE_array) == self.NWAVE,'WAVE should be (NWAVE)'

        self.WAVE = WAVE_array

    ######################################################################################################
    def edit_K(self, K_array):
        """
        Edit the k-coefficients (ILBL=0) or absorption cross sections (ILBL=2)
        @param K_array: 5D array (NWAVE,NG,NP,NT,NGAS) or 4D array (NWAVE,NP,NT,NGAS)
            K-coefficients or absorption cross sections
        """
        K_array = np.array(K_array)

        if self.ILBL==SpectralCalculationMode.K_TABLES: #K-tables
            assert K_array.shape == (self.NWAVE, self.NG, self.NP, self.NT, self.NGAS),\
                'K should be (NWAVE,NG,NP,NT,NGAS) if ILBL=0 (K-tables)'
        elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES: #LBL-tables
            assert K_array.shape == (self.NWAVE, self.NP, abs(self.NT), self.NGAS),\
                'K should be (NWAVE,NP,NT,NGAS) if ILBL=2 (LBL-tables)'
        else:
            raise ValueError('ILBL needs to be either 0 (K-tables) or 2 (LBL-tables)')

        self.K = K_array


    ######################################################################################################
    def set_table_location_redirects(self, redirects: tuple[tuple[str,str],...]):
        """
        Adds a group of redirects that means we can look in a different place for a file instead of
        having to copy files into different places.
        """
        for old_path, new_path in redirects:
            self._locations._path_redirects[old_path] = new_path
        
        return self

    def add_table_location(self, *new_locations : str) -> None:
        """
        Adds the location of a table to this class, will not add the same table twice.
        """
        
        for new_location in new_locations:
            _lgr.debug(f'Adding table location {new_location=}')
            if self.LOCATION is None or self.NGAS is None:
                _lgr.error('Spectroscopy_0 instance cannot add a table location before it has been fully initialised. Run one of the following routines first: `self.read_hdf5(...)`, `self.read_lls(...)`, `self.read_kls(...)`.')
                return
            
            add_new_loc_flag = True
            
            # Check that the new_location is not already in our list of locations
            for location in self.LOCATION:
                if os.path.samefile(self._locations._get_redirected_path(new_location), location): # NOTE: must apply redirects here as well otherwise we will be comparing the wrong files
                    _lgr.warning(f'Spectral table file "{new_location}" is the same file as "{location}". Spectroscopy_0 instance will not add the same file twice as the data is already included')
                    add_new_loc_flag = False
            
            if add_new_loc_flag:
                self.LOCATION.append(new_location)
                self.NGAS = len(self.LOCATION)
                _lgr.info(f'Added new table location "{new_location}" to the Spectroscopy_0 instance. To make this change permanent do one of the following: [LEGACY INPUT] add the string to "{self.runname}.lls" or "{self.runname}.kls"; [HDF5 INPUT] run the snippet `spectroscopy.write_hdf5(spectroscopy.runname)` where `spectroscopy` is this Spectroscopy_0 instance.')
            else:
                if self.NGAS != len(self.LOCATION):
                    _lgr.warning(f'Table file "{new_location}" was not added due to the above reason. However, the number of active gasses ({self.NGAS}) does not match the number of table locations ({len(self.LOCATION)}), whereas it is expected that they should. Updating the number of active gasses to be the same as the number of table locations...')
                    self.NGAS = len(self.LOCATION)
        
        
        # If tables have already been read, re-read them
        if self.ID is not None:
            self.read_header()
        
        return

    def write_hdf5(self, runname, inside_telluric=False):
        """
        Write the information about the k-tables or lbl-tables into the HDF5 file

        @param runname: str
            Name of the Nemesis run
        """

        import h5py

        #Assessing that everything is correct
        self.assess()

        with h5py.File(runname+'.h5','a') as f:
            if inside_telluric is False:
                #Checking if Spectroscopy already exists
                if ('/Spectroscopy' in f)==True:
                    del f['Spectroscopy']   #Deleting the Spectroscopy information that was previously written in the file

                grp = f.create_group("Spectroscopy")
            else:
                #The Spectroscopy class must be inserted inside the Telluric class
                if ('/Telluric/Spectroscopy' in f)==True:
                    del f['Telluric/Spectroscopy']   #Deleting the Spectroscopy information that was previously written in the file

                grp = f.create_group("Telluric/Spectroscopy")

            #Writing the main dimensions
            dset = h5py_helper.store_data(grp, 'NGAS', self.NGAS)
            dset.attrs['title'] = "Number of radiatively active gases in atmosphere"

            dset = h5py_helper.store_data(grp, 'ILBL', int(self.ILBL))
            dset.attrs['title'] = "Spectroscopy calculation type"
            if self.ILBL==SpectralCalculationMode.K_TABLES:
                dset.attrs['type'] = 'Correlated-k pre-tabulated look-up tables'
            elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:
                dset.attrs['type'] = 'Line-by-line pre-tabulated look-up tables'
            else:
                raise ValueError('error :: ILBL must be 0 or 2')

            if self.NGAS>0:
                if((self.ILBL==SpectralCalculationMode.K_TABLES) or (self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES)):
                    dt = h5py.special_dtype(vlen=str)
                    dset = h5py_helper.store_data(grp, 'LOCATION', self._locations._raw_paths,dtype=dt) # do not save the redirected paths.
                    dset.attrs['title'] = "Location of the pre-tabulated tables"

    ######################################################################################################
    def read_hdf5(self,runname,inside_telluric=False):
        """
        Read the information about the Spectroscopy class from the HDF5 file

        @param runname: str
            Name of the Nemesis run
        """

        import h5py
        
        with h5py.File(runname+'.h5','r') as f:
            if inside_telluric is True:
                name = '/Telluric/Spectroscopy'
            else:
                name = '/Spectroscopy'

            #Checking if Spectroscopy exists
            e = name in f
            if e==False:
                raise ValueError('error :: Spectroscopy is not defined in HDF5 file')
            else:
                self.NGAS = h5py_helper.retrieve_data(f, name+'/NGAS', np.int32, default=0)
                self.ILBL = SpectralCalculationMode(h5py_helper.retrieve_data(f, name+'/ILBL', np.int32))

                if self.NGAS>0:
                    LOCATION1 = h5py_helper.retrieve_data(f, name+'/LOCATION', default=tuple())
                    if LOCATION1 is None:
                        LOCATION1 = []
                    LOCATION = ['']*self.NGAS
                    for igas in range(self.NGAS):
                        LOCATION[igas] = LOCATION1[igas].decode('ascii')
                    self.LOCATION = LOCATION
                    
                    #Reading the header information
                    self.read_header()
                    
    ######################################################################################################
    def read_lls(self, runname):
        """
        Read the .lls file and store the parameters into the Spectroscopy Class

        @param runname: str
            Name of the Nemesis run
        """

        ngasact = len(open(runname+'.lls').readlines(  ))

        #Opening .lls file
        f = open(runname+'.lls','r')
        strlta = [''] * ngasact
        for i in range(ngasact):
            s = f.readline().split()
            strlta[i] = s[0]

        self.NGAS = ngasact
        self.LOCATION = strlta

        #Now reading the head of the binary files included in the .lls file
        nwavelta = np.zeros([ngasact],dtype='int')
        npresslta = np.zeros([ngasact],dtype='int')
        ntemplta = np.zeros([ngasact],dtype='int')
        gasIDlta = np.zeros([ngasact],dtype='int')
        isoIDlta = np.zeros([ngasact],dtype='int')
        for i in range(ngasact):
            nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels = read_ltahead(strlta[i])
            nwavelta[i] = nwave
            npresslta[i] = npress
            ntemplta[i] = ntemp
            gasIDlta[i] = gasID
            isoIDlta[i] = isoID

        if len(np.unique(nwavelta)) != 1:
            raise ValueError('error :: Number of wavenumbers in all .lta files must be the same')
        if len(np.unique(npresslta)) != 1:
            raise ValueError('error :: Number of pressure levels in all .lta files must be the same')
        if len(np.unique(ntemplta)) != 1:
            raise ValueError('error :: Number of temperature levels in all .lta files must be the same')

        self.ID = gasIDlta
        self.ISO = isoIDlta
        self.NP = npress
        self.NG = 1
        self.G_ORD = np.array([0.])
        self.DELG = np.array([1.0])
        self.NT = ntemp
        self.PRESS = presslevels
        self.TEMP = templevels
        self.NWAVE = nwave

        vmax = vmin + delv * (nwave-1)
        wavelta = np.linspace(vmin,vmax,nwave)
        self.WAVE = wavelta

    ######################################################################################################
    def read_kls(self, runname):
        """
        Read the .kls file and store the parameters into the Spectroscopy Class

        @param runname: str
            Name of the Nemesis run
        """

        from archnemesis import read_ktahead

        ngasact = len(open(runname+'.kls').readlines(  ))

        #Opening file
        f = open(runname+'.kls','r')
        strkta = [''] * ngasact
        for i in range(ngasact):
            s = f.readline().split()
            strkta[i] = s[0]

        self.NGAS = ngasact
        self.LOCATION = strkta

        #Now reading the head of the binary files included in the .kta file
        nwavekta = np.zeros([ngasact],dtype='int')
        npresskta = np.zeros([ngasact],dtype='int')
        ntempkta = np.zeros([ngasact],dtype='int')
        ngkta = np.zeros([ngasact],dtype='int')
        gasIDkta = np.zeros([ngasact],dtype='int')
        isoIDkta = np.zeros([ngasact],dtype='int')
        for i in range(ngasact):
            nwave,wavekta,fwhmk,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(strkta[i])
            nwavekta[i] = nwave
            npresskta[i] = npress
            ntempkta[i] = ntemp
            ngkta[i] = ng
            gasIDkta[i] = gasID
            isoIDkta[i] = isoID

        if len(np.unique(nwavekta)) != 1:
            raise ValueError('error :: Number of wavenumbers in all .kta files must be the same')
        if len(np.unique(npresskta)) != 1:
            raise ValueError('error :: Number of pressure levels in all .kta files must be the same')
        if len(np.unique(ntempkta)) != 1:
            raise ValueError('error :: Number of temperature levels in all .kta files must be the same')
        if len(np.unique(ngkta)) != 1:
            raise ValueError('error :: Number of g-ordinates in all .kta files must be the same')

        self.ID = gasIDkta
        self.ISO = isoIDkta
        self.NP = npress
        self.NT = ntemp
        self.PRESS = presslevels
        self.TEMP = templevels
        self.NWAVE = nwave
        self.NG = ng
        self.DELG = del_g
        self.G_ORD = g_ord
        self.FWHM = fwhmk
        self.WAVE = wavekta


    ######################################################################################################
    def read_header(self):
        """
        Given the LOCATION of the look-up tables, reads the header information
        """
        
        _lgr.warning(f'{self.NGAS=} {self.LOCATION=}')
        if self.NGAS>0:

            if self.ILBL==SpectralCalculationMode.K_TABLES:

                #Getting the extension of the look-up tables to see whether they are in HDF5 or binary formats
                ext = np.zeros(self.NGAS,dtype='int32')
                for i in range(self.NGAS):
                    tablex = self.LOCATION[i]
                    extx = tablex[len(tablex)-3:len(tablex)]
                    if extx=='kta':
                        ext[i] = 0
                    elif extx=='.h5':
                        ext[i] = 1
                    else:
                        raise ValueError('error in read_header :: The extention of the look-up tables must be .kta or .h5')
                
                if len(np.unique(ext)) != 1:
                    raise ValueError('error in read_header:: all look-up tables must be defined in the same format (with same extension)')
                    
                extx = np.unique(ext)[0]
                
                if extx==0:

                    #Now reading the head of the binary files included in the .kls file
                    nwavekta = np.zeros(self.NGAS,dtype='int')
                    npresskta = np.zeros(self.NGAS,dtype='int')
                    ntempkta = np.zeros(self.NGAS,dtype='int')
                    ngkta = np.zeros(self.NGAS,dtype='int')
                    gasIDkta = np.zeros(self.NGAS,dtype='int')
                    isoIDkta = np.zeros(self.NGAS,dtype='int')
                    for i in range(self.NGAS):
                        nwave,wavekta,fwhmk,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(self.LOCATION[i])
                        nwavekta[i] = nwave
                        npresskta[i] = npress
                        ntempkta[i] = ntemp
                        ngkta[i] = ng
                        gasIDkta[i] = gasID
                        isoIDkta[i] = isoID

                    if len(np.unique(nwavekta)) != 1:
                        raise ValueError('error :: Number of wavenumbers in all .kta files must be the same')
                    if len(np.unique(npresskta)) != 1:
                        raise ValueError('error :: Number of pressure levels in all .kta files must be the same')
                    if len(np.unique(ntempkta)) != 1:
                        raise ValueError('error :: Number of temperature levels in all .kta files must be the same')
                    if len(np.unique(ngkta)) != 1:
                        raise ValueError('error :: Number of g-ordinates in all .kta files must be the same')

                    self.ID = gasIDkta
                    self.ISO = isoIDkta
                    self.NP = npress
                    self.NT = ntemp
                    self.PRESS = presslevels
                    self.TEMP = templevels
                    self.NWAVE = nwave
                    self.NG = ng
                    self.DELG = del_g
                    self.G_ORD = g_ord
                    self.FWHM = fwhmk
                    self.WAVE = wavekta
                    
                else:
                    
                    raise ValueError('error in read_header:: HDF5 correlated-k look-up tables have not yet been implemented')

            elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:

                #Getting the extension of the look-up tables to see whether they are in HDF5 or binary formats
                ext = np.zeros(self.NGAS,dtype='int32')
                for i in range(self.NGAS):
                    tablex = self.LOCATION[i]
                    extx = tablex[len(tablex)-3:len(tablex)]
                    if extx=='lta':
                        ext[i] = 0
                    elif extx=='.h5':
                        ext[i] = 1
                    else:
                        raise ValueError('error in read_header :: The extention of the look-up tables must be .lta or .h5')
                
                if len(np.unique(ext)) != 1:
                    raise ValueError('error in read_header :: all look-up tables must be defined in the same format (with same extension)')
                    
                extx = np.unique(ext)[0]

                if extx==0:
                    
                    self.ONLINE = False  #With .lta tables we read them and store them on memory
                    
                    #Now reading the head of the binary files included in the .lls file
                    nwavelta = np.zeros(self.NGAS,dtype='int')
                    npresslta = np.zeros(self.NGAS,dtype='int')
                    ntemplta = np.zeros(self.NGAS,dtype='int')
                    gasIDlta = np.zeros(self.NGAS,dtype='int')
                    isoIDlta = np.zeros(self.NGAS,dtype='int')
                    for i in range(self.NGAS):
                        nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels = read_ltahead(self.LOCATION[i])
                        nwavelta[i] = nwave
                        npresslta[i] = npress
                        ntemplta[i] = ntemp
                        gasIDlta[i] = gasID
                        isoIDlta[i] = isoID

                    if len(np.unique(nwavelta)) != 1:
                        raise ValueError('error in read_header :: Number of wavenumbers in all .lta files must be the same')
                    if len(np.unique(npresslta)) != 1:
                        raise ValueError('error in read_header :: Number of pressure levels in all .lta files must be the same')
                    if len(np.unique(ntemplta)) != 1:
                        raise ValueError('error in read_header :: Number of temperature levels in all .lta files must be the same')

                    self.ID = gasIDlta
                    self.ISO = isoIDlta
                    self.NP = npress
                    self.NG = 1
                    self.G_ORD = np.array([0.])
                    self.DELG = np.array([1.0])
                    self.NT = ntemp
                    self.PRESS = presslevels
                    self.TEMP = templevels
                    self.NWAVE = nwave

                    vmax = vmin + delv * (nwave-1)
                    wavelta = np.linspace(vmin,vmax,nwave)
                    #wavelta = np.round(wavelta,5)
                    self.WAVE = wavelta
                    
                elif extx==1:
                    
                    self.ONLINE = True   #With .h5 tables we read them online when making the calculations
                    
                    #Now reading the head of the HDF5 files
                    nwavelta = np.zeros(self.NGAS,dtype='int')
                    npresslta = np.zeros(self.NGAS,dtype='int')
                    ntemplta = np.zeros(self.NGAS,dtype='int')
                    gasIDlta = np.zeros(self.NGAS,dtype='int')
                    isoIDlta = np.zeros(self.NGAS,dtype='int')
                    for i in range(self.NGAS):
                        ilbl,wave,npress,ntemp,gasID,isoID,presslevels,templevels = read_header_lta_hdf5(self.LOCATION[i])
                        if ilbl!=SpectralCalculationMode.LINE_BY_LINE_TABLES:
                            raise ValueError('error :: ILBL in look-up tables must be the same as in Spectroscopy class')
                        nwavelta[i] = len(wave)
                        npresslta[i] = npress
                        ntemplta[i] = ntemp
                        gasIDlta[i] = gasID
                        isoIDlta[i] = isoID
                        
                    if len(np.unique(nwavelta)) != 1:
                        raise ValueError('error in read_header :: Number of wavenumbers in all look-up tables must be the same')
                    if len(np.unique(npresslta)) != 1:
                        raise ValueError('error in read_header :: Number of pressure levels in all look-up tables must be the same')
                    if len(np.unique(ntemplta)) != 1:
                        raise ValueError('error in read_header :: Number of temperature levels in all look-up tables must be the same')
                    
                    self.ID = gasIDlta
                    self.ISO = isoIDlta
                    self.NP = npress
                    self.NG = 1
                    self.G_ORD = np.array([0.])
                    self.DELG = np.array([1.0])
                    self.NT = ntemp
                    self.PRESS = presslevels
                    self.TEMP = templevels
                    self.NWAVE = len(wave)
                    self.WAVE = wave
        else:
            self.ID = np.zeros((0,),dtype=int)
            self.ISO = np.zeros((0,),dtype=int)

    ######################################################################################################
    def read_tables(self, wavemin=0., wavemax=1.0e10):
        """
        Reads the .kta or .lta tables and stores the results into this class
        
        In the case that the look-up tables are stored in HDF5 format to be read online, 
        we just constrain the size of the wavelength array

        Optional parameters
        -----------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)
        """
        
        if self.LOCATION is None:
            raise ValueError('error in Spectroscopy.read_tables() :: LOCATION is not defined')
            
        if self.WAVE is None:
            #In this case the headers have not been read so we need to read them
            self.read_header()

        iwl = np.searchsorted(self.WAVE, wavemin, side='right') - 1
        if iwl < 0:
            iwl = 0

        iwh = np.searchsorted(self.WAVE, wavemax, side='left')
        if iwh >= self.NWAVE:
            iwh = self.NWAVE - 1

        wave1 = self.WAVE[iwl:iwh + 1]
        self.NWAVE = len(wave1)
        self.WAVE = wave1

        if self.ONLINE==False:
            #Tables must be read and stored on memory

            if self.ILBL==SpectralCalculationMode.K_TABLES: #K-tables

                kstore = np.zeros([self.NWAVE,self.NG,self.NP,self.NT,self.NGAS])
                for igas in range(self.NGAS):
                    gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(self.LOCATION[igas],self.WAVE.min(),self.WAVE.max())
                    kstore[:,:,:,:,igas] = k_g[:,:,:,:]
                self.edit_K(kstore)


            elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES: #LBL-tables
                kstore = np.zeros([self.NWAVE,self.NP,abs(self.NT),self.NGAS])
                for igas in range(self.NGAS):
                    npress,ntemp,gasID,isoID,presslevels,templevels,nwave,wave,k = read_lbltable(self.LOCATION[igas],self.WAVE.min(),self.WAVE.max())
                    kstore[:,:,:,igas] = k[:,:,:]
                self.edit_K(kstore)

            else:
                raise ValueError('error in Spectroscopy :: ILBL must be either 0 (K-tables) or 2 (LBL-tables)')

    ######################################################################################################
    def write_table_hdf5(self,ID,ISO,filename):
        """
        Write information on the look-up tables loaded in the Spectroscopy class into an HDF5 file
        
        Inputs
        ------
        
        ID :: Radtran ID of the table to write
        ISO :: Radtran isotope ID of the table to write
        filename :: Name of the look-up table file (without .h5)
        """
        
        import h5py
        
        #Identifying the location of the gas in the Spectroscopy class
        igas = np.where( (self.ID==ID) & (self.ISO==ISO) )[0]
        if len(igas)==0:
            raise ValueError('error in write_table_hdf5 :: The specified gas is not defined in the Spectroscopy class')
        
        
        if self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:
            
            if os.path.exists(filename+'.h5')==True:
                os.remove(filename+'.h5')
            
            with h5py.File(filename+'.h5','w') as f:
            
                #Writing the header information
                dset = h5py_helper.store_data(f, 'ILBL', data=self.ILBL)
                dset.attrs['title'] = "Spectroscopy calculation type"
                if self.ILBL==SpectralCalculationMode.K_TABLES:
                    dset.attrs['type'] = 'Correlated-k pre-tabulated look-up tables'
                elif self.ILBL==SpectralCalculationMode.LINE_BY_LINE_TABLES:
                    dset.attrs['type'] = 'Line-by-line pre-tabulated look-up tables'
                else:
                    raise ValueError('error :: ILBL must be 0 or 2')
                    
                dset = h5py_helper.store_data(f, 'ID', data=ID)
                dset.attrs['title'] = "ID of the gaseous species"

                dset = h5py_helper.store_data(f, 'ISO', data=ISO)
                dset.attrs['title'] = "Isotope ID of the gaseous species"
                
                dset = h5py_helper.store_data(f, 'WAVE', data=self.WAVE)
                dset.attrs['title'] = "Spectral points at which the cross sections are defined"
                
                dset = h5py_helper.store_data(f, 'NP', data=self.NP)
                dset.attrs['title'] = "Number of pressure levels at which the look-up table is tabulated"
                
                dset = h5py_helper.store_data(f, 'NT', data=self.NT)
                dset.attrs['title'] = "Number of temperature levels at which the look-up table is tabulated"
                
                dset = h5py_helper.store_data(f, 'PRESS', data=self.PRESS)
                dset.attrs['title'] = "Pressure levels at which the look-up table is tabulated / atm"
                
                dset = h5py_helper.store_data(f, 'TEMP', data=self.TEMP)
                dset.attrs['title'] = "Temperature levels at which the look-up table is tabulated / K"
                
                #Writing the coefficients
                dset = h5py_helper.store_data(f, 'K', data=self.K[:,:,:,igas])
                dset.attrs['title'] = "Tabulated cross sections / cm2 multiplied by a factor of 1.0 x 10^20"
            
        else:
            
            raise ValueError('error in write_table_hdf5 :: selected ILBL has not been implemented yet (only ILBL=2 is currently working)')

    ######################################################################################################
    def calc_klblg(self,npoints,press,temp,WAVECALC=[12345678.],MakePlot=False):
        """
        Calculate the absorption coefficient at a given pressure and temperature
        looking at pre-tabulated line-by-line tables (assumed to be already stored in this class)

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters
        ---------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)


        Outputs
        ---------

        K(NWAVE,NPOINTS,NGAS) :: Absorption cross sections of each gas in each p-T point
        dKdT(NWAVE,NPOINTS,NGAS) :: Rate of change of the absorption cross section with temperature for each gas in each p-T point

        """

        #Interpolating to the correct pressure and temperature
        ########################################################

        #K (NWAVE,NP,NT,NGAS)

        PRESS = np.log(self.PRESS)
        TEMP = self.TEMP
        kgood = np.zeros([self.NWAVE,npoints,self.NGAS])
        dkgooddT = np.zeros([self.NWAVE,npoints,self.NGAS])
        for ipoint in range(npoints):

            p_l = np.log(press[ipoint])
            if p_l < np.min(PRESS):
                p_l = np.min(PRESS)
            if p_l > np.max(PRESS):
                p_l = np.max(PRESS)

            t_l = temp[ipoint]

            if t_l < np.min(TEMP):
                t_l = np.min(TEMP)
            if t_l > np.max(TEMP):
                t_l = np.max(TEMP)

            ip = np.searchsorted(PRESS, p_l) - 1
            if ip < 0:
                ip = 0
            if ip >= len(PRESS) - 1:
                ip = len(PRESS) - 2

            v = (p_l - PRESS[ip]) / (PRESS[ip + 1] - PRESS[ip])


            if self.NT < 0:
                Tn = TEMP[ip]
                Tn2 = TEMP[ip + 1]
            else:
                Tn = TEMP
                Tn2 = TEMP
            
            
            it1 = np.searchsorted(Tn, t_l) - 1
            if it1 >= len(Tn)-1:
                it1 = len(Tn)-2
            u1 = (t_l-Tn[it1])/(Tn[it1+1]-Tn[it1])
            du1dt = 1./(Tn[it1+1]-Tn[it1])
            
            it2 = np.searchsorted(Tn2, t_l) - 1
            if it2 >= len(Tn2)-1:
                it2 = len(Tn2)-2
            u2 = (t_l-Tn2[it2])/(Tn2[it2+1]-Tn2[it2])
            du2dt = 1./(Tn2[it2+1]-Tn2[it2])
            
            klo1 = np.zeros((self.NWAVE,self.NGAS))
            klo2 = np.zeros((self.NWAVE,self.NGAS))
            khi1 = np.zeros((self.NWAVE,self.NGAS))
            khi2 = np.zeros((self.NWAVE,self.NGAS))
            
            if self.K is not None:
                #In this case the look-up tables are stored in memory
            
                klo1[:,:] = self.K[:,ip,it1,:]
                klo2[:,:] = self.K[:,ip,it1+1,:]
                khi1[:,:] = self.K[:,ip+1,it2,:]
                khi2[:,:] = self.K[:,ip+1,it2+1,:]
                
            else:
                
                #In this case the look-up tables are not stored in memory and need to be read online    
                #It is assumed that in this case they are HDF5 tables            
                import h5py
                
                for igas in range(self.NGAS):
                    
                    with h5py.File(self.LOCATION[igas],'r') as f:
                        kfile = f['K']
                        wave = f['WAVE']
                        
                        #Creating new array to make sure it matches the resolution of self.WAVE
                        vmin = np.round(np.float64(wave[0]), decimals=7)
                        delv = np.round(np.float64(wave[1] - wave[0]), decimals=7)
                        nwave = len(wave)
                        vmax = delv*(nwave-1) + vmin
                        wave = np.linspace(vmin,vmax,nwave)
                        
                        #Calculating the wavelengths to read
                        iwl = np.searchsorted(np.array(wave), np.min(self.WAVE), side='right') - 1
                        if iwl < 0:
                            iwl = 0

                        iwh = np.searchsorted(np.array(wave), np.max(self.WAVE), side='left')
                        if iwh >= nwave:
                            iwh = nwave - 1
                        
                        klo1[:,igas] = kfile[iwl:iwh+1,ip,it1,0]
                        klo2[:,igas] = kfile[iwl:iwh+1,ip,it1+1,0]
                        khi1[:,igas] = kfile[iwl:iwh+1,ip+1,it2,0]
                        khi2[:,igas] = kfile[iwl:iwh+1,ip+1,it2+1,0]
                        
            #Interpolating to get the k-coefficients at desired p-T
            igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
            
            kgood[igood[0],ipoint,igood[1]] = (1.0-v)*(1.0-u1)*np.log(klo1[igood[0],igood[1]])\
                                                  + v*(1.0-u2)*np.log(khi1[igood[0],igood[1]])\
                                                        + v*u2*np.log(khi2[igood[0],igood[1]])\
                                                  + (1.0-v)*u1*np.log(klo2[igood[0],igood[1]])
            
                                    
            kgood[igood[0],ipoint,igood[1]] = np.exp(kgood[igood[0],ipoint,igood[1]])
            
            dxdt =  -np.log(klo1[igood[0],igood[1]])*(1.0-v)*du1dt\
                    -np.log(khi1[igood[0],igood[1]])*v*du2dt\
                    +np.log(khi2[igood[0],igood[1]])*v*du2dt\
                    +np.log(klo2[igood[0],igood[1]])*(1.0-v)*du1dt
            
            
            dkgooddT[igood[0],ipoint,igood[1]] = kgood[igood[0],ipoint,igood[1]] * dxdt
            

            ibad = np.where( (klo1<=0.0) & (klo2<=0.0) & (khi1<=0.0) & (khi2<=0.0) )
            
            kgood[ibad[0],ipoint,ibad[1]] = (1.0-v)*(1.0-u1)*(klo1[ibad[0],ibad[1]])\
                                                  + v*(1.0-u2)*(khi1[ibad[0],ibad[1]])\
                                                        + v*u2*(khi2[ibad[0],ibad[1]])\
                                                  + (1.0-v)*u1*(klo2[ibad[0],ibad[1]])

            dxdt =  -klo1[ibad[0],ibad[1]]*(1.0-v)*du1dt\
                    -khi1[ibad[0],ibad[1]]*v*du2dt\
                    +khi2[ibad[0],ibad[1]]*v*du2dt\
                    +klo2[ibad[0],ibad[1]]*(1.0-v)*du1dt
            
            dkgooddT[ibad[0],ipoint,ibad[1]] = dxdt
            

        return kgood,dkgooddT

    ######################################################################################################
    def calc_klbl(self,npoints,press,temp,WAVECALC=[12345678.],MakePlot=False):
        """
        Calculate the absorption coefficient at a given pressure and temperature
        looking at pre-tabulated line-by-line tables (assumed to be already stored in this class)

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters
        ---------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)


        Outputs
        ---------

        K(NWAVE,NPOINTS,NGAS) :: Absorption cross sections of each gas in each p-T point

        """

        #Interpolating to the correct pressure and temperature
        ########################################################

        #K (NWAVE,NP,NT,NGAS)
        PRESS = np.log(self.PRESS)
        TEMP = self.TEMP
        kgood = np.zeros((self.NWAVE, npoints, self.NGAS))
        for ipoint in range(npoints):

            p_l = np.log(press[ipoint])
            if p_l < np.min(PRESS):
                p_l = np.min(PRESS)
            if p_l > np.max(PRESS):
                p_l = np.max(PRESS)

            t_l = temp[ipoint]

            if t_l < np.min(TEMP):
                t_l = np.min(TEMP)
            if t_l > np.max(TEMP):
                t_l = np.max(TEMP)

            ip = np.searchsorted(PRESS, p_l) - 1
            if ip < 0:
                ip = 0
            if ip >= len(PRESS) - 1:
                ip = len(PRESS) - 2

            v = (p_l - PRESS[ip]) / (PRESS[ip + 1] - PRESS[ip])


            if self.NT < 0:
                Tn = TEMP[ip]
                Tn2 = TEMP[ip + 1]
            else:
                Tn = TEMP
                Tn2 = TEMP
            
                
            it1 = np.searchsorted(Tn, t_l) - 1
            if it1 < 0:
                it1 = 0
            if it1 >= len(Tn) - 1:
                it1 = len(Tn) - 2
            u1 = (t_l - Tn[it1]) / (Tn[it1 + 1] - Tn[it1])
                
                
            it2 = np.searchsorted(Tn2, t_l) - 1
            if it2 < 0:
                it2 = 0
            if it2 >= len(Tn2) - 1:
                it2 = len(Tn2) - 2
            u2 = (t_l - Tn2[it2]) / (Tn2[it2 + 1] - Tn2[it2])

            
            klo1 = np.zeros((self.NWAVE, self.NGAS))
            klo2 = np.zeros((self.NWAVE, self.NGAS))
            khi1 = np.zeros((self.NWAVE, self.NGAS))
            khi2 = np.zeros((self.NWAVE, self.NGAS))

            if self.K is not None:
                # Look-up tables are stored in memory
                klo1[:, :] = self.K[:, ip, it1, :]
                klo2[:, :] = self.K[:, ip, it1 + 1, :]
                khi1[:, :] = self.K[:, ip + 1, it2, :]
                khi2[:, :] = self.K[:, ip + 1, it2 + 1, :]
            else:
                
                #In this case the look-up tables are not stored in memory and need to be read online    
                #It is assumed that in this case they are HDF5 tables            
                import h5py
                
                for igas in range(self.NGAS):
                    
                    with h5py.File(self.LOCATION[igas],'r') as f:
                        kfile = f['K']
                        wave = f['WAVE']
                        
                        #Creating new array to make sure it matches the resolution of self.WAVE
                        vmin = np.round(np.float64(wave[0]), decimals=7)
                        delv = np.round(np.float64(wave[1] - wave[0]), decimals=7)
                        nwave = len(wave)
                        vmax = delv*(nwave-1) + vmin
                        wave = np.linspace(vmin,vmax,nwave)
                        
                        #Calculating the wavelengths to read
                        iwl = np.searchsorted(np.array(wave), np.min(self.WAVE), side='right') - 1
                        if iwl < 0:
                            iwl = 0

                        iwh = np.searchsorted(np.array(wave), np.max(self.WAVE), side='left')
                        if iwh >= nwave:
                            iwh = nwave - 1
                        
                        klo1[:,igas] = kfile[iwl:iwh+1,ip,it1,0]
                        klo2[:,igas] = kfile[iwl:iwh+1,ip,it1+1,0]
                        khi1[:,igas] = kfile[iwl:iwh+1,ip+1,it2,0]
                        khi2[:,igas] = kfile[iwl:iwh+1,ip+1,it2+1,0]
                    
            
            
            # Interpolating to get the k-coefficients at desired p-T
            igood = np.where((klo1 > 0.0) & (klo2 > 0.0) & (khi1 > 0.0) & (khi2 > 0.0))

            kgood[igood[0], ipoint, igood[1]] = (
                (1.0 - v) * (1.0 - u1) * np.log(klo1[igood[0], igood[1]])
                + v * (1.0 - u2) * np.log(khi1[igood[0], igood[1]])
                + v * u2 * np.log(khi2[igood[0], igood[1]])
                + (1.0 - v) * u1 * np.log(klo2[igood[0], igood[1]])
            )

            kgood[igood[0], ipoint, igood[1]] = np.exp(kgood[igood[0], ipoint, igood[1]])

            ibad = np.where((klo1 <= 0.0) & (klo2 <= 0.0) & (khi1 <= 0.0) & (khi2 <= 0.0))

            kgood[ibad[0], ipoint, ibad[1]] = (
                (1.0 - v) * (1.0 - u1) * klo1[ibad[0], ibad[1]]
                + v * (1.0 - u2) * khi1[ibad[0], ibad[1]]
                + v * u2 * khi2[ibad[0], ibad[1]]
                + (1.0 - v) * u1 * klo2[ibad[0], ibad[1]]
            )
            
        return kgood


    ######################################################################################################
    def calc_klbl_online(self,npoints,press,temp,self_frac=1.0,wave=None,add_pressure_shift=True):
        """
        Calculate the absorption coefficient at a given pressure and temperature
        from the LineData class

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters
        ---------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)
        @param self_frac: real
            Fraction of the line broadening that is due to self-broadening (as opposed to broadening by the ambient gas)
        @param add_pressure_shift: bool
            Whether to include pressure shift in the line positions (NEMESIS does not include it)


        Outputs
        ---------

        K(NWAVE,NPOINTS,NGAS) :: Absorption cross sections of each gas in each p-T point

        """

        if wave is None:
            wave = self.WAVE
            nwave = self.NWAVE
        else:
            wave = np.array(wave)
            nwave = len(wave)

        k = np.zeros((nwave, npoints, self.NGAS))
        for igas in range(self.NGAS):

            _lgr.info(f'Gas {self.ID[igas]}, Isotope {self.ISO[igas]} - Calculating line-by-line cross sections...')

            linedata = ans.LineData_1(
                self.ID[igas], #ID of the gas
                self.ISO[igas], #Isotope ID of the gas
                LINE_DATABASE=self.LOCATION[igas] # Different database location
            )

            if self.IPROC[igas]==SpectroscopicLineProfile.VOIGT:
                lineshape = ans.Data.lineshapes.voigt
            elif self.IPROC[igas]==SpectroscopicLineProfile.DOPPLER:
                lineshape = ans.Data.lineshapes.doppler
            elif self.IPROC[igas]==SpectroscopicLineProfile.LORENTZ:
                lineshape = ans.Data.lineshapes.lorentz
            else:
                raise ValueError('error in calc_klbl_online :: selected IPROC has not been implemented yet')

            # Download partition function tables for the gas isotopes
            linedata.fetch_partition_function()

            for ipoint in range(npoints):

                p_l = press[ipoint]
                t_l = temp[ipoint]

                k[:,ipoint,igas] = linedata.calculate_monochromatic_absorption(
                            waves=wave,        # wavenumbers in cm^{-1}
                            temp=t_l,               # kelvin
                            press=p_l,              # Atmospheres
                            amb_frac=1.-self_frac,  # fraction of broadening due to ambient gas
                            wave_unit=self.ISPACE,  # unit of `waves` argument
                            lineshape_fn=lineshape, # lineshape function to use
                            line_calculation_wavenumber_window=self.VREL, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
                            add_pressure_shift=add_pressure_shift, # whether to include pressure shift in the line positions
                )

        return k


    ######################################################################################################
    def calc_kg(self,npoints,press,temp,WAVECALC=[12345678.],MakePlot=False):
        """
        Calculate the k-coefficients at a given pressure and temperature
        looking at pre-tabulated k-tables (assumed to be already stored in this class)

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters
        ---------------------
        @param wavemin: real
            Minimum wavenumber (cm-1) or wavelength (um)
        @param wavemax: real
            Maximum wavenumber (cm-1) or wavelength (um)
        """

        #Interpolating the k-coefficients to the correct pressure and temperature
        #############################################################################

        #K (NWAVE,NG,NPOINTS,NGAS)

        kgood = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
        dkgooddT = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
        for ipoint in range(npoints):
            press1 = press[ipoint]
            temp1 = temp[ipoint]

            #Getting the levels just above and below the desired points
            lpress  = np.log(press1)
            ip = np.argmin(np.abs(self.PRESS-press1))

            if self.PRESS[ip]>=press1:
                iphi = ip
                if ip==0:
                    lpress = np.log(self.PRESS[0])
                    ipl = 0
                    iphi = 1
                else:
                    ipl = ip - 1
            elif self.PRESS[ip]<press1:
                ipl = ip
                if ip==self.NP-1:
                    lpress = np.log(self.PRESS[self.NP-1])
                    iphi = self.NP - 1
                    ipl = self.NP - 2
                else:
                    iphi = ip + 1

            it = np.argmin(np.abs(self.TEMP-temp1))

            if self.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    temp1 = self.TEMP[0]
                    itl = 0
                    ithi = 1
                else:
                    itl = it - 1
            elif self.TEMP[it]<temp1:
                itl = it
                if it==self.NT-1:
                    temp1 = self.TEMP[self.NT-1]
                    ithi = self.NT - 1
                    itl = self.NT -2
                else:
                    ithi = it + 1

            plo = np.log(self.PRESS[ipl])
            phi = np.log(self.PRESS[iphi])
            tlo = self.TEMP[itl]
            thi = self.TEMP[ithi]
            klo1 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            klo2 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            khi1 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            khi2 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            klo1[:] = self.K[:,:,ipl,itl,:]
            klo2[:] = self.K[:,:,ipl,ithi,:]
            khi2[:] = self.K[:,:,iphi,ithi,:]
            khi1[:] = self.K[:,:,iphi,itl,:]

            #Interpolating to get the k-coefficients at desired p-T
            v = (lpress-plo)/(phi-plo)
            u = (temp1-tlo)/(thi-tlo)
            dudt = 1./(thi-tlo)

            igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
            kgood[igood[0],igood[1],ipoint,igood[2]] = (1.0-v)*(1.0-u)*np.log(klo1[igood[0],igood[1],igood[2]]) + v*(1.0-u)*np.log(khi1[igood[0],igood[1],igood[2]]) + v*u*np.log(khi2[igood[0],igood[1],igood[2]]) + (1.0-v)*u*np.log(klo2[igood[0],igood[1],igood[2]])
            kgood[igood[0],igood[1],ipoint,igood[2]] = np.exp(kgood[igood[0],igood[1],ipoint,igood[2]])
            dxdt = (-np.log(klo1[igood[0],igood[1],igood[2]])*(1.0-v) - np.log(khi1[igood[0],igood[1],igood[2]])*v + np.log(khi2[igood[0],igood[1],igood[2]])*v + np.log(klo2[igood[0],igood[1],igood[2]]) * (1.0-v))*dudt
            dkgooddT[igood[0],igood[1],ipoint,igood[2]] = kgood[igood[0],igood[1],ipoint,igood[2]] * dxdt

            ibad = np.where( (klo1<=0.0) & (klo2<=0.0) & (khi1<=0.0) & (khi2<=0.0) )
            kgood[ibad[0],ibad[1],ipoint,ibad[2]] = (1.0-v)*(1.0-u)*klo1[ibad[0],ibad[1],ibad[2]] + v*(1.0-u)*khi1[ibad[0],ibad[1],ibad[2]] + v*u*khi2[ibad[0],ibad[1],ibad[2]] + (1.0-v)*u*klo2[ibad[0],ibad[1],ibad[2]]
            dxdt = (-klo1[ibad[0],ibad[1],ibad[2]]*(1.0-v) - khi1[ibad[0],ibad[1],ibad[2]]*v + khi2[ibad[0],ibad[1],ibad[2]]*v + klo2[ibad[0],ibad[1],ibad[2]] * (1.0-v))*dudt
            dkgooddT[ibad[0],ibad[1],ipoint,ibad[2]] = dxdt


        #Checking that the calculation wavenumbers coincide with the wavenumbers in the k-tables
        ##########################################################################################

        if WAVECALC[0]!=12345678.:

            NWAVEC = len(WAVECALC)
            kret = np.zeros([NWAVEC,self.NG,npoints,self.NGAS])
            dkretdT = np.zeros([NWAVEC,self.NG,npoints,self.NGAS])

            #Checking if k-tables are defined in irregularly spaced wavenumber grid
            delv = 0.0
            Irr = 0
            for iv in range(self.NWAVE-1):
                delv1 = self.WAVE[iv+1] - self.WAVE[iv]
                if iv==0:
                    delv = delv1
                    pass

                if abs((delv1-delv)/(delv))>0.001:
                    Irr = 1
                    break
                else:
                    delv = delv1
                    continue

            #If they are defined in a regular grid, we interpolate to the nearest value
            if Irr==0:
                for i in range(npoints):
                    for j in range(self.NGAS):
                        for k in range(self.NG):
                            f = scipy.interpolate.interp1d(self.WAVE,kgood[:,k,i,j])
                            kret[:,k,i,j] = f(WAVECALC)
                            f = scipy.interpolate.interp1d(self.WAVE,dkgooddT[:,k,i,j])
                            dkretdT[:,k,i,j] = f(WAVECALC)
            else:
                for i in range(NWAVEC):
                    iv = np.argmin(np.abs(self.WAVE-WAVECALC[i]))
                    kret[i,:,:,:] = kgood[iv,:,:,:]
                    dkretdT[i,:,:,:] = dkgooddT[iv,:,:,:]

        else:

            kret = kgood
            dkretdT = dkgooddT

        return kret,dkretdT

    ######################################################################################################
    def calc_k(self,npoints,press,temp,WAVECALC=[12345678.],MakePlot=False,linear=False):
        """
        Calculate the k-coefficients at a given pressure and temperature
        looking at pre-tabulated k-tables (assumed to be already stored in this class)

        Input parameters
        -------------------
        @param npoints: int
            Number of p-T points at which to calculate the cross sections
        @param press: 1D array
            Pressure levels (atm)
        @param temp: 1D array
            Temperature levels (K)

        Optional parameters
        ---------------------
        @param WAVECALC: 1D array
            Wavenumbers or wavelengths at which to calculate the k-coefficients
        @param linear: bool
            If True, the interpolation is done linearly. If False, it is done in log-space
        """

        #Interpolating the k-coefficients to the correct pressure and temperature
        #############################################################################

        #K (NWAVE,NG,NPOINTS,NGAS)
        TEMP = self.TEMP
        PRESS = self.PRESS
        NP = self.NP
        NT = self.NT
        
        kgood = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
        #dkgooddT = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
        for ipoint in range(npoints):
            press1 = press[ipoint]
            temp1 = temp[ipoint]

            # Find pressure grid points above and below current layer pressure
            ip = np.abs(PRESS - press1).argmin()
            if PRESS[ip] >= press1:
                ip_high = ip
                if ip == 0:
                    press1 = PRESS[0]
                    ip_low = 0
                    ip_high = 1
                else:
                    ip_low = ip - 1
            elif PRESS[ip] < press1:
                ip_low = ip
                if ip == NP - 1:
                    press1 = PRESS[NP - 1]
                    ip_high = NP - 1
                    ip_low = NP - 2
                else:
                    ip_high = ip + 1

            # Find temperature grid points above and below current layer temperature
            it = np.abs(TEMP - temp1).argmin()
            if TEMP[it] >= temp1:
                it_high = it
                if it == 0:
                    temp1 = TEMP[0]
                    it_low = 0
                    it_high = 1
                else:
                    it_low = it - 1
            elif TEMP[it] < temp1:
                it_low = it
                if it == NT - 1:
                    temp1 = TEMP[NT - 1]
                    it_high = NT - 1
                    it_low = NT - 2
                else:
                    it_high = it + 1

            lpress = np.log(press1)
            plo = np.log(self.PRESS[ip_low])
            phi = np.log(self.PRESS[ip_high])
            tlo = self.TEMP[it_low]
            thi = self.TEMP[it_high]
            klo1 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            klo2 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            khi1 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            khi2 = np.zeros([self.NWAVE,self.NG,self.NGAS])
            klo1[:] = self.K[:,:,ip_low,it_low,:]
            klo2[:] = self.K[:,:,ip_low,it_high,:]
            khi2[:] = self.K[:,:,ip_high,it_high,:]
            khi1[:] = self.K[:,:,ip_high,it_low,:]

            #Interpolating to get the k-coefficients at desired p-T
            v = (lpress-plo)/(phi-plo)
            u = (temp1-tlo)/(thi-tlo)

            igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
            
            
            if linear is True:
                #Linear interpolation
                kgood[igood[0],igood[1],ipoint,igood[2]] = (1.0-v)*(1.0-u)*(klo1[igood[0],igood[1],igood[2]]) + v*(1.0-u)*(khi1[igood[0],igood[1],igood[2]]) + v*u*(khi2[igood[0],igood[1],igood[2]]) + (1.0-v)*u*(klo2[igood[0],igood[1],igood[2]])
            else:
                #Logarithmic interpolation
                kgood[igood[0],igood[1],ipoint,igood[2]] = (1.0-v)*(1.0-u)*(np.log(klo1[igood[0],igood[1],igood[2]])) + v*(1.0-u)*(np.log(khi1[igood[0],igood[1],igood[2]])) + v*u*(np.log(khi2[igood[0],igood[1],igood[2]])) + (1.0-v)*u*(np.log(klo2[igood[0],igood[1],igood[2]]))
                kgood[igood[0],igood[1],ipoint,igood[2]] = np.exp(kgood[igood[0],igood[1],ipoint,igood[2]])
            
            ibad = np.where( (klo1<=0.0) & (klo2<=0.0) & (khi1<=0.0) & (khi2<=0.0) )
            kgood[ibad[0],ibad[1],ipoint,ibad[2]] = (1.0-v)*(1.0-u)*klo1[ibad[0],ibad[1],ibad[2]] + v*(1.0-u)*khi1[ibad[0],ibad[1],ibad[2]] + v*u*khi2[ibad[0],ibad[1],ibad[2]] + (1.0-v)*u*klo2[ibad[0],ibad[1],ibad[2]]


        #Checking that the calculation wavenumbers coincide with the wavenumbers in the k-tables
        ##########################################################################################
        
        NWAVEC = len(WAVECALC)
        kret = np.zeros([NWAVEC,self.NG,npoints,self.NGAS])
        # Precompute indices and weights for WAVECALC
        precomputed_indices = np.zeros((NWAVEC,2),dtype=int)
        precomputed_weights = np.zeros(NWAVEC)

        for iwave in range(NWAVEC):
            wave = WAVECALC[iwave]
            iw_closest = np.searchsorted(self.WAVE, wave)  # Find insertion point

            iw_low = max(iw_closest - 1, 0)
            iw_high = min(iw_closest, len(self.WAVE) - 1)
            if iw_high == iw_low:
                iw_high = min(iw_high + 1, len(self.WAVE) - 1)

            wave_low = self.WAVE[iw_low]
            wave_high = self.WAVE[iw_high]
            w = (wave - wave_low) / (wave_high - wave_low) if wave_high != wave_low else 0

            precomputed_indices[iwave] = ((iw_low, iw_high))
            precomputed_weights[iwave] = (w)

        kret = interpolate_k_values(npoints, self.NGAS, NWAVEC, precomputed_indices,
                                             precomputed_weights, kgood, self.DELG, kret)
        return kret


###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

Other functions interacting with the Spectroscopy class
"""


def read_ltahead(filename):
    """
    Read the header information in a line-by-line look-up table
    written with the standard format of Nemesis

    @param filename: str
        Name of the .lta file
    """

    #Opening file
    if not filename.endswith('.lta'):
        filename += '.lta'
    
    with open(filename, 'rb') as f:
        
        _ = int(np.fromfile(f,dtype='int32',count=1)[0]) # irec0
        nwave = np.fromfile(f,dtype='int32',count=1)[0]
        vmin = np.fromfile(f,dtype='float32',count=1)[0]
        delv = np.fromfile(f,dtype='float32',count=1)[0]
        npress = int(np.fromfile(f,dtype='int32',count=1)[0])
        ntemp = int(np.fromfile(f,dtype='int32',count=1)[0])
        gasID = int(np.fromfile(f,dtype='int32',count=1)[0])
        isoID = int(np.fromfile(f,dtype='int32',count=1)[0])
        
        # Convert explicitly rounding to 7 decimals (float32 precision)
        vmin = np.round(np.float64(vmin), decimals=7)
        delv = np.round(np.float64(delv), decimals=7)

        presslevels = np.fromfile(f,dtype='float32',count=npress)
        if ntemp > 0:
            templevels = np.fromfile(f,dtype='float32',count=ntemp)
        else:
            templevels = np.zeros((npress,2))
            for i in range(npress):
                templevels[i] = np.fromfile(f,dtype='float32',count=-ntemp)

    return nwave,vmin,delv,npress,ntemp,gasID,isoID,presslevels,templevels


###############################################################################################

def read_ktahead(filename):

    """
        FUNCTION NAME : read_ktahead_nemesis()

        DESCRIPTION : Read the header information in a correlated-k look-up table written with the standard format of Nemesis

        INPUTS :

            filename :: Name of the file (supposed to have a .kta extension)

        OPTIONAL INPUTS: none

        OUTPUTS :

            nwave :: Number of wavelength points
            wave :: Wavelength (um) / Wavenumber (cm-1) array
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            pressleves(np) :: Pressure levels (atm)
            templeves(np) :: Temperature levels (K)

        CALLING SEQUENCE:

            nwave,wave,fwhm,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels = read_ktahead(filename)

        MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """
    #Opening file
    if not filename.endswith('.kta'):
        filename += '.kta'
    
    with open(filename, 'rb') as f:

        _ = int(np.fromfile(f,dtype='int32',count=1)[0]) # irec0
        nwave = int(np.fromfile(f,dtype='int32',count=1)[0])
        vmin = np.fromfile(f,dtype='float32',count=1)[0]
        delv = np.fromfile(f,dtype='float32',count=1)[0]
        fwhm = np.fromfile(f,dtype='float32',count=1)[0]
        npress = int(np.fromfile(f,dtype='int32',count=1)[0])
        ntemp = int(np.fromfile(f,dtype='int32',count=1)[0])
        ng = int(np.fromfile(f,dtype='int32',count=1)[0])
        gasID = int(np.fromfile(f,dtype='int32',count=1)[0])
        isoID = int(np.fromfile(f,dtype='int32',count=1)[0])

        # Convert explicitly rounding to 7 decimals (float32 precision)
        vmin = np.round(np.float64(vmin), decimals=7)
        delv = np.round(np.float64(delv), decimals=7)

        g_ord = np.fromfile(f,dtype='float32',count=ng)
        del_g = np.fromfile(f,dtype='float32',count=ng)

        _ = np.fromfile(f,dtype='float32',count=1)
        _ = np.fromfile(f,dtype='float32',count=1)

        presslevels = np.fromfile(f,dtype='float32',count=npress)

        N1 = abs(ntemp)
        if ntemp < 0:
            templevels = np.zeros([npress,N1])
            for i in range(npress):
                for j in range(N1):
                    templevels[i,j] =  np.fromfile(f,dtype='float32',count=1)
        else:
            templevels = np.fromfile(f,dtype='float32',count=ntemp)

        #Reading central wavelengths in non-uniform grid
        if delv>0.0:
            vmax = delv*(nwave-1) + vmin
            wavetot = np.linspace(vmin,vmax,nwave)
        else:
            wavetot = np.zeros(nwave)
            wavetot[:] = np.fromfile(f,dtype='float32',count=nwave)

    return nwave,wavetot,fwhm,npress,ntemp,ng,gasID,isoID,g_ord,del_g,presslevels,templevels


def read_header_lta_hdf5(filename):
    """
        FUNCTION NAME : read_header_lta_hdf5()

        DESCRIPTION : Read the header of the look-up line-by-line tables stored in HDF5 files

        INPUTS :

            filename :: Name of the look-up table file

        OPTIONAL INPUTS: none

        OUTPUTS :

            ilbl :: Look-up table type (0 - correlated-k ; 2 - line-by-line)
            wave :: Wavelength (um) / Wavenumber (cm-1) array
            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            pressleves(np) :: Pressure levels (atm)
            templeves(np) :: Temperature levels (K)
            
        CALLING SEQUENCE:

            ilbl,wave,npress,ntemp,gasID,isoID,presslevels,templevels = read_header_lta_hdf5(filename)

        MODIFICATION HISTORY : Juan Alday (29/04/2023)
    """
    
    import h5py
    
    #Opening file
    filename = filename if filename.endswith('.h5') else (filename+'.h5')

    with h5py.File(filename,'r') as f:

        ilbl = h5py_helper.retrieve_data(f, 'ILBL', np.int32)
        if ilbl==SpectralCalculationMode.LINE_BY_LINE_TABLES:
            wave = h5py_helper.retrieve_data(f, 'WAVE', np.array)
            npress = h5py_helper.retrieve_data(f, 'NP', np.int32)
            ntemp = h5py_helper.retrieve_data(f, 'NT', np.int32)
            gasID = h5py_helper.retrieve_data(f, 'ID', np.int32)
            isoID = h5py_helper.retrieve_data(f, 'ISO', np.int32)
            presslevels = h5py_helper.retrieve_data(f, 'PRESS', np.array)
            templevels = h5py_helper.retrieve_data(f, 'TEMP', np.array)
        else:
            raise ValueError('error in read_header_lta_hdf5 :: the defined ilbl in the look-up table must be 2')
        

    return ilbl,wave,npress,ntemp,gasID,isoID,presslevels,templevels


###############################################################################################
def read_lbltable(filename,wavemin,wavemax):

    """
        FUNCTION NAME : read_lbltable()

        DESCRIPTION : Read the line-by-line look-up table written with the standard format of Nemesis

        INPUTS :

            filename :: Name of the file (supposed to have a .lta extension)
            wavemin :: Minimum wavenumber to read (cm-1)
            wavemax :: Maximum wavenumber to read (cm-1)

        OPTIONAL INPUTS: none

        OUTPUTS :

            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: RADTRAN gas ID
            isoID :: RADTRAN isotopologue ID
            presslevels(np) :: Pressure levels (atm)
            templevels(np) :: Temperature levels (K)
            nwave :: Number of wavenumbers
            wave :: Wavenumber array (cm-1)
            k(nwave,np,nt) :: Absorption coefficient at each p-T point (cm2)

        CALLING SEQUENCE:

            npress,ntemp,gasID,isoID,presslevels,templevels,nwave,wave,k = read_lbltable(filename,wavemin,wavemax)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """
    #Opening file
    if not filename.endswith('.lta'):
        filename += '.lta'
    
    with open(filename, 'rb') as f:

        #nbytes_int32 = 4
        nbytes_float32 = 4

        #Reading header
        irec0 = np.fromfile(f,dtype='int32',count=1)[0]
        _lgr.debug(f'{irec0=}')
        nwavelta = np.fromfile(f,dtype='int32',count=1)[0]
        _lgr.debug(f'{nwavelta=}')
        vmin = np.fromfile(f,dtype='float32',count=1)[0]
        _lgr.debug(f'{vmin=}')
        delv = np.fromfile(f,dtype='float32',count=1)[0]
        _lgr.debug(f'{delv=}')
        npress = np.fromfile(f,dtype='int32',count=1)[0]
        _lgr.debug(f'{npress=}')
        ntemp = np.fromfile(f,dtype='int32',count=1)[0]
        _lgr.debug(f'{ntemp=}')
        gasID = np.fromfile(f,dtype='int32',count=1)[0]
        _lgr.debug(f'{gasID=}')
        isoID = np.fromfile(f,dtype='int32',count=1)[0]
        _lgr.debug(f'{isoID=}')

        # Convert explicitly rounding to 7 decimals (float32 precision)
        vmin = np.round(np.float64(vmin), decimals=7)
        delv = np.round(np.float64(delv), decimals=7)

        presslevels = np.fromfile(f,dtype='float32',count=npress)
        
        if ntemp > 0:
            templevels = np.fromfile(f,dtype='float32',count=ntemp)
        else:
            templevels = np.zeros((npress,2))
            for i in range(npress):
                templevels[i] = np.fromfile(f,dtype='float32',count=-ntemp)

        #Calculating the wavenumbers to be read
        vmax = vmin + delv * (nwavelta-1)
        wavelta = np.linspace(vmin,vmax,nwavelta)

        wn_idxs = np.nonzero( (wavemin<=wavelta) & (wavelta<=wavemax) )[0]
        _lgr.debug(f'{wn_idxs=}')
        
        nwave = len(wn_idxs)
        wave = np.zeros(nwave)
        wave[:] = wavelta[wn_idxs]

        #Reading the absorption coefficients
        #######################################
        k = np.zeros([nwave,npress,abs(ntemp)])

        #Jumping until we get to the minimum wavenumber
        njump = npress*abs(ntemp)*(wn_idxs[0])
        ioff = njump*nbytes_float32 + (irec0-1)*nbytes_float32
        f.seek(ioff,0)

        #Reading the coefficients we require
        k_out = np.fromfile(f,dtype='float32',count=abs(ntemp)*npress*nwave)
        il = 0
        for ik in range(nwave):
            for i in range(npress):
                k[ik,i,:] = k_out[il:il+abs(ntemp)]
                il = il + abs(ntemp)
    """
    _lgr.debug(f'{filename=}')
    _lgr.debug(f'{npress=}')
    _lgr.debug(f'{ntemp=}')
    _lgr.debug(f'{gasID=}')
    _lgr.debug(f'{isoID=}')
    _lgr.debug(f'{presslevels=}')
    _lgr.debug(f'{templevels=}')
    _lgr.debug(f'{nwave=}')
    _lgr.debug(f'{wave=}')
    _lgr.debug(f'{k=}')
    """
    
    return npress,ntemp,gasID,isoID,presslevels,templevels,nwave,wave,k


###############################################################################################
def read_ktable(filename,wavemin,wavemax):

    """
        FUNCTION NAME : read_ktable()

        DESCRIPTION : Read the correlated-k look-up table written with the standard format of Nemesis

        INPUTS :

            filename :: Name of the file (supposed to have a .kta extension)
            wavemin :: Minimum wavenumber to read (cm-1)
            wavemax :: Maximum wavenumber to read (cm-1)

        OPTIONAL INPUTS: none

        OUTPUTS :

            gasID :: Nemesis gas identifier
            isoID :: Nemesis isotopologue identifier
            nwave :: Number of wavenumbers
            wave(nwave) :: Wavenumbers or wavelengths
            fwhm :: Full width at half maximum
            ng :: Number of g-ordinates
            g_ord(ng) :: g-ordinates
            del_g(ng) :: Intervals of g-ordinates
            npress :: Number of pressure levels
            presslevels(npress) :: Pressure levels (atm)
            ntemp :: Number of temperature levels
            templevels(ntemp) :: Temperature levels (K)
            k_g(nwave,ng,npress,ntemp) :: K coefficients

        CALLING SEQUENCE:

            gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(filename,wavemin,wavemax)

        MODIFICATION HISTORY : Juan Alday (05/03/2021)

    """
    #Opening file
    if not filename.endswith('.kta'):
        filename += '.kta'
    
    with open(filename, 'rb') as f:

        nbytes_int32 = 4
        nbytes_float32 = 4
        ioff = 0

        #Reading header
        irec0 = int(np.fromfile(f,dtype='int32',count=1)[0])
        nwavekta = int(np.fromfile(f,dtype='int32',count=1)[0])
        vmin = np.fromfile(f,dtype='float32',count=1)[0]
        delv = np.fromfile(f,dtype='float32',count=1)[0]
        fwhm = float(np.fromfile(f,dtype='float32',count=1)[0])
        npress = int(np.fromfile(f,dtype='int32',count=1)[0])
        ntemp = int(np.fromfile(f,dtype='int32',count=1)[0])
        ng = int(np.fromfile(f,dtype='int32',count=1)[0])
        gasID = int(np.fromfile(f,dtype='int32',count=1)[0])
        isoID = int(np.fromfile(f,dtype='int32',count=1)[0])

        # Convert explicitly rounding to 7 decimals (float32 precision)
        vmin = np.round(np.float64(vmin), decimals=7)
        delv = np.round(np.float64(delv), decimals=7)

        ioff = ioff + 10 * nbytes_int32

        g_ord = np.zeros(ng)
        del_g = np.zeros(ng)
        templevels = np.zeros(ntemp)
        presslevels = np.zeros(npress)
        g_ord[:] = np.fromfile(f,dtype='float32',count=ng)
        del_g[:] = np.fromfile(f,dtype='float32',count=ng)

        ioff = ioff + 2*ng*nbytes_float32

        _ = np.fromfile(f,dtype='float32',count=1)[0]
        _ = np.fromfile(f,dtype='float32',count=1)[0]

        ioff = ioff + 2*nbytes_float32

        presslevels[:] = np.fromfile(f,dtype='float32',count=npress)
        templevels[:] = np.fromfile(f,dtype='float32',count=ntemp)

        ioff = ioff + npress*nbytes_float32+ntemp*nbytes_float32

        #Reading central wavelengths in non-uniform grid
        if delv>0.0:
            vmax = delv*(nwavekta-1) + vmin
            wavetot = np.linspace(vmin,vmax,nwavekta)
        else:
            wavetot = np.zeros([nwavekta])
            wavetot[:] = np.fromfile(f,dtype='float32',count=nwavekta)
            ioff = ioff + nwavekta*nbytes_float32

        #Calculating the wavenumbers to be read
        ins = np.where( (wavetot>=wavemin) & (wavetot<=wavemax) )[0]
        nwave = len(ins)
        wave = np.zeros([nwave])
        wave[:] = wavetot[ins]

        #Reading the k-coefficients
        #######################################

        k_g = np.zeros([nwave,ng,npress,ntemp])

        #Jumping until we get to the minimum wavenumber
        njump = npress*ntemp*ng*ins[0]
        ioff = njump*nbytes_float32 + (irec0-1)*nbytes_float32
        f.seek(ioff,0)

        #Reading the coefficients we require
        k_out = np.fromfile(f,dtype='float32',count=ntemp*npress*ng*nwave)
        il = 0
        for ik in range(nwave):
            for i in range(npress):
                for j in range(ntemp):
                    k_g[ik,:,i,j] = k_out[il:il+ng]
                    il = il + ng

    return gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g

######################################################################################################

def write_lbltable(filename,npress,ntemp,gasID,isoID,presslevels,templevels,nwave,vmin,delv,k,DOUBLE=False):

    """
        FUNCTION NAME : write_lbltable()

        DESCRIPTION : Read a .lta file (binary file) with the information about the absorption cross-section
                      of a given gas at different pressure and temperature levels

        INPUTS :

            filename :: Name of the file (supposed to have a .kta extension)

        OPTIONAL INPUTS:

            DOUBLE :: If True, the parameters are written with double precision (double) rather than single (float)

        OUTPUTS :

            npress :: Number of pressure levels
            ntemp :: Number of temperature levels
            gasID :: NEMESIS gas ID (see manual)
            isoID :: NEMESIS isotopologue ID (0 for all isotopes)
            presslevels(npress) :: Pressure levels (atm)
            templevels(ntemp) :: Temperature levels (K)
            nwave :: Number of spectral points in lbl-table
            vmin :: Minimum wavelength/wavenumber (um/cm-1)
            delv :: Wavelength/wavenumber step (um/cm-1)
            k(nwave,npress,ntemp) :: Absorption cross-section (cm2)

        CALLING SEQUENCE:

            write_lbltable(filename,npress,ntemp,gasID,isoID,presslevels,templevels,nwave,vmin,delv,k)

        MODIFICATION HISTORY : Juan Alday (06/08/2021)

    """

    import struct

    #Opening file
    if not filename.endswith('.lta'):
        filename += '.lta'
    
    if np.any(presslevels < 0):
        raise ValueError("error in write_lbltable :: Pressure levels must be non-negative")

    with open(filename, 'wb') as f:

        irec0 = 9 + npress + ntemp    #Don't know why this 9 is like this, but it works for a Linux/Ubuntu machine
        bin=struct.pack('i',irec0) #IREC0
        f.write(bin)

        bin=struct.pack('i',nwave) #NWAVE
        f.write(bin)

        if DOUBLE==True:
            df = 'd'
        else:
            df = 'f'

        bin=struct.pack(df,vmin) #VMIN
        f.write(bin)

        bin=struct.pack(df,delv) #DELV
        f.write(bin)

        bin=struct.pack('i',npress) #NPRESS
        f.write(bin)

        bin=struct.pack('i',ntemp) #NTEMP
        f.write(bin)

        bin=struct.pack('i',gasID) #GASID
        f.write(bin)

        bin=struct.pack('i',isoID) #ISOID
        f.write(bin)

        myfmt=df*len(presslevels)
        bin=struct.pack(myfmt,*presslevels) #PRESSLEVELS
        f.write(bin)

        myfmt=df*len(templevels)
        bin=struct.pack(myfmt,*templevels) #TEMPLEVELS
        f.write(bin)

        for i in range(nwave):
            for j in range(npress):
                tmp = k[i,j,:] * 1.0e20
                myfmt=df*len(tmp)
                bin=struct.pack(myfmt,*tmp) #K
                f.write(bin)

    
###############################################################################################
def write_ktable(filename,gasID,isoID,g_ord,del_g,presslevels,templevels,nwave,vmin,delv,fwhm,k_g,wave=None):
    """
    FUNCTION NAME : write_ktable()

    DESCRIPTION : Write a correlated-k look-up table in standard Nemesis .kta format

    INPUTS :

        filename :: Name of the output file ('.kta' appended if missing)
        gasID :: Nemesis gas identifier
        isoID :: Nemesis isotopologue identifier
        g_ord(ng) :: g-ordinates
        del_g(ng) :: g-intervals
        presslevels(npress) :: Pressure levels (atm)
        templevels(ntemp) :: Temperature levels (K)
        nwave :: Number of spectral points
        vmin :: Minimum wavenumber/wavelength
        delv :: Increment in wavenumber/wavelength
        fwhm :: Full width at half maximum of instrument lineshape
        k_g(nwave,ng,npress,ntemp) :: K coefficients

    """

    if not filename.endswith('.kta'):
        filename += '.kta'

    # Dimensions
    ng = g_ord.size
    npress = presslevels.size
    ntemp = templevels.size

    # -------------------------
    # Compute IREC0 (in float32 words)
    # -------------------------
    irec0 = (10 + 2*ng + 3 + npress + ntemp)

    if delv <= 0.0:
        irec0 += nwave

    with open(filename, 'wb') as f:

        # Header
        np.int32(irec0).tofile(f)
        np.int32(nwave).tofile(f)
        np.float32(vmin).tofile(f)
        np.float32(delv).tofile(f)
        np.float32(fwhm).tofile(f)
        np.int32(npress).tofile(f)
        np.int32(ntemp).tofile(f)
        np.int32(ng).tofile(f)
        np.int32(gasID).tofile(f)
        np.int32(isoID).tofile(f)

        # g-ordinates
        np.asarray(g_ord, dtype='float32').tofile(f)
        np.asarray(del_g, dtype='float32').tofile(f)

        # Dummy padding (Nemesis legacy)
        np.float32(0.0).tofile(f)
        np.float32(0.0).tofile(f)

        # Pressure & temperature
        np.asarray(presslevels, dtype='float32').tofile(f)
        np.asarray(templevels, dtype='float32').tofile(f)

        # Non-uniform spectral grid
        if delv <= 0.0:
            np.asarray(wave, dtype='float32').tofile(f)

        # -------------------------
        # k-coefficients
        # Order MUST match read_ktable():
        # wave → press → temp → g
        # -------------------------
        for iw in range(nwave):
            for ip in range(npress):
                for it in range(ntemp):
                    np.asarray(
                        k_g[iw, :, ip, it],
                        dtype='float32'
                    ).tofile(f)


######################################################################################################
    
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

            ig = ig +1
            if ig < ng:
                sum1 = (1.0-frac)*weight[iloop]
                k_g[ig] = (1.0-frac)*cont_weight[iloop]

    if ig == ng-1:
        k_g[ig] = k_g[ig]/sum1

    return k_g

######################################################################################################

@njit
def interpolate_k_values(npoints, NGAS, NWAVEC, precomputed_indices, precomputed_weights, kgood, del_g, kret):
    for ipoint in range(npoints):
        for igas in range(NGAS):
            for iwave in range(NWAVEC):
                iw_low, iw_high = precomputed_indices[iwave]
                w = precomputed_weights[iwave]

                # Interpolate k-values across pressure, temperature, and wavenumber
                k_interpolated_1 = kgood[iw_low, :, ipoint, igas]
                k_interpolated_2 = kgood[iw_high, :, ipoint, igas]

                k_interp = np.concatenate((k_interpolated_1, k_interpolated_2))
                weight = np.concatenate((del_g * (1 - w), del_g * w))

                if 0 < w < 1:
                    kret[iwave, :, ipoint, igas] = rank(weight, k_interp, del_g)
                elif w == 0:
                    kret[iwave, :, ipoint, igas] = k_interpolated_1
                else:  # w == 1
                    kret[iwave, :, ipoint, igas] = k_interpolated_2

    return kret

######################################################################################################

def calc_lbltable(outname,                       #Name of the output .lta file
                  gasID,isoID,                   #Gas information
                  npress,p0,pn,                  #Pressure grid
                  ntemp,t0,tn,                   #Temperature grid
                  ispace,nwave,wavemin,delwave,  #Wavenumber grid
                  iproc,                         #Lineshape
                  vrel,                          #Wavenumber window 
                  self_frac,                     #Self-broadening fraction
                  database,                      #Database
                  add_pressure_shift=True,       #Flag to include pressure shift in the waveumbers
):
    """
    Calculate a line-by-line look-up table for a given gas
    at specified pressure and temperature levels

    Input parameters
    -----------------
    @param gasID: int
        Nemesis gas identifier
    @param isoID: int
        Nemesis isotopologue identifier
    @param npress: int
        Number of pressure levels
    @param p0: float
        Minimum pressure level (atm)
    @param pn: float
        Maximum pressure level (atm)
    @param ntemp: int
        Number of temperature levels
    @param t0: float
        Minimum temperature level (K)
    @param tn: float
        Maximum temperature level (K)
    @param ispace: int
        Spectral unit (0 - wavenumbers (cm-1); 1 - wavelength (um))
    @param nwave: int
        Number of wavenumber/wavelength points
    @param wavemin: float
        Minimum wavenumber/wavelength (cm-1/um)
    @param delwave: float
        Wavenumber/wavelength step (cm-1/um)
    @param iproc: int
        Lineshape processing mode (see Nemesis manual)
    @param vrel: float
        Wavenumber window for lineshape calculation (cm-1)
    @param self_frac: float
        Self-broadening fraction (0 - complete self-broadening, 1 - complete air broadening)
    @param database: str
        Path to archnemesis spectroscopic database to use ('HITRAN','GEISA', etc.)

    """

    #Initialising spectroscopy class
    Spectroscopy  = ans.Spectroscopy_0()
    Spectroscopy.NGAS = 1
    Spectroscopy.ID = [gasID]
    Spectroscopy.ISO = [isoID]
    Spectroscopy.LOCATION = [database]

    #Calculating the pressure grid
    presslevels = np.linspace( np.log(p0) , np.log(pn), npress )
    Spectroscopy.NP = npress
    Spectroscopy.PRESS = np.exp(presslevels)

    #Calculating the temperature grid
    templevels = np.linspace( t0 , tn, ntemp )
    Spectroscopy.NT = ntemp
    Spectroscopy.TEMP = templevels

    #Calculating the spectral grid
    wavemax = wavemin + delwave * (nwave - 1)
    wave = np.linspace( wavemin , wavemax , nwave )
    Spectroscopy.NWAVE = nwave
    Spectroscopy.WAVE = wave

    #Other parameters
    Spectroscopy.IPROC = np.array([iproc],dtype="int32")
    Spectroscopy.VREL = vrel

    Spectroscopy.assess()

    k = np.zeros((Spectroscopy.NWAVE, Spectroscopy.NP, Spectroscopy.NT, Spectroscopy.NGAS))
    for i in range(Spectroscopy.NP):
        pressx = np.ones(Spectroscopy.NT) * Spectroscopy.PRESS[i]
        k[:,i,:,0] = Spectroscopy.calc_klbl_online(Spectroscopy.NT,pressx,Spectroscopy.TEMP,self_frac=self_frac,add_pressure_shift=add_pressure_shift)[:,:,0]

    #Writing the look-up table
    write_lbltable(outname,npress,ntemp,gasID,isoID,Spectroscopy.PRESS,Spectroscopy.TEMP,nwave,wavemin,delwave,k*1.0e-20,DOUBLE=False)


def calc_ktable(outname,                       #Name of the output .lta file
                gasID,isoID,                   #Gas information
                npress,p0,pn,                  #Pressure grid
                ntemp,t0,tn,                   #Temperature grid
                ispace,nwave,wavemin,delwave,  #Wavenumber grid
                ng,                            #Number of g-ordinates
                iproc,                         #Lineshape
                vrel,                          #Wavenumber window 
                self_frac,                     #Self-broadening fraction
                database,                      #Database
                Measurement=None,              #Measurement class to read the ILS (NFIL,VFIL,AFIL parameters) 
                add_pressure_shift=True,       #Flag to include pressure shift in the waveumbers
):


    #Initialising spectroscopy class
    Spectroscopy  = ans.Spectroscopy_0()
    Spectroscopy.NGAS = 1
    Spectroscopy.ID = [gasID]
    Spectroscopy.ISO = [isoID]
    Spectroscopy.LOCATION = [database]

    #Calculating the pressure grid
    presslevels = np.linspace( np.log(p0) , np.log(pn), npress )
    Spectroscopy.NP = npress
    Spectroscopy.PRESS = np.exp(presslevels)

    #Calculating the temperature grid
    templevels = np.linspace( t0 , tn, ntemp )
    Spectroscopy.NT = ntemp
    Spectroscopy.TEMP = templevels

    #Calculating the spectral grid
    wavemax = wavemin + delwave * (nwave - 1)
    wave = np.linspace( wavemin , wavemax , nwave )
    Spectroscopy.NWAVE = nwave
    Spectroscopy.WAVE = wave

    #Other parameters
    Spectroscopy.IPROC = np.array([iproc],dtype="int32")
    Spectroscopy.VREL = vrel

    # Gauss–Legendre points and weights for g-ordinates
    Spectroscopy.NG = ng
    x, w = np.polynomial.legendre.leggauss(ng)
    Spectroscopy.G_ORD = 0.5 * (x + 1.0)
    Spectroscopy.DELG = 0.5 * w

    Spectroscopy.assess()


    #Starting calculations
    ############################################################################

    #Initialising the LineData class
    linedata = ans.LineData_1(
        gasID, #ID of the gas
        isoID, #Isotope ID of the gas
        LINE_DATABASE=database # Different database location
    )

    #Defining the lineshape function to use in the line-by-line calculations
    if Spectroscopy.IPROC[0]==SpectroscopicLineProfile.VOIGT:
        lineshape = ans.Data.lineshapes.voigt
    elif Spectroscopy.IPROC[0]==SpectroscopicLineProfile.DOPPLER:
        lineshape = ans.Data.lineshapes.doppler
    elif Spectroscopy.IPROC[0]==SpectroscopicLineProfile.LORENTZ:
        lineshape = ans.Data.lineshapes.lorentz
    else:
        raise ValueError('error in calc_klbl_online :: selected IPROC has not been implemented yet')

    #Selecting the spectral range needed for the calculations
    if Measurement is not None:
        assert np.allclose(Measurement.VCONV[0:Measurement.NCONV[0],0], Spectroscopy.WAVE, rtol=0.5)
        vbinmin = Spectroscopy.WAVE.min() - (Measurement.VFIL[0:Measurement.NFIL[0],0]-Measurement.VCONV[0,0]).max()
        vbinmax = Spectroscopy.WAVE.max() + (Measurement.VFIL[0:Measurement.NFIL[-1],-1]-Measurement.VCONV[-1,0]).max()
    else:
        vbinmin = Spectroscopy.WAVE.min() - delwave / 2.
        vbinmax = Spectroscopy.WAVE.max() + delwave / 2.

    _lgr.info(f'Selecting lines in the spectral range {vbinmin:.2f} - {vbinmax:.2f} for the calculations')

    linedata.fetch_linedata(
        vmin = vbinmin, 
        vmax = vbinmax, 
        wave_unit = ispace
    )

    # Download partition function tables for the gas isotopes
    linedata.fetch_partition_function()

    #Looping through the pressure and temperature levels to calculate the k-coefficients
    k_coefficients = np.zeros((Spectroscopy.NWAVE, Spectroscopy.NG, Spectroscopy.NP, Spectroscopy.NT))
    for ip in range(Spectroscopy.NP):

        for it in range(Spectroscopy.NT):

            pressx = Spectroscopy.PRESS[ip]
            tempx = Spectroscopy.TEMP[it]

            _lgr.info(f'Calculating k-coefficients at p = {pressx} atm and t = {tempx} K')

            if ispace == WaveUnit.Wavenumber_cm:
                nus = linedata.line_data.NU
            elif ispace == WaveUnit.Wavelength_um:
                nus = 1e4 / linedata.line_data.NU

            #Estimating the spacing in the cross section calculations
            alpha_d = linedata.calculate_doppler_width(tempx)
            gamma_l = linedata.calculate_lorentz_width(pressx,tempx,amb_frac=1.-self_frac)

            delv_calc_doppler = np.min(alpha_d * nus / linedata.line_data.NU)
            delv_calc_lorentz = np.min(gamma_l * nus / linedata.line_data.NU) 
            delv_calc = np.min([delv_calc_doppler]) / 5.

            ncalc = int((vbinmax-vbinmin)/delv_calc)
            wavecalc = np.linspace(vbinmin,vbinmax,ncalc)

            _lgr.info(f'Number of absorption lines in the range = {len(linedata.line_data.NU)}')
            _lgr.info(f'Number of spectral points in the range = {ncalc} - delta_wave = {delv_calc}')    
            _lgr.info(f'Calculating line-by-line absorption coefficients')        

            #Calculating the absorption coefficients at the line-by-line level for the given pressure and temperature
            kabs = linedata.calculate_monochromatic_absorption(
                        waves=wavecalc,     # spectral points
                        temp=tempx,         # kelvin
                        press=pressx,       # Atmospheres
                        amb_frac=1.-self_frac,  # fraction of broadening due to ambient gas
                        wave_unit=ispace,   # unit of `waves` argument
                        lineshape_fn=lineshape, # lineshape function to use
                        line_calculation_wavenumber_window=vrel, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
                        add_pressure_shift=add_pressure_shift, # whether to include pressure shift in the line positions
            )


            #Calculating fo each wavelength bin
            _lgr.info(f'Calculating the cumulative distributions and k-coefficients for each bin')
            for iwave in range(Spectroscopy.NWAVE):

                #Calculating the bin size
                if Measurement is not None:
                    vbinminx = Spectroscopy.WAVE[iwave] - (Measurement.VFIL[0:Measurement.NFIL[iwave],iwave]-Measurement.VCONV[iwave,0]).max()
                    vbinmaxx = Spectroscopy.WAVE[iwave] + (Measurement.VFIL[0:Measurement.NFIL[iwave],iwave]-Measurement.VCONV[iwave,0]).max()
                else:
                    vbinminx = Spectroscopy.WAVE[iwave] - delwave / 2.
                    vbinmaxx = Spectroscopy.WAVE[iwave] + delwave / 2.

                #Sorting the absorption coefficients in the bin
                mask = (wavecalc >= vbinminx) & (wavecalc <= vbinmaxx)
                idx = np.argsort(kabs[mask])
                wavesel = wavecalc[mask]
                k_sorted = kabs[mask][idx]

                #Considering the instrument lineshape if needed
                if Measurement is not None:
                    delta_wave = wavesel[idx] - Spectroscopy.WAVE[iwave]
                    ils_sorted = np.interp(delta_wave,Measurement.VFIL[0:Measurement.NFIL[iwave],iwave]-Measurement.VCONV[iwave,0],Measurement.AFIL[0:Measurement.NFIL[iwave],iwave])
                else:
                    ils_sorted = np.ones_like(wavesel)

                #Calculating the cumulative distribution function of the absorption coefficients in the bin
                delvarray = np.zeros_like(k_sorted) + (wavecalc[1]-wavecalc[0])
                g_sorted = np.cumsum(ils_sorted * delvarray) / np.sum(ils_sorted * delvarray)

                #Interpolate to get the k-coefficients at the g-ordinates
                k_coefficients[iwave,:,ip,it] = np.interp(Spectroscopy.G_ORD, g_sorted, k_sorted)


    Spectroscopy.K = k_coefficients[:,:,:,:,np.newaxis]

    if Measurement is not None:
        fwhm = Measurement.FWHM
    else:
        fwhm = 0.0

    #Writing the look-up table
    write_ktable(outname,gasID,isoID,Spectroscopy.G_ORD,Spectroscopy.DELG,Spectroscopy.PRESS,Spectroscopy.TEMP,Spectroscopy.NWAVE,Spectroscopy.WAVE.min(),delwave,fwhm,k_coefficients)

