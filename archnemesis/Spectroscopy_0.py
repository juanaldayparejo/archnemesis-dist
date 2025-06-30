from __future__ import annotations #  for 3.9 compatability

from archnemesis import *
from archnemesis.enums import (
    WaveUnit,
    SpectraUnit,
    SpectralCalculationMode,
    Gas
)
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from numba import jit, njit

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

State Vector Class.
"""

class Spectroscopy_0:

    def __init__(self, RUNNAME: str = '', ILBL: SpectralCalculationMode = SpectralCalculationMode.LINE_BY_LINE_TABLES, NGAS: int = 2, ONLINE: bool = False):

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
        self.RUNNAME = RUNNAME
        #self.ILBL = SpectralCalculationMode(ILBL) if not isinstance(ILBL, SpectralCalculationMode) else ILBL
        self.NGAS = NGAS
        self.ONLINE = ONLINE

        # Attributes with proper typing
        #self.ISPACE: Optional[WaveUnit] = None
        self.ID: Optional[np.ndarray] = None  # Array of Gas enum values (NGAS)
        self.ISO = None       #(NGAS)
        self.LOCATION = None  #(NGAS)
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
        
        # set property values
        self.ILBL = ILBL
        self.ISPACE = WaveUnit.Wavenumber_cm  # Default value
    
    @property
    def ILBL(self) -> SpectralCalculationMode:
        return self._ilbl
    
    @ILBL.setter
    def ILBL(self, value):
        self._ilbl = SpectralCalculationMode(value)
    
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
            dset = grp.create_dataset('NGAS',data=self.NGAS)
            dset.attrs['title'] = "Number of radiatively active gases in atmosphere"

            dset = grp.create_dataset('ILBL',data=int(self.ILBL))
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
                    dset = grp.create_dataset('LOCATION',data=self.LOCATION,dtype=dt)
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
                f.close()
                raise ValueError('error :: Spectroscopy is not defined in HDF5 file')
            else:
                self.NGAS = np.int32(f.get(name+'/NGAS'))
                self.ILBL = SpectralCalculationMode(np.int32(f.get(name+'/ILBL')))

                if self.NGAS>0:
                    LOCATION1 = f.get(name+'/LOCATION')
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
            
            f = h5py.File(filename+'.h5','w')
            
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
            
            f.close()

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
                    
                    f = h5py.File(self.LOCATION[igas],'r')
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
                    
                    f.close()

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
                    
                    f = h5py.File(self.LOCATION[igas],'r')
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
                    
                    f.close()
            
            
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
            press0 = self.PRESS[ip]

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
            temp0 = self.TEMP[it]

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

        from scipy import interpolate

        #Interpolating the k-coefficients to the correct pressure and temperature
        #############################################################################

        #K (NWAVE,NG,NPOINTS,NGAS)
        TEMP = self.TEMP
        PRESS = self.PRESS
        NP = self.NP
        NT = self.NT
        
        kgood = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
        dkgooddT = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS])
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
        NG = self.NG
        del_g = self.DELG
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
                                             precomputed_weights, kgood, del_g, kret)
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
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'r')
    else:
        f = open(filename+'.lta','r')

    irec0 = int(np.fromfile(f,dtype='int32',count=1)[0])
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
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'kta':
        f = open(filename,'r')
    else:
        f = open(filename+'.kta','r')

    irec0 = int(np.fromfile(f,dtype='int32',count=1)[0])
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

    dummy = np.fromfile(f,dtype='float32',count=1)
    dummy = np.fromfile(f,dtype='float32',count=1)

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
    strlen = len(filename)
    if filename[strlen-2:strlen] == 'h5':
        f = h5py.File(filename,'r')
    else:
        f = h5py.File(filename+'.h5','r')

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
    
    f.close()

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
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'rb')
    else:
        f = open(filename+'.lta','rb')

    nbytes_int32 = 4
    nbytes_float32 = 4

    #Reading header
    irec0 = np.fromfile(f,dtype='int32',count=1)[0]
    nwavelta = np.fromfile(f,dtype='int32',count=1)[0]
    vmin = np.fromfile(f,dtype='float32',count=1)[0]
    delv = np.fromfile(f,dtype='float32',count=1)[0]
    npress = np.fromfile(f,dtype='int32',count=1)[0]
    ntemp = np.fromfile(f,dtype='int32',count=1)[0]
    gasID = np.fromfile(f,dtype='int32',count=1)[0]
    isoID = np.fromfile(f,dtype='int32',count=1)[0]

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

    ins = np.where( (wavelta>=wavemin) & (wavelta<=wavemax) )[0]
    nwave = len(ins)
    wave = np.zeros(nwave)
    wave[:] = wavelta[ins]

    #Reading the absorption coefficients
    #######################################
    k = np.zeros([nwave,npress,abs(ntemp)])

    #Jumping until we get to the minimum wavenumber
    njump = npress*abs(ntemp)*(ins[0])
    ioff = njump*nbytes_float32 + (irec0-1)*nbytes_float32
    f.seek(ioff,0)

    #Reading the coefficients we require
    k_out = np.fromfile(f,dtype='float32',count=abs(ntemp)*npress*nwave)
    il = 0
    for ik in range(nwave):
        for i in range(npress):
            k[ik,i,:] = k_out[il:il+abs(ntemp)]
            il = il + abs(ntemp)

    f.close()
    
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
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'kta':
        f = open(filename,'rb')
    else:
        f = open(filename+'.kta','rb')

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

    dummy = np.fromfile(f,dtype='float32',count=1)[0]
    dummy = np.fromfile(f,dtype='float32',count=1)[0]

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

    f.close()

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
    strlen = len(filename)
    if filename[strlen-3:strlen] == 'lta':
        f = open(filename,'w+b')
    else:
        f = open(filename+'.lta','w+b')

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

    f.close()
    
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

