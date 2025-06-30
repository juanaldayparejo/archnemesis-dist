from __future__ import annotations #  for 3.9 compatability

from archnemesis import *
from archnemesis.enums import WaveUnit
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

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

Stellar Class.
"""

class Stellar_0:

    def __init__(self, SOLEXIST=True, DIST=None, RADIUS=None, ISPACE=None, NWAVE=None):

        """
        Attributes
        ----------
        @attribute SOLEXIST: log,
            Flag indicating whether sunlight needs to be included (SOLEXIST=True) or not (SOLEXIST=False)
        @attribute DIST: float,
            Distance between star and planet (AU) 
        @attribute RADIUS: float,
            Radius of the star (km)       
        @attribute ISPACE: int,
            Spectral units in which the solar spectrum is defined (0) Wavenumber (1) Wavelength              
        @attribute NWAVE: int,
            Number of spectral points in which the stellar spectrum is defined
        @attribute WAVE: 1D array
            Wavelength array at which the stellar file is defined
        @attribute SOLSPEC: 1D array
            Stellar luminosity spectrum (W/(cm-1) or W/um)
        @attribute SOLFLUX: 1D array
            Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)
        @attribute STELLARDATA: str
            String indicating where the STELLAR data files are stored

        Methods
        -------
        Stellar_0.assess
        Stellar_0.edit_WAVE
        Stellar_0.edit_SOLSPEC
        Stellar_0.read_hdf5
        Stellar_0.write_hdf5
        Stellar_0.read_sol
        Stellar_0.write_sol
        Stellar_0.calc_solar_flux
        Stellar_0.calc_solar_power
        Stellar_0.write_solar_file
        """

        from archnemesis.Data.path_data import archnemesis_path

        #Input parameters
        self.SOLEXIST = SOLEXIST
        self.DIST = DIST
        self.RADIUS = RADIUS
        #self.ISPACE = WaveUnit(ISPACE) if ISPACE is not None and not isinstance(ISPACE, WaveUnit) else ISPACE
        self.NWAVE = NWAVE

        # Input the following profiles using the edit_ methods.
        self.WAVE = None # np.zeros(NWAVE)
        self.SOLSPEC = None # np.zeros(NWAVE)
        self.SOLFLUX = None #np.zeros(NWAVE)

        self.STELLARDATA = archnemesis_path()+'archnemesis/Data/stellar/'
        
        # private attributes
        self._ispace = None
        
        # set property values
        self.ISPACE = ISPACE if ISPACE is not None else WaveUnit.Wavenumber_cm
    
    
    @property
    def ISPACE(self) -> WaveUnit:
        return self._ispace
    
    @ISPACE.setter
    def ISPACE(self, value):
        self._ispace = WaveUnit(value)


    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        if self.SOLEXIST is True:

            #Checking some common parameters to all cases
            assert isinstance(self.ISPACE, (int, np.integer, WaveUnit)), \
                'ISPACE must be int or WaveUnit'
            assert self.ISPACE in WaveUnit, \
                f'ISPACE must be one of {tuple(WaveUnit)}'

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.DIST), np.float64) == True , \
                'DIST must be float'

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.RADIUS), np.float64) == True , \
                'RADIUS must be float'

            #Checking some common parameters to all cases
            assert np.issubdtype(type(self.NWAVE), np.integer) == True , \
                'NWAVE must be int'
            assert self.NWAVE >= 0 , \
                'NWAVE must be >=0'
             
            assert len(self.WAVE) == self.NWAVE , \
                'WAVE must have size (NWAVE)'
            
            assert len(self.SOLSPEC) == self.NWAVE , \
                'SOLSPEC must have size (NWAVE)'
            

    def edit_WAVE(self, WAVE):
        """
        Edit the wavelength array 
        @param WAVE: 1D array
            Array defining the wavelengths at which the solar spectrum is defined
        """
        WAVE_array = np.array(WAVE)
        assert len(WAVE_array) == self.NWAVE, 'WAVE should have NWAVE elements'
        self.WAVE = WAVE_array

    def edit_SOLSPEC(self, SOLSPEC):
        """
        Edit the solar spectrum 
        @param SOLSPEC: 1D array
            Array defining the solar spectrum
        """
        SOLSPEC_array = np.array(SOLSPEC)
        assert len(SOLSPEC_array) == self.NWAVE, 'SOLSPEC should have NWAVE elements'
        self.SOLSPEC = SOLSPEC_array

    def write_hdf5(self,runname,solfile=None):
        """
        Write the information about the solar spectrum in the HDF5 file

        If the optional input solfile is defined, then the information is read from the
        specified file (assumed to be stored in the Data/stellar directory).

        If solfile is not defined, then the information about the solar spectrum is assumed
        to be defined in the class
        """

        from archnemesis.Files import file_lines

        if solfile is not None:

            #Reading the solar spectrum file

            nlines = file_lines(self.STELLARDATA+solfile)

            #Reading buffer
            ibuff = 0
            with open(self.STELLARDATA+solfile,'r') as fsol:
                for curline in fsol:
                    if curline.startswith("#"):
                        ibuff = ibuff + 1
                    else:
                        break

            nvsol = nlines - ibuff - 2
            
            #Reading file
            fsol = open(self.STELLARDATA+solfile,'r')
            for i in range(ibuff):
                s = fsol.readline().split()
        
            s = fsol.readline().split()
            ispace = WaveUnit(int(s[0]))
            s = fsol.readline().split()
            solrad = float(s[0])
            vsol = np.zeros(nvsol)
            rad = np.zeros(nvsol)
            for i in range(nvsol):
                s = fsol.readline().split()
                vsol[i] = float(s[0])
                rad[i] = float(s[1])
        
            fsol.close()


            self.RADIUS = solrad
            self.ISPACE = ispace
            self.NWAVE = nvsol
            self.edit_WAVE(vsol)
            self.edit_SOLSPEC(rad)


        self.assess()

        #Writing the information into the HDF5 file
        f = h5py.File(runname+'.h5','a')
        #Checking if Stellar already exists
        if ('/Stellar' in f)==True:
            del f['Stellar']   #Deleting the Stellar information that was previously written in the file

        if self.SOLEXIST is True:

            grp = f.create_group("Stellar")

            #Writing the spectral units
            dset = grp.create_dataset('ISPACE',data=int(self.ISPACE))
            dset.attrs['title'] = "Spectral units"
            if self.ISPACE == WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE == WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um'

            #Writing the Planet-Star distance
            dset = grp.create_dataset('DIST',data=self.DIST)
            dset.attrs['title'] = "Planet-Star distance"
            dset.attrs['units'] = 'Astronomical Units'

            #Writing the Star radius
            dset = grp.create_dataset('RADIUS',data=self.RADIUS)
            dset.attrs['title'] = "Star radius"
            dset.attrs['units'] = 'km'

            #Writing the number of points in stellar spectrum
            dset = grp.create_dataset('NWAVE',data=self.NWAVE)
            dset.attrs['title'] = "Number of spectral points in stellar spectrum"

            #Writing the spectral array
            dset = grp.create_dataset('WAVE',data=self.WAVE)
            dset.attrs['title'] = "Spectral array"
            if self.ISPACE == WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'Wavenumber / cm-1'
            elif self.ISPACE == WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'Wavelength / um' 

            #Writing the solar spectrum
            dset = grp.create_dataset('SOLSPEC',data=self.SOLSPEC)
            dset.attrs['title'] = "Stellar power spectrum"
            if self.ISPACE == WaveUnit.Wavenumber_cm:
                dset.attrs['units'] = 'W (cm-1)-1'
            elif self.ISPACE == WaveUnit.Wavelength_um:
                dset.attrs['units'] = 'W um-1'     

        f.close()

    def read_hdf5(self,runname):
        """
        Read the Stellar properties from an HDF5 file
        """

        f = h5py.File(runname+'.h5','r')

        #Checking if Surface exists
        e = "/Stellar" in f
        if e is False:
            self.SOLEXIST = False
        else:
            self.SOLEXIST = True
            self.ISPACE = h5py_helper.retrieve_data(f, 'Stellar/ISPACE', lambda x:  WaveUnit(np.int32(x)))
            self.DIST = h5py_helper.retrieve_data(f, 'Stellar/DIST', np.float64)
            self.RADIUS = h5py_helper.retrieve_data(f, 'Stellar/RADIUS', np.float64)
            self.NWAVE = h5py_helper.retrieve_data(f, 'Stellar/NWAVE', np.int32)
            self.WAVE = h5py_helper.retrieve_data(f, 'Stellar/WAVE', np.array)
            self.SOLSPEC = h5py_helper.retrieve_data(f, 'Stellar/SOLSPEC', np.array)

        f.close()

    def read_sol(self, runname, MakePlot=False):
        """
        Read the solar spectrum from the .sol file. There are two options for this file:

            - The only line in the file is the name of another file including the solar power spectrum,
              assumed to be stored in the Data/stellar/ directory

            - The first line is equal to -1. Then the stellar spectrum is read from the solar file.

        @param runname: str
            Name of the NEMESIS run
        """

        from archnemesis.Files import file_lines

        #Opening file
        f = open(runname+'.sol','r')
        s = f.readline().split()
        solname = s[0]
        f.close()

        if solname=='-1':
            #Information about stellar spectrum is stored in this same file
            nlines = file_lines(runname+'.sol')
            nvsol = nlines - 3


            f = open(runname+'.sol','r')
            s = f.readline().split()
            solname = s[0]

            s = f.readline().split()
            ispace = WaveUnit(int(s[0]))
            s = f.readline().split()
            solrad = float(s[0])
            vsol = np.zeros(nvsol)
            rad = np.zeros(nvsol)
            for i in range(nvsol):
                s = f.readline().split()
                vsol[i] = float(s[0])
                rad[i] = float(s[1])
            f.close()

        else:

            nlines = file_lines(self.STELLARDATA+solname)

            #Reading buffer
            ibuff = 0
            with open(self.STELLARDATA+solname,'r') as fsol:
                for curline in fsol:
                    if curline.startswith("#"):
                        ibuff = ibuff + 1
                    else:
                        break

            nvsol = nlines - ibuff - 2
            
            #Reading file
            fsol = open(self.STELLARDATA+solname,'r')
            for i in range(ibuff):
                s = fsol.readline().split()
        
            s = fsol.readline().split()
            ispace = WaveUnit(int(s[0]))
            s = fsol.readline().split()
            solrad = float(s[0])
            vsol = np.zeros(nvsol)
            rad = np.zeros(nvsol)
            for i in range(nvsol):
                s = fsol.readline().split()
                vsol[i] = float(s[0])
                rad[i] = float(s[1])
        
            fsol.close()

        self.RADIUS = solrad
        self.ISPACE = ispace
        self.NWAVE = nvsol
        self.edit_WAVE(vsol)
        self.edit_SOLSPEC(rad)

        if MakePlot is True:
            fig,ax1=plt.subplots(1,1,figsize=(8,3))
            ax1.plot(vsol,rad)
            #ax1.set_yscale('log')
            plt.tight_layout()
            plt.show()

    def write_sol(self,runname):
        """
        Write the solar power into a .sol file with the format required by NEMESISpy, in the case
        that the solar power spectrum is stored directy in the .sol file
        """

        f = open(runname+'.sol','w')

        #Defining errors while writing file
        if self.ISPACE is None:
            raise ValueError('error :: ISPACE must be defined in Stellar class to write Stellar power to file')

        if self.RADIUS is None:
            raise ValueError('error :: RADIUS must be defined in Stellar class to write Stellar power to file')

        if self.NWAVE is None:
            raise ValueError('error :: NWAVE must be defined in Stellar class to write Stellar power to file')

        if self.WAVE is None:
            raise ValueError('error :: WAVE must be defined in Stellar class to write Stellar power to file')

        if self.SOLSPEC is None:
            raise ValueError('error :: SOLSPEC Must be defined in Stellar class to write Stellar power to file')


        header = '-1'
        f.write(header+' \n')
        f.write('\t %i \n' % (int(self.ISPACE)))
        f.write('\t %7.3e \n' % (self.RADIUS))
        for i in range(self.NWAVE):
            f.write('\t %7.6f \t %7.5e \n' % (self.WAVE[i],self.SOLSPEC[i]))
        f.close()


    def calc_solar_flux(self):
        """
        Calculate the stellar flux at the planet's distance (W cm-2 (cm-1)-1 or W cm-2 um-1)
        """

        AU = 1.49598e11
        area = 4.*np.pi*(self.DIST * AU * 100. )**2.
        self.SOLFLUX = self.SOLSPEC / area   #W cm-2 (cm-1)-1 or W cm-2 um-1


    def calc_solar_power(self):
        """
        Calculate the stellar power based on the solar flux measured at a given distance
        """

        AU = 1.49598e11
        area = 4.*np.pi*(self.DIST * AU * 100. )**2.
        self.SOLSPEC = self.SOLFLUX * area   #W (cm-1)-1 or W um-1


    def write_solar_file(self,filename,header=None):
        """
        Write the solar power into a file with the format required by NEMESIS
        """

        f = open(filename,'w')

        #Defining errors while writing file
        if self.ISPACE is None:
            raise ValueError('error :: ISPACE must be defined in Stellar class to write Stellar power to file')

        if self.RADIUS is None:
            raise ValueError('error :: RADIUS must be defined in Stellar class to write Stellar power to file')

        if self.NWAVE is None:
            raise ValueError('error :: NWAVE must be defined in Stellar class to write Stellar power to file')

        if self.WAVE is None:
            raise ValueError('error :: WAVE must be defined in Stellar class to write Stellar power to file')

        if self.SOLSPEC is None:
            raise ValueError('error :: SOLSPEC Must be defined in Stellar class to write Stellar power to file')


        if header is None:
            if self.ISPACE == WaveUnit.Wavenumber_cm:
                header = '# Stellar power in W (cm-1)-1'
            elif self.ISPACE == WaveUnit.Wavelength_um:
                header = '# Stellar power in W um-1' 
        else:
            if header[0]!='#':
                header = '#'+header
        
        f.write(header+' \n')
        f.write('\t %i \n' % (int(self.ISPACE)))
        f.write('\t %7.3e \n' % (self.RADIUS))
        for i in range(self.NWAVE):
            f.write('\t %7.6f \t %7.5e \n' % (self.WAVE[i],self.SOLSPEC[i]))

        f.close()
        
###############################################################################################