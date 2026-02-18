#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# Emissions_0.py - Subroutines to store the information about the spectroscopy for atmospheric emissions
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
    EmissionType,
)
import numpy as np
import scipy
import os
import os.path
import archnemesis as ans
from numba import jit, njit
from archnemesis.database.datatypes.wave_range import WaveRange

from archnemesis.helpers import h5py_helper, path_redirect
import matplotlib.pyplot as plt
from typing import Optional


import logging
_lgr = logging.getLogger(__name__)

###############################################################################################

"""
Created on Feb 16 10:00:00 2026

@author: juanalday

Emissions Class.
"""

class Emissions_0:

    def __init__(
            self,
            NEM: int = 0, 
        ):

        """
        Inputs
        ------
        @param NEM: int,
            Number of atmospheric emissions included
        @param EMTYPE: 1D array (NEM),
            Type of atmospheric emission for each gas
                - FLUORESCENCE (0) - Fluorescence (produced by solar pumping)
                - CHEMICAL (1) - Prompt (produced by chemical reactions)
                - PHOTOLYSIS (2) - Prompt (produced by photolysis reactions)
        @param NGAS: 1D array (NEM),
            Number of gases included in each atmospheric emission type
            For fluorescence this will be 1, but for chemical emissions this might involve two or more species.
        @param ID: 2D array (NGAS,NEM),
            Gas ID for each active gas (using Gas enum)
        @param ISO: 2D array (NGAS,NEM),
            Isotope ID for each gas, default 0 for all isotopes
        @param ISPACE: int,
            Flag indicating the units of the spectral coordinate:
            (0) Wavenumber (cm-1) 
            (1) Wavelength (um)
        @param LOCATION: 1D array,
            List of strings indicating where the data for each of the gases is stored
        @param NWAVE: int,
            Number of spectral points included in the tables
        @param WAVE: 1D array,
            Spectral points at which the tables are computed
        @param NT: int,
            Number of temperature levels at which the K-tables or LBL-tables were computed
        @param TEMP: 1D array
            Temperature levels at which the K-tables or LBL-tables were computed (K)
        @param K: 3D array (NWAVE,NT,NEM)
            Emission rates 
                - FLUORESCENCE: Units must be [photons molecule-1 s-1 (cm-1)-1] or [photons molecule-1 s-1 (um)-1] 
                - CHEMICAL: Units must be [photons molecule-1 (cm-1)-1] or [photons molecule-1 (um)-1]
        @param RATE_COEFF: 2D array (NT,NEM)
            Reaction rate producing the atmospheric emission. This will only be used for chemical emissions.
            Depending on the number of species involved, the units will be 
             - For 1-species emission (e.g., H2O + hv ---> OH that emits), units will be s-1, so that RRATE x n_H2O = [cm-3 s-1]
             - For 2-species emission (e.h., H + O3 ---> OH that emits), units will be cm3 s-1, so that RRATE x n_H x n_O3 = [cm-3 s-1]
        @param DIST_REF: 1D array (NEM)
            Stellar distance at which the emission coefficients are tabulated (AU)
            (Required if EMTYPE=0 or EMTYPE=2)
            
        Methods
        -------

        """


        # Input parameters with validation
        self.NEM = NEM
        
        # Attributes with proper typing
        self.NGAS: None | np.ndarray = None 
        self.ID: None | np.ndarray = None  # Array of Gas enum values (NGAS,NEM)
        self.ISO = None       #(NGAS)
        self.EMTYPE: None | np.ndarray = None  # Array of EmissionType enum values (NEM)
        self._locations = path_redirect.PathRedirectList() #(NEM)
        self.NWAVE = None     
        self.WAVE = None      #(NWAVE)
        self.NT = None
        self.TEMP = None      #(NT)
        self.DIST_REF = None
        
        self.K = None #(NWAVE,NT,NEM)
        self.RATE_COEFF = None #(NT,NEM)
        
        # private attributes
        self._ispace = None
        self._locations_initialised = False
        
        # set property values
        self.ISPACE = WaveUnit.Wavelength_um  # Default value

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
    def EMTYPE(self) -> list[EmissionType]:
        return self._iemi

    @EMTYPE.setter
    def EMTYPE(self, value):
        if value is None:
            self._iemi = None
        else:
            self._iemi = [EmissionType(v) for v in value]

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
        if self.ISPACE is not None:
            assert isinstance(self.ISPACE, WaveUnit), \
                'ISPACE must be WaveUnit enum'
            assert self.ISPACE in (WaveUnit.Wavenumber_cm, WaveUnit.Wavelength_um), \
                'ISPACE must be Wavenumber_cm or Wavelength_um'

        if self.NEM>0:

            assert len(self.EMTYPE) == self.NEM, \
                'EMTYPE must have size (NEM)'

            assert len(self.LOCATION) == self.NEM , \
                'LOCATION must have size (NEM)'

            assert len(self.DIST_REF) == self.NEM , \
                'DIST_REF must have size (NEM)'

            for i in range(self.NEM):

                if( (self.EMTYPE[i]==0) & (self.NGAS[i]>1) ):
                    raise ValueError("If EMTYPE=2 then NGAS must be 1 (fluorescence of one species)")

                if( (self.EMTYPE[i]==2) & (self.NGAS[i]>1) ):
                    raise ValueError("If EMTYPE=2 then NGAS must be 1 (photolysis of one species)")


    def write_hdf5(self,runname):
        """
        Subroutine to write the input HDF5 file
        """   

        import h5py

        #Assessing that everything is correct
        self.assess()

        with h5py.File(runname+'.h5','a') as f:

            #Checking if Emissions already exists
            if ('/Emissions' in f)==True:
                del f['Emissions']   #Deleting the Emissions information that was previously written in the file

            grp = f.create_group("Emissions")

            #Writing the main dimensions
            dset = h5py_helper.store_data(grp, 'NEM', self.NEM)
            dset.attrs['title'] = "Number of atmospheric emissions considered"

            if self.NEM>0:

                dt = h5py.special_dtype(vlen=str)
                dset = h5py_helper.store_data(grp, 'LOCATION', self._locations._raw_paths,dtype=dt) # do not save the redirected paths.
                dset.attrs['title'] = "Location of the pre-tabulated tables"



    ######################################################################################################

    def read_hdf5(self,runname):
        """
        Subroutine to read the input HDF5 file
        """   

        import h5py
        
        with h5py.File(runname+'.h5','r') as f:

            name = '/Emissions'

            #Checking if Spectroscopy exists
            e = name in f
            if e==False:
                
                self.NEM = 0

            else:

                #Reading the number of emissions specified in the file
                self.NEM = h5py_helper.retrieve_data(f, name+'/NEM', np.int32, default=0)

                if self.NEM>0:
                    LOCATION1 = h5py_helper.retrieve_data(f, name+'/LOCATION', default=tuple())
                    if LOCATION1 is None:
                        LOCATION1 = []
                    LOCATION = ['']*self.NEM
                    for igas in range(self.NEM):
                        LOCATION[igas] = LOCATION1[igas].decode('ascii')
                    self.LOCATION = LOCATION
                    
        if self.NEM > 0: #Reading header of the tables
            self.read_header_table_hdf5()


    ######################################################################################################
    def calc_rates_hdf5(self,temps,dist=None,vmin=0.,vmax=1.0e12):
        """
        Calculate the emission rates at different temperatures

        The emission rates here are the product of the line strengths times the rate coefficients, so
        the units of the emission rates are [photons molecule-1 s-1 (um)-1] (or in wavenumber units)

        Optional inputs
        ---------------

        vmin,vmax :: Bounds of the spectral range that wants to be modelled
        dist :: Distance between the planet at the parent star (AU) 
                This is important for fluorescence and emissions initiated by photolysis
                (for the proper scaling of the stellar flux)
        """

        import h5py

        inwave = np.where( (self.WAVE>=vmin) & (self.WAVE<=vmax) )[0]

        nwavex = len(inwave)
        npoints = len(temps)

        emrate = np.zeros((nwavex,npoints,self.NEM))

        for iemi in range(self.NEM):

            #Opening file
            filename = self.LOCATION[iemi]
            filename = filename if filename.endswith('.h5') else (filename+'.h5')

            #Interpolating the parameters to the temperature
            with h5py.File(filename,'r') as f:

                mask = ( (self.WAVE >= vmin) & (self.WAVE <= vmax) )
                
                if self.NT == 1:

                    #There is no information on the temperature
                    k0 = f["K"][:][mask,0]
                    emrate[:,:,iemi] = k0[:,np.newaxis]
                    c0 = f["RATE_COEFF"][:][0]
                    emrate *= c0

                    #Scaling for the planet-star distance
                    if dist is not None:
                        if( (self.EMTYPE[iemi]==0) | (self.EMTYPE[iemi]==2) ):
                            emrate *= (self.DIST_REF[iemi]/dist)**2.

                else:

                    #Interpolating the temperature
                    for ipoint in range(npoints):

                        t_l = temps[ipoint]

                        if t_l < np.min(self.TEMP):
                            t_l = np.min(self.TEMP)
                        if t_l > np.max(self.TEMP):
                            t_l = np.max(self.TEMP)

                        it = np.searchsorted(self.TEMP, t_l) - 1
                        if it < 0:
                            it = 0
                        if it >= self.NT - 1:
                            it = self.NT - 2

                        u = (t_l - self.TEMP[it]) / (self.TEMP[it + 1] - self.TEMP[it])
                        klo = f['K'][:][mask,it]
                        khi = f['K'][:][mask,it+1]

                        k = klo * (1. - u) + khi * u

                        if self.EMTYPE[iemi]==0:
                            c = 1.
                        else:
                            clo = f['RATE_COEFF'][:][it]
                            chi = f['RATE_COEFF'][:][it+1]
                            c = clo * (1. - u) + chi * u

                        emrate[:,ipoint,iemi] = k[:] * c

                        #Scaling for the planet-star distance
                        if dist is not None:
                            if( (self.EMTYPE[iemi]==0) | (self.EMTYPE[iemi]==2) ):
                                emrate[:,ipoint,iemi] *= (self.DIST_REF[iemi]/dist)**2.

        return emrate



    ######################################################################################################
    def read_header_table_hdf5(self):
        """
        Read the header information of a HDF5 look-up table with the emission rates
        """

        import h5py

        mgas = 4
        mtemp = 10
        mwave = 20000000
        
        emtype = np.zeros(self.NEM,dtype="int32")
        ngas = np.zeros(self.NEM,dtype="int32")
        idx = np.zeros((mgas,self.NEM),dtype="int32")
        isox = np.zeros((mgas,self.NEM),dtype="int32")
        ntemp = np.zeros(self.NEM,dtype="int32")
        temp = np.zeros((mtemp,self.NEM))
        nwave = np.zeros(self.NEM,dtype="int32")
        wave = np.zeros((mwave,self.NEM))
        ispace = np.zeros(self.NEM,dtype="int32")
        dist_ref = np.zeros(self.NEM)
        
        for iemi in range(self.NEM):
    
            #Opening file
            filename = self.LOCATION[iemi]
            filename = filename if filename.endswith('.h5') else (filename+'.h5')

            with h5py.File(filename,'r') as f:

                emtype[iemi] = h5py_helper.retrieve_data(f, 'EMTYPE', np.int32)
                ngas[iemi] = h5py_helper.retrieve_data(f, 'NGAS', np.array)
                idx[0:ngas[iemi],iemi] = h5py_helper.retrieve_data(f, 'ID', np.array)
                isox[0:ngas[iemi],iemi] = h5py_helper.retrieve_data(f, 'ISO', np.array)
                ntemp[iemi] = h5py_helper.retrieve_data(f, 'NT', np.int32)
                temp[0:ntemp[iemi],iemi] = h5py_helper.retrieve_data(f, 'TEMP', np.array)
                ispace[iemi] = h5py_helper.retrieve_data(f, 'ISPACE', np.int32)
                nwave[iemi] = h5py_helper.retrieve_data(f, 'NWAVE', np.int32)
                wave[0:nwave[iemi],iemi] = h5py_helper.retrieve_data(f, 'WAVE', np.array)
                dist_ref[iemi] = h5py_helper.retrieve_data(f, 'DIST_REF', np.float32)


        if len(np.unique(ispace))>1:
            print("ispace",ispace)
            raise ValueError("error :: all tables must be defined in the same spectral units")

        if len(np.unique(ntemp))>1:
            print("ntemp",ntemp)
            raise ValueError("error :: all tables must be defined at the same temperature levels")

        if len(np.unique(nwave))>1:
            print("nwave",nwave)
            raise ValueError("error :: all tables must be defined at the same spectral grid")

        self.EMTYPE = emtype
        self.NGAS = ngas
        self.ID = idx[0:self.NGAS.max(),:]
        self.ISO = isox[0:self.NGAS.max(),:]
        self.ISPACE = ispace[0]
        self.NT = ntemp[0]
        self.TEMP = temp[0:self.NT,0]
        self.NWAVE = nwave[0]
        self.WAVE = wave[0:self.NWAVE,0]
        self.DIST_REF = dist_ref


    ######################################################################################################
    def write_table_hdf5(self,iemi=0):
        """
        Write information on the look-up tables loaded in the Emissions class into an HDF5 file
        
        Inputs
        ------
        
        filename :: Name of the look-up table file (without .h5)
        iemi :: Index of the emission that wants to be written into the file
        """
        
        import h5py
        
        self.assess()

        filename = self.LOCATION[iemi]

        # Ensure filename ends with .h5 only once
        root, ext = os.path.splitext(filename)
        if ext != '.h5':
            filename = root + '.h5'

        # Now safely check and remove
        if os.path.exists(filename)==True:
            os.remove(filename)
        
        with h5py.File(filename,'w') as f:
        
            #Writing the header information
            dset = h5py_helper.store_data(f, 'NGAS', data=self.NGAS[iemi])
            dset.attrs['title'] = "Number of gases involved in the emission"

            dset = h5py_helper.store_data(f, 'ID', data=self.ID[0:self.NGAS[iemi],iemi])
            dset.attrs['title'] = "ID of the gaseous species involved in the emission process"

            dset = h5py_helper.store_data(f, 'ISO', data=self.ISO[0:self.NGAS[iemi],iemi])
            dset.attrs['title'] = "Isotope ID of the gaseous species involved in the emission process"

            dset = h5py_helper.store_data(f, 'EMTYPE', data=self.EMTYPE[iemi])
            dset.attrs['title'] = "Emission type"
            if self.EMTYPE[iemi]==EmissionType.FLUORESCENCE:
                dset.attrs['type'] = 'Fluorescent emission'
            elif self.EMTYPE[iemi]==EmissionType.CHEMICAL:
                dset.attrs['type'] = 'Emission initiated by chemical reaction'
            elif self.EMTYPE[iemi]==EmissionType.PHOTOLYSIS:
                dset.attrs['type'] = 'Emission initiated by photolysis'
            else:
                raise ValueError('error :: EMTYPE must be 0 or 1')
                
            dset = h5py_helper.store_data(f, 'ISPACE', data=self.ISPACE)
            dset.attrs['title'] = "Spectral unit"   
            if self.ISPACE==0:
                dset.attrs['type'] = 'Wavenumber (cm-1)'
            elif self.ISPACE==1:
                dset.attrs['type'] = 'Wavelength (um)'
            else:
                raise ValueError('error :: EMTYPE must be 0 or 1')

            dset = h5py_helper.store_data(f, 'NWAVE', data=self.NWAVE)
            dset.attrs['title'] = "Number of spectral points at which tables are defined"

            dset = h5py_helper.store_data(f, 'WAVE', data=self.WAVE)
            dset.attrs['title'] = "Spectral points at which the tables are defined"
            
            dset = h5py_helper.store_data(f, 'NT', data=self.NT)
            dset.attrs['title'] = "Number of temperature levels at which the look-up table is tabulated"
            
            dset = h5py_helper.store_data(f, 'TEMP', data=self.TEMP)
            dset.attrs['title'] = "Temperature levels at which the look-up table is tabulated / K"
            
            #Writing the coefficients
            dset = h5py_helper.store_data(f, 'K', data=self.K[:,:,iemi])
            if self.EMTYPE[iemi]==0:  #Fluorescent emission
                if self.ISPACE==0:
                    dset.attrs['title'] = "Emission g-factors [photon molecule-1 s-1 (cm-1)-1]"
                elif self.ISPACE==1:
                    dset.attrs['title'] = "Emission g-factors [photon molecule-1 s-1 um-1]"
            elif self.EMTYPE[iemi]==1:  #Chemical emission
                if self.ISPACE==0:
                    dset.attrs['title'] = "Emission rate [photon molecule-1 (cm-1)-1]"
                elif self.ISPACE==1:
                    dset.attrs['title'] = "Emission rate [photon molecule-1 um-1]"
            elif self.EMTYPE[iemi]==2:  #Photolysis emission
                if self.ISPACE==0:
                    dset.attrs['title'] = "Emission rate [photon molecule-1 (cm-1)-1]"
                elif self.ISPACE==1:
                    dset.attrs['title'] = "Emission rate [photon molecule-1 um-1]"

            #Writing the reaction rates
            if self.EMTYPE[iemi]==1:

                if self.RATE_COEFF is None:
                    raise ValueError("error :: Chemical emissions require the definition of the reaction rate coefficient")
                else:

                    dset = h5py_helper.store_data(f, 'RATE_COEFF', data=self.RATE_COEFF[:,iemi])

                    if self.NGAS[iemi]==1:  #rate coefficient in s-1
                        dset.attrs['title'] = "Reacton rate coefficient (s-1)"
                    elif self.NGAS[iemi]==2:
                        dset.attrs['title'] = "Reacton rate coefficient (cm3 s-1)"
                    
            elif self.EMTYPE[iemi]==2:

                if self.RATE_COEFF is None:
                    raise ValueError("error :: Photolysis emissions require the definition of the reaction rate coefficient")
                else:

                    dset = h5py_helper.store_data(f, 'RATE_COEFF', data=self.RATE_COEFF[:,iemi])

                    if self.NGAS[iemi]==1:  #rate coefficient in s-1
                        dset.attrs['title'] = "Reacton rate coefficient (s-1)"

            #Writing the reference stellar distance in the tables
            dset = h5py_helper.store_data(f, 'DIST_REF', data=self.DIST_REF[iemi])
            dset.attrs['title'] = "Stellar distance used in the calculations (AU)"

