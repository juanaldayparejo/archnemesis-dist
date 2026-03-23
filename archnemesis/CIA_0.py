#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# CIA_0.py - Object to represent the collision-induced absorption (CIA) properties of gases.
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



from archnemesis.Data.path_data import archnemesis_resolve_path, archnemesis_indirect_path
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit

from archnemesis.helpers import h5py_helper
from archnemesis.enums import Gas, ParaH2Ratio

import logging
_lgr = logging.getLogger(__name__)

###############################################################################################

"""
Created on Tue Aug 09 17:27:12 2021

@author: juanalday

Collision-Induced Absorption Class.
"""

class CIA_0:

    def __init__(
            self, 
            runname='', 
            INORMAL=ParaH2Ratio.EQUILIBRIUM, 
            NPAIR=9, 
            NT=25, 
            CIADATA=None, 
            CIATABLE='CO2-CO2_HITRAN.h5', 
            NWAVE=1501, 
            NPARA=0, 
            IPAIRG1=[
                Gas.H2, Gas.H2, Gas.H2, Gas.H2, Gas.H2, Gas.N2, Gas.N2, Gas.CH4, Gas.H2], 
            IPAIRG2=[
                Gas.H2, Gas.He, Gas.H2, Gas.He, Gas.N2, Gas.CH4, Gas.N2, Gas.CH4, Gas.CH4], 
            INORMALT=[
                ParaH2Ratio.EQUILIBRIUM,
                ParaH2Ratio.EQUILIBRIUM,
                ParaH2Ratio.NORMAL,
                ParaH2Ratio.NORMAL,
                ParaH2Ratio.EQUILIBRIUM,
                ParaH2Ratio.EQUILIBRIUM,
                ParaH2Ratio.EQUILIBRIUM,
                ParaH2Ratio.EQUILIBRIUM,
                ParaH2Ratio.EQUILIBRIUM
            ]
        ):
        """
        Inputs
        ------
        @param runname: str,
            Name of the Nemesis run
        @param INORMAL: int,
            Flag indicating whether the ortho/para-H2 ratio is in equilibrium (0 for 1:1) or normal (1 for 3:1)
        @param NPAIR: int,
            Number of gaseous pairs listed 
            (Default = 9 in .cia from Fortran NEMESIS : H2-H2 (eqm), H2-He (eqm), H2-H2 (normal), H2-He (normal), H2-N2, N2-CH4, N2-N2, CH4-CH4, H2-CH4)
        @param NT: int,
            Number of temperature levels over which the CIA data is defined 
        @param NWAVE: int,
            Number of spectral points over which the CIA data is defined
        @param NPARA: int,
            Number of para-H2 fractions listed in the CIA table
        @param FRAC: 1D array (NPARA),
            Fraction of para-H2 in the CIA table (only valid for H2-H2 and H2-He pairs)
        @param IPAIRG1: 1D array (NPAIR),
            First gas of each of the listed pairs (e.g., H2-He ; IPAIRG1 = H2 = 39)
        @param IPAIRG2: 1D array (NPAIR),
            Second gas of each of the listed pairs (e.g., H2-He ; IPAIRG2 = He = 40)
        @param INORMALT: 1D array (NPAIR),
            Flag indicating the equilibrium/normal hydrogen listed in the CIA table (only valid for H2-He and H2-H2, for rest of gases it is not used but needs to be defined)
        @param CIADATA: str
            String indicating where the CIA data files are stored (NOTE: Default location is the Data/cia/ directory)
        @param CIATABLE: str
            String indicating the name of the file storing the CIA table

        Attributes
        ----------
        @attribute WAVEN: 1D array
            Wavenumber array (NOTE: ALWAYS IN WAVENUMBER, NOT WAVELENGTH)
        @attribute TEMP: 1D array
            Temperature levels at which the CIA data is defined (K)
        @attribute K_CIA: 3D array
            CIA cross sections for each pair at each wavenumber and temperature level (cm5 molecule-2 ; NOTE: THIS IS DIFFERENT FROM FORTRAN NEMESIS WHERE THEY ARE LISTED IN CM-1 AMAGAT-2)

        Methods
        ----------
        CIA_0.assess()
        CIA_0.summary_info()
        CIA_0.write_hdf5()
        CIA_0.read_hdf5()
        CIA_0.read_cia()
        CIA_0.plot_cia()
        CIA_0.calc_tau_cia()
        CIA_0.locate_INORMAL_pairs()
        CIA_0.read_ciatable()
        CIA_0.read_ciatable_tab()
        CIA_0.write_ciatable_hdf5()
        CIA_0.read_ciatable_hdf5()
        """

        from archnemesis.Data.path_data import archnemesis_path 

        #Input parameters
        self.runname = runname
        #self.INORMAL : ParaH2Ratio = INORMAL
        self.NPAIR = NPAIR
        self.NPARA = NPARA
        self.IPAIRG1 : list[Gas] = IPAIRG1
        self.IPAIRG2 : list[Gas] = IPAIRG2
        self.INORMALT : list[ParaH2Ratio] = INORMALT
        self.NT = NT
        self.NWAVE = NWAVE
        self.FRAC = np.array([0])
        
        if CIADATA is None:
            self.CIADATA = archnemesis_path()+'archnemesis/Data/cia/'
        else:
            self.CIADATA = CIADATA
            
        self.CIATABLE = CIATABLE

        # Input the following profiles using the edit_ methods.
        self.WAVEN = None # np.zeros(NWAVE)
        self.TEMP = None # np.zeros(NT)
        self.K_CIA = None #np.zeros(NPAIR,NPARA,NT,NWAVE)
        
        # private attributes
        self._inormal = None
        
        # set properties
        self.INORMAL = INORMAL
    
    @property
    def INORMAL(self) -> ParaH2Ratio:
        return self._inormal
    
    @INORMAL.setter
    def INORMAL(self, value):
        self._inormal = ParaH2Ratio(value)
        
    ##################################################################################

    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        #Checking some common parameters to all cases
        assert np.issubdtype(type(self.NPAIR), np.integer) == True , \
            'NPAIR must be int'
        assert self.NPAIR > 0 , \
            'NPAIR must be >0'
            

        assert np.issubdtype(type(self.NT), np.integer) == True , \
            'NT must be int'
        assert self.NT > 0 , \
            'NT must be >0'
            
        assert np.issubdtype(type(self.NWAVE), np.integer) == True , \
            'NWAVE must be int'
        assert self.NWAVE > 0 , \
            'NWAVE must be >0'
        
        assert isinstance(self.INORMAL, ParaH2Ratio) , \
            'INORMAL must be ParaH2Ratio'
            
        assert np.issubdtype(type(self.NPARA), np.integer) == True , \
            'NPARA must be int'
        assert self.NPARA >= 0 , \
            'NPARA must be >= 0'
            
        assert len(self.FRAC) == max(self.NPARA,1) , \
            'FRAC must have size (NPARA) or (1)'
            
        assert len(self.IPAIRG1) == self.NPAIR , \
            'IPAIRG1 must have size (NPAIR)'
            
        assert len(self.IPAIRG2) == self.NPAIR , \
            'IPAIRG2 must have size (NPAIR)'
            
        assert len(self.INORMALT) == self.NPAIR , \
            'INORMALT must have size (NPAIR)'
            
        if self.WAVEN is not None:
            assert len(self.WAVEN) == self.NWAVE , \
                'WAVEN must have size (NWAVE)'
            
        if self.TEMP is not None:
            assert len(self.TEMP) == self.NT , \
                'TEMP must have size (NT)'
                
        if self.K_CIA is not None:
            assert self.K_CIA.shape == (self.NPAIR,max(self.NPARA,1),self.NT,self.NWAVE) , \
                'K_CIA must have size (NPAIR,NPARA or 1,NT,NWAVE)'

    ##################################################################################

    def summary_info(self):
        """
        Print summary information about the CIA class
        """
        
        from archnemesis.Data.gas_data import gas_info
        
        _lgr.info(f'Wavelength range ::  {(self.WAVEN.min(),self.WAVEN.max())}')
        _lgr.info(f'Temperature range ::  {(self.TEMP.min(),self.TEMP.max())}')
        _lgr.info(f'Number of CIA pairs ::  {(self.NPAIR)}')
        _lgr.info('Pairs included in CIA class :: ')
        for i in range(self.NPAIR):
            
            gasname1 = gas_info[str(self.IPAIRG1[i])]['name']
            gasname2 = gas_info[str(self.IPAIRG2[i])]['name']

            label = gasname1+'-'+gasname2
            if self.INORMALT[i]== ParaH2Ratio.NORMAL:
                label = label + " ('normal')"
                
            _lgr.info(label)



    ##################################################################################

    def read_hdf5(self,runname):
        """
        Read the information about the CIA class from the inpit HDF5 file
        @param runname: str
            Name of the NEMESIS run
        """
        
        import h5py

        with h5py.File(runname+'.h5','r') as f:
            #Checking if Spectroscopy exists
            e = "/CIA" in f
            if e==False:
                raise ValueError('error :: CIA is not defined in HDF5 file')
            else:
                self.CIADATA = f['CIA/CIADATA'][0].decode('ascii')
                self.CIATABLE = f['CIA/CIATABLE'][0].decode('ascii')
                self.INORMAL = h5py_helper.retrieve_data(f, 'CIA/INORMAL', lambda x:  ParaH2Ratio(np.int32(x)))
    
        # Resolve archnemesis path if it has been indirected
        self.CIADATA = archnemesis_resolve_path(self.CIADATA)
        
        #Reading the CIA table from the name specified
        self.read_ciatable(self.CIADATA+self.CIATABLE)
    
    ##################################################################################
        
    def write_hdf5(self,runname):
        """
        Write the information about the CIA class from the inpit HDF5 file
        Since the CIA information is actually stored in look-up tables
        @param runname: str
            Name of the NEMESIS run
        @param ciatable: str
            Name of the CIA table (expected to be stored in the Data/cia/ directory)
        """
        
        import h5py

        with h5py.File(runname+'.h5','a') as f:
            #Checking if Spectroscopy already exists
            if ('/CIA' in f)==True:
                del f['CIA']   #Deleting the Spectroscopy information that was previously written in the file

            grp = f.create_group("CIA")

            #Writing the necessary flags
            dset = h5py_helper.store_data(grp, 'INORMAL', int(self.INORMAL))
            dset.attrs['title'] = "Flag indicating whether the ortho/para-H2 ratio is in equilibrium (0 for 1:1) or normal (1 for 3:1)"
            
            #Write the directory where CIA tables are stored
            dt = h5py.special_dtype(vlen=str)
            CIADATA = ['']*1
            
            # Indirect the archnemesis path so it works on other systems
            CIADATA[0] = archnemesis_indirect_path(self.CIADATA)
            
            dset = h5py_helper.store_data(grp, 'CIADATA', CIADATA,dtype=dt)
            dset.attrs['title'] = "Path to directory where CIA table is stored"
            
            # Convert CIATABLE to HDF5 format if we do not have an HDF5 format of this table
            CIATABLE_str = self.CIATABLE
            if CIATABLE_str.endswith('.tab'):
                CIATABLE_str = CIATABLE_str[:-4]+'.h5'
            
            if not os.path.exists(self.CIADATA+CIATABLE_str):
                self.write_ciatable_hdf5(self.CIADATA+CIATABLE_str)
            
            # Write the name of the CIATABLE
            CIATABLE = [CIATABLE_str]
            dset = h5py_helper.store_data(grp, 'CIATABLE', CIATABLE,dtype=dt)
            dset.attrs['title'] = "Name of the CIA table file"
        
        
    ##################################################################################

    def read_cia(self):
        """
        Read the .cia file
        @param runname: str
            Name of the NEMESIS run
        """
        
        #Reading .cia file
        f = open(self.runname+'.cia','r')
        s = f.readline().split()
        cianame = s[0]
        s = f.readline().split()
        dnu = float(s[0])
        s = f.readline().split()
        NPARA = int(s[0])
        f.close()

        self.CIATABLE=cianame

        self.read_ciatable(self.CIADATA+self.CIATABLE, dnu, NPARA)

    ##################################################################################

    def plot_cia(self,logscale=False):
        """
        Subroutine to make a summary plot of the contents of the .cia file
        """

        from archnemesis.Data.gas_data import gas_info

        fig,ax1 = plt.subplots(1,1,figsize=(9,4))

        #labels = ['H$_2$-H$_2$ w equilibrium ortho/para-H$_2$','He-H$_2$ w equilibrium ortho/para-H$_2$','H$_2$-H$_2$ w normal ortho/para-H$_2$','He-H$_2$ w normal ortho/para-H$_2$','H$_2$-N$_2$','N$_2$-CH$_4$','N$_2$-N$_2$','CH$_4$-CH$_4$','H$_2$-CH$_4$']
        for i in range(self.NPAIR):

            gasname1 = gas_info[str(self.IPAIRG1[i])]['name']
            gasname2 = gas_info[str(self.IPAIRG2[i])]['name']

            label = gasname1+'-'+gasname2
            if self.INORMALT[i]==ParaH2Ratio.NORMAL:
                label = label + " ('normal')"

            iTEMP = np.argmin(np.abs(self.TEMP-296.))
            ax1.plot(self.WAVEN,self.K_CIA[i,0,iTEMP,:],label=label)

        ax1.legend()
        ax1.set_facecolor('lightgray')
        ax1.set_xlabel('Wavenumber (cm$^{-1}$')
        ax1.set_ylabel('CIA cross section (cm$^{5}$ molec$^{-2}$)')
        ax1.grid()
        if logscale==True:
            ax1.set_yscale('log')
        plt.tight_layout()
        plt.show()
        
    ##################################################################################
        
    def locate_INORMAL_pairs(self):
        """
        Subroutine to locate which pairs in the class are dependent on the para/ortho-H2 ratio (i.e., INORMAL = 0 or 1)
        
        Outputs
        -------
        INORMALD(NPAIR) :: Flag indicating whether the pair depends on the ortho/para-H2 ratio (True if it depends)
        """
        
        #We locate the pairs affected by INORMAL by seeing whether some of the IDs for the pairs are repeated
        #If they are repeated, we make sure that they must be defined one with INORMAL = 0 and one with INORMAL = 1
        
        arr = np.vstack([self.IPAIRG1,self.IPAIRG2])
        _, ind = np.unique(arr, axis=1, return_index=True)
        out = np.zeros(shape=arr.shape[1], dtype=bool)
        out[ind] = True
        
        #if out=False then it means that pair is repeated
        iFalse = np.where(out==False)[0]
        
        outx = [False]*self.NPAIR
        for i in range(len(iFalse)):
            
            for j in range(self.NPAIR):
                
                if((self.IPAIRG1[j]==self.IPAIRG1[iFalse[i]]) & (self.IPAIRG2[j]==self.IPAIRG2[iFalse[i]]) ):
                    outx[j] = True
            
        
        #Making sure there are no repeated cases (i.e., the repeated cases have a different INORMAL flag)    
        arr = np.vstack([self.IPAIRG1,self.IPAIRG2,self.INORMALT])
        _, ind = np.unique(arr, axis=1, return_index=True)
        out2 = np.zeros(shape=arr.shape[1], dtype=bool)
        out2[ind] = True
            
        iFalse = np.where(out2==False)[0]
        if len(iFalse)>0:
            raise ValueError('error in locate_INORMAL_pairs :: It appears that there are repeated pairs with the same INORMAL flag')            
            
        return outx
        
    ##################################################################################
    
    def read_ciatable(self, filename, *args):
        """
        Dispatch to correct read_ciatable_X function
        """
        file_read_successfully = False
        
        if filename.endswith(".tab"):
            self.read_ciatable_tab(filename, *args)
            file_read_successfully = True
        elif filename.endswith(".h5"):
            self.read_ciatable_hdf5(filename)
            file_read_successfully = True
        else:
            # Try opening files in different modes use first that works
            if not file_read_successfully:
                try:
                    self.read_ciatable_hdf5(filename+'.h5')
                    file_read_successfully = True
                except Exception:
                    file_read_successfully = False
            if not file_read_successfully:
                try:
                    self.read_ciatable_tab(filename+'.tab', *args)
                    file_read_successfully = True
                except Exception:
                    file_read_successfully = False
        
        if not file_read_successfully:
            raise RuntimeError(f"Failed to read ciatable '{filename}'")
    
    ##################################################################################
    
    def read_ciatable_tab(self, filename, dnu, NPARA):
        """
        Read the CIA look-up table in .tab format
        """
        
        from scipy.io import FortranFile
        
        try:
            f = FortranFile(filename, 'r')
            
            if NPARA != 0:
                NPAIR = 2
                TEMPS = f.read_reals(dtype='float32')
                FRAC = np.abs(f.read_reals(dtype='float32'))
                K_H2H2 = f.read_reals(dtype='float32')
                K_H2HE = f.read_reals(dtype='float32')
                KCIA_list = np.vstack([K_H2H2, K_H2HE]).reshape((-1,), order='F')
                IPAIRG1 = np.array([Gas.H2, Gas.H2])
                IPAIRG2 = np.array([Gas.H2, Gas.He])
                INORMALT = np.array([ParaH2Ratio.EQUILIBRIUM, ParaH2Ratio.EQUILIBRIUM])
                
                self.FRAC = FRAC
                
            # Reading the actual CIA file
            if NPARA == 0:
                NPAIR = 9  # 9 pairs of collision-induced absorption opacities
                TEMPS = f.read_reals(dtype='float64')
                KCIA_list = f.read_reals(dtype='float32')
                IPAIRG1 = np.array([Gas.H2, Gas.H2, Gas.H2, Gas.H2, Gas.H2, Gas.N2, Gas.N2, Gas.CH4, Gas.H2])
                IPAIRG2 = np.array([Gas.H2, Gas.He, Gas.H2, Gas.He, Gas.N2, Gas.CH4, Gas.N2, Gas.CH4, Gas.CH4])
                INORMALT = np.array([
                    ParaH2Ratio.EQUILIBRIUM,
                    ParaH2Ratio.EQUILIBRIUM,
                    ParaH2Ratio.NORMAL,
                    ParaH2Ratio.NORMAL,
                    ParaH2Ratio.EQUILIBRIUM,
                    ParaH2Ratio.EQUILIBRIUM,
                    ParaH2Ratio.EQUILIBRIUM,
                    ParaH2Ratio.EQUILIBRIUM,
                    ParaH2Ratio.EQUILIBRIUM
                ])
        finally:
            f.close()

        NT = len(TEMPS)
        NWAVE = int(len(KCIA_list) / NT / NPAIR / max(NPARA, 1))
        NU_GRID = np.linspace(0, dnu * (NWAVE - 1), NWAVE)
        K_CIA = np.zeros((NPAIR, max(NPARA, 1), NT, NWAVE))  # NPAIR x NPARA x NT x NWAVE

        index = 0
        for iwn in range(NWAVE):
            for itemp in range(NT):
                for ipara in range(max(NPARA, 1)):
                    for ipair in range(NPAIR):
                        K_CIA[ipair, ipara, itemp, iwn] = KCIA_list[index]
                        index += 1

        # Changing the units of the CIA table (NEMESIS format) from cm-1 amagat-2 to cm5 molecule-2
        AMAGAT = 2.68675E19  # molecule cm-3 (definition of amagat unit)
        K_CIA = K_CIA / (AMAGAT**2.)  # cm5 molecule-2

        self.NWAVE = NWAVE
        self.NT = NT
        self.NPAIR = NPAIR
        self.NPARA = NPARA
        self.IPAIRG1 = IPAIRG1
        self.IPAIRG2 = IPAIRG2
        self.INORMALT = INORMALT
        self.WAVEN = NU_GRID
        self.TEMP = TEMPS
        self.K_CIA = K_CIA
    
    ##################################################################################
    
    def write_ciatable_hdf5(self,filename):
        """
        Write the CIA look-up table in an HDF5 file
        """
        
        import h5py

        #Assessing that all the parameters have the correct type and dimension
        self.assess()
        
        if not filename.endswith('.h5'):
            filename += '.h5'
            
        with h5py.File(filename,'w') as f:
            #Writing the main dimensions
            dset = h5py_helper.store_data(f, 'NPARA', data=self.NPARA)
            dset.attrs['title'] = "Number of para-H2 fractions listed in the CIA table"
            
            dset = h5py_helper.store_data(f, 'NPAIR', data=self.NPAIR)
            dset.attrs['title'] = "Number of CIA pairs included in the look-up table"

            dset = h5py_helper.store_data(f, 'NWAVE', data=self.NWAVE)
            dset.attrs['title'] = "Number of wavenumber points in the look-up table"

            dset = h5py_helper.store_data(f, 'NT', data=self.NT)
            dset.attrs['title'] = "Number of temperatures at which the CIA cross sections are tabulated"
            
            dset = h5py_helper.store_data(f, 'IPAIRG1', data=self.IPAIRG1)
            dset.attrs['title'] = "ID of the first gas of each CIA pair (e.g., N2-CO2; IPAIRG1 = N2 = 22)"
            
            dset = h5py_helper.store_data(f, 'IPAIRG2', data=self.IPAIRG2)
            dset.attrs['title'] = "ID of the second gas of each CIA pair (e.g., N2-CO2; IPAIRG2 = CO2 = 2)"
            
            dset = h5py_helper.store_data(f, 'INORMALT', data=self.INORMALT)
            dset.attrs['title'] = "Flag indicating whether the cross sections correspond to equilibrium or normal hydrogen"
            
            dset = h5py_helper.store_data(f, 'WAVEN', data=self.WAVEN)
            dset.attrs['title'] = "Wavenumber"
            dset.attrs['units'] = "cm-1"
            
            dset = h5py_helper.store_data(f, 'TEMP', data=self.TEMP)
            dset.attrs['title'] = "Temperature"
            dset.attrs['units'] = "K"
            
            dset = h5py_helper.store_data(f, 'FRAC', data=self.FRAC)
            dset.attrs['title'] = "Para-H2 fractions"
            
            dset = h5py_helper.store_data(f, 'K_CIA', data=self.K_CIA)
            dset.attrs['title'] = "CIA cross sections"
            dset.attrs['units'] = "cm5 molecule-2"


        
        
    ##################################################################################
        
    def read_ciatable_hdf5(self,filename):
        """
        Read the CIA look-up table from an HDF5 file
        """
        
        import h5py

        if not filename.endswith('.h5'):
            filename += '.h5'
        
        with h5py.File(filename, 'r') as f:
            self.NPARA = np.int32(f.get('NPARA', 0))
            self.NPAIR = h5py_helper.retrieve_data(f, 'NPAIR', np.int32)
            self.NT = h5py_helper.retrieve_data(f, 'NT', np.int32)
            self.NWAVE = h5py_helper.retrieve_data(f, 'NWAVE', np.int32)
                
            self.IPAIRG1 = h5py_helper.retrieve_data(f, 'IPAIRG1', np.array)
            self.IPAIRG2 = h5py_helper.retrieve_data(f, 'IPAIRG2', np.array)
            self.INORMALT = h5py_helper.retrieve_data(f, 'INORMALT', np.array)
            
            self.WAVEN = h5py_helper.retrieve_data(f, 'WAVEN', np.array)
            self.TEMP = h5py_helper.retrieve_data(f, 'TEMP', np.array)
            
            if self.NPARA != 0:
                self.FRAC = h5py_helper.retrieve_data(f, 'FRAC', np.array)
            
            K_CIA = np.zeros((self.NPAIR,max(self.NPARA,1),self.NT,self.NWAVE)) # NPAIR x NPARA x NT x NWAVE
            K_CIA[:,:,:,:] = h5py_helper.retrieve_data(f, 'K_CIA', np.array)
            
            self.K_CIA = K_CIA
        
        self.assess()
        


###############################################################################################

"""
Created on Tue Jul 22 17:27:12 2021

@author: juanalday

Other functions interacting with the CIA class
"""

@jit(nopython=True)
def co2cia(WAVEN):
    """
    Subroutine to return CIA absorption coefficients for CO2-CO2

    @param WAVEN: 1D array
        Wavenumber array (cm-1)
    """

    WAVEL = 1.0e4/WAVEN
    CO2CIA = np.zeros(len(WAVEN))

    #2.3 micron window. Assume de Bergh 1995 a = 4e-8 cm-1/amagat^2 (NEMESIS implementation)
    #iin = np.where((WAVEL>=2.15) & (WAVEL<=2.55))
    #iin = iin[0]
    #if len(iin)>0:
    #    CO2CIA[iin] = 4.0e-8

    #2.3 micron window. Use model from Tran+2025
    nu_2p3 = np.arange(3950.,4500.+1,1)
    cia_xs_2p3 = np.array([6.53776740e-10, 7.07294844e-10, 7.48768045e-10, 7.97590633e-10, 8.32921525e-10, 8.51278057e-10, 8.98218264e-10, 9.23611791e-10, 9.74157222e-10, 1.02240920e-09, 1.06654576e-09, 1.12422975e-09, 1.17946422e-09, 1.24460516e-09, 1.29618189e-09, 1.36925371e-09, 1.43481578e-09, 1.50565883e-09, 1.56542509e-09, 1.64934629e-09, 1.73015100e-09, 1.81745205e-09, 1.89455525e-09, 1.99230177e-09, 2.06982261e-09, 2.19599842e-09, 2.30768868e-09, 2.38964708e-09, 2.51766290e-09, 2.65430083e-09, 2.76492371e-09, 2.89111661e-09,
    3.05871238e-09, 3.23515903e-09, 3.36327433e-09, 3.44948408e-09, 3.60940500e-09, 3.77548401e-09, 3.99234147e-09, 4.14271111e-09, 4.29837295e-09, 4.50505245e-09, 4.71151479e-09, 4.95926907e-09, 5.15243286e-09, 5.38416393e-09, 5.64218310e-09, 5.88825866e-09, 6.12006446e-09, 6.44447252e-09, 6.71250328e-09, 7.04469774e-09, 7.36491070e-09, 7.73378486e-09, 8.09946042e-09, 8.38541157e-09, 8.81582109e-09, 9.23486158e-09, 9.56508383e-09, 1.01102492e-08, 1.04675870e-08, 1.10708366e-08, 1.15351984e-08, 1.23887631e-08,1.29645731e-08, 1.35695170e-08, 1.38257264e-08, 1.47405265e-08, 1.52072388e-08, 1.60419160e-08, 1.67096551e-08, 1.76249564e-08,
    1.82098878e-08, 1.91367574e-08, 1.98566292e-08, 2.08572183e-08, 2.15246616e-08, 2.23733668e-08, 2.31112797e-08, 2.41358771e-08, 2.48742084e-08, 2.59073134e-08, 2.69542748e-08, 2.82675769e-08, 2.97318842e-08, 3.12042738e-08, 3.26155257e-08, 3.38897979e-08, 3.49283074e-08, 3.68730009e-08, 3.85826260e-08, 4.00960415e-08, 4.22147689e-08, 4.43374127e-08, 4.58212127e-08, 4.68191261e-08, 4.78890528e-08, 4.84543088e-08, 4.90877297e-08, 4.98314555e-08, 5.04568992e-08, 5.21683210e-08, 5.51469441e-08, 6.09743032e-08, 6.89365776e-08, 7.67473316e-08, 8.07667089e-08, 7.85247111e-08, 7.00774460e-08, 6.69171360e-08, 7.97939062e-08, 6.60764837e-08, 7.39909277e-08, 8.16464643e-08, 8.23584901e-08, 7.69689392e-08, 6.85769204e-08, 6.11716864e-08, 5.63517904e-08, 5.39650633e-08,
    5.29202241e-08, 5.23665475e-08, 5.20495814e-08, 5.14296069e-08, 5.05605639e-08, 4.95490015e-08, 4.82554969e-08, 4.69422630e-08, 4.54741539e-08, 4.04952166e-08, 3.89882341e-08, 3.72815476e-08, 3.53355902e-08, 3.43042697e-08, 3.30334286e-08, 3.16227048e-08, 3.01487051e-08, 2.86815159e-08, 2.73660293e-08, 2.63194888e-08, 2.52860518e-08, 2.45509861e-08, 2.35244533e-08, 2.27883883e-08, 2.19389953e-08, 2.12733055e-08, 2.02679038e-08, 1.95475255e-08,
    1.86157092e-08, 1.80315602e-08, 1.71100008e-08, 1.64402807e-08, 1.55995815e-08, 1.51340115e-08, 1.42102440e-08, 1.39590295e-08, 1.33509888e-08, 1.27723045e-08, 1.19088010e-08, 1.14433174e-08, 1.08444438e-08, 1.04922021e-08, 9.94517631e-09, 9.61935874e-09,
    9.20007262e-09, 8.76624798e-09, 8.48472651e-09, 8.11782581e-09, 7.75100823e-09, 7.43364460e-09, 7.10307826e-09, 6.84035213e-09, 6.51894231e-09, 6.29425492e-09, 6.05269129e-09, 5.80179167e-09, 5.57675863e-09, 5.39222322e-09, 5.14956512e-09, 4.95294934e-09, 4.75554256e-09, 4.61183171e-09, 4.47205050e-09, 4.26633145e-09,
    4.11022377e-09, 3.96758068e-09, 3.89930589e-09, 3.78319113e-09, 3.62360992e-09, 3.47445381e-09, 3.36761585e-09, 3.27758899e-09, 3.16598180e-09, 3.06498231e-09, 3.00421670e-09, 2.90555625e-09, 2.80396186e-09, 2.75302495e-09, 2.68970297e-09, 2.63816461e-09, 2.57660671e-09, 2.53008903e-09, 2.48018826e-09, 2.46226398e-09,
    2.42412554e-09, 2.39840745e-09, 2.36838763e-09, 2.35954087e-09, 2.33398587e-09, 2.33376603e-09, 2.32270332e-09, 2.33573504e-09, 2.34246187e-09, 2.35504826e-09, 2.39286178e-09, 2.39891585e-09, 2.45573697e-09, 2.49182275e-09, 2.52322603e-09, 2.58344948e-09, 2.59832225e-09, 2.61638291e-09, 2.67943629e-09, 2.79545645e-09, 2.87834876e-09, 2.95743970e-09, 2.99790648e-09, 3.13496854e-09,
    3.20864542e-09, 3.33790765e-09, 3.44284138e-09, 3.59050637e-09, 3.68617339e-09, 3.84139080e-09, 3.96174292e-09, 4.12761470e-09, 4.24280155e-09, 4.39244799e-09, 4.52455258e-09, 4.70588303e-09, 4.83980522e-09, 5.02710729e-09, 5.20993971e-09, 5.45008680e-09, 5.70792619e-09, 5.95415263e-09, 6.21343572e-09, 6.45522759e-09, 6.65040709e-09, 6.99405683e-09, 7.30763496e-09, 7.60763147e-09, 7.97447413e-09, 8.34789810e-09, 8.63798599e-09, 8.86064674e-09,
    9.09323760e-09, 9.26331445e-09, 9.42343667e-09, 9.63415402e-09, 9.81960321e-09, 1.01894028e-08, 1.07114936e-08, 1.16796647e-08, 1.29587251e-08, 1.42257506e-08, 1.49173506e-08, 1.47310482e-08, 1.36135792e-08, 1.33043988e-08, 1.53245490e-08, 1.34828983e-08, 1.48067393e-08, 1.60805698e-08, 1.63916963e-08, 1.58069254e-08, 1.47518290e-08, 1.39028723e-08, 1.33613226e-08, 1.32982661e-08, 1.33762446e-08, 1.36401688e-08, 1.38803466e-08, 1.40858084e-08, 1.41132151e-08, 1.43051912e-08, 1.43174868e-08, 1.44191948e-08,
    1.44488263e-08, 1.40488849e-08, 1.40465800e-08, 1.41767438e-08, 1.42682618e-08, 1.47218612e-08, 1.51390819e-08, 1.55861708e-08, 1.58041549e-08, 1.58089157e-08, 1.54426060e-08, 1.54672676e-08, 1.62808584e-08, 1.59255073e-08, 1.66309150e-08, 1.73690362e-08, 1.77328027e-08, 1.77826515e-08, 1.75619778e-08, 1.77187483e-08, 1.78966092e-08, 1.81819759e-08, 1.86759005e-08, 1.92321274e-08,
    1.95726547e-08, 1.97995715e-08, 1.99706296e-08, 2.00734341e-08, 2.01322806e-08, 2.02300343e-08, 2.02413999e-08, 2.04846171e-08, 2.12785886e-08, 2.30034257e-08, 2.53658470e-08, 2.77960903e-08, 2.89782555e-08, 2.81590471e-08, 2.53953475e-08, 2.42881434e-08, 2.82623917e-08, 2.38421791e-08, 2.62664171e-08, 2.86480354e-08,
    2.88039412e-08, 2.70508835e-08, 2.43464904e-08, 2.19606315e-08, 2.03879063e-08, 1.95980234e-08, 1.92281052e-08, 1.90324739e-08, 1.88985879e-08, 1.86818382e-08, 1.83933145e-08, 1.80658881e-08, 1.76383673e-08, 1.72320567e-08, 1.67830172e-08, 1.52037433e-08, 1.47335388e-08, 1.42118969e-08, 1.35955580e-08, 1.33118820e-08,
    1.29441422e-08, 1.25281894e-08, 1.21042183e-08, 1.17025363e-08, 1.13593257e-08, 1.10752263e-08, 1.07663944e-08, 1.06026809e-08, 1.03481376e-08, 1.02208186e-08, 1.00136307e-08, 9.87618243e-09, 9.65976562e-09, 9.53369193e-09, 9.36587803e-09, 9.28016097e-09, 9.10617330e-09, 9.03158019e-09, 8.89506799e-09, 8.87332523e-09, 8.76247072e-09, 8.82656663e-09, 8.82311950e-09, 8.81859361e-09,
    8.76157962e-09, 8.82393892e-09, 8.78783573e-09, 8.92118135e-09, 8.98663127e-09, 9.06313434e-09, 9.25036574e-09, 9.31968574e-09, 9.58844907e-09, 9.74497017e-09, 1.01372274e-08, 1.03782721e-08, 1.06321902e-08, 1.06952655e-08, 1.11412984e-08, 1.13444869e-08, 1.17673989e-08, 1.20857625e-08, 1.25631937e-08, 1.28519638e-08, 1.33309072e-08, 1.36983714e-08, 1.42363151e-08, 1.45882423e-08,
    1.50523732e-08, 1.54277736e-08, 1.59945351e-08, 1.63885459e-08, 1.69878982e-08, 1.75808313e-08, 1.83217815e-08, 1.91568572e-08, 2.00097909e-08, 2.08309138e-08, 2.15592539e-08, 2.21448436e-08, 2.33016763e-08, 2.43053160e-08, 2.51844894e-08, 2.64489665e-08, 2.77091453e-08, 2.85864865e-08, 2.91641169e-08, 2.97878350e-08,
    3.01027270e-08, 3.04672235e-08, 3.08953461e-08, 3.12530951e-08, 3.22706523e-08, 3.40681440e-08, 3.75983342e-08, 4.24336445e-08, 4.71755767e-08, 4.96098838e-08, 4.82246230e-08, 4.30548244e-08, 4.11174306e-08, 4.89516634e-08, 4.05794502e-08, 4.53916012e-08, 5.00412141e-08, 5.04595677e-08, 4.71537442e-08, 4.20061050e-08,
    3.74816868e-08, 3.45339358e-08, 3.30699619e-08, 3.24243760e-08, 3.20785156e-08, 3.18771637e-08, 3.14914256e-08, 3.09540744e-08, 3.03300537e-08, 2.95344502e-08, 2.87270510e-08, 2.78254698e-08,
    2.47840420e-08, 2.38592599e-08, 2.28128647e-08, 2.16209112e-08, 2.09865647e-08, 2.02064297e-08, 1.93411913e-08, 1.84375657e-08, 1.75382382e-08, 1.67315289e-08, 1.60888190e-08, 1.54542399e-08, 1.50015698e-08, 1.43713661e-08, 1.39182880e-08, 1.33962483e-08, 1.29862024e-08, 1.23692330e-08, 1.19259938e-08, 1.13539145e-08,
    1.09937857e-08, 1.04281040e-08, 1.00158941e-08, 9.49954219e-09, 9.21177057e-09, 8.64482293e-09, 8.48766165e-09, 8.11307954e-09, 7.75638447e-09, 7.22614914e-09, 6.93837491e-09, 6.56367081e-09, 6.34229203e-09, 6.00336432e-09, 5.79857341e-09, 5.53807361e-09,
    5.27025769e-09, 5.09269311e-09, 4.86506282e-09, 4.63525163e-09, 4.43577771e-09, 4.22863665e-09, 4.06160809e-09, 3.85905636e-09, 3.71452066e-09, 3.56091292e-09, 3.39968281e-09, 3.25486283e-09, 3.13421677e-09, 2.97913020e-09, 2.84990941e-09, 2.72045322e-09, 2.62307218e-09, 2.52896022e-09, 2.39281079e-09, 2.28862014e-09, 2.18823735e-09, 2.13438193e-09, 2.05398832e-09, 1.94295407e-09, 1.83742934e-09, 1.75804456e-09, 1.68846853e-09, 1.60234444e-09,
    1.52162144e-09, 1.47006250e-09, 1.39959031e-09, 1.31986188e-09, 1.27098824e-09, 1.20921959e-09, 1.16053316e-09, 1.10532086e-09, 1.05420178e-09, 1.00105723e-09, 9.63264348e-10, 9.18381341e-10, 8.76832150e-10, 8.30462835e-10, 7.97776788e-10, 7.56400860e-10, 7.21322774e-10, 6.84651802e-10, 6.56618567e-10, 6.25930140e-10, 5.93748807e-10, 5.77655539e-10, 5.47741646e-10, 5.36133381e-10, 5.13628380e-10, 4.82445904e-10, 4.55958108e-10, 4.21709746e-10, 3.65609785e-10, 3.50644308e-10, 3.36253441e-10, 3.22362783e-10, 3.08971472e-10, 2.96094515e-10, 2.83585096e-10
    ])
    iin = np.where( (WAVEN>=nu_2p3.min()) & (WAVEN<=nu_2p3.max()) )[0]
    CO2CIA[iin] = np.interp(WAVEN[iin],nu_2p3,cia_xs_2p3)

    #1.73 micron window. Assume mean a = 6e-9 cm-1/amagat^2
    iin = np.where((WAVEL>=1.7) & (WAVEL<=1.76))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 6.0e-9

    #1.28 micron window. Update from Federova et al. (2014) to
    #aco2 = 1.5e-9 cm-1/amagat^2
    iin = np.where((WAVEL>=1.25) & (WAVEL<=1.35))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 1.5e-9

    #1.18 micron window. Assume a mean a = 1.5e-9 cm-1/amagat^2
    #if(xl.ge.1.05.and.xl.le.1.35)aco2 = 1.5e-9
    #Update from Federova et al. (2014)
    iin = np.where((WAVEL>=1.125) & (WAVEL<=1.225))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 0.5*(0.31+0.79)*1e-9

    #1.10 micron window. Update from Federova et al. (2014)
    iin = np.where((WAVEL>=1.06) & (WAVEL<=1.125))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 0.5*(0.29+0.67)*1e-9
        
    #Changing the units from cm-1 amagat-2 (NEMESIS format) to cm5 molecule-2
    AMAGAT = 2.68675E19 #molecule cm-3 (definition of amagat unit)
    CO2CIA = CO2CIA / (AMAGAT**2.) #cm5 molecule-2

    return CO2CIA

@jit(nopython=True)
def n2n2cia(WAVEN):
    """
    Subroutine to return CIA absorption coefficients for N2-N2
 
    Overtone absorption coef. (km-1 amagat-2) Bob McKellar
    Kindly provided by Caitlin Griffith


    @param WAVEN: 1D array
        Wavenumber array (cm-1)
    """


    WAVEN1 = [4500.0,4505.0,4510.0,4515.0,4520.0,4525.0,4530.0,4535.0,\
    4540.0,4545.0,4550.0,4555.0,4560.0,4565.0,4570.0,4575.0,\
    4580.0,4585.0,4590.0,4595.0,4600.0,4605.0,4610.0,4615.0,\
    4620.0,4625.0,4630.0,4635.0,4640.0,4645.0,4650.0,4655.0,\
    4660.0,4665.0,4670.0,4675.0,4680.0,4685.0,4690.0,4695.0,\
    4700.0,4705.0,4710.0,4715.0,4720.0,4725.0,4730.0,4735.0,\
    4740.0,4745.0,4750.0,4755.0,4760.0,4765.0,4770.0,4775.0,\
    4780.0,4785.0,4790.0,4795.0,4800.0,4805.0,4810.0,4815.0,\
    4820.0,4825.0]
    WAVEN1 = np.array(WAVEN1)

    N2COEF1 = [1.5478185E-05,3.4825567E-05,5.4172953E-05,7.3520343E-05,\
    9.2867725E-05,1.1221511E-04,1.3156250E-04,1.5090988E-04,\
    1.7025726E-04,1.8960465E-04,2.0895203E-04,2.3593617E-04,\
    2.9850862E-04,3.6948317E-04,4.4885988E-04,5.4001610E-04,\
    6.4105232E-04,7.5234997E-04,8.7262847E-04,9.9942752E-04,\
    1.1362602E-03,1.2936132E-03,1.5176521E-03,1.7954395E-03,\
    2.1481151E-03,2.6931590E-03,3.1120952E-03,2.7946872E-03,\
    2.5185575E-03,2.4253442E-03,2.4188559E-03,2.4769977E-03,\
    2.4829037E-03,2.3845681E-03,2.2442993E-03,2.1040305E-03,\
    1.9726211E-03,1.8545000E-03,1.7363789E-03,1.6182578E-03,\
    1.5128252E-03,1.4635258E-03,1.2099572E-03,1.0359654E-03,\
    9.1723543E-04,7.5135247E-04,6.0498451E-04,5.0746030E-04,\
    4.0987082E-04,3.2203691E-04,2.5376283E-04,2.0496233E-04,\
    1.5671484E-04,1.1761552E-04,9.7678370E-05,7.8062728E-05,\
    5.8552457E-05,4.8789554E-05,4.1275161E-05,3.9085765E-05,\
    3.9056369E-05,3.5796973E-05,3.0637581E-05,2.5478185E-05,\
    2.0318790E-05,5.1593952E-06]
    N2COEF1 = np.array(N2COEF1)

    #from scipy.interpolate import interp1d
    #f = interp1d(WAVEN1,N2COEF1)

    #Finding the range within the defined wavenumbers
    N2N2CIA = np.zeros(len(WAVEN))
    iin = np.where((WAVEN>=np.min(WAVEN1)) & (WAVEN<=np.max(WAVEN1)))
    iin = iin[0]

    #N2N2CIA[iin] = f(WAVEN[iin])

    N2N2CIA[iin] = np.interp(WAVEN[iin],WAVEN1,N2COEF1)

    #Convert to cm-1 (amagat)-2
    N2N2CIA = N2N2CIA * 1.0e-5
    
    #Changing the units from cm-1 amagat-2 (NEMESIS format) to cm5 molecule-2
    AMAGAT = 2.68675E19 #molecule cm-3 (definition of amagat unit)
    N2N2CIA = N2N2CIA / (AMAGAT**2.) #cm5 molecule-2

    return N2N2CIA

@jit(nopython=True)
def n2h2cia(WAVEN):
    """
    Subroutine to return CIA absorption coefficients for H2-N2
 
    Absorption coef. (km-1 amagat-2) from McKellar et al.
    Kindly provided by Caitlin Griffith

    @param WAVEN: 1D array
        Wavenumber array (cm-1)
    """

    WAVEN1 = [3995.00,4000.00,4005.00,4010.00,4015.00,4020.00,\
    4025.00,4030.00,4035.00,4040.00,4045.00,4050.00,\
    4055.00,4060.00,4065.00,4070.00,4075.00,4080.00,\
    4085.00,4090.00,4095.00,4100.00,4105.00,4110.00,\
    4115.00,4120.00,4125.00,4130.00,4135.00,4140.00,\
    4145.00,4150.00,4155.00,4160.00,4165.00,4170.00,\
    4175.00,4180.00,4185.00,4190.00,4195.00,4200.00,\
    4205.00,4210.00,4215.00,4220.00,4225.00,4230.00,\
    4235.00,4240.00,4245.00,4250.00,4255.00,4260.00,\
    4265.00,4270.00,4275.00,4280.00,4285.00,4290.00,\
    4295.00,4300.00,4305.00,4310.00,4315.00,4320.00,\
    4325.00,4330.00,4335.00,4340.00,4345.00,4350.00,\
    4355.00,4360.00,4365.00,4370.00,4375.00,4380.00,\
    4385.00,4390.00,4395.00,4400.00,4405.00,4410.00,\
    4415.00,4420.00,4425.00,4430.00,4435.00,4440.00,\
    4445.00,4450.00,4455.00,4460.00,4465.00,4470.00,\
    4475.00,4480.00,4485.00,4490.00,4495.00,4500.00,\
    4505.00,4510.00,4515.00,4520.00,4525.00,4530.00,\
    4535.00,4540.00,4545.00,4550.00,4555.00,4560.00,\
    4565.00,4570.00,4575.00,4580.00,4585.00,4590.00,\
    4595.00,4600.00,4605.00,4610.00,4615.00,4620.00,\
    4625.00,4630.00,4635.00,4640.00,4645.00,4650.00,\
    4655.00,4660.00,4665.00,4670.00,4675.00,4680.00,\
    4685.00,4690.00,4695.00,4700.00,4705.00,4710.00,\
    4715.00,4720.00,4725.00,4730.00,4735.00,4740.00,\
    4745.00,4750.00,4755.00,4760.00,4765.00,4770.00,\
    4775.00,4780.00,4785.00,4790.00,4795.00,4800.00,\
    4805.00,4810.00,4815.00,4820.00,4825.00,4830.00,\
    4835.00,4840.00,4845.00,4850.00,4855.00,4860.00,\
    4865.00,4870.00,4875.00,4880.00,4885.00,4890.00,\
    4895.00,4900.00,4905.00,4910.00,4915.00,4920.00,\
    4925.00,4930.00,4935.00,4940.00,4945.00,4950.00,\
    4955.00,4960.00,4965.00,4970.00,4975.00,4980.00,\
    4985.00,4990.00,4995.00]
    WAVEN1 = np.array(WAVEN1)

    H2N2COEF1 = [3.69231E-04,3.60000E-03,6.83077E-03,1.00615E-02,\
    1.36610E-02,1.84067E-02,2.40000E-02,3.18526E-02,\
    3.97052E-02,4.75578E-02,4.88968E-02,7.44768E-02,\
    9.08708E-02,0.108070,0.139377,0.155680,0.195880,0.228788,\
    0.267880,0.324936,0.367100,0.436444,0.500482,0.577078,\
    0.656174,0.762064,0.853292,0.986708,1.12556,1.22017,\
    1.33110,1.65591,1.69356,1.91446,1.75494,1.63788,\
    1.67026,1.62200,1.60460,1.54774,1.52408,1.48716,\
    1.43510,1.42334,1.34482,1.28970,1.24494,1.16838,\
    1.11038,1.06030,0.977912,0.924116,0.860958,0.807182,\
    0.759858,0.705942,0.680112,0.619298,0.597530,0.550046,\
    0.512880,0.489128,0.454720,0.432634,0.404038,0.378780,\
    0.359632,0.333034,0.317658,0.293554,0.277882,0.262120,\
    0.240452,0.231128,0.210256,0.202584,0.192098,0.181876,\
    0.178396,0.167158,0.171314,0.165576,0.166146,0.170206,\
    0.171386,0.181330,0.188274,0.205804,0.223392,0.253012,\
    0.292670,0.337776,0.413258,0.490366,0.600940,0.726022,\
    0.890254,1.14016,1.21950,1.45480,1.35675,1.53680,\
    1.50765,1.45149,1.38065,1.19780,1.08241,0.977574,\
    0.878010,0.787324,0.708668,0.639210,0.578290,0.524698,\
    0.473266,0.431024,0.392020,0.357620,0.331398,0.299684,\
    0.282366,0.260752,0.242422,0.234518,0.217008,0.212732,\
    0.204464,0.198802,0.199584,0.188652,0.195038,0.191616,\
    0.200324,0.213712,0.224948,0.252292,0.276978,0.318584,\
    0.369182,0.432017,0.527234,0.567386,0.655152,0.660094,\
    0.739228,0.698344,0.662759,0.663277,0.584378,0.535622,\
    0.481566,0.443086,0.400727,0.364086,0.338196,0.303834,\
    0.289236,0.262176,0.247296,0.231594,0.211104,0.205644,\
    0.185118,0.178470,0.170610,0.152406,0.153222,0.132552,\
    0.131400,0.122286,0.109758,0.107472,9.21480E-02,9.09240E-02,\
    8.40520E-02,7.71800E-02,7.03080E-02,6.34360E-02,5.76892E-02,\
    5.32345E-02,4.90027E-02,4.49936E-02,4.12073E-02,3.76437E-02,\
    3.43029E-02,3.11848E-02,2.80457E-02,2.49195E-02,2.19570E-02,\
    1.91581E-02,1.65230E-02,1.40517E-02,1.17440E-02,9.60000E-03,\
    8.40000E-03,7.20000E-03,6.00000E-03,4.80000E-03,3.60000E-03,\
    2.40000E-03,1.20000E-03]
    H2N2COEF1 = np.array(H2N2COEF1)

    #from scipy.interpolate import interp1d
    #f = interp1d(WAVEN1,H2N2COEF1)

    #Finding the range within the defined wavenumbers
    N2H2CIA = np.zeros(len(WAVEN))
    iin = np.where((WAVEN>=np.min(WAVEN1)) & (WAVEN<=np.max(WAVEN1)))[0]

    N2H2CIA[iin] = np.interp(WAVEN[iin],WAVEN1,H2N2COEF1)

    #Convert to cm-1 (amagat)-2
    N2H2CIA = N2H2CIA * 1.0e-5
    
    #Changing the units from cm-1 amagat-2 (NEMESIS format) to cm5 molecule-2
    AMAGAT = 2.68675E19 #molecule cm-3 (definition of amagat unit)
    N2H2CIA = N2H2CIA / (AMAGAT**2.) #cm5 molecule-2

    return N2H2CIA

###############################################################################################

def read_cia_hitran_file(filename):
    """
    Subroutine to read the CIA cross sections from a file written in the HITRAN CIA format
 
    Inputs
    --------
    @param filename: str
        Name of the file
        
    Outputs
    ---------
    @param gasID1: int
        First gas of the pair (e.g., CO2-O2) ; gasID1 = CO2 = 2
    @param gasID2: int
        Second gas of the pair (e.g., CO2-O2) ; gasID2 = O2 = 7
    @param ncases: int
        Number of cases (temperature or spectral ranges) tabulated in the file
    @param temp(ncases): 1D array
        Temperature for each of the cases
    @param nwave(ncases): 1D array
        Number of wavenumbers for each of the cases
    @param wave(nwave,ncases) :: 2D array
        Wavenumber array for each of the cases (cm-1)
    @param k(nwave,ncases): 2D array
        CIA cross section (cm5 molecule-2)
    """
    
    with open(filename,'r') as file1:
    
        
        temp = []
        nwave = []
        wave = []
        k = []
        
        ix = 0
        while True:
            
                    
            if ix==0:
                #Header line for each case
                line = file1.readline()
                # if line is empty
                # end of file is reached
                if not line:
                    break
        
                il = 0
                #paircase = line[il:il+20]
                il = il + 20
                
                #wavemin = float(line[il:il+10])
                il = il + 10
                #wavemax = float(line[il:il+10])
                il = il + 10
                nwavex = int(line[il:il+7])
                il = il + 7
                tempx = float(line[il:il+7])
                il = il + 7
                #ciamax = float(line[il:il+10])
                il = il + 10
                #dwave = float(line[il:il+6])
                il = il + 6
                #comments = line[il:il+27]
                il = il + 27
                #reference = line[il:il+3]
                il = il + 3
                
                ix = 1
                
            else:
            
                temp.append(tempx)
                nwave.append(nwavex)
            
                #Data with cross sections
                for iwave in range(nwavex):
                    line = file1.readline()
                    vals = line.split()
                    wave.append(float(vals[0]))
                    k.append(float(vals[1]))
                    
                ix = 0


    #Re-shaping arrays
    temp = np.array(temp)
    nwave = np.array(nwave,dtype='int32')
    ncases = len(temp)
    waven = np.zeros((nwave.max(),ncases))
    kn = np.zeros((nwave.max(),ncases))
    
    ix = 0
    for i in range(ncases):
        
        waven[0:nwave[i],i] = wave[ix:ix+nwave[i]]
        kn[0:nwave[i],i] = k[ix:ix+nwave[i]]
        ix = ix + nwave[i]
        
    return ncases,temp,nwave,waven,kn

