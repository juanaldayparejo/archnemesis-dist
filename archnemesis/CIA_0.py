from __future__ import annotations #  for 3.9 compatability
from archnemesis import *
from archnemesis.Data.path_data import *
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit

from archnemesis.helpers import h5py_helper
from archnemesis.enums import Gas, ParaH2Ratio

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

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

        f = h5py.File(runname+'.h5','a')
        #Checking if Spectroscopy already exists
        if ('/CIA' in f)==True:
            del f['CIA']   #Deleting the Spectroscopy information that was previously written in the file

        grp = f.create_group("CIA")

        #Writing the necessary flags
        dset = grp.create_dataset('INORMAL',data=int(self.INORMAL))
        dset.attrs['title'] = "Flag indicating whether the ortho/para-H2 ratio is in equilibrium (0 for 1:1) or normal (1 for 3:1)"
        
        #Write the directory where CIA tables are stored
        dt = h5py.special_dtype(vlen=str)
        CIADATA = ['']*1
        
        # Indirect the archnemesis path so it works on other systems
        CIADATA[0] = archnemesis_indirect_path(self.CIADATA)
        
        dset = grp.create_dataset('CIADATA',data=CIADATA,dtype=dt)
        dset.attrs['title'] = "Path to directory where CIA table is stored"
        
        # Convert CIATABLE to HDF5 format if we do not have an HDF5 format of this table
        CIATABLE_str = self.CIATABLE
        if CIATABLE_str.endswith('.tab'):
            CIATABLE_str = CIATABLE_str[:-4]+'.h5'
        
        if not os.path.exists(self.CIADATA+CIATABLE_str):
            self.write_ciatable_hdf5(self.CIADATA+CIATABLE_str)
        
        # Write the name of the CIATABLE
        CIATABLE = [CIATABLE_str]
        dset = grp.create_dataset('CIATABLE',data=CIATABLE,dtype=dt)
        dset.attrs['title'] = "Name of the CIA table file"
        
        f.close()
        
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
            ax1.plot(self.WAVEN,self.K_CIA[i,iTEMP,:],label=label)

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
                    read_ciatable_hdf5(filename+'.h5')
                    file_read_successfully = True
                except Exception:
                    file_read_successfully = False
            if not file_read_successfully:
                try:
                    read_ciatable_tab(filename+'.tab', *args)
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
            self.NPARA = np.int32(f.get('NPARA', 1))
            self.NPAIR = h5py_helper.retrieve_data(f, 'NPAIR', np.int32)
            self.NT = h5py_helper.retrieve_data(f, 'NT', np.int32)
            self.NWAVE = h5py_helper.retrieve_data(f, 'NWAVE', np.int32)
                
            self.IPAIRG1 = h5py_helper.retrieve_data(f, 'IPAIRG1', np.array)
            self.IPAIRG2 = h5py_helper.retrieve_data(f, 'IPAIRG2', np.array)
            self.INORMALT = h5py_helper.retrieve_data(f, 'INORMALT', np.array)
            
            self.WAVEN = h5py_helper.retrieve_data(f, 'WAVEN', np.array)
            self.TEMP = h5py_helper.retrieve_data(f, 'TEMP', np.array)
            
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

    #2.3 micron window. Assume de Bergh 1995 a = 4e-8 cm-1/amagat^2
    iin = np.where((WAVEL>=2.15) & (WAVEL<=2.55))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 4.0e-8

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
    
    from archnemesis.Files import file_lines
    
    file1 = open(filename,'r')
    
    
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
            paircase = line[il:il+20]
            il = il + 20
            
            wavemin = float(line[il:il+10])
            il = il + 10
            wavemax = float(line[il:il+10])
            il = il + 10
            nwavex = int(line[il:il+7])
            il = il + 7
            tempx = float(line[il:il+7])
            il = il + 7
            ciamax = float(line[il:il+10])
            il = il + 10
            dwave = float(line[il:il+6])
            il = il + 6
            comments = line[il:il+27]
            il = il + 27
            reference = line[il:il+3]
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
                
                
    file1.close()

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

