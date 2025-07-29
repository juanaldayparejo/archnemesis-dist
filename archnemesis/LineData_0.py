#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# LineData_0.py - Class to store line data for a specific gas and isotope.
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

from archnemesis import Data
from archnemesis import *
from archnemesis.enums import SpectroscopicLineList, AmbientGas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

class LineData_0:
    """
    Clear class for storing line data.
    """
    def __init__(
            self, 
            ID: int = 1,
            ISO: int = 0,
            DATAFILE: str = None,
            DATABASE=SpectroscopicLineList.HITRAN,
            ambient_gas=AmbientGas.AIR,
        ):
        """
        Class to store line data for a specific gas and isotope.

        Inputs
        ------
        @attribute ID: int
            Radtran Gas ID
        @attribute ISO: int
            Radtran Isotope ID for each gas, default 0 for all
            isotopes in terrestrial relative abundance
        @attribute DATAFILE: str
            Path to the data file containing the line data.
            If it is None and DATABASE is HITRAN, it will
            automatically download the data from HITRAN.
        @attribute DATABASE: enum
            Name of the database to use, default HITRAN.
        @attribute ambient_gas: enum
            Name of the ambient gas, default AIR. This is used to
            determine the pressure-broadening coefficients

        Attributes
        ----------
        
        @attribute isotope_abundance: np.array
            Array containing the isotope abundance for the gas.
            This is only useful if calculating linedata for all isotopes (i.e., ISO=0). 
            If ISO=0 and isotope_abundance is None, then we will assume the HITRAN terrestrial abundance for the gas.
            
        @attribute NLINES: int
            Number of lines in the line data.
        @attribute IDLINE: np.array
            Radtran ID for each of the lines (this will be the same as ID)
        @attribute ISOLINE: np.array
            Radtran isotope ID for each of the lines (this will be the same as ISO if ISO!=0)
        @attribute NU: np.array
            Wavenumbers of the lines in cm^-1.
        @attribute SW: np.array
            Line strengths in cm/molecule.
        @attribute A: np.array
            Einstein A coefficients in s^-1.
        @attribute GAMMA_AIR: np.array
            Air-broadening coefficients in cm^-1/atm.
        @attribute N_AIR: np.array
            Air-broadening exponents.
        @attribute DELTA_AIR: np.array
            Air-broadening pressure shift coefficients in cm^-1/atm.
        @attribute GAMMA_SELF: np.array
            Self-broadening coefficients in cm^-1/atm.
        @attribute ELOWER: np.array
            Lower state energies in cm^-1.
            
        @attribute NISOQ: int
            Number of isotopes for which the partition sums are stored
        @attribute IDQ: np.array(NISOQ)
            Radtran ID for the gases whose partition sums are stored
        @attribute ISOQ: np.array(NISOQ)
            Radtran isotope ID for the gases whose partition sums are stored
        @attribute NTQ: np.array(NISOQ)
            Number of temperatures at which the partition sums are tabulated
        @attribute TT: np.array(NTQ,NISOQ)
            Temperatures at which the partition sums are tabulated
        @attribute QT: np.array(NTQ,NISOQ)
            Tabulated partition functions

        Methods
        -------
        LineData_0.assess()
        LineData_0.fetch_linedata()
        """
        
        #Spectroscopic parameters
        self.NLINES = 0      #Number of lines
        self.NU = None               #np.array(NLINES)
        self.IDLINE = None           #np.array(NLINES)
        self.ISOLINE = None          #np.array(NLINES)
        self.SW = None               #np.array(NLINES)
        self.A = None                #np.array(NLINES)
        self.GAMMA_AIR = None        #np.array(NLINES)
        self.N_AIR = None            #np.array(NLINES)
        self.DELTA_AIR = None        #np.array(NLINES)
        self.GAMMA_SELF = None       #np.array(NLINES)
        self.ELOWER = None           #np.array(NLINES)
        
        #Partition functions
        self.NISOQ = None            #int 
        self.IDQ = None              #np.array(NISOQ)
        self.ISOQ = None             #np.array(NISOQ)
        self.NTQ = None              #np.array(NISOQ)
        self.TT = None               #np.array((NTQ,NISOQ))
        self.QT = None               #np.array((NTQ,NISOQ))
        
        # private attributes
        self._database = None
        self._ambient_gas = None
        
        #General inputs
        self.ID = ID
        self.ISO = ISO
        self.isotope_abundance = None  #np.array([NISOTOPES])  
        self.DATABASE = DATABASE 
        self.DATAFILE = DATAFILE
        self.ambient_gas = ambient_gas
        
    ##################################################################################

    @property
    def DATABASE(self) -> SpectroscopicLineList:
        return self._database

    @DATABASE.setter
    def DATABASE(self, value):
        self._database = SpectroscopicLineList(value)

    @property
    def ambient_gas(self) -> AmbientGas:
        return self._ambient_gas

    @ambient_gas.setter
    def ambient_gas(self, value):
        self._ambient_gas = AmbientGas(value)


    ##################################################################################
 
    def assess(self):
        """
        Assess whether the different variables have the correct dimensions and types
        """

        if not isinstance(self.ID, int):
            raise TypeError(f"ID must be an integer, got {type(self.ID)}")
        if not isinstance(self.ISO, int):
            raise TypeError(f"ISO must be an integer, got {type(self.ISO)}")

        assert self.ambient_gas in AmbientGas, \
            f"ambient_gas must be one of {tuple(AmbientGas)}"

        if self.ID < 1:
            raise ValueError(f"ID must be greater than 0, got {self.ID}")
        if self.ISO < 0:   
            raise ValueError(f"ISO must be greater than or equal to 0, got {self.ISO}")
        
    ###########################################################################################################################
    
    def fetch_linedata(self,vmin,vmax,keep_data=True):
        """
        Fetch the line data from the specified database.
        If DATAFILE is provided, it will read the data from that file.
        If DATABASE is SpectroscopicLineList.HITRAN, it will download the data from HITRAN.
        If keep_data is False, then the downloaded file from HITRAN will be removed.
        """
        
        import hapi
        
        #Getting the gas name from the Radtran ID
        if self.ISO == 0:
            gasname = Data.gas_data.gas_info[str(self.ID)]['name']            
        else:
            gasname = Data.gas_data.gas_info[str(self.ID)]['isotope'][str(self.ISO)]['name']
        
        #Getting information about the Radtran and HITRAN IDs
        if self.ISO == 0:
            
            niso_radtran = Data.gas_data.count_isotopes(self.ID)
            
            #Searching for the isotopes in HITRAN
            iso_global_ids = []
            mol_id = []
            iso_local_ids = []
            if niso_radtran > 1:
                for i in range(1, niso_radtran + 1):
                    hitran_id = radtran_to_hitran.get((self.ID, i), None)
                    if hitran_id is not None:
                        mol_id.append(hitran_id[0])  #HITRAN molecule ID
                        iso_local_ids.append(hitran_id[1])  #HITRAN isotope ID
                        iso_global_ids.append(hapi.ISO[hitran_id][0])
            else:
                raise ValueError(f"Gas ID {self.ID} does not exist in archNEMESIS.")
            
        else:
            
            #Searching for the isotopes in HITRAN
            hitran_id = radtran_to_hitran.get((self.ID, self.ISO), None)
            if hitran_id is not None:
                mol_id = [hitran_id[0]]  #HITRAN molecule ID
                iso_local_ids = [hitran_id[1]]  #Local isotopologue ID
                iso_global_ids = [hapi.ISO[hitran_id][0]]  #Global isotopologue ID
            
            
        if len(mol_id) == 0:
            raise ValueError(f"Gas and Isotope ID not present in HITRAN: {self.ID}, {self.ISO}")
            
        
        #DOWNLOADING THE DATA IF NEEDED
        #####################################
        
        if self.DATAFILE is None: #We need to download the data from the HITRAN servers
            
            if self.DATABASE == SpectroscopicLineList.HITRAN: #Download data from HITRAN database
                
                retrieve_hitran = True
                
                # Downloading the data from HITRAN
                hapi.fetch_by_ids(gasname,iso_global_ids,vmin,vmax)
                self.DATAFILE = os.getcwd()+'/'+gasname
                
            else:
                raise ValueError(f"Unsupported DATABASE for downloading data: {self.DATABASE}")
        
        #READING THE DATA FROM THE FILE
        #####################################
            
        #Splitting the directory name and the file name
        if self.DATAFILE is None:
            raise ValueError("error in fetch_linedata :: DATAFILE must be provided")
        folder, filename = os.path.split(self.DATAFILE)
        
        #Starting the HAPI database
        hapi.db_begin(folder)
        
        #Selecting only the data we want
        if self.ISO == 0:
            Conditions = ('and',('between', 'nu', vmin, vmax),('==','molec_id',mol_id[0]))
        else:
            Conditions = ('and',('between', 'nu', vmin, vmax),('==','local_iso_id',iso_local_ids[0]))
        hapi.select(filename,Conditions=Conditions,DestinationTableName='tmp')
        
        #Extracting the relevant parameters
        if self.ambient_gas == AmbientGas.AIR:
            gamma_str = 'gamma_air'
            n_str = 'n_air'
            delta_str = 'delta_air'
        elif self.ambient_gas == AmbientGas.CO2:
            gamma_str = 'gamma_co2'
            n_str = 'n_co2'
            delta_str = 'delta_co2'
            
        idline,isoline,nu,sw,a,gamma_air,n_air,delta_air,gamma_self,elower = hapi.getColumns('tmp',['molec_id','local_iso_id','nu','sw','a',gamma_str,n_str,delta_str,'gamma_self','elower']) 

        #Converting to numpy arrays
        idline = np.array(idline,dtype='int32')
        isoline = np.array(isoline,dtype='int32')
        nu = np.array(nu)
        sw = np.array(sw)
        a = np.array(a)
        gamma_air = np.array(gamma_air)
        n_air = np.array(n_air)
        delta_air = np.array(delta_air)
        gamma_self = np.array(gamma_self)
        elower = np.array(elower)

        #REFORMATTING DATA INTO RADTRAN FORMATS
        ########################################

        #Renormalising the line strengths based on the required isotopic abundances
        if self.DATABASE == SpectroscopicLineList.HITRAN:

            #Reading the abundance of each of the isotopes and normalising the line strengths
            for i in range(len(mol_id)):
                abundance = hapi.ISO[(mol_id[i],iso_local_ids[i])][2]   #molecular abundance in HITRAN database
                sw[ (idline==mol_id[i]) & (isoline==iso_local_ids[i]) ] /= abundance
                
            #Let's translate the ID and ISO of each line back to Radtran 
            for i in range(len(mol_id)):
                radtran_ids = hitran_to_radtran.get((mol_id[i],iso_local_ids[i]))
                idline[ (idline==mol_id[i]) ] = radtran_ids[0]
                isoline[ (isoline==iso_local_ids[i]) ] = radtran_ids[1]
                
        #Removing the HITRAN folder and file if required
        if keep_data is False:
            try:
                os.remove(self.DATAFILE)
            except FileNotFoundError:
                print(f"Warning: File {self.DATAFILE} not found for deletion.")
            
                
        self.IDLINE = idline
        self.ISOLINE = isoline
        self.NU = nu
        self.SW = sw
        self.A = a
        self.GAMMA_AIR = gamma_air
        self.N_AIR = n_air
        self.DELTA_AIR = delta_air
        self.GAMMA_SELF = gamma_self
        self.ELOWER = elower
        self.NLINES = len(nu)
        
    ###########################################################################################################################
    
    def fetch_partition_function_tips(self):
        """
        Get partition function data from TIPS.
        
        In archNEMESIS, we fit the dependency of the partition functions with the temperature using a polynomial.
        The polynomial is fitted in log(T)-log(QT) space to be able to capture 
        """
        
        import hapi
    
        #Checking whether it is a single isotope or all of them
        if self.ISO == 0:
            
            niso = Data.gas_data.count_isotopes(self.ID)
            
            #Searching for the isotopes in HITRAN
            iso_global_ids = []
            mol_id = []
            iso_local_ids = []
            if niso > 0:
                for i in range(1, niso + 1):
                    hitran_id = radtran_to_hitran.get((self.ID, i), None)
                    if hitran_id is not None:
                        mol_id.append(hitran_id[0])  #HITRAN molecule ID
                        iso_local_ids.append(hitran_id[1])  #HITRAN isotope ID
                        iso_global_ids.append(hapi.ISO[hitran_id][0])
                    else:
                        print('warning :: Radtran ID',(self.ID, i),'is not listed in HAPI')

        else:
            
            niso = 1
            
            #Searching for the isotopes in HITRAN
            hitran_id = radtran_to_hitran.get((self.ID, self.ISO), None)
            if hitran_id is not None:
                mol_id = [hitran_id[0]]  #HITRAN molecule ID
                iso_local_ids = [hitran_id[1]]  #Local isotopologue ID
                iso_global_ids = [hapi.ISO[hitran_id][0]]  #Global isotopologue ID
            else:
                print('warning :: Radtran ID',(self.ID, self.ISO),'is not listed in HAPI')
    
        #Initialising arrays
        idq = np.zeros(niso,dtype='int32') ; isoq = np.zeros(niso,dtype='int32')
        idq[:] = self.ID
        if self.ISO == 0:
            isoq[:] = np.arange(1,niso+1,1)
        else:
            isoq[:] = self.ISO
            
        ntq = np.zeros(niso,dtype='int32')
        mtq = 6000
        tt = np.ones((mtq,niso)) * np.nan
        qt = np.ones((mtq,niso)) * np.nan
        
        for i in range(len(mol_id)):
            
            # get temperature grid
            TT = hapi.TIPS_2021_ISOT_HASH[(mol_id[i],iso_local_ids[i])]
            Tmin = min(TT); Tmax = max(TT)
            #get the values of the tabulated partition functions
            QT = hapi.TIPS_2021_ISOQ_HASH[(mol_id[i],iso_local_ids[i])]
            
            #Converting from HITRAN ID to Radtran ID
            radtran_ids = hitran_to_radtran.get((mol_id[i],iso_local_ids[i]))
            iiso = np.where( (idq==radtran_ids[0]) & (isoq==radtran_ids[1]) )[0][0]            
            ntq[ iiso ] = len(TT)
            tt[0:ntq[iiso],iiso] = TT
            qt[0:ntq[iiso],iiso] = QT
            
        tt = tt[0:ntq.max(),:]
        qt = qt[0:ntq.max(),:]
        
        self.NISOQ = niso
        self.IDQ = idq
        self.ISOQ = isoq
        self.NTQ = ntq
        self.TT = tt
        self.QT = qt
    
    ###########################################################################################################################
    
    def calculate_partition_sums(self,T):
        """
        Calculate the partition functions at any arbitrary temperature

        Args:
            T (float): Temperature (K)

        Returns:
            QT (NISOQ): Partition functions for each of the isotopes at temperature T 
        """
        
        QTs = np.zeros(self.NISOQ)
        for i in range(self.NISOQ):
            
            if( (T>self.TT[0:self.NTQ[i],i].max()) or (T<self.TT[0:self.NTQ[i],i].min()) ):
                raise ValueError('error in calculate_parameter_qt :: T is outside the range of temperatures tabulated in the class')
            
            QTs[i] = np.interp(T,self.TT[0:self.NTQ[i],i],self.QT[0:self.NTQ[i],i])
            
        return QTs
        
    ###########################################################################################################################
    
    def calculate_line_strength(self,T,Tref=296.):
        """
        Calculate the line strengths at any arbitrary temperature

        Args:
            T (float): Temperature (K)
            Tref (float) :: Reference temperature at which the line strengths are listed (default 296 K)

        Returns:
            SW(NLINES) :: Line strengths at temperature T
        """
        
        c2 = 1.4388028496642257  #cm K
        
        #Calculating the partition function
        QTs = self.calculate_partition_sums(T)
        QTrefs = self.calculate_partition_sums(Tref)
        
        #Mapping the partition functions for each of the lines
        if self.ISO==0:
            QTx = QTs[ self.ISOLINE - 1 ] #Here we assume that ISOQ includes all possible isotopes
            QTrefx = QTrefs[ self.ISOLINE - 1 ]
        else:
            QTx = QTs[ 0 ] #Here we assume that ISOQ includes all possible isotopes
            QTrefx = QTrefs[ 0 ]
        
        #Calculating the rest of the factors
        num = np.exp(-c2*self.ELOWER/T) * (1. - np.exp(-c2*self.NU/T))
        den = np.exp(-c2*self.ELOWER/Tref) * (1. - np.exp(-c2*self.NU/Tref))
        
        #Calculating the line strength
        SW = self.SW * QTrefx / QTx * num / den

        return SW
        
    ###########################################################################################################################
    
    def plot_linedata(self,smin=None,logscale=True):
        """
        Create diagnostic plots of the line data.
        
        Args:
            smin (float): Minimum line strength to plot. If None, all lines will be plotted.
            logscale (bool): If True, the y-axis will be in logarithmic scale.
        """
        
        import matplotlib.pyplot as plt
        
        if self.NLINES == 0:
            raise ValueError("No line data available. Please fetch the line data first.")
        
        if smin is None:
            smin = 1.0e-32  # Default minimum line strength
        
        
        #Plotting line strengths vs wavenumber coloured by lower state energy
        #############################################################################
        
        fig,ax1 = plt.subplots(1,1,figsize=(10,5))
        psize = 15.
        im = ax1.scatter(self.NU[self.SW > smin],self.SW[self.SW > smin],s=psize,c=self.ELOWER[self.SW > smin],cmap='turbo',vmin=0.,edgecolor='black',linewidth=0.5)
        
        if logscale:
            ax1.set_yscale('log')
            ax1.set_ylim(smin, self.SW.max() * 10)
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
        ax1.set_facecolor('lightgray')
        
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Lower state energy (cm$^{-1}$)')
        
        #Getting the name of the molecule
        gasname = ans.Data.gas_data.id_to_name(self.ID,self.ISO)
        gaslabel = ans.Data.gas_data.molecule_to_latex(gasname)
        
        ax1.set_title(f'Line data for ${gaslabel}$ (ID={self.ID}, ISO={self.ISO})')
        
        #Plotting line strengths vs wavenumber coloured by isotope
        #############################################################################
            
        fig,ax1 = plt.subplots(1,1,figsize=(10,5))
        
        isotopes = np.unique(self.ISOLINE[self.SW > smin])
        for i in range(len(isotopes)):
        
            isoname = ans.Data.gas_data.id_to_name(self.ID,isotopes[i])
            isolabel = ans.Data.gas_data.molecule_to_latex(isoname)
        
            im = ax1.scatter(self.NU[ (self.SW > smin) & (self.ISOLINE == isotopes[i])],self.SW[ (self.SW > smin) & (self.ISOLINE == isotopes[i])],s=psize,edgecolor='black',linewidth=0.5,label='$'+isolabel+'$')

        ax1.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1),  # Shift legend to the right
            fontsize=12, 
            title='Isotope', 
            title_fontsize='15'
        )
        
        if logscale:
            ax1.set_yscale('log')
            ax1.set_ylim(smin, self.SW.max() * 10)
        ax1.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
        ax1.set_facecolor('lightgray')
        
        #Getting the name of the molecule
        gasname = ans.Data.gas_data.id_to_name(self.ID,self.ISO)
        gaslabel = ans.Data.gas_data.molecule_to_latex(gasname)
        
        ax1.set_title(f'Line data for ${gaslabel}$ (ID={self.ID}, ISO={self.ISO})')
    
        plt.tight_layout()
    
    
##################################################################################
##################################################################################
##################################################################################
#                           Mapping IDs across databases
##################################################################################
##################################################################################
##################################################################################
    
radtran_to_hitran = {
    
    (1, 1): (1, 1),   # H2(16O)
    (1, 2): (1, 2),   # H2(18O)
    (1, 3): (1, 3),   # H2(17O)
    (1, 4): (1, 4),   # HD(16O)
    (1, 5): (1, 5),   # HD(18O)
    (1, 6): (1, 6),   # HD(17O)
    (1, 7): (1, 7),   # D2(16O)
    
    (2, 1): (2, 1),   # (12C)(16O)2
    (2, 2): (2, 2),   # (13C)(16O)2
    (2, 3): (2, 3),   # (16O)(12C)(18O)
    (2, 4): (2, 4),   # (16O)(12C)(17O)
    (2, 5): (2, 5),   # (16O)(13C)(18O)
    (2, 6): (2, 6),   # (16O)(13C)(17O)
    (2, 7): (2, 7),   # (12C)(18O)2
    (2, 8): (2, 8),   # (17O)(12C)(18O)
    (2, 9): (2, 9),   # (12C)(17O)2
    (2, 10): (2, 10), # (13C)(18O)2
    (2, 11): (2, 11), # (18O)(13C)(17O)
    (2, 12): (2, 12), # (13C)(17O)2
    
    (3, 1): (3, 1),   #(16O)3
    (3, 2): (3, 2),   #(16O)(16O)(18O)
    (3, 3): (3, 3),   #(16O)(18O)(16O)
    (3, 4): (3, 4),   #(16O)(16O)(17O)
    (3, 5): (3, 5),   #(16O)(17O)(16O)
    
    (4, 1): (4, 1),   #(14N)2(16O)
    (4, 2): (4, 2),   #(14N)(15N)(16O)
    (4, 3): (4, 3),   #(15N)(14N)(16O)
    (4, 4): (4, 4),   #(14N)2(18O)
    (4, 5): (4, 5),   #(14N)2(17O)
    
    (5, 1): (5, 1),   #(12C)(16O)
    (5, 2): (5, 2),   #(13C)(16O)
    (5, 3): (5, 3),   #(12C)(18O)
    (5, 4): (5, 4),   #(12C)(17O)
    (5, 5): (5, 5),   #(13C)(18O)
    (5, 6): (5, 6),   #(13C)(17O)
    
    (6, 1): (6, 1),   #(12C)H4
    (6, 2): (6, 2),   #(13C)H4
    (6, 3): (6, 3),   #(12C)H3D
    (6, 4): (6, 4),   #(13C)H3D
    
    (7, 1): (7, 1),   #(16O)2
    (7, 2): (7, 2),   #(16O)(18O)
    (7, 3): (7, 3),   #(16O)(17O)
    
    (8, 1): (8, 1),   #(14N)(16O)
    (8, 2): (8, 2),   #(15N)(16O)
    (8, 3): (8, 3),   #(14N)(18O)
    
    (10, 1): (10, 1), #(14N)(16O)2
    (10, 2): (10, 2), #(15N)(16O)2
    
    (11, 1): (11, 1), #(14N)H3
    (11, 2): (11, 2), #(15N)H3
    
    (12, 1): (12, 1), #H(14N)(16O)3
    (12, 2): (12, 2), #H(15N)(16O)3
    
    (13, 1): (13, 1), #(16O)H
    (13, 2): (13, 2), #(18O)H
    (13, 3): (13, 3), #(16O)D
    
    (14, 1): (14, 1), #H(19F)
    (14, 2): (14, 2), #D(19F)
    
    (15, 1): (15, 1), #H(35Cl)
    (15, 2): (15, 2), #H(37Cl)
    (15, 3): (15, 3), #D(35Cl)
    (15, 4): (15, 4), #D(37Cl)
    
    (16, 1): (16, 1), #H(79Br)
    (16, 2): (16, 2), #H(81Br)
    (16, 3): (16, 3), #D(79Br)
    (16, 4): (16, 4), #D(81Br)
    
    (17, 1): (17, 1), #H(127I)
    (17, 2): (17, 2), #D(127I)
    
    (18, 1): (18, 1), #(35Cl)(16O)
    (18, 2): (18, 2), #(37Cl)(16O)
    
    (19, 1): (19, 1), #(16O)(12C)(32S)
    (19, 2): (19, 2), #(16O)(12C)(34S)
    (19, 3): (19, 3), #(16O)(13C)(32S)
    (19, 4): (19, 5), #(18O)(12C)(32S) ---> Changes in RADTRAN wrt HITRAN
    (19, 5): (19, 4), #(16O)(12C)(33S) ---> Changes in RADTRAN wrt HITRAN
    (19, 6): (19, 6), #(16O)(13C)(34S)
    #(19, 7) Does not exist in HITRAN, but is in RADTRAN
    
    (20, 1): (20, 1), #H2(12C)(16O)
    (20, 2): (20, 2), #H2(13C)(16O)
    (20, 3): (20, 3), #H2(12C)(18O)
    
    (21, 1): (21, 1), #H(16O)(35Cl)
    (21, 2): (21, 2), #H(16O)(37Cl)
    
    (22, 1): (22, 1), #(14N)2
    (22, 2): (22, 2), #(14N)(15N)
    
    (23, 1): (23, 1), #H(12C)(14N)
    (23, 2): (23, 2), #H(13C)(14N)
    (23, 3): (23, 3), #H(12C)(15N)
    
    (24, 1): (24, 1), #(12C)H3(35Cl)
    (24, 2): (24, 2), #(12C)H3(37Cl)
    
    (25, 1): (25, 1), #H2(16O)2
    
    (26, 1): (26, 1), #(12C)2H2
    (26, 2): (26, 2), #(12C)(13C)H2
    (26, 3): (26, 3), #(12C)2HD
    
    (27, 1): (27, 1), #(12C)2H6
    (27, 2): (27, 2), #(12C)H3(13C)H3
    
    (28, 1): (28, 1), #(31P)H3
    
    (29, 1): (48, 1), #(12C)2(14N)2
    #(29, 2) Does not exist in HITRAN, but is in RADTRAN
    #(29, 3) Does not exist in HITRAN, but is in RADTRAN
    
    (30, 1): (43, 1), #(12C)4H2
    #(30, 2) Does not exist in HITRAN, but is in RADTRAN
    
    (31, 1): (44, 1), #H(12C)3(14N)
    #(31, 2) Does not exist in HITRAN, but is in RADTRAN
    #(31, 3) Does not exist in HITRAN, but is in RADTRAN
    
    (32, 1): (38, 1), #(12C)2H4
    (32, 2): (38, 2), #(12C)H2(13C)H2
    
    (33, 1): (52, 1), #(74Ge)H4
    (33, 2): (52, 2), #(72Ge)H4
    (33, 3): (52, 3), #(70Ge)H4
    (33, 4): (52, 4), #(73Ge)H4
    (33, 5): (52, 5), #(76Ge)H4
    
    #(34, 1) Does not exist in HITRAN, but is in RADTRAN
    
    (35, 1): (32, 1), #H(12C)(16O)(16O)H
    
    (36, 1): (31, 1), #H2(32S)
    (36, 2): (31, 3), #H2(33S) ---> Changes in RADTRAN wrt HITRAN
    (36, 3): (31, 2), #H2(34S) ---> Changes in RADTRAN wrt HITRAN
    
    (37, 1): (29, 1), #(12C)(16O)(19F)2
    (37, 2): (29, 2), #(13C)(16O)(19F)2
    
    (38, 1): (30, 1), #(32S)(19F)6
    
    (39, 1): (45, 1), #H2
    (39, 2): (45, 2), #HD
    
    #(40, 1) Does not exist in HITRAN, but is in RADTRAN
    
    #(41, 1) Does not exist in HITRAN, but is in RADTRAN
    
    #(42, 1) Does not exist in HITRAN, but is in RADTRAN
    
    (43, 1): (35, 1), #(35Cl)(16O)(14N)(16O)2
    (43, 2): (35, 2), #(37Cl)(16O)(14N)(16O)2
    
    (44, 1): (33, 1), #H(16O)2
    
    (45, 1): (34, 1), #(16O)
    
    (46, 1): (36, 1), #(14N)(16O)+
    
    (47, 1): (39, 1), #(12C)H3(16O)H
    
    #(48, 1) Does not exist in HITRAN, but is in RADTRAN
    
    #(49, 1) Does not exist in HITRAN, but is in RADTRAN
    
    (50, 1): (41, 1), #(12C)H3(12C)(14N)
    
    #(51, 1) Does not exist in HITRAN, but is in RADTRAN
    
    (72, 1): (40, 1), #(12C)H3(79Br)
    (72, 2): (40, 2), #(12C)H3(81Br)
    
    (73, 1): (42, 1), #(12C)(19F)4
    
    (74, 1): (47, 1), #(32S)(16O)3
    
    #(75, 1) Does not exist in HITRAN, but is in RADTRAN
    #(75, 2) Does not exist in HITRAN, but is in RADTRAN
    #(75, 3) Does not exist in HITRAN, but is in RADTRAN
    
    (77, 1): (49, 1), #(12C)(16O)(35Cl)2
    (77, 2): (49, 2), #(12C)(16O)(35Cl)(37Cl)
    
    (78, 1): (50, 1), #(32S)(16O)
    (78, 2): (50, 2), #(34S)(16O)
    (78, 3): (50, 3), #(32S)(18O)
    
    (95, 1): (51, 1), #(12C)H3(19F)
    
    (138, 1): (53, 1), #(12C)(32S)2
    (138, 2): (53, 2), #(32S)(12C)(34S)
    (138, 3): (53, 3), #(32S)(12C)(33S)
    (138, 4): (53, 4), #(13C)(32S)2
    
    (139, 1): (54, 1), #(12C)H3(127I)
    
    (140, 1): (55, 1), #(14N)(19F)3
    
}

hitran_to_radtran = {v: k for k, v in radtran_to_hitran.items()}
    
##################################################################################
##################################################################################
##################################################################################
#                    FUNCTIONS FOR CROSS SECTION CALCULATION
##################################################################################
##################################################################################
##################################################################################
    
def calculate_line_strength(NU,SW0,Elower,T,Tref=296.,pref=1.):
    """
    Calculate the line strength of a spectral transition at a given temperature.

    Args:
        T (float): Temperature at which to compute the line strength (K).
        SW0 (float): Line intensity at the reference temperature.
        E (float): Lower state energy in cm⁻¹.
        nu (float): Line center wavenumber in cm⁻¹.
        Tref (float, optional): Reference temperature (K). Defaults to 296.0.

    Returns:
        float: Line strength at temperature T in cm⁻¹/(molecule·cm⁻²).
    """
    
