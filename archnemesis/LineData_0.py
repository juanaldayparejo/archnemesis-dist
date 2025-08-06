from __future__ import annotations #  for 3.9 compatability

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


## Imports
# Standard library
import time
import dataclasses as dc
from typing import Self, Iterator, Any
# Third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hapi
# This package
from archnemesis import Data
from archnemesis import *
from archnemesis.enums import SpectroscopicLineList, AmbientGas
import archnemesis.database
from archnemesis.database.datatypes.wave_range import WaveRange
from archnemesis.database.datatypes.gas_isotopes import GasIsotopes
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)





class LineData_0:
    """
    Clear class for storing line data.
    """
    def __init__(
            self, 
            ID: int = 1,
            ISO: int = 0,
            ambient_gas=AmbientGas.AIR,
            DATABASE : None | archnemsis.database.LineDatabaseProtocol = None
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
        """
        self.ID = ID
        self.ISO = ISO
        self._ambient_gas = ambient_gas
        self._database = DATABASE
        self.line_data = None
        self.partition_data = None
        
        
    ##################################################################################

    @property
    def DATABASE(self) -> archnemsis.database.LineDatabaseProtocol:
        if self._database is None:
            raise RuntimeError('No database attached to LineData_0 instance')
        return self._database

    @DATABASE.setter
    def DATABASE(self, value : archnemsis.database.LineDatabaseProtocol):
        self._database = value

    @property
    def ambient_gas(self) -> AmbientGas:
        return self._ambient_gas

    @ambient_gas.setter
    def ambient_gas(self, value):
        self._ambient_gas = AmbientGas(value)

    def __repr__(self):
        return f'LineData_0(ID={self.ID}, ISO={self.ISO}, ambient_gas={self.ambient_gas}, line_data_ready={self.line_data_ready()}, partition_function_ready={self.partition_function_ready()}, database={self._database})'

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
    
    def line_data_ready(self):
        return self.line_data is not None
    
    def partition_function_ready(self):
        return self.partition_data is not None
    
    def fetch_linedata(self,vmin,vmax):
        """
        Fetch the line data from the specified database.
        If DATAFILE is provided, it will read the data from that file.
        If DATABASE is SpectroscopicLineList.HITRAN, it will download the data from HITRAN.
        If keep_data is False, then the downloaded file from HITRAN will be removed.
        """
        self.line_data = self.DATABASE.get_line_data(
            GasIsotopes(self.ID, self.ISO).as_radtran_gasses(), 
            WaveRange(vmin, vmax, ans.enums.WaveUnit.Wavenumber_cm), 
            self.ambient_gas
        )

        #self.line_data = {}
        #for k,v in self.DATABASE.get_line_data(
        #        GasIsotopes(self.ID, self.ISO).as_radtran_gasses(), 
        #        WaveRange(vmin, vmax, ans.enums.WaveUnit.Wavenumber_cm), 
        #        self.ambient_gas
        #    ).items():
        #    self.line_data[k] = v.view(np.recarray)
        
    ###########################################################################################################################
    
    def fetch_partition_function(self):
        """
        Get partition function data
        """
        self.partition_data = self.DATABASE.get_partition_function_data(
            GasIsotopes(self.ID, self.ISO).as_radtran_gasses()
        )
    
    ###########################################################################################################################
    
    def calculate_partition_sums(self,T):
        """
        Calculate the partition functions at any arbitrary temperature

        Args:
            T (float): Temperature (K)

        Returns:
            QT (NISOQ): Partition functions for each of the isotopes at temperature T 
        """
        gas_isotopes = GasIsotopes(self.ID, self.ISO)
        QTs = np.zeros(gas_isotopes.n_isotopes)
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            QTs[i] = np.interp(T, self.partition_data[gas_desc]['t'], self.partition_data[gas_desc]['q'])
        
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
        
        gas_isotopes = GasIsotopes(self.ID, self.ISO)
        line_strengths = dict()
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            num = np.exp(-c2*self.line_data[gas_desc]['elower']/T) * ( 1 - np.exp(-c2*self.line_data[gas_desc]['nu']/T))
            den = np.exp(-c2*self.line_data[gas_desc]['elower']/Tref) * ( 1 - np.exp(-c2*self.line_data[gas_desc]['nu']/Tref))
        
            line_strengths[gas_desc] = self.line_data[gas_desc]['sw'] * QTrefs[i]/QTs[i] * num / den
        
        return line_strengths
        
    ###########################################################################################################################
    
    def plot_linedata(
            self, 
            smin : float = 1E-32, 
            logscale : bool = True, 
            scatter_style_kw : dict[str,Any] = {},
            ax_style_kw : dict[str,Any] = {},
            legend_style_kw : dict[str,Any] = {},
        ):
        """
        Create diagnostic plots of the line data.
        
        ## ARGUMENTS ##
        
            smin : float = 1E-32
                Minimum line strength to plot.
                
            logscale : bool = True
                If True, the y-axis will be in logarithmic scale, else will be linear.
            
            scatter_style_kw : dict[str,Any] = {}
                Dictionary to pass to scatter plots that will set style parameters (e.g. `s` for size, `edgecolor`, `linewidth`,...)
                
            ax_style_kw : dict[str,Any] = {},
                Dictionary to pass to axes that will set style parameters (e.g. `facecolor`,...)
                
            legend_style_kw : dict[str,Any] = {},
                Dictionary to pass to legend that will set style parameters (e.g. `fontsize`, `title_fontsize`, ...)
        """
        
        if not self.line_data_ready():
            raise RuntimeError(f'No line data ready in {self}')
        
        scatter_style_defaults = dict(
            s = 15,
            edgecolor='black',
            linewidth=-.2
        )
        scatter_style_defaults.update(scatter_style_kw)
        
        ax_style_defaults = dict(
            facecolor='#EEEEEE'
        )
        ax_style_defaults.update(ax_style_kw)
        
        legend_style_defaults = dict(
            fontsize = 10,
            title_fontsize=12
        )
        legend_style_defaults.update(legend_style_kw)
        
        gas_isotopes = GasIsotopes(self.ID, self.ISO)
        
        f, ax_array = plt.subplots(
            gas_isotopes.n_isotopes+1,1, 
            figsize=(12,4*(gas_isotopes.n_isotopes+1)), 
            gridspec_kw={'hspace':0.3},
            squeeze=False
        )
        ax_array = ax_array.flatten()
        
        combined_ax = ax_array[0]
        combined_ax.set_title(f'Line data coloured by isotopologue')
        
        line_strengths_max = 0
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            gas_linedata = self.line_data[gas_desc]
            line_strength_mask = gas_linedata['sw'] >= smin
            wavenumbers = gas_linedata['nu'][line_strength_mask]
            line_strengths = gas_linedata['sw'][line_strength_mask]
            
            ls_max = line_strengths.max()
            line_strengths_max = ls_max if ls_max > line_strengths_max else line_strengths_max
            
            # Combined plot, all isotopes on one figure, coloured by isotope
            p0 = combined_ax.scatter(
                wavenumbers,
                line_strengths,
                label=f'${ans.Data.gas_data.molecule_to_latex(gas_desc.isotope_name)}$ (ID={int(gas_desc.gas_id)}, ISO={gas_desc.iso_id})',
                **scatter_style_defaults
            )
        
        combined_ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1.05),  # Shift legend to the right
            title='Isotope', 
            **legend_style_defaults
        )
        
        if logscale:
            combined_ax.set_yscale('log')
            combined_ax.set_ylim(smin, line_strengths_max * 10)
        combined_ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        combined_ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
        combined_ax.set(**ax_style_defaults)
    
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            gas_linedata = self.line_data[gas_desc]
            line_strength_mask = gas_linedata['sw'] >= smin
            wavenumbers = gas_linedata['nu'][line_strength_mask]
            line_strengths = gas_linedata['sw'][line_strength_mask]
            
            # Plots for specific isotopes, coloured by lower energy state
            ax = ax_array[i+1]
            
            ax.set_title(f'Line data for ${ans.Data.gas_data.molecule_to_latex(gas_desc.isotope_name)}$ (ID={int(gas_desc.gas_id)}, ISO={gas_desc.iso_id})')
            p1 = ax.scatter(
                wavenumbers,
                line_strengths,
                c = gas_linedata['elower'][line_strength_mask],
                cmap = 'turbo',
                vmin = 0,
                **scatter_style_defaults
            )
            
            # Create a colourbar axes on the right side of ax.
            x_pad = 0.01
            x0, y0, w, h = ax.get_position().bounds
            x1 = x0+w+x_pad
            cax = f.add_axes([x1,y0,0.25*(1-x1),h])
            cbar = plt.colorbar(p1, cax=cax)
            cbar.set_label('Lower state energy (cm$^{-1}$)')
        
            if logscale:
                ax.set_yscale('log')
                ax.set_ylim(smin, line_strengths.max() * 10)
            ax.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
            ax.set(**ax_style_defaults)











