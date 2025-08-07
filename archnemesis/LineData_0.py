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
from typing import Self, Iterator, Any
# Third party
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# This package
from archnemesis import Data
from archnemesis import *
from archnemesis.enums import SpectroscopicLineList, AmbientGas
import archnemesis.database
import archnemesis.database.hitran
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
        
        @attribute ambient_gas: enum
            Name of the ambient gas, default AIR. This is used to
            determine the pressure-broadening coefficients
        
        @attribute DATABASE: None | archnemsis.database.LineDatabaseProtocol
            Instance of a class that implements the `archnemsis.database.LineDatabaseProtocol` protocol.
            If `None` will use HITRAN as backend.
        

        Attributes
        ----------
        
        @attribute line_data : None | archnemsis.database.line_database_protocol.LineDataProtocol
            An object (normally a numpy record array) that implements the `archnemsis.database.line_database_protocol.LineDataProtocol`
            protocol. If `None` the data has not been retrieved from the database yet.
            
            If not `None` will have the following attributes:
                
                NU : np.ndarray[['N_LINES_OF_GAS'],float]
                    Transition wavenumber (cm^{-1})
                
                SW : np.ndarray[['N_LINES_OF_GAS'],float]
                    Transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                
                A : np.ndarray[['N_LINES_OF_GAS'],float]
                    Einstein-A coeifficient (s^{-1})
                
                GAMMA_AMB : np.ndarray[['N_LINES_OF_GAS'],float]
                    Ambient gas broadening coefficient (cm^{-1} atm^{-1})
                
                N_AMB : np.ndarray[['N_LINES_OF_GAS'],float] 
                    Temperature dependent exponent for `GAMMA_AMB` (NUMBER)
                
                DELTA_AMB : np.ndarray[['N_LINES_OF_GAS'],float] 
                    Ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                
                GAMMA_SELF : np.ndarray[['N_LINES_OF_GAS'],float] 
                    Self broadening coefficient (cm^{-1} atm^{-1})
                
                ELOWER : np.ndarray[['N_LINES_OF_GAS'],float] 
                    Lower state energy (cm^{-1})
        
        @attribute partition_data : None | archnemsis.database.line_database_protocol.PartitionFunctionDataProtocol
            An object (normally a numpy record array) that implements the `archnemsis.database.line_database_protocol.PartitionFunctionDataProtocol`
            protocol. If `None` the data has not been retrieved from the database yet.
            
            If not `None` will have the following attributes:
            
                TEMP : np.ndarray[['N_TEMPS_OF_GAS'],float]
                    Temperature of tablulated partition function (Kelvin)
                
                Q : np.ndarray[['N_TEMPS_OF_GAS'],float]
                    Tabulated partition function value
        

        Methods
        -------
        LineData_0.assess()
        LineData_0.fetch_linedata(...)
        LineData_0.fetch_partition_function(...)
        """
        
        self.ID = ID
        self.ISO = ISO
        self._ambient_gas = ambient_gas
        self.DATABASE = DATABASE
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
        if value is None:
            db = archnemesis.database.hitran.HITRAN()
            _lgr.info(f'Using default database {db}')
            self._database = db
        else:
            self._database = value

    @property
    def ambient_gas(self) -> AmbientGas:
        return self._ambient_gas

    @ambient_gas.setter
    def ambient_gas(self, value : int | AmbientGas):
        self._ambient_gas = AmbientGas(value)

    def __repr__(self) -> str:
        return f'LineData_0(ID={self.ID}, ISO={self.ISO}, ambient_gas={self.ambient_gas}, is_line_data_ready={self.is_line_data_ready()}, is_partition_function_ready={self.is_partition_function_ready()}, database={self._database})'

    ##################################################################################
 
    def assess(self) -> None:
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
    
    def is_line_data_ready(self) -> bool:
        return self.line_data is not None
    
    def fetch_linedata(self, vmin : float , vmax : float, refresh : bool = False) -> None:
        """
        Fetch the line data from the specified database, if `refresh` then get the data even if we already have some.
        NOTE: Does not check that the data we already have is valid for the wavelength range (vmin, vmax), only
        checks if `self.is_line_data_ready()` is True or False.
        
        # ARGUMENTS #
            vmin : float
                Minimum wavenumber to get line data for (cm^{-1})
            vmax : float
                Maximum wavenumber to get line data for (cm^{-1})
            refresh : bool = False
                If True will retrieve data from database again, even if data
                is already present.
        """
        if refresh or not self.is_line_data_ready():
            self.line_data = self.DATABASE.get_line_data(
                GasIsotopes(self.ID, self.ISO).as_radtran_gasses(), 
                WaveRange(vmin, vmax, ans.enums.WaveUnit.Wavenumber_cm), 
                self.ambient_gas
            )
            _lgr.info(f'Retrieved line data from database {self.DATABASE}')
        else:
            _lgr.info(f'Line data already loaded')
        
    ###########################################################################################################################
    
    def is_partition_function_ready(self) -> bool:
        return self.partition_data is not None
    
    def fetch_partition_function(self, refresh : bool = False) -> None:
        """
        Get partition function data, if `refresh` then get the data again even if `self.is_partition_function_ready()` is True.
        
        # ARGUMENTS #
            refresh : bool = False
                If True will retrieve data from database again, even if data
                is already present.
        """
        if refresh or not self.is_partition_function_ready():
            self.partition_data = self.DATABASE.get_partition_function_data(
                GasIsotopes(self.ID, self.ISO).as_radtran_gasses()
            )
            _lgr.info(f'Retrieved partition function data from database {self.DATABASE}')
        else:
            _lgr.info(f'Partition function data already loaded')
    
    ###########################################################################################################################
    
    def calculate_partition_sums(self,T):
        """
        Calculate the partition functions at any arbitrary temperature

        Args:
            T (float): Temperature (K)

        Returns:
            QT : np.ndarray[['N_GAS_ISOTOPES'], float]
                Partition functions for each of the isotopes at temperature T 
        """
        gas_isotopes = GasIsotopes(self.ID, self.ISO)
        QTs = np.zeros(gas_isotopes.n_isotopes)
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            gas_partition_data = self.partition_data[gas_desc]
            QTs[i] = np.interp(T, gas_partition_data.TEMP, gas_partition_data.Q)
        
        return QTs
        
    ###########################################################################################################################
    
    def calculate_line_strength(self,T,Tref=296.):
        """
        Calculate the line strengths at any arbitrary temperature

        Args:
            T (float): Temperature (K)
            Tref (float) :: Reference temperature at which the line strengths are listed (default 296 K)

        Returns:
            line_strengths : dict[RadtranGasDescriptor, np.ndarray[['N_GAS_LINES'], float]
                Line strengths at temperature T
        """
        
        c2 = 1.4388028496642257  #cm K
        
        #Calculating the partition function
        QTs = self.calculate_partition_sums(T)
        QTrefs = self.calculate_partition_sums(Tref)
        
        gas_isotopes = GasIsotopes(self.ID, self.ISO)
        line_strengths = dict()
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            gas_line_data = self.line_data[gas_desc]
            num = np.exp(-c2*gas_line_data.ELOWER/T) * ( 1 - np.exp(-c2*gas_line_data.NU/T))
            den = np.exp(-c2*gas_line_data.ELOWER/Tref) * ( 1 - np.exp(-c2*gas_line_data.NU/Tref))
        
            line_strengths[gas_desc] = gas_line_data.SW * QTrefs[i]/QTs[i] * num / den
        
        return line_strengths
        
    ###########################################################################################################################
    
    def plot_linedata(
            self, 
            smin : float = 1E-32, 
            logscale : bool = True, 
            scatter_style_kw : dict[str,Any] = {},
            ax_style_kw : dict[str,Any] = {},
            legend_style_kw : dict[str,Any] = {},
        ) -> None:
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
        
        if not self.is_line_data_ready():
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
            line_strength_mask = gas_linedata.SW >= smin
            wavenumbers = gas_linedata.NU[line_strength_mask]
            line_strengths = gas_linedata.SW[line_strength_mask]
            
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
            line_strength_mask = gas_linedata.SW >= smin
            wavenumbers = gas_linedata.NU[line_strength_mask]
            line_strengths = gas_linedata.SW[line_strength_mask]
            
            # Plots for specific isotopes, coloured by lower energy state
            ax = ax_array[i+1]
            
            ax.set_title(f'Line data for ${ans.Data.gas_data.molecule_to_latex(gas_desc.isotope_name)}$ (ID={int(gas_desc.gas_id)}, ISO={gas_desc.iso_id})')
            p1 = ax.scatter(
                wavenumbers,
                line_strengths,
                c = gas_linedata.ELOWER[line_strength_mask],
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











