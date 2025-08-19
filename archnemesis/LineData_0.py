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
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# This package
from archnemesis import Data
from archnemesis import *
from archnemesis.enums import SpectroscopicLineList, AmbientGas
import archnemesis.helpers.maths_helper as maths_helper
import archnemesis.database
import archnemesis.database.hitran
from archnemesis.database.datatypes.wave_point import WavePoint
from archnemesis.database.datatypes.wave_range import WaveRange
from archnemesis.database.datatypes.gas_isotopes import GasIsotopes
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



# TODO: 
# * Account for HITRAN weighting by terrestrial abundances .
#   NOTE: do this in the HITRAN.py file, not here as ideally LineData_0 would give 
#         line strengths and absorption coefficients per unit of gas (e.g. per column density,
#         per gram, per mole, something like that) the exact unit to be worked out later.
# * Natural Broadening
# * Pressure shift
# * Multiple ambient gasses. At the moment can only have one, but should be able to define
#   a gas mixture.
# * Additional lineshapes, have a look at https://www.degruyterbrill.com/document/doi/10.1515/pac-2014-0208/html



class LineData_0:
    """
    Clear class for storing line data.
    """
    def __init__(
            self, 
            ID: int = 1,
            ISO: int = 0,
            ambient_gas=AmbientGas.H2,
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
    
    @property
    def gas_isotopes(self) -> GasIsotopes:
        return GasIsotopes(self.ID, self.ISO)

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
        assert vmin < vmax, f'Mimimum wavenumber ({vmin}) must be less than maximum wavenumber ({vmax})'
        
        wave_range = WaveRange(vmin, vmax, ans.enums.WaveUnit.Wavenumber_cm)
        
    
        if refresh or not self.is_line_data_ready():
            self.line_data = self.DATABASE.get_line_data(
                self.gas_isotopes.as_radtran_gasses(), 
                wave_range,
                self.ambient_gas, 
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
                self.gas_isotopes.as_radtran_gasses()
            )
            _lgr.info(f'Retrieved partition function data from database {self.DATABASE}')
        else:
            _lgr.info(f'Partition function data already loaded')
    
    ###########################################################################################################################
    
    def calculate_doppler_width(
            self, 
            temp: float, 
        ) -> dict[RadtranGasDescriptor, np.ndarray[['NWAVE'], float]]:
        """
        Calculate Doppler width (HWHM), broadening due to thermal motion.
        NOTE: To get the standard deviation of the gaussian, multiply HWHM by 1/sqrt(2*ln(2))
        
            dlambda/lambda_0 = sqrt(2 ln(2) * (k_b * T)/(m_0 * c^2) )
                             = sqrt(T/m_0) * 1/c * sqrt(2 ln(2) k_b)
                             = sqrt(T/M_0) * 1/c * sqrt(2 ln(2) N_A k_b)
        dlambda - half-width-half-maximum in wavelength space
        lambda_0 - wavelength of line transition
        k_b - boltzmann const
        T - temperature (Kelvin)
        m_0 - molecular mass
        c - speed of light
        M_0 - molecular mass
        N_A - avogadro's constant
        
        """
        
        doppler_width_const : float  = 1/Data.constants.c_light_cgs * np.sqrt(2*np.log(2)*Data.constants.N_avogadro*Data.constants.k_boltzmann_cgs)
        
        dws = dict() # result
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            gas_line_data = self.line_data[gas_desc]
            
            dws[gas_desc] = gas_line_data.NU * np.sqrt( temp / (gas_desc.molecular_mass*1E-3) ) * doppler_width_const
        
        return dws
    
    def calculate_lorentz_width(
            self, 
            press: float, 
            temp: float,
            amb_frac: float, # fraction of ambient gas
            tref : float = 296,
        ) -> dict[RadtranGasDescriptor, np.ndarray[['NWAVE'], float]]:
        """
        Calculate pressure-broadened width HWHM (half-width-half-maximum) of cauchy-lorentz distribution.
        """
        _lgr.debug(f'{press=} {temp=} {amb_frac=} {tref=}')
        
        lws = dict() # result
        
        tratio = temp/tref
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            gas_line_data = self.line_data[gas_desc]
            
            lws[gas_desc] = (
                gas_line_data.GAMMA_AMB * tratio**gas_line_data.N_AMB * amb_frac 
                + gas_line_data.GAMMA_SELF * tratio**gas_line_data.N_SELF * (1-amb_frac)
            ) * press
        
        return lws
    
    
    def calculate_partition_sums(self,T) -> dict[RadtranGasDescriptor, float]:
        """
        Calculate the partition functions at any arbitrary temperature

        Args:
            T (float): Temperature (K)

        Returns:
            QT : np.ndarray[['N_GAS_ISOTOPES'], float]
                Partition functions for each of the isotopes at temperature T 
        """
        QTs = dict() # result
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            gas_partition_data = self.partition_data[gas_desc]
            QTs[gas_desc] = np.interp(T, gas_partition_data.TEMP, gas_partition_data.Q)
        
        return QTs
        
    ###########################################################################################################################
    
    def calculate_line_strength(self,T,Tref=296.) -> dict[RadtranGasDescriptor, np.ndarray[['NWAVE'], float]]:
        """
        Calculate the line strengths at any arbitrary temperature

        Args:
            T (float): Temperature (K)
            Tref (float) :: Reference temperature at which the line strengths are listed (default 296 K)

        Returns:
            line_strengths : dict[RadtranGasDescriptor, np.ndarray[['N_GAS_LINES'], float]
                Line strengths at temperature T
        """
        
        line_strengths = dict() # result
        
        qT = self.calculate_partition_sums(T)
        qTref = self.calculate_partition_sums(Tref)
        
        _lgr.debug(f'{Data.constants.c2_cgs=}')
        
        
        t_factor = Data.constants.c2_cgs * (T - Tref)/(T*Tref)
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            gas_line_data = self.line_data[gas_desc]
            
            # Partition function ratio
            q_ratio = qTref[gas_desc] / qT[gas_desc]
            
            boltz = np.exp(t_factor*gas_line_data.ELOWER)
            
            stim = ( 1 - np.exp(-Data.constants.c2_cgs*gas_line_data.NU/T)) / ( 1 - np.exp(-Data.constants.c2_cgs*gas_line_data.NU/Tref))
            
            line_strengths[gas_desc] = gas_line_data.SW * q_ratio * boltz * stim
            
            # Take into account abundances here?
        
        return line_strengths
    
    
    def calculate_line_absorption_separate_bins(
            self,
            delta_wn_edges : np.ndarray, # wavenumber difference from line center (cm^{-1}), (NWAVE+1) edges of wavenumber bin
            strength : float, # line strength
            alpha_d : float, # line doppler width (gaussinan HWHM)
            gamma_l : float, # line lorentz width (cauchy-lorentz HWHM)
            lineshape_fn : Callable[[np.ndarray, float], np.ndarray], # function that calculates line shape
            line_integral_points_delta_wn : np.ndarray,
            TEST : bool = False,
        ) -> np.ndarray:
        """
        Calculates line absorbtion at `delta_wn` wavenumber difference from line center
        for a `lineshape_fn`. Return absorption coefficient (cm^2 at `delta_wn` cm^{-1})
        """
        #_lgr.debug(f'{delta_wn_edges=}')
        
        # Get value of absorption coefficient at all integration points
        # do this here so we don't have to repeat the calculation of endpoints.
        ls_at_ip = lineshape_fn(line_integral_points_delta_wn, alpha_d, gamma_l)
        integral_ls_at_ip = sp.integrate.cumulative_simpson(ls_at_ip, x=line_integral_points_delta_wn, initial=0)
        
        #_lgr.debug(f'{line_integral_points_delta_wn=}')
        #_lgr.debug(f'{ls_at_ip=}')
        #_lgr.debug(f'{integral_ls_at_ip=}')
        
        # Assume symmetric so perform small fix
        integral_ls_at_ip += (1 - integral_ls_at_ip[-1])/2
        
        #_lgr.debug(f'after fix: {integral_ls_at_ip=}')
        
        
        wn_edges_int = np.interp(
            delta_wn_edges,
            line_integral_points_delta_wn,
            integral_ls_at_ip,
            left=0,
            right=1
        )
        #_lgr.debug(f'{cumulative_low=}')
        #_lgr.debug(f'{cumulative_high=}')
        
        lineshape = np.diff(wn_edges_int) / np.diff(delta_wn_edges) #/ (delta_wn_edges[:,1] - delta_wn_edges[:,0])
        #_lgr.debug(f'{lineshape=}')
        
        if TEST:
            fix, ax = plt.subplots(1,2, figsize=(12,12), squeeze=False)
            ax = ax.flatten()
        
            delta_wn_midpoints = 0.5*(delta_wn_edges[1:] + delta_wn_edges[:-1])
            _lgr.debug(f'{strength=}')
            _lgr.debug(f'{alpha_d=}')
            _lgr.debug(f'{gamma_l=}')
            _lgr.debug(f'{delta_wn_edges=}')
            _lgr.debug(f'{wn_edges_int=}')
            _lgr.debug(f'{delta_wn_midpoints=}')
            _lgr.debug(f'{lineshape=}')
            
            ax[0].set_title('lineshape and frac of lineshape in bin')
            ax[0].plot(
                line_integral_points_delta_wn,
                ls_at_ip,
                '.-',
                color='tab:blue',
                linewidth=1,
                alpha=0.6,
            )
            
            ax[0].plot(
                delta_wn_midpoints,
                lineshape,
                '.-',
                color='tab:orange',
                linewidth=1,
                alpha=0.6,
            )
            
            for x in delta_wn_edges:
                ax[0].axvline(
                    x, 
                    color='tab:red',
                    linewidth=1,
                    alpha=0.6,
                )
                
            ax[0].set_yscale('log')
            
            
            ax[1].set_title('Cumulative prob.')
            ax[1].plot(
                line_integral_points_delta_wn,
                integral_ls_at_ip,
                '.-',
                color='tab:blue',
                linewidth=1,
                alpha=0.6,
            )
            
            ax[1].plot(
                delta_wn_edges,
                wn_edges_int,
                '.-',
                color='tab:red',
                linewidth=1,
                alpha=0.6,
            )
            
            plt.show()
            
            raise RuntimeError("TESTING")
        
        #sys.exit()
        return strength * lineshape# / alpha_d # absorption coefficient (cm^2)
    
    
    def calculate_line_absorption_overlapping_bins(
            self,
            delta_wn_edges : np.ndarray, # wavenumber difference from line center (cm^{-1}), (NWAVE,2) edges of wavenumber bin
            strength : float, # line strength
            alpha_d : float, # line doppler width (gaussinan HWHM)
            gamma_l : float, # line lorentz width (cauchy-lorentz HWHM)
            lineshape_fn : Callable[[np.ndarray, float], np.ndarray], # function that calculates line shape
            line_integral_points_delta_wn : np.ndarray,
            TEST : bool = False,
        ) -> np.ndarray:
        """
        Calculates line absorbtion at `delta_wn` wavenumber difference from line center
        for a `lineshape_fn`. Return absorption coefficient (cm^2 at `delta_wn` cm^{-1})
        """
        #return strength* lineshape_fn(0.5*np.sum(delta_wn_edges,axis=0), alpha_d, gamma_l) # TESTING just using lineshape, no integral
        
        # Get value of absorption coefficient at all integration points
        # do this here so we don't have to repeat the calculation of endpoints.
        ls_at_ip = lineshape_fn(line_integral_points_delta_wn, alpha_d, gamma_l)
        integral_ls_at_ip = sp.integrate.cumulative_simpson(ls_at_ip, x=line_integral_points_delta_wn, initial=0)
        
        # Assume symmetric so perform small fix
        integral_ls_at_ip += (1 - integral_ls_at_ip[-1])/2
        
        wn_edges_int = np.interp(
            delta_wn_edges,
            line_integral_points_delta_wn,
            integral_ls_at_ip,
            left=0,
            right=1
        )
        
        lineshape = (wn_edges_int[1] - wn_edges_int[0]) / (delta_wn_edges[1]-delta_wn_edges[0])
        
        if TEST:
            fix, ax = plt.subplots(1,2, figsize=(12,12), squeeze=False)
            ax = ax.flatten()
        
            delta_wn_midpoints = 0.5*np.sum(delta_wn_edges, axis=0)
            _lgr.debug(f'{strength=}')
            _lgr.debug(f'{alpha_d=}')
            _lgr.debug(f'{gamma_l=}')
            _lgr.debug(f'{delta_wn_edges=}')
            _lgr.debug(f'{wn_edges_int.shape=}')
            _lgr.debug(f'{wn_edges_int=}')
            _lgr.debug(f'{delta_wn_midpoints=}')
            _lgr.debug(f'{(wn_edges_int[1] - wn_edges_int[0])=}')
            _lgr.debug(f'{(delta_wn_edges[1] - delta_wn_edges[0])=}')
            _lgr.debug(f'{lineshape=}')
            
            ax[0].set_title('lineshape and frac of lineshape in bin')
            ax[0].plot(
                line_integral_points_delta_wn,
                ls_at_ip,
                '.-',
                color='tab:blue',
                linewidth=1,
                alpha=0.6,
            )
            
            ax[0].plot(
                delta_wn_midpoints,
                lineshape,
                '.-',
                color='tab:orange',
                linewidth=1,
                alpha=0.6,
            )
            
            """
            for x in delta_wn_edges[0]:
                ax[0].axvline(
                    x, 
                    color='tab:red',
                    linewidth=0.5,
                    alpha=0.2,
                )
            for x in delta_wn_edges[1]:
                ax[0].axvline(
                    x, 
                    color='tab:red',
                    linestyle=':',
                    linewidth=0.5,
                    alpha=0.2,
                )
            """
            ax[0].set_yscale('log')
            
            
            ax[1].set_title('Cumulative prob.')
            ax[1].plot(
                line_integral_points_delta_wn,
                integral_ls_at_ip,
                '.-',
                color='tab:blue',
                linewidth=1,
                alpha=0.6,
            )
            
            ax[1].plot(
                delta_wn_edges[0],
                wn_edges_int[0],
                '.-',
                color='tab:red',
                linewidth=1,
                alpha=0.6,
            )
            
            ax[1].plot(
                delta_wn_edges[1],
                wn_edges_int[1],
                'x-',
                color='tab:red',
                linewidth=1,
                alpha=0.6,
            )
            
            plt.show()
            
            raise RuntimeError("TESTING")
        
        #sys.exit()
        return strength * lineshape# / alpha_d # absorption coefficient (cm^2)
    
    
    def calculate_absorption(
            self,
            waves : np.ndarray,
            temp: float, 
            press: float,
            amb_frac: float = 1, # fraction of ambient gas
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,
            lineshape_fn: Callable[[np.ndarray, float], np.ndarray] = Data.lineshapes.voigt,
            line_calculation_wavenumber_window: float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
            tref : float = 296,
            line_strength_cutoff : float = 1E-32,
            n_line_integral_points : int = 51,
            wn_bin : float = 1E-3,
            TEST : bool = False,
        ) -> dict[RadtranGasDescriptor, np.ndarray]:
        """
        Calculate total absorption coefficient (cm^2) for wavenumbers (cm^{-1}). Returns the value for a single molecule
        at the specified temperature, pressure, and ambient gas fraction.
        
        For details see "applications" section (at bottom) of https://hitran.org/docs/definitions-and-units/
        """
        warn_once = False
        info_once = False
        time_last = 0
        time_this = None
        n_line_progress = 1000
        
        abs_coeffs = dict()
        
        waves = WavePoint(waves, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value
        wav_sort_idxs = np.argsort(waves)
        waves = waves[wav_sort_idxs]
        
        _lgr.debug(f'{waves.size=} [{waves[0]}, ..., {waves[-1]}]')
        
        assert waves.size >= 3, "Must have at least three wavenumbers in wavenumber grid"
        
        wave_edges = np.empty((2,waves.size), dtype=float)
        if wn_bin < 0:
            wn_bin = (-1*wn_bin)*(waves[1:]-waves[:-1])*0.5
            wave_edges[0,:-1] = waves[:-1] - wn_bin
            wave_edges[0,-1] = waves[-1] - wn_bin[-1]
            wave_edges[1,1:] = waves[1:] + wn_bin
            wave_edges[1,0] = waves[0] + wn_bin[0]
        else:
            wave_edges[0] = waves - (wn_bin/2)
            wave_edges[1] = waves + (wn_bin/2)
        
        _lgr.debug(f'{wave_edges=}')
        
        lip_factor = 1/(np.e*np.log(2*line_calculation_wavenumber_window))
        lip_n1 = n_line_integral_points//2
        lip_n2 = n_line_integral_points - (lip_n1 + 1)
        min_step = lip_factor*(1+line_calculation_wavenumber_window/(2*n_line_integral_points))
        line_integral_points_delta_wn = np.zeros(n_line_integral_points, dtype=float)
        line_integral_points_delta_wn[:lip_n1] = -(np.geomspace(min_step, line_calculation_wavenumber_window+lip_factor, num=lip_n1) - lip_factor)[::-1]
        line_integral_points_delta_wn[lip_n1 + 1:] = np.geomspace(min_step, line_calculation_wavenumber_window+lip_factor, num=lip_n2) - lip_factor

        _lgr.debug(f'line_integral_points_delta_wn={line_integral_points_delta_wn}')
        #for i, x in enumerate(line_integral_points_delta_wn):
        #    _lgr.debug(f'\t{i} : {x}')
        #_lgr.debug(f'{line_integral_points_delta_wn=}')
        
        
        strengths = self.calculate_line_strength(temp, tref)
        alpha_ds = self.calculate_doppler_width(temp)
        gamma_ls = self.calculate_lorentz_width(press, temp, amb_frac, tref)
        
        k_total = np.zeros_like(waves, dtype=float)
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            _lgr.debug(f'Getting absorbtion coefficient for {i}^th gas {gas_desc=} (of {self.gas_isotopes.n_isotopes} gasses)')
            gas_line_data = self.line_data[gas_desc]
        
            abs_coeffs[gas_desc] = np.zeros_like(waves, dtype=float)
            
            k_total *= 0.0 # reset 
            
        
            strength = strengths[gas_desc]
            alpha_d = alpha_ds[gas_desc]
            gamma_l = gamma_ls[gas_desc]
            
            idx_min = np.min(np.nonzero(gas_line_data.NU >= (np.min(waves) - line_calculation_wavenumber_window)))
            idx_max = np.max(np.nonzero(gas_line_data.NU <= (np.max(waves) + line_calculation_wavenumber_window)))
            n_lines = idx_max - idx_min
            _lgr.debug(f'{idx_min=} {idx_max=} {n_lines=}')
            
            time_last = time.time()
            
            for line_idx in range(idx_min, idx_max):
                if ((line_idx-idx_min) % n_line_progress == 0):
                    if ((line_idx-idx_min)!=0):
                        time_this = time.time()
                        tuc = (n_lines - (line_idx-idx_min)) * (time_this - time_last)/(line_idx-idx_min) # time until completion
                        tuc_h = int(tuc // 3600)
                        tuc -= 3600 * tuc_h
                        tuc_m = int(tuc // 60)
                        tuc -= 60*tuc_m
                        _lgr.info(f'Computing contribution from line {line_idx-idx_min} / {n_lines}, {tuc_h:02}:{tuc_m:02}:{tuc:06.3f}(sec) remaining...')
                        #time_last = time_this
                
                if strength[line_idx] < line_strength_cutoff:
                    continue
            
                line_wn_mask = np.abs(waves - gas_line_data.NU[line_idx]) < line_calculation_wavenumber_window
                n_line_wn_mask = np.count_nonzero(line_wn_mask)
                 
                
                #line_wn_mask_idxs = np.nonzero(line_wn_mask)
                
                """
                if ((line_idx-idx_min) % n_line_progress == 0):
                    _lgr.debug(f'{n_line_wn_mask=}')
                    #_lgr.debug(f'{line_wn_mask_idxs=}')
                    _lgr.debug(f'{waves[line_wn_mask]=}')
                    _lgr.debug(f'{line_integral_points_delta_wn=}')
                """
                
                if n_line_wn_mask > 0:
                    # as edges are contiguous can do this
                    """
                    line_wn_mask_idxs = np.nonzero(line_wn_mask)[0]
                    wave_edges_idx_min, wave_edges_idx_max = np.min(line_wn_mask_idxs), np.max(line_wn_mask_idxs) + 2 # always add 2 as we are creating the limits of a slice
                
                    wn = wave_edges[wave_edges_idx_min:wave_edges_idx_max]
                    
                    k_total[line_wn_mask] += self.calculate_line_absorption_separate_bins(
                        (wn - gas_line_data.NU[line_idx]),
                        strength[line_idx],
                        alpha_d[line_idx],
                        gamma_l[line_idx],
                        lineshape_fn,
                        line_integral_points_delta_wn,
                        TEST = TEST and (np.abs(1/(2.371 * 1E-4) - gas_line_data.NU[line_idx]) < 0.001)
                    ) 
                    """
                    
                    wn = wave_edges[:, line_wn_mask]
                    
                    k_total[line_wn_mask] += self.calculate_line_absorption_overlapping_bins(
                        (wn - gas_line_data.NU[line_idx]),
                        strength[line_idx],
                        alpha_d[line_idx],
                        gamma_l[line_idx],
                        lineshape_fn,
                        line_integral_points_delta_wn,
                        TEST = TEST and (np.abs(1/(2.371 * 1E-4) - gas_line_data.NU[line_idx]) < 0.001)
                    ) 
                
                # Add in continuum absorption here if required
                if (10*gamma_l[line_idx]) > line_calculation_wavenumber_window:
                    if not warn_once:
                        _lgr.warning(f'Continuum calculation not implemented yet, increase `line_calculation_wavenumber_window` until >= 10 * `gamma_l`.  (10*gamma_l = {10*gamma_l[line_idx]})')
                        warn_once = True
                else:
                    if not info_once:
                        _lgr.info(f'No continuum calculation performed as `line_calculation_wavenumber_window` >= 10 * `gamma_l`. ({line_calculation_wavenumber_window} >= {10*gamma_l[line_idx]})')
                        info_once = True
                
                
            # put into absorption coefficient dictionary
            abs_coeffs[gas_desc][wav_sort_idxs] = k_total
        
        return abs_coeffs
    
    
    
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











