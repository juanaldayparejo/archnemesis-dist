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
# This file is part of ans.
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
from typing import Any, Callable, TYPE_CHECKING
# Third party
import numpy as np
import scipy as sp
import scipy.integrate
import matplotlib.pyplot as plt
# This package
from archnemesis import Data
#from archnemesis import *
import archnemesis as ans
import archnemesis.enums

#import archnemesis.helpers.maths_helper as maths_helper
from archnemesis.helpers.io_helper import SimpleProgressTracker
import archnemesis.database
import archnemesis.database.line_database.hitran
import archnemesis.database.partition_function_database.hitran
from archnemesis.database.filetypes.lbltable import LblDataTProfilesAtPressure#, LblDataTPGrid
from archnemesis.database.datatypes.wave_point import WavePoint
from archnemesis.database.datatypes.wave_range import WaveRange
from archnemesis.database.datatypes.gas_isotopes import GasIsotopes
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    NWAVE = "Number of wave points"

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
            ambient_gas=ans.enums.AmbientGas.AIR,
            LINE_DATABASE : None | ans.database.protocols.LineDatabaseProtocol = None,
            PARTITION_FUNCTION_DATABASE : None | ans.database.protocols.PartitionFunctionDatabaseProtocol = None,
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
        
        @attribute LINE_DATABASE: None | ans.database.protocols.LineDatabaseProtocol
            Instance of a class that implements the `ans.database.protocols.LineDatabaseProtocol` protocol.
            If `None` will use HITRAN as backend.
        
        @attribute PARTITION_FUNCTION_DATABASE: None | archnemsis.database.protocols.PartitionFunctionDatabaseProtocol
            Instance of a class that implements the `archnemsis.database.protocols.PartitionFunctionDatabaseProtocol` protocol.
            If `None` will use HITRAN as backend.
        

        Attributes
        ----------
        
        @attribute line_data : None | archnemsis.database.protocols.LineDataProtocol
            An object (normally a numpy record array) that implements the `archnemsis.database.protocols.LineDataProtocol`
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
        
        @attribute partition_data : None | archnemsis.database.protocols.PartitionFunctionDataProtocol
            An object (normally a numpy record array) that implements the `archnemsis.database.protocols.PartitionFunctionDataProtocol`
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
        
        self._ambient_gas = ambient_gas
        self._line_database = None
        self._partition_function_database = None
        
        self.ID = ID
        self.ISO = ISO
        self.LINE_DATABASE = LINE_DATABASE
        self.PARTITION_FUNCTION_DATABASE = PARTITION_FUNCTION_DATABASE
        
        self.line_data = None
        self.partition_data = None
        
    ##################################################################################

    @property
    def LINE_DATABASE(self) -> ans.database.protocols.LineDatabaseProtocol:
        if self._line_database is None:
            raise RuntimeError('No line database attached to LineData_0 instance')
        return self._line_database

    @LINE_DATABASE.setter
    def LINE_DATABASE(self, value : ans.database.protocols.LineDatabaseProtocol):
        if value is None:
            db = ans.database.line_database.hitran.HITRAN()
            _lgr.info(f'Using default line database {db}')
            self._line_database = db
        else:
            self._line_database = value
    
    @property
    def PARTITION_FUNCTION_DATABASE(self) -> ans.database.protocols.PartitionFunctionDatabaseProtocol:
        if self._partition_function_database is None:
            raise RuntimeError('No partition function database attached to LineData_0 instance')
        return self._partition_function_database

    @PARTITION_FUNCTION_DATABASE.setter
    def PARTITION_FUNCTION_DATABASE(self, value : ans.database.protocols.PartitionFunctionDatabaseProtocol):
        if value is None:
            db = ans.database.partition_function_database.hitran.HITRAN()
            _lgr.info(f'Using default partition function database {db}')
            self._partition_function_database = db
        else:
            self._partition_function_database = value

    @property
    def ambient_gas(self) -> ans.enums.AmbientGas:
        return self._ambient_gas

    @ambient_gas.setter
    def ambient_gas(self, value : int | ans.enums.AmbientGas):
        self._ambient_gas = ans.enums.AmbientGas(value)
    
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

        assert self.ambient_gas in ans.enums.AmbientGas, \
            f"ambient_gas must be one of {tuple(ans.enums.AmbientGas)}"

        if self.ID < 1:
            raise ValueError(f"ID must be greater than 0, got {self.ID}")
        if self.ISO < 0:   
            raise ValueError(f"ISO must be greater than or equal to 0, got {self.ISO}")
        
    ###########################################################################################################################
    
    def is_line_data_ready(self) -> bool:
        """
        """
        if self.line_data is None:
            return False
        return True
    
    def fetch_linedata(
            self, 
            vmin : float , 
            vmax : float, 
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,
            refresh : bool = False, 
    ) -> None:
        """
        Fetch the line data from the specified database, if `refresh` then get the data even if we already have some.
        NOTE: Does not check that the data we already have is valid for the wavelength range (vmin, vmax), only
        checks if `self.is_line_data_ready()` is True or False.
        
        # ARGUMENTS #
            vmin : float
                Minimum wavenumber to get line data for
            vmax : float
                Maximum wavenumber to get line data for
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm
                Unit of `vmin` and `vmax`, default is wavenumbers in cm^{-1}.
            refresh : bool = False
                If True will retrieve data from database again, even if data
                is already present. NOTE: this will not neccessarily trigger a download
                as the database may cache data.
        """
        assert vmin < vmax, f'Mimimum wave ({vmin}) must be less than maximum wave ({vmax})'
        
        # Turn wavelength range in to Wavenumbers cm^{-1} for internal use
        wave_range = WaveRange(vmin, vmax, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm)
        
    
        if refresh or not self.is_line_data_ready():
            self.line_data = self.LINE_DATABASE.get_line_data(
                self.gas_isotopes.as_radtran_gasses(), 
                wave_range,
                self.ambient_gas, 
            )
            _lgr.info(f'Retrieved line data from database {self.LINE_DATABASE}')
        else:
            _lgr.info('Line data already loaded')
        
        for gas_desc,v in self.line_data:
            if v is None:
                _lgr.warning(f'Database does not recognise gas {gas_desc} returned `None` for line_data, it should not be included further calculations. If strange things happen, try not including this isotope.')
        
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
            self.partition_data = self.PARTITION_FUNCTION_DATABASE.get_partition_function_data(
                self.gas_isotopes.as_radtran_gasses()
            )
            _lgr.info(f'Retrieved partition function data from database {self.PARTITION_FUNCTION_DATABASE}')
        else:
            _lgr.info('Partition function data already loaded')
    
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
        m_0 - mass of a single molecule
        c - speed of light
        M_0 - molecular mass (mass per mole of molecules)
        N_A - avogadro's constant
        
        """
        doppler_width_const_cgs : float  = (1.0 / Data.constants.c_light_cgs) * np.sqrt(2 * np.log(2) * Data.constants.N_avogadro * Data.constants.k_boltzmann_cgs)
        _lgr.debug(f'{doppler_width_const_cgs=}')
        dws = dict() # result
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            gas_line_data = self.line_data[gas_desc]
            if gas_line_data is None:
                dws[gas_desc] = np.nan
                continue
            
            dws[gas_desc] = doppler_width_const_cgs * gas_line_data.NU * np.sqrt( temp / gas_desc.molecular_mass)
        
        _lgr.debug(dws)
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
        
        tratio = tref/temp
        
        ONCE_FLAG=True
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            gas_line_data = self.line_data[gas_desc]
            
            
            if (_lgr.level == logging.DEBUG) and ONCE_FLAG:
                _lgr.debug(f'{gas_line_data.N_AMB=}')
                _lgr.debug(f'{gas_line_data.GAMMA_AMB=}')
                _lgr.debug(f'{gas_line_data.N_SELF=}')
                _lgr.debug(f'{gas_line_data.GAMMA_SELF=}')
                ONCE_FLAG=False
            
            if gas_line_data is None:
                lws[gas_desc] = np.nan
                continue
            
            lws[gas_desc] = (
                (tratio**gas_line_data.N_AMB) * gas_line_data.GAMMA_AMB * amb_frac 
                 + (tratio**gas_line_data.N_SELF) * gas_line_data.GAMMA_SELF *  (1-amb_frac)
            ) * press
            
        
        _lgr.debug(lws)
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
            if gas_partition_data is None:
                QTs[gas_desc] = np.nan
                continue
            
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
            if gas_line_data is None:
                line_strengths[gas_desc] = np.zeros((0,),dtype=float)
                continue
            
            # Partition function ratio
            q_ratio = qTref[gas_desc] / qT[gas_desc]
            
            boltz = np.exp(t_factor*gas_line_data.ELOWER)
            
            stim = ( 1 - np.exp(-Data.constants.c2_cgs*gas_line_data.NU/T)) / ( 1 - np.exp(-Data.constants.c2_cgs*gas_line_data.NU/Tref))
            
            line_strengths[gas_desc] = gas_line_data.SW * q_ratio * boltz * stim
            
            # Take into account abundances here?
        
        return line_strengths
    

    
    def calculate_line_absorption_in_bins(
            self,
            delta_wn_edges : np.ndarray, # wavenumber bin edges difference from line center (cm^{-1}), (2,NWAVE) low, high
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
    
    
    def calculate_absorption_in_bins(
            self,
            waves : np.ndarray, # wavenumber midpoints (if 1D with shape [NWAVE], will calculate bin edges from difference between values), wavenumber bin edges (if 2D with shape [2,NWAVE], index 0 are "left" edges, index 1 are "right" edges)
            temp: float, # Kelvin
            press: float, # Atmospheres
            amb_frac: float = 1, # fraction of ambient gas
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm, # Unit of input `waves`
            lineshape_fn: Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, #  lineshape function to use, will be evaluated at `n_line_integral_points`
            line_calculation_wavenumber_window: float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
            tref : float = 296, # Reference temperature (Kelvin). TODO: This should be set by the database used
            line_strength_cutoff : float = 1E-32, # Strength below which a line is ignored.
            n_line_integral_points : int = 51, # Number of points to use in lineshape integral
            wn_bin : float = -1.0, # Wavenumber (cm^{-1}) bin size. If negative, the bin size will be multiplied by the absolute value (when `waves` is 1-dimensional will calculate bin edges and then apply this scaling). If positive bin size will be set to this value.
            TEST : bool = False, # FOR DEBUGGING: Set to true to turn on testing, at the moment will display plots of the lineshape calculation
            TEST_PREDICATE : Callable[[int,float],bool] = lambda line_index, line_wavenumber : (np.abs(1/(2.371 * 1E-4) - line_wavenumber) < 0.001), # FOR DEBUGGING: Callable that must return True for testing to happen. First argument is the current line index, second is the current line wavenumber. This is here so we can choose which line to apply tests to
            ensure_linedata_downloaded : bool = True,
    ) -> dict[RadtranGasDescriptor, np.ndarray]:
        """
        Calculate total absorption coefficient (cm^2) for wavenumbers (cm^{-1}) multiplied by a factor of 1E20. 
        Returns the value for a single molecule at the specified temperature, pressure, and ambient gas fraction.
        
        For details see "applications" section (at bottom) of https://hitran.org/docs/definitions-and-units/
        """
        INFO_ONCE_FLAG = False
        n_line_progress = 1000
        
        abs_coeffs = dict()
        
        # Convert waves to wavenumber (cm^{-1})
        waves = np.array(WavePoint(waves, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value)
        
        
        
        if waves.ndim == 1:
            _lgr.info(f'Input waves ({waves.shape=}) are midpoints, calculating wave bin edges...')
            wave_midpoints = np.array(waves)
            wave_edges = np.empty((2,wave_midpoints.size), dtype=float)
            wave_edges[0,1:] =  (wave_midpoints[1:]+wave_midpoints[:-1])*0.5 # "Left" bin edges are midpoints except for the first one
            wave_edges[1,:-1] =  (wave_midpoints[1:]+wave_midpoints[:-1])*0.5 # "Right" bin edges are midpoints except for the last one
            
            wave_edges[0,0] = 2*wave_midpoints[0] - wave_edges[1,0] # First "Left" bin edge is guessed to be the same distance from midpoint as first "Right" bin edge
            wave_edges[1,-1] = 2*wave_midpoints[-1] - wave_edges[0,-1] # Last "Right" bin edge is guessed to be the same distance from midpoint as last "Left" bin edge
            
        elif waves.ndim == 2:
            if waves.shape[0] != 2:
                raise ValueError(f'`waves` argument must be either 1-dimensional (interpreted as wave-point midpoints) or 2-dimensional (interpreted as wave-point bin edges). If 2-dimensional must have shape (2,NWAVE). Current `waves` has {waves.ndim=}, {waves.shape=}')
            _lgr.info(f'Input waves ({waves.shape=}) are edges, calculating wave bin midpoints...')
            wave_midpoints = 0.5*np.sum(waves, axis=0)
            wave_edges = np.array(waves)
        else:
            raise ValueError(f'`waves` argument must be either 1-dimensional (interpreted as wave-point midpoints) or 2-dimensional (interpreted as wave-point bin edges). If 2-dimensional must have shape (2,NWAVE). Current `waves` has {waves.ndim=}, {waves.shape=}')
        
        assert wave_midpoints.size >= 3, "Must have at least three wavenumbers in wavenumber grid"
        
        # Sort the wavenumber bins so their midpoints are in ascending order
        # NOTE: `wav_sort_idxs` is used later to put the results into the same order
        # as the input `waves` array.
        wav_sort_idxs = np.argsort(wave_midpoints)
        wave_midpoints = wave_midpoints[wav_sort_idxs]
        wave_edges[0] = wave_edges[0,wav_sort_idxs]
        wave_edges[1] = wave_edges[1,wav_sort_idxs]
        
        
        if wn_bin != -1.0:
            _lgr.debug(f'Applying {wn_bin=}')
            if wn_bin > 0:
                wave_edges[0] = wave_midpoints - wn_bin/2
                wave_edges[1] = wave_midpoints + wn_bin/2
            else:
                wave_edges = wave_midpoints + (wave_midpoints - wave_edges)*wn_bin
        else:
            _lgr.debug(f'{wn_bin=}, no need to alter passed/calculated bins')
        
        
        if wave_midpoints.size == 3:
            _lgr.debug(f'{wave_midpoints.size=} [{wave_midpoints[0]:012.6E}, {wave_midpoints[1]:012.6E}, {wave_midpoints[2]:012.6E}]')
            _lgr.debug(f'"Left" edges : [({wave_edges[0,0]:012.6E}, {wave_edges[0,1]:012.6E}, {wave_edges[0,2]:012.6E}]')
            _lgr.debug(f'"Right" edges: [({wave_edges[1,0]:012.6E}, {wave_edges[1,1]:012.6E}, {wave_edges[1,2]:012.6E}]')
        elif wave_midpoints.size == 4:
            _lgr.debug(f'{wave_midpoints.size=} [{wave_midpoints[0]:012.6E}, {wave_midpoints[1]}, {wave_midpoints[2]:012.6E}, {wave_midpoints[3]:012.6E}]')
            _lgr.debug(f'"Left" edges : [({wave_edges[0,0]:012.6E}, {wave_edges[0,1]:012.6E}, {wave_edges[0,2]:012.6E}, {wave_edges[0,3]:012.6E}]')
            _lgr.debug(f'"Right" edges: [({wave_edges[1,0]:012.6E}, {wave_edges[1,1]:012.6E}, {wave_edges[1,2]:012.6E}, {wave_edges[1,3]:012.6E}]')
        else:
            _lgr.debug(f'{wave_midpoints.size=} [{wave_midpoints[0]:012.6E}, ..., {wave_midpoints[wave_midpoints.size//2]:012.6E}, ..., {wave_midpoints[-1]:012.6E}]')
            _lgr.debug(f'"Left" edges : [({wave_edges[0,0]:012.6E}, ..., {wave_edges[0,wave_midpoints.size//2]:012.6E}, ..., {wave_edges[0,-1]:012.6E}]')
            _lgr.debug(f'"Right" edges: [({wave_edges[1,0]:012.6E}, ..., {wave_edges[1,wave_midpoints.size//2]:012.6E}, ..., {wave_edges[1,-1]:012.6E}]')
            
        
        self.fetch_linedata(vmin = wave_edges[0,0], vmax = wave_edges[1,-1], refresh=ensure_linedata_downloaded)
        
        _lgr.info('Getting lineshape integral points...')
        lip_factor = 1/(np.e*line_calculation_wavenumber_window)
        lip_n1 = n_line_integral_points//2
        lip_n2 = n_line_integral_points - (lip_n1 + 1)
        min_step = lip_factor*(1+line_calculation_wavenumber_window/(2*n_line_integral_points))
        line_integral_points_delta_wn = np.zeros(n_line_integral_points, dtype=float)
        line_integral_points_delta_wn[:lip_n1] = -(np.geomspace(min_step, line_calculation_wavenumber_window+lip_factor, num=lip_n1) - lip_factor)[::-1]
        line_integral_points_delta_wn[-lip_n2:] = np.geomspace(min_step, line_calculation_wavenumber_window+lip_factor, num=lip_n2) - lip_factor

        _lgr.debug(f'line_integral_points_delta_wn={line_integral_points_delta_wn}')
        
        strengths = self.calculate_line_strength(temp, tref)
        alpha_ds = self.calculate_doppler_width(temp)
        gamma_ls = self.calculate_lorentz_width(press, temp, amb_frac, tref)
        
        k_total = np.zeros_like(wave_midpoints, dtype=float) # define here and re-use the memory
        
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            _lgr.debug(f'Getting absorbtion coefficient for {i}^th gas {gas_desc=} (of {self.gas_isotopes.n_isotopes} gasses)')
            gas_line_data = self.line_data[gas_desc]
        
            if gas_line_data is None:
                abs_coeffs[gas_desc] = np.zeros_like(wave_midpoints, dtype=float) # All zeros for gasses that are not present in database
                continue
            else:
                # need a new array for each gas, need to define it before using as want to place values in in a specific order
                abs_coeffs[gas_desc] = np.zeros_like(wave_midpoints, dtype=float) 
            
            k_total *= 0.0 # reset 
            
        
            strength = strengths[gas_desc]
            alpha_d = alpha_ds[gas_desc]
            gamma_l = gamma_ls[gas_desc]
            
            mask_lines_above_min_wave= gas_line_data.NU >= (wave_edges[0,0] - line_calculation_wavenumber_window)
            mask_lines_below_max_wave = gas_line_data.NU <= (wave_edges[1,-1] - line_calculation_wavenumber_window)
            line_idxs_to_include = np.nonzero(mask_lines_above_min_wave & mask_lines_below_max_wave)
            idx_min = np.min(line_idxs_to_include)
            idx_max = np.max(line_idxs_to_include)
            n_lines = idx_max - idx_min
            _lgr.debug(f'{idx_min=} {idx_max=} {n_lines=}')
            
            progress_tracker = SimpleProgressTracker(n_lines, f"Computing line contributions from {gas_desc}. ", 5, target_logger=_lgr)
            
            for _j, line_idx in enumerate(range(idx_min, idx_max)):
                if _j % n_line_progress == 0:
                    progress_tracker.n = _j
                    progress_tracker.log_at(logging.INFO)
                
                if strength[line_idx] < line_strength_cutoff:
                    continue
            
                line_wn_mask = np.abs(wave_midpoints - gas_line_data.NU[line_idx]) < line_calculation_wavenumber_window
                n_line_wn_mask = np.count_nonzero(line_wn_mask)

                
                if n_line_wn_mask > 0:
                    delta_wn_edges = wave_edges[:, line_wn_mask] - gas_line_data.NU[line_idx]
                    
                    k_total[line_wn_mask] += self.calculate_line_absorption_in_bins(
                        delta_wn_edges,
                        strength[line_idx],
                        alpha_d[line_idx],
                        gamma_l[line_idx],
                        lineshape_fn,
                        line_integral_points_delta_wn,
                        TEST = TEST and TEST_PREDICATE(line_idx, gas_line_data.NU[line_idx])
                    ) 
                
                # Add in continuum absorption here if required
                if not INFO_ONCE_FLAG:
                    _lgr.info('NOTE: Continuum absorbtion is handled in the lineshape calculation for now, if required may want to separate it for efficiency')
                    INFO_ONCE_FLAG = True
            
            # Ensure abs coeff are not less than zero
            k_total[k_total<0] = 0
            
            # put into absorption coefficient dictionary, multiply by 1E20 factor here
            abs_coeffs[gas_desc][wav_sort_idxs] = k_total*1E20
        
        return abs_coeffs
    
    
    def calculate_monochromatic_line_absorption(
            self,
            delta_wn : np.ndarray, # wavenumber difference from line center (cm^{-1}), (NWAVE)
            mask : np.ndarray, # Boolean mask (NWAVE) that determines if absorption coefficient will be calculated from lineshape (True) or from a 1/(delta_wn^2) fit (False).
            wide_mask : np.ndarray, # Boolean mask (NWAVE) that determines if absorption coefficient will be at all.
            strength : float, # line strength
            alpha_d : float, # line doppler width (gaussinan HWHM)
            gamma_l : float, # line lorentz width (cauchy-lorentz HWHM)
            lineshape_fn : Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, # function that calculates line shape
            line_calculation_wavenumber_window : float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption, should be the same as the window used to make `mask`
            out : np.ndarray | None = None, # if not None, will place the result into this array and return it, otherwise will create an array and return it.
    ) -> np.ndarray:
    
        # At `line_calculation_wavenumber_window` the lineshape_fn and 1/x^2 fit should be equal
        # requires that`line_calculation_wavenumber_window` was applied symmetrically around zero.
        constant = lineshape_fn(np.array([line_calculation_wavenumber_window]), alpha_d, gamma_l) * (line_calculation_wavenumber_window**2)
        
        if out is None:
            out = np.empty_like(delta_wn)
        
        out[wide_mask] = strength * constant / (delta_wn[wide_mask]**2) # calculate "continuum" contribution for the wide mask
        out[mask] = strength * lineshape_fn(delta_wn[mask], alpha_d, gamma_l) # overwrite the "center" with lineshape contribution
        return out
    
    
    def calculate_monochromatic_absorption(
            self,
            waves : np.ndarray, # 1D array with shape [NWAVE]
            temp : float, # kelvin
            press : float, # Atmospheres
            amb_frac : float = 1, # fraction of broadening due to ambient gas
            wave_unit : ans.enums.waveUnit = ans.enums.WaveUnit.Wavenumber_cm,  # unit of `waves` argument
            lineshape_fn : Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, # lineshape function to use
            line_calculation_wavenumber_window: float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
            tref : float = 296, # Reference temperature (Kelvin). TODO: This should be set by the database used
            line_strength_cutoff : float = 1E-32, # Strength below which a line is ignored.
            ensure_linedata_downloaded : bool = True,
    ) -> dict[RadtranGasDescriptor, np.ndarray]:
        """
        Calculate total absorption coefficient (cm^2) for wavenumbers (cm^{-1}) multiplied by a factor of 1E20. 
        Returns the value for a single molecule at the specified temperature, pressure, and ambient gas fraction.
        
        For details see "applications" section (at bottom) of https://hitran.org/docs/definitions-and-units/
        """
        INFO_ONCE_FLAG = False
        n_line_progress = 1000
        
        abs_coeffs = dict()
        
        # Convert waves to wavenumber (cm^{-1})
        #in_waves = np.array(waves)
        waves = np.array(WavePoint(waves, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value)
        
        # Remember the ordering we got as input
        wav_sort_idxs = np.argsort(waves)
        
        waves = waves[wav_sort_idxs]
        
        # Download data if we need it
        self.fetch_linedata(
            vmin = waves[0]-line_calculation_wavenumber_window, 
            vmax = waves[-1]+line_calculation_wavenumber_window, 
            refresh=ensure_linedata_downloaded
        )
        
        strengths = self.calculate_line_strength(temp, tref)
        alpha_ds = self.calculate_doppler_width(temp)
        gamma_ls = self.calculate_lorentz_width(press, temp, amb_frac, tref)
        
        # Define arrays here to re-use memory
        delta_wn = np.zeros_like(waves, dtype=float)
        scratch = np.zeros_like(waves, dtype=float)
        mask_leq = np.zeros_like(waves, dtype=bool)
        mask_geq = np.zeros_like(waves, dtype=bool)
        mask = np.zeros_like(waves, dtype=bool)
        wide_mask_leq = np.zeros_like(waves, dtype=bool)
        wide_mask_geq = np.zeros_like(waves, dtype=bool)
        wide_mask = np.zeros_like(waves, dtype=bool)
        k_total = np.zeros_like(waves, dtype=float)
        
        
        for i, gas_desc in enumerate(self.gas_isotopes.as_radtran_gasses()):
            _lgr.debug(f'Getting absorbtion coefficient for {i}^th gas {gas_desc=} (of {self.gas_isotopes.n_isotopes} gasses)')
            gas_line_data = self.line_data[gas_desc]
        
            if gas_line_data is None:
                abs_coeffs[gas_desc] = np.zeros_like(waves, dtype=float) # All zeros for gasses that are not present in database
                continue
            else:
                # need a new array for each gas, need to define it before using as want to place values in in a specific order
                abs_coeffs[gas_desc] = np.zeros_like(waves, dtype=float) 
            
            k_total.fill(0.0) # reset to all zeros
            
        
            strength = strengths[gas_desc]
            alpha_d = alpha_ds[gas_desc]
            gamma_l = gamma_ls[gas_desc]
            
            line_idxs_to_include = np.nonzero((waves[0] <= gas_line_data.NU) & (gas_line_data.NU <= waves[-1]))
            idx_min = np.min(line_idxs_to_include)
            idx_max = np.max(line_idxs_to_include)
            n_lines = idx_max - idx_min
            _lgr.debug(f'{idx_min=} {idx_max=} {n_lines=}')
            
            progress_tracker = SimpleProgressTracker(n_lines, f"Computing line contributions from {gas_desc}. ", 5, target_logger=_lgr)
            
            for _j, line_idx in enumerate(range(idx_min, idx_max)):
                if _j % n_line_progress == 0:
                    progress_tracker.n = _j
                    progress_tracker.log_at(logging.INFO)
                
                if strength[line_idx] < line_strength_cutoff:
                    continue
            
                np.subtract(waves, gas_line_data.NU[line_idx], out=delta_wn)
                
                np.less_equal(delta_wn, line_calculation_wavenumber_window, out=mask_leq)
                np.greater_equal(delta_wn, -line_calculation_wavenumber_window, out=mask_geq)
                np.logical_and(mask_leq, mask_geq, out=mask)
                
                np.less_equal(delta_wn, 10*line_calculation_wavenumber_window, out=wide_mask_leq)
                np.greater_equal(delta_wn, -10*line_calculation_wavenumber_window, out=wide_mask_geq)
                np.logical_and(mask_leq, mask_geq, out=wide_mask)

                self.calculate_monochromatic_line_absorption(
                    delta_wn,
                    mask, 
                    wide_mask,
                    strength[line_idx],
                    alpha_d[line_idx],
                    gamma_l[line_idx],
                    lineshape_fn,
                    line_calculation_wavenumber_window,
                    out=scratch
                )
                
                k_total = np.add(k_total, scratch, out=k_total)
                
                # Add in continuum absorption here if required
                if not INFO_ONCE_FLAG:
                    _lgr.info('NOTE: Continuum absorbtion is handled in the lineshape calculation for now, if required may want to separate it for efficiency')
                    INFO_ONCE_FLAG = True
            
            # Ensure abs coeff are not less than zero
            k_total[k_total<0] = 0

            # put into absorption coefficient dictionary, multiply by 1E20 factor here
            abs_coeffs[gas_desc][wav_sort_idxs] = k_total*1E20
        
        return abs_coeffs
        
        
    
    def calculate_absorption_at_temp_pressure_profile(
            self,
            waves : np.ndarray,
            temp_profile : np.ndarray, # Kelvin
            pressure_profile : np.ndarray, # Atmospheres
            delta_temp : float = 5, # Kelvin. Temperature differential
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavelength_um, # Unit of input `waves`
            wn_bin : float = 0.0, # Wavenumber (cm^{-1}) bin size. If zero will perform monochromatic calculation, if -ve will multiply `waves` spacing by absolute value to get bin size, if +ve specifies an exact bin size.
            **kwargs # Other keyword arguments passed through to `self.calculate_absorption(...)`
    ) -> tuple[LblDataTProfilesAtPressure,...]:
        
        # Need to check that `waves` is evenly spaced and monotonically increasing
        if waves.ndim == 1:
            mid_w = waves
            delta_w = np.diff(waves)
        elif (waves.ndim == 2) and (waves.shape[0] == 2):
            mid_w = 0.5*np.sum(waves, axis=0)
            delta_w = waves
        else:
            raise ValueError(f'`waves` must have one of the following shapes: (NWAVE,) or (2,NWAVE). But {waves.shape=}')
        
        assert np.all(delta_w == delta_w[0]), "`wave` must be an evenly spaced grid"
        
        # Are we performing a monochromatic calculation or a binned calculation
        monochromatic_flag = True if (np.abs(wn_bin) <= 1E-30) else False
        
        # If we are performing a monochromatic calculation, `waves` must only have 1 dimension
        if monochromatic_flag:
            assert waves.ndim == 1, "Monochromatic calculation can only be perfomed when `waves` has a single dimension"
        
        # Define the function that will calculate the absorption
        if monochromatic_flag:
            _lgr.info('Calculating monochromatic absorption...')
            calc_absorption_fn = self.calculate_monochromatic_absorption
        else:
            _lgr.info('Calculating absorption in bins...')
            calc_absorption_fn = lambda *args, **kwargs: self.calculate_absorption_in_bins(*args, wn_bin=wn_bin, **kwargs)
        
        # Get the number of wave points or wave bins we are working with
        n_waves = waves.size if waves.ndim == 1 else waves.shape[1]
        
        # Get temperature points
        temp_points = np.empty((temp_profile.size,2), dtype=float)
        temp_points[:,0] = temp_profile - delta_temp
        temp_points[:,1] = temp_profile + delta_temp
        
        # Get gas descriptors
        gas_descs = tuple(self.gas_isotopes.as_radtran_gasses())
        
        # Allocate result array
        k_total = np.zeros((len(gas_descs), n_waves, pressure_profile.size, 2), dtype=float)
        
        progress_tracker = SimpleProgressTracker(temp_points.size, "Computing absorption coefficients of temperature-pressure profile. ", target_logger=_lgr)
        
        for i, press in enumerate(pressure_profile):
            for j, temp in enumerate(temp_points[i]):
                progress_tracker.log_at_and_increment(logging.INFO)
                #_lgr.info(f'Calculating absorption at temperature-pressure profile point ({i},{j}) of ({pressure_profile.size},{temp_points.shape[1]}). Progress: {i*2+j} / {temp_points.size} [{100.0*(i*2+j)/temp_points.size: 6.2f} %]')
                abs_coeffs = calc_absorption_fn(
                    waves, 
                    temp, 
                    press, 
                    wave_unit = wave_unit,
                    **kwargs
                )
                for k, v in abs_coeffs.items():
                    x = gas_descs.index(k)
                    k_total[x, :, i, j] = v
                    
                    """
                    if (temp > 100) and (press > 7):
                        plt.title(f'{press=}\n{temp=}')
                        plt.plot(waves, v)
                        plt.xlim((1.64,1.68))
                        plt.show()
                        sys.exit()
                    """
        
        result = []
        
        if self.ISO == 0: # In this case, we can put all of the absorption coefficients into a single file
            result.append(
                LblDataTProfilesAtPressure(
                    self.ID,
                    self.ISO,
                    wave_unit,
                    mid_w,
                    pressure_profile,
                    temp_points,
                    np.sum(k_total, axis=0) # add absorption coefficients of different gas isotopes for the combined file
                )
            )
        else: # Otherwise we make a separate file for each gas isotope
            for x, gas_desc in enumerate(gas_descs):
                result.append(
                    LblDataTProfilesAtPressure(
                        gas_desc.gas_id,
                        gas_desc.iso_id,
                        wave_unit,
                        mid_w,
                        pressure_profile,
                        temp_points,
                        k_total[x]
                    )
                )
        
        return result
        
        
        
    
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
        combined_ax.set_title('Line data coloured by isotopologue')
        
        line_strengths_max = 0
        for i, gas_desc in enumerate(gas_isotopes.as_radtran_gasses()):
            gas_linedata = self.line_data[gas_desc]
            line_strength_mask = gas_linedata.SW >= smin
            wavenumbers = gas_linedata.NU[line_strength_mask]
            line_strengths = gas_linedata.SW[line_strength_mask]
            
            ls_max = line_strengths.max()
            line_strengths_max = ls_max if ls_max > line_strengths_max else line_strengths_max
            
            # Combined plot, all isotopes on one figure, coloured by isotope
            combined_ax.scatter(
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











