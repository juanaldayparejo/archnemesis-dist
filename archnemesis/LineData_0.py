

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

from archnemesis.database.filetypes.ans_line_data_file import AnsLineDataFile
from archnemesis.database.filetypes.ans_partition_fn_data_file import AnsPartitionFunctionDataFile


# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    NWAVE = "Number of wave points"

# TODO: 
# * Account for HITRAN weighting by terrestrial abundances .
#   NOTE: do this correction in the HITRAN.py file, not here as ideally LineData_0 would give 
#         line strengths and absorption coefficients per unit of gas (e.g. per column density,
#         per gram, per mole, something like that) the exact unit to be worked out later.
# * Natural Broadening
# * Pressure shift
# * Multiple ambient gasses. At the moment can only have one, but should be able to define
#   a gas mixture.
# * Additional lineshapes, have a look at https://www.degruyterbrill.com/document/doi/10.1515/pac-2014-0208/html

# TODO: Instead of dictionaries change `line_data` and `partition_data` to be available as big arrays with `gas_id` and `iso_id` fields
# or maybe just `iso_id` fields as these are only for one gas at a time.
# 
# Either `partition_data` is combined into the giant array, or it can be in a seprate array and looked up based on `gas_id` and `iso_id`.
#
# Only partition functions and abundances are isotope dependent so after those are accounted for, one large array is a perfectly good
# structure to have the data in.

# TODO: Speed up molecular mass lookup for each isotopologue

if TYPE_CHECKING:
    N_LINES_OF_GAS = 'Number of lines for a gas isotopologue'
    N_TEMPS_OF_GAS = 'Number of temperature points for a gas isotopologue'


class LineData_0:
    """
    Clear class for storing line data.
    """
    def __init__(
            self, 
            ID: int = 1,
            ISO: int = 0,
            ambient_gas=ans.enums.AmbientGas.AIR,
            LINE_DATABASE : None | str = None,
            PARTITION_FUNCTION_DATABASE : None | str = None,
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
        
        @attribute LINE_DATABASE: None | str
            String indicating the path of the line database to use (it must be an archNEMESIS HDF5 line database).
        
        @attribute PARTITION_FUNCTION_DATABASE: None | archnemsis.database.protocols.PartitionFunctionDatabaseProtocol
            Instance of a class that implements the `archnemsis.database.protocols.PartitionFunctionDatabaseProtocol` protocol.
            If `None` will use HITRAN as backend.
        

        Attributes
        ----------
        
        @attribute line_data : None | dict[RadtranGasDescriptor, archnemsis.database.protocols.LineDataProtocol]
            An object (normally a numpy record array) that implements the `archnemsis.database.protocols.LineDataProtocol`
            protocol. If `None` the data has not been retrieved from the database yet.
            
            If not `None` will have the following attributes:
                RT_GAS_DESC : np.array[['N_LINES_OF_GAS'], (int,int)]
                    Radtran gas descriptor
                
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
        
        @attribute partition_data : None | dict[GasDescriptor, archnemsis.database.protocols.PartitionFunctionDataProtocol]
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
        self.partition_data_dict = None
        self._total_line_data_wave_lims = np.array([np.inf, -np.inf], dtype=float) # limits on fetched wavelengths of line data
        
        assert isinstance(self.ID, int), 'ID must be an integer (molecule id)'
        #assert isinstance(self.ISO, int), "ISO must be an integer (isotope id), or 0 to select all isotopes"
        
    ##################################################################################

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
    
    @property
    def line_data_dict(self) -> dict[RadtranGasDescriptor, ans.database.protocols.LineDataProtocol]:
        if self.line_data is None:
            return dict((rt_gas_desc, None) for rt_gas_desc in self.gas_isotopes.as_radtran_gasses())
        
        _lgr.debug(f'{self.line_data.RT_GAS_DESC.dtype=}')
        _lgr.debug(f'{self.line_data.RT_GAS_DESC.shape=}')
        _lgr.debug(f'{self.line_data.RT_GAS_DESC=}')
        
        _ldd = {}
        for rt_gas_desc in self.gas_isotopes.as_radtran_gasses():
            _lgr.debug(f'{len(self.get_line_data_gas_desc_mask(rt_gas_desc))=}')
            _lgr.debug(f'{type(self.get_line_data_gas_desc_mask(rt_gas_desc))=}')
            _lgr.debug(f'{self.get_line_data_gas_desc_mask(rt_gas_desc).dtype=}')
            _lgr.debug(f'{np.count_nonzero(self.get_line_data_gas_desc_mask(rt_gas_desc))=}')
            
            _ldd[rt_gas_desc] = self.line_data[self.get_line_data_gas_desc_mask(rt_gas_desc)]
        
        return _ldd
    
    
    @property
    def partition_data(self):# -> tuple[tuple[RadtranGasDescriptor, PFList]]:
        if self.partition_data_dict is None:
            return None
        
        return tuple((rt_gas_desc, self.partition_data_dict[rt_gas_desc]) for rt_gas_desc in self.gas_isotopes.as_radtran_gasses())
        


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
    
    def get_line_data_gas_desc_mask(self, rt_gas_desc : RadtranGasDescriptor) -> np.ndarray[bool]:
        """
        Returns a boolean mask of the same shape as `self.line_data` that is True where the isotopologues is the same as in `rt_gas_desc`
        """
        return np.logical_and((self.line_data.RT_GAS_DESC[:,0] == rt_gas_desc.gas_id),(self.line_data.RT_GAS_DESC[:,1] == rt_gas_desc.iso_id))
    
    def group_line_derived_array_by_gas_desc(self, array : np.ndarray[['N_LINES_OF_GAS',...],Any]) -> dict[RadtranGasDescriptor, np.ndarray]:
        """
        Take an `array` and group it in the same way as `self.line_data` would be grouped by `RadtranGasDescriptor` to form `self.line_data_dict`.
        Used to separate e.g. calculated Line Strengths into a dictionary that has Line Strengths for each constitudent isotope.
        """
        _ldd = {}
        assert array.shape[0] == self.line_data.shape[0], "Array to be grouped must have the same shape in the 1st dimension as `self.line_data`"
        for rt_gas_desc in self.gas_isotopes.as_radtran_gasses():
            _ldd[rt_gas_desc] = array[self.get_line_data_gas_desc_mask(rt_gas_desc)]
        return _ldd
    
    def is_line_data_ready(self) -> bool:
        """
        Tests if `self.line_data` has data stored in it or not.
        """
        if self.line_data is None:
            return False
        return True

    def restrict_linedata(
        self,
        vmin : float,
        vmax : float,
        wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,
    ) -> None:
        """
        Restricts the self.line_data attribute to ONLY contain values within (vmin,vmax).
        Effectively forgets the union of previously fetched wavelength ranges, and just
        re-fetches the range (vmin, vmax).
        """
        self._total_line_data_wave_lims = np.array([np.inf, -np.inf], dtype=float)
        self.fetch_linedata(vmin, vmax, wave_unit, refresh=True)

    def fetch_linedata(
            self, 
            vmin : float, 
            vmax : float, 
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,
            refresh : bool = False,
    ) -> None:
        """
        Fetch the line data from the specified database. 
        The database is assumed to be an archNEMESIS HDF5 line database.
        
        # ARGUMENTS #
            vmin : float
                Minimum wavenumber to get line data for
            vmax : float
                Maximum wavenumber to get line data for
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm
                Unit of `vmin` and `vmax`, default is wavenumbers in cm^{-1}.
            refresh : bool
                If False, will only load more data from file if (vmin, vmax) 
                is not entirely within the union of all previously fetched 
                wavelength ranges.
                If True, will reload data in the union of (vmin,vmax) with 
                all previously fetched wavelength ranges, regardless of if 
                (vmin, vmax) is wholly within the already loaded wavelength 
                range.
        """
        
        
        assert vmin < vmax, f'Mimimum wave ({vmin}) must be less than maximum wave ({vmax})'
        
        # Turn wavelength range in to Wavenumbers cm^{-1} for internal use
        wave_range = WaveRange(vmin, vmax, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm)
        vmin, vmax = wave_range.values()
        
        # Only get the wavelengths we don't already have
        already_contain_wave_range = ((self._total_line_data_wave_lims[0] < vmin) and (vmax < self.total_line_data_wave_lims[1]))
            
        
        if already_contain_wave_range and self.is_line_data_ready() and not refresh:
            _lgr.debug(f'Line data already loaded. {already_contain_wave_range=} {self.is_line_data_ready()=} {refresh=}')
            return
        
        # retrieve the superset of the existing interval and the desired interval
        self._total_line_data_wave_lims[0] = min(self._total_line_data_wave_lims[0], vmin)
        self._total_line_data_wave_lims[1] = max(self._total_line_data_wave_lims[1], vmax)
        
        vmin, vmax = self._total_line_data_wave_lims
        
        # get line database file
        ans_line_data_file = AnsLineDataFile(self.LINE_DATABASE)
        
        mol_id_list = []
        local_iso_id_list = []
        nu_list = []
        sw_list = []
        a_list = []
        elower_list = []
        gamma_self_list = []
        n_self_list = []
        gamma_amb_list = []
        n_amb_list = []
        delta_amb_list = []

        # NOTE: abundances are accounted for in `self.calculate_monochromatic_absorption(...)`
        #scale_strength_by_abundance = self.ISO == 0
            

        for gas_desc in self.gas_isotopes.as_radtran_gasses():
        
            (
                mol_id,
                local_iso_id,
                nu,
                sw,
                a,
                elower,
                gamma_self,
                n_self,
                gamma_amb,
                n_amb,
                delta_amb,
            ) = ans_line_data_file.get_line_data(
                gas_desc.gas_name, 
                local_iso_id=gas_desc.iso_id, 
                ambient_gas=self.ambient_gas,
                wavelength_mask_fn=lambda nu: ((vmin < nu) & (nu <= vmax))
            )
            
            mol_id_list.append(mol_id)
            local_iso_id_list.append(local_iso_id)
            nu_list.append(nu)
            sw_list.append(sw) #sw_list.append((sw*gas_desc.abundance) if scale_strength_by_abundance else sw) # NOTE: abundances are accounted for in `self.calculate_monochromatic_absorption(...)`
            a_list.append(a)
            elower_list.append(elower)
            gamma_self_list.append(gamma_self)
            n_self_list.append(n_self)
            gamma_amb_list.append(gamma_amb)
            n_amb_list.append(n_amb)
            delta_amb_list.append(delta_amb)


        dtype = [
            ('RT_GAS_DESC', int, (2,)),
            ('NU', float),
            ('SW', float),
            ('A', float),
            ('GAMMA_AMB', float),
            ('N_AMB', float),
            ('DELTA_AMB', float),
            ('GAMMA_SELF', float),
            ('N_SELF', float),
            ('ELOWER', float),
        ]
        
        self.line_data = np.rec.fromarrays(
            (
                np.array((np.concatenate(mol_id_list),np.concatenate(local_iso_id_list)), dtype=int).T,
                np.concatenate(nu_list),
                np.concatenate(sw_list),
                np.concatenate(a_list),
                np.concatenate(gamma_amb_list),
                np.concatenate(n_amb_list),
                np.concatenate(delta_amb_list),
                np.concatenate(gamma_self_list),
                np.concatenate(n_self_list),
                np.concatenate(elower_list),
            ),
            dtype=dtype
        )


    ###########################################################################################################################
    
    def get_partition_data_gas_desc_mask(self, rt_gas_desc : RadtranGasDescriptor) -> np.ndarray[bool]:
        """
        Returns a boolean mask of the same shape as `self.partition_data` that is True where the isotopologue is the same as in `rt_gas_desc`
        """
        return np.logical_and((self.partition_data.RT_GAS_DESC[:,0] == rt_gas_desc.gas_id), (self.partition_data.RT_GAS_DESC[:,1] == rt_gas_desc.iso_id))
    
    def is_partition_function_ready(self) -> bool:
        """
        Tests if `self.partition_data` has data stored it in or not
        """
        return self.partition_data_dict is not None
    
    def fetch_partition_function(self, refresh : bool = False) -> None:
        """
        Get partition function data, if `refresh` then get the data again even if `self.is_partition_function_ready()` is True.
        
        # ARGUMENTS #
            refresh : bool = False
                If True will retrieve data from database again, even if data
                is already present.
        """
        
        if self.is_partition_function_ready() and not refresh:
            _lgr.info('Partition function data already loaded')
            return
        
        
        ans_pf_file = AnsPartitionFunctionDataFile(self.PARTITION_FUNCTION_DATABASE)
        self.partition_data_dict : dict[RadtranGasDescriptor, Callable[[float|np.ndarray],float|np.ndarray]] = dict()
        
        for gas_desc in self.gas_isotopes.as_radtran_gasses():
            self.partition_data_dict[gas_desc] = ans_pf_file.get_partition_function_data(gas_desc.gas_name, gas_desc.iso_id)
        _lgr.info(f'Retrieved partition function data from database {self.PARTITION_FUNCTION_DATABASE}')

    
    ###########################################################################################################################
    
    def calculate_doppler_width(
            self, 
            temp: float, 
    ) -> np.ndarray[['N_LINES_OF_GAS'], float]:
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
        
        mol_masses = np.array([RadtranGasDescriptor(*x).molecular_mass for x in self.line_data.RT_GAS_DESC]) # This lookup is slow
        dws = doppler_width_const_cgs * self.line_data.NU * np.sqrt( temp / mol_masses)
        
        _lgr.debug(dws)
        return dws
    
    def calculate_lorentz_width(
            self, 
            press: float, 
            temp: float,
            amb_frac: float, # fraction of ambient gas
            tref : float = 296,
    ) -> np.ndarray[['N_LINES_OF_GAS'], float]:
        """
        Calculate pressure-broadened width HWHM (half-width-half-maximum) of cauchy-lorentz distribution.
        """
        _lgr.debug(f'{press=} {temp=} {amb_frac=} {tref=}')
        
        tratio = tref/temp

        lws = (
            (tratio**self.line_data.N_AMB) 
            * ( self.line_data.GAMMA_AMB * amb_frac + self.line_data.GAMMA_SELF * (1-amb_frac) )
        ) * press
        
        _lgr.debug(lws)
        return lws
    
    
    def calculate_partition_sums(self,T) -> np.ndarray[['N_LINES_OF_GAS'], float]:
        """
        Calculate the partition functions at any arbitrary temperature

        Args:
            T (float): Temperature (K)

        Returns:
            QT : np.ndarray[['N_LINES_OF_GAS'], float]
                Partition functions for lines at temperature T 
        """
        QTs = np.empty((self.line_data.shape[0],), dtype=float)
        
        for gas_desc, gas_partition_data in self.partition_data_dict.items():
            if gas_partition_data is None:
                continue
            QTs[self.get_line_data_gas_desc_mask(gas_desc)] = gas_partition_data(T)
            
        return QTs
        
    ###########################################################################################################################
    
    def calculate_line_strength(
            self,
            T,
            Tref=296.0
    ) -> np.ndarray[['N_LINES_OF_GAS'], float]:
        """
        Calculate the line strengths at any arbitrary temperature

        Args:
            T (float): Temperature (K)
            Tref (float) :: Reference temperature at which the line strengths are listed (default 296 K)

        Returns:
            line_strengths : np.ndarray[['N_GAS_LINES'], float]
                Line strengths at temperature T
        """
        
        qT = self.calculate_partition_sums(T)
        qTref = self.calculate_partition_sums(Tref)
        
        _lgr.debug(f'{qT.shape=}')
        _lgr.debug(f'{qTref.shape=}')
        
        _lgr.debug(f'{Data.constants.c2_cgs=}')
        
        
        t_factor = Data.constants.c2_cgs * (T - Tref)/(T*Tref)
        _lgr.debug(f'{t_factor=}')
        
        q_ratio = qTref / qT
        _lgr.debug(f'{q_ratio.shape=}')
            
        boltz = np.exp(t_factor*self.line_data.ELOWER)
        _lgr.debug(f'{boltz.shape=}')
        
        stim = ( 1 - np.exp(-Data.constants.c2_cgs*self.line_data.NU/T)) / ( 1 - np.exp(-Data.constants.c2_cgs*self.line_data.NU/Tref))
        _lgr.debug(f'{stim.shape=}')
        
        line_strengths = self.line_data.SW * q_ratio * boltz * stim
        _lgr.debug(f'{line_strengths.shape=}')
        
        return line_strengths
    
    def calculate_monochromatic_line_absorption(
            self,
            delta_wn : np.ndarray, # wavenumber difference from line center (cm^{-1}), (NWAVE)
            mask : np.ndarray, # Boolean mask (NWAVE) that determines if absorption coefficient will be calculated from lineshape (True) or from a 1/(delta_wn^2) fit (False).
            wide_mask : np.ndarray, # Boolean mask (NWAVE) that determines if absorption coefficient will be calculated from a 1/(delta_wn^2) fit (True) or not accounted for at all (False).
            strength : float, # line strength
            alpha_d : float, # line doppler width (gaussinan HWHM)
            gamma_l : float, # line lorentz width (cauchy-lorentz HWHM)
            lineshape_fn : Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, # function that calculates line shape
            line_calculation_wavenumber_window : float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption, should be the same as the window used to make `mask`
            out : np.ndarray | None = None, # if not None, will place the result into this array and return it, otherwise will create an array and return it. Allocating memory is slow, so if possible try to pre-allocate the output array and pass it in here.
    ) -> np.ndarray:
        """
        Calculates absorption coefficient per wavenumber (cm^2 per cm^{-1}) for a single line at 
        a specific set of `delta_wn` wavenumber difference from the center of the line. 
        
        Does not use wavenumber bins, so `delta_wn` must be close enough together to 
        approximate a continuous set of values.
        
        Used inside `calculate_monochromatic_absorption`
        """
    
        # At `line_calculation_wavenumber_window` the lineshape_fn and 1/x^2 fit should be equal
        # requires that`line_calculation_wavenumber_window` was applied symmetrically around zero.
        constant = lineshape_fn(np.array([line_calculation_wavenumber_window]), alpha_d, gamma_l) * (line_calculation_wavenumber_window**2)
        
        if out is None:
            out = np.zeros_like(delta_wn)
        
        out[wide_mask] = strength * constant / (delta_wn[wide_mask]**2) # calculate "continuum" contribution for the wide mask
        out[mask] = strength * lineshape_fn(delta_wn[mask], alpha_d, gamma_l) # overwrite the "center" with lineshape contribution
        return out
    
    def calculate_monochromatic_absorption(
            self,
            waves : np.ndarray, # 1D array with shape [NWAVE]
            temp : float, # kelvin
            press : float, # Atmospheres
            amb_frac : float = 1, # fraction of broadening due to ambient gas
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,  # unit of `waves` argument
            lineshape_fn : Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, # lineshape function to use
            line_calculation_wavenumber_window: float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
            tref : float = 296, # Reference temperature (Kelvin). TODO: This should be set by the database used
            line_strength_cutoff : float = 1E-32, # Strength below which a line is ignored.
            ensure_linedata_downloaded : bool = True,
            isotopic_abundances : None | dict[RadtranGasDescriptor, float] = None, # If not None, use these abundances for each isotopologue instead of the default terrestrial ones. 
    ) -> np.ndarray:
        """
        Calculate total absorption coefficient (cm^2) for wavenumbers (cm^{-1}) multiplied by a factor of 1E20. 
        Returns the value for a single molecule at the specified temperature, pressure, and ambient gas fraction.
        
        Faster than `calculate_absorption_in_bins` but as this function only calculates at specific wavelengths
        not over wavelength bins, must ensure that `waves` is a fine enough grid that no important spectral
        features are missed.
        
        For details see "applications" section (at bottom) of https://hitran.org/docs/definitions-and-units/
        """
        CONT_CALC_MSG_ONCE_FLAG = False
        n_line_progress = 1000
        
        # Convert waves to wavenumber (cm^{-1})
        #in_waves = np.array(waves)
        waves = np.array(WavePoint(waves, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value)
        
        # Remember the ordering we got as input
        wav_sort_idxs = np.argsort(waves)
        
        waves = waves[wav_sort_idxs]
        
        # Download data
        self.fetch_linedata(
            vmin = waves[0]-line_calculation_wavenumber_window, 
            vmax = waves[-1]+line_calculation_wavenumber_window, 
            wave_unit = ans.enums.WaveUnit.Wavenumber_cm
        )

        #Applying default abundances if not specified
        strength = self.calculate_line_strength(temp, tref)
        alpha_d = self.calculate_doppler_width(temp)
        gamma_l = self.calculate_lorentz_width(press, temp, amb_frac, tref)


        #Weighting line strengths with isotopic abundances if needed
        if self.ISO == 0:
            if isotopic_abundances is None:
                iso_abu = np.array([RadtranGasDescriptor(*x).abundance for x in self.line_data.RT_GAS_DESC]) # This lookup is slow
                strength *= iso_abu
            else:
                iso_abu = np.array([isotopic_abundances[RadtranGasDescriptor(*x)] for x in self.line_data.RT_GAS_DESC])
                strength *= iso_abu

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
        abs_coeffs= np.zeros_like(waves, dtype=float) # Final output array
        
        line_idxs_to_include = np.flatnonzero((waves[0] <= self.line_data.NU) & (self.line_data.NU <= waves[-1]))
        n_lines = line_idxs_to_include.size
        _lgr.debug(f'{n_lines=}')
        
        with SimpleProgressTracker("Computing line contributions. ", n_lines, display_interval_n=n_line_progress, output_target=_lgr) as line_contrib_prog:
            #for _j, line_idx in enumerate(range(idx_min, idx_max)):
            for _j, line_idx in enumerate(line_idxs_to_include):

                if strength[line_idx] < line_strength_cutoff:
                    continue
                
                line_contrib_prog.display()
                
                scratch.fill(0.0)
                
                np.subtract(waves, self.line_data.NU[line_idx], out=delta_wn)
                
                np.less_equal(delta_wn, line_calculation_wavenumber_window, out=mask_leq)
                np.greater_equal(delta_wn, -line_calculation_wavenumber_window, out=mask_geq)
                np.logical_and(mask_leq, mask_geq, out=mask)
                
                np.less_equal(delta_wn, 10*line_calculation_wavenumber_window, out=wide_mask_leq)
                np.greater_equal(delta_wn, -10*line_calculation_wavenumber_window, out=wide_mask_geq)
                np.logical_and(wide_mask_leq, wide_mask_geq, out=wide_mask)

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
                if not CONT_CALC_MSG_ONCE_FLAG:
                    _lgr.debug('NOTE: Continuum absorbtion is handled in the lineshape calculation for now, if required may want to separate it for efficiency')
                    CONT_CALC_MSG_ONCE_FLAG = True
                
                
                # Ensure abs coeff are not less than zero
                k_total[k_total<0] = 0

                # put into absorption coefficient dictionary, multiply by 1E20 factor here
                abs_coeffs[wav_sort_idxs] = k_total*1E20
            
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
        _lgr.critical('`LineData_0.calculate_absorption_at_temp_pressure_profile` is not quite ready for production use yet. Having trouble with the linetable file format.')
        
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
        
        with SimpleProgressTracker("Computing absorption coefficients of temperature-pressure profile", temp_points.size, output_target=_lgr) as pt_profile_progress:
        
            for i, press in enumerate(pressure_profile):
                for j, temp in enumerate(temp_points[i]):
                    pt_profile_progress.display()
                    
                    #_lgr.info(f'Calculating absorption at temperature-pressure profile point ({i},{j}) of ({pressure_profile.size},{temp_points.shape[1]}). Progress: {i*2+j} / {temp_points.size} [{100.0*(i*2+j)/temp_points.size: 6.2f} %]')
                    abs_coeffs = self.calculate_monochromatic_absorption(
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
        
        line_data_dict = self.line_data_dict
        
        for i, (gas_desc, gas_linedata) in enumerate(line_data_dict.items()):
            _lgr.info(f'{gas_desc=} {gas_linedata.shape=}')
            
            #if gas_linedata is None or gas_linedata.size == 0:
            #    continue
            
            line_strength_mask = gas_linedata.SW >= smin
            wavenumbers = gas_linedata.NU[line_strength_mask]
            line_strengths = gas_linedata.SW[line_strength_mask]
            
            if gas_linedata.size == 0:
                ls_max = 0
                no_data_str = ' [NO DATA]'
            else:
                ls_max = line_strengths.max()
                no_data_str = ''
            line_strengths_max = ls_max if ls_max > line_strengths_max else line_strengths_max
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            # Combined plot, all isotopes on one figure, coloured by isotope
            combined_ax.scatter(
                wavenumbers,
                line_strengths,
                label=f'${gas_name_latex}$ (ID={int(gas_desc.gas_id)}, ISO={gas_desc.iso_id}){no_data_str}',
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
    
        for i, (gas_desc, gas_linedata) in enumerate(line_data_dict.items()):
            ax = ax_array[i+1]
            
            line_strength_mask = gas_linedata.SW >= smin
            wavenumbers = gas_linedata.NU[line_strength_mask]
            line_strengths = gas_linedata.SW[line_strength_mask]
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            
            if gas_linedata is None or gas_linedata.size == 0:
                no_data_str = ' [NO DATA]'
                line_strengths_max = 0
            else:
                no_data_str = ''
                line_strengths_max = line_strengths.max()
            
            ax.set_title(f'Line data for ${gas_name_latex}$ (ID={int(gas_desc.gas_id)}, ISO={gas_desc.iso_id}){no_data_str}')
            
            
            # Plots for specific isotopes, coloured by lower energy state
            
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
                ax.set_ylim(smin, line_strengths_max * 10)
            ax.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
            ax.set(**ax_style_defaults)











