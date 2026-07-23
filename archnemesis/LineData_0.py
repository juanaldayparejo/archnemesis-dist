

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
from contextlib import contextmanager
# Third party
import numpy as np
import matplotlib as mpl
import matplotlib.collections
import matplotlib.pyplot as plt
# This package
from archnemesis import Data
#from archnemesis import *
import archnemesis as ans
import archnemesis.enum

#import archnemesis.helpers.maths_helper as maths_helper
#from archnemesis.helpers.io_helper import SimpleProgressTracker
import archnemesis.database
#from archnemesis.database.filetypes.lbltable import LblDataTProfilesAtPressure#, LblDataTPGrid
from archnemesis.database.datatypes.wave_point import WavePoint
#from archnemesis.database.datatypes.wave_range import WaveRange
from archnemesis.database.datatypes.gas_isotopes import GasIsotopes
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor

from archnemesis.database.datatypes.line_set_data import LineSetData
from archnemesis.database.datatypes.pseudo_continuum_data import PseudoContinuumData
from archnemesis.database.datatypes.pf_list import PFList

from archnemesis.database.filetypes.ans_line_data_file import AnsLineDataFile
from archnemesis.database.filetypes.ans_partition_fn_data_file import AnsPartitionFunctionDataFile
from archnemesis.database.filetypes.ans_pseudo_continuum_file import AnsPseudoContinuumFile

from archnemesis.download import get_reference_database_downloader_for

from archnemesis.Data.path_data import archnemesis_path 

# Logging
import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)

if TYPE_CHECKING:
    NWAVE = "Number of wave points"
    N_LINES_OF_GAS = 'Number of lines for a gas isotopologue'
    N_TEMPS_OF_GAS = 'Number of temperature points for a gas isotopologue'

import dataclasses as dc
from typing import Self#, NamedTuple

from numba import njit, prange

INVALID_MOLECULE_ID = -1
INVALID_ISOTOPOLOGUE_ID = -1

def mol_id_and_iso_id_to_arrays(
        mol_id : int | tuple[int,...] | np.ndarray,
        iso_id : int | tuple[int,...] | tuple[tuple[int,...]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(mol_id, int):
        mol_id = np.array([mol_id], dtype=int)
    elif not isinstance(mol_id, np.ndarray):
        mol_id = np.array(mol_id, dtype=int)
    
    n_mols = len(mol_id)
    
    if isinstance(iso_id, int):
        iso_id = np.array([iso_id], dtype=int)
    elif not isinstance(iso_id, np.ndarray):
        _max_isos_for_mol = 0
        _iso_id_temp = [[]]*n_mols
        for i, x in enumerate(iso_id):
            if isinstance(x, int):
                n = 1
                _iso_id_temp[i].append(x)
            else:
                n = len(iso_id)
                _iso_id_temp[i].extend(x)
                
            if (n > _max_isos_for_mol):
                _max_isos_for_mol = n
        
        iso_id = np.ones((n_mols,_max_isos_for_mol), dtype=int) * INVALID_ISOTOPOLOGUE_ID
        for i, x in enumerate(_iso_id_temp):
            for j, y in enumerate(x):
                iso_id[i,j] = y
    
    return mol_id, iso_id
    
def get_container_for_iso_id_array(
        iso_id : np.ndarray,
) -> list | tuple[list,...]:
    return tuple([None]*np.count_nonzero(x != INVALID_ISOTOPOLOGUE_ID) for x in iso_id) if iso_id.shape[0] > 1 else ([None]*np.count_nonzero(iso_id[0] != INVALID_ISOTOPOLOGUE_ID))

def wn_range_is_within(
        wn_range,
        wn_range_reference
) -> bool:
    return (wn_range_reference[0] <= wn_range[0]) and (wn_range[1] <= wn_range_reference[1])

MODULE_NUMBA_CACHE = True

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def stimulated_emission(
    t_calc : float,
    nu : np.ndarray, # [N] wavenumber
    out : np.ndarray, # [N] stimulated emission factor
):
    for i in prange(nu.shape[0]):
        out[i] = 1 - np.exp(-Data.constants.c2_cgs * nu[i] / t_calc)

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def boltzmann_factor(
    t_calc : float,
    t_ref : float,
    e_lower : np.ndarray, #[N]
    out : np.ndarray, #[N]
):
    boltz_const_factor = Data.constants.c2_cgs * (t_calc - t_ref)/(t_calc*t_ref)
    for i in prange(e_lower.shape[0]):
        out[i] = np.exp(boltz_const_factor*e_lower[i])

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def doppler_width(
        t_calc : float, 
        molecular_mass : float,
        nu : np.ndarray, #[N] wavenumber
        out : np.ndarray, #[N] doppler width
):
    """
    Computes Half-width-half-maximum (HWHM) of doppler broadening component
    """
    doppler_width_const_cgs = (1.0 / Data.constants.c_light_cgs) * np.sqrt(2 * np.log(2) * Data.constants.N_avogadro * Data.constants.k_boltzmann_cgs)
    
    for i in prange(nu.shape[0]):
        out[i] = doppler_width_const_cgs * nu[i] * np.sqrt( t_calc / molecular_mass)

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def lorentz_width(
        t_ratio : float,
        p_ratio : float,
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        broadening_params : np.ndarray, #[M,N] M = N_mol * 3, note: N_mol = 1 + number of ambient molecules (self counts as 1), M is arranged (gamma, n, delta) for each molecule
        out : np.ndarray, #[N]
):
    """
    Computes Half-width-half-maximum (HWHM) of pressure broadening component
    """

    #NOTE : This version considers both n_self and n_air. While there is nothing wrong with it in principle,
    #        it is not how it is implemented in NEMESIS
    for i in prange(broadening_params.shape[1]):
        gamma_combined = 0
        for j in prange(mol_mix_frac.shape[0]):
            gamma_combined += (t_ratio**broadening_params[3*j+1,i])*broadening_params[3*j,i] * mol_mix_frac[j] * p_ratio
        out[i] = gamma_combined

    #NOTE : This version considers n_air only. This is how NEMESIS implements it
    #for i in prange(broadening_params.shape[1]):
    #    gamma_combined = 0
    #    for j in prange(mol_mix_frac.shape[0]):
    #        if j==0:
    #            gamma_combined += (t_ratio**broadening_params[3*(j+1)+1,i])*broadening_params[3*j,i] * mol_mix_frac[j] * p_ratio
    #        else:
    #            gamma_combined += (t_ratio**broadening_params[3*j+1,i])*broadening_params[3*j,i] * mol_mix_frac[j] * p_ratio
    #    out[i] = gamma_combined

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def line_shift(
        t_ratio : float,
        p_ratio : float,
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        broadening_params : np.ndarray, #[M,N] M = N_mol * 3, note: N_mol = 1 + number of ambient molecules (self counts as 1), M is arranged (gamma, n, delta) for each molecule
        out : np.ndarray, #[N]
):
    """
    Computes shift of lines
    """
    for i in prange(broadening_params.shape[1]):
        shift_combined = 0
        for j in prange(mol_mix_frac.shape[0]):
            shift_combined += (p_ratio*broadening_params[3*j+2,i]) * mol_mix_frac[j]
        out[i] = shift_combined

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def line_strength(
        t_calc : float,
        t_ref : float,
        q_ratio : float,
        
        nu : np.ndarray, #[N]
        sw : np.ndarray, #[N]
        e_lower : np.ndarray, #[N]
        stimulated_emission_at_t_ref : np.ndarray, #[N]
        
        out : np.ndarray, #[N]
):
    boltz_const_factor = Data.constants.c2_cgs * (t_calc - t_ref)/(t_calc*t_ref)
    for i in prange(nu.shape[0]):
        
        out[i] = (
            sw[i]
            * ((1 - np.exp(-Data.constants.c2_cgs * nu[i] / t_calc)) / stimulated_emission_at_t_ref[i]) # stimulated_emission
            * np.exp(boltz_const_factor*e_lower[i]) # boltzmann pop
            * q_ratio
        )

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def add_line_set_monochromatic_spectrum(
        wn_grid : np.ndarray, #[N_waves] (cm^{-1}) must be in ascending order
        lineshape_fn : Callable[[float,float,float], float],
        nu : np.ndarray, #[N_lines] (cm^{-1}) ideally sorted into ascending order
        strength : np.ndarray, #[N_lines]
        alpha_d : np.ndarray, #[N_lines]
        gamma_l : np.ndarray, #[N_lines]
        line_shift : np.ndarray, #[N_lines]
        factor : float, # Factor to apply to final result (e.g. a factor to account for isotopic abundance)
        
        out : np.ndarray, #[N_waves] should be all zeros, as result will be ADDED to this not overwritten
        
        s_floor : float = 0.0,
        wn_calc_window : float = 25.0,
        wn_approx_window : float = 75.0
):
    """
    Calculates the combined spectrum from many absorption lines
    """
    wn_delta : float = 0.0
    wn_calc_window_min :float = -1*wn_calc_window
    wn_calc_window_max : float = wn_calc_window
    wn_approx_window_min :float = -1*wn_approx_window
    wn_approx_window_max : float = wn_approx_window
    line_approx_const : float = 0.0
    
    for i in range(nu.shape[0]):
        
        if strength[i] < s_floor:
            continue
            
        line_approx_const = lineshape_fn(wn_calc_window_max, alpha_d[i], gamma_l[i])
        
        for j in range(wn_grid.shape[0]):
            wn_delta = wn_grid[j] - (nu[i] + line_shift[i])
            
            if wn_delta >= wn_approx_window_max:
                    break
            if wn_delta < wn_approx_window_min:
                    continue
            
            if wn_calc_window_min <= wn_delta and wn_delta < wn_calc_window_max:
                # Lineshape calculation
                out[j] += factor * strength[i] * lineshape_fn(wn_delta, alpha_d[i], gamma_l[i])
            else:
                # 1/x**2 approximation
                out[j] += factor * strength[i] * line_approx_const * wn_calc_window_max**2. / (wn_delta * wn_delta)
    
    return

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def add_line_set_monochromatic_absorption(
        wn_grid : np.ndarray, #[N_waves]
        lineshape_fn : Callable[[float,float,float], float],
        t_calc : float,
        t_ref : float,
        p_calc : float,
        p_ref : float,
        q_ratio : float,
        isotopic_abundance : float,
        isotopic_mass : float,
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        broadening_params : np.ndarray, #[M,N] M = N_mol * 3, note: N_mol = 1 + number of ambient molecules (self counts as 1), M is arranged (gamma, n, delta) for each molecule
        nu : np.ndarray, #[N]
        sw : np.ndarray, #[N]
        e_lower : np.ndarray, #[N]
        stimulated_emission_at_t_ref : np.ndarray, #[N]
        
        out : np.ndarray,
        
        store : np.ndarray | None = None, #[4,N]
        
        s_floor : float = 0,
        wn_calc_window : float = 25.0, # (cm^{-1})
        wn_approx_window : float = 75.0, # (cm^{-1})
):
    if store is None:
        store = np.empty((4, nu.shape[0]), dtype=float)
    
    line_strength(
        t_calc,
        t_ref,
        q_ratio,
        nu,
        sw,
        e_lower,
        stimulated_emission_at_t_ref,
        out = store[0]
    )
    
    doppler_width( # alpha_d
        t_calc,
        isotopic_mass,
        nu,
        out = store[1]
    )
    
    lorentz_width( # gamma_l
        t_ref / t_calc,
        p_calc / p_ref,
        mol_mix_frac,
        broadening_params,
        out = store[2]
    )
    
    line_shift(
        t_ref / t_calc,
        p_calc / p_ref,
        mol_mix_frac,
        broadening_params,
        out = store[3]
    )
    
    
    add_line_set_monochromatic_spectrum(
        wn_grid,
        lineshape_fn,
        nu,
        store[0],
        store[1],
        store[2],
        store[3],
        factor = isotopic_abundance,
        out = out,
        s_floor = s_floor,
        wn_calc_window = wn_calc_window,
        wn_approx_window = wn_approx_window,
    )
    
    return

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def add_pseudo_continuum_monochromatic_spectrum(
        wn_grid : np.ndarray, #[N_waves] (cm^{-1}) must be in ascending order
        lineshape_fn : Callable[[float,float,float], float],
        wn_bin_centers : np.ndarray, #[N] (cm^{-1}) must be in ascending order
        wn_bin_widths : np.ndarray, #[N] (cm^{-1}) in same order as `wn_bin_centers`
        strength_sum : np.ndarray, #[N]
        mean_alpha_d : np.ndarray, #[N]
        mean_gamma_l : np.ndarray, #[N]
        factor : float, # Factor to apply to final result (e.g. a factor to account for isotopic abundance)
        
        out : np.ndarray, #[N_waves] should be all zeros, as result will be ADDED to this not overwritten
        
        store_x : None | np.ndarray = None, #[N]
        store_y : None | np.ndarray = None, #[2*n_neighbour_bins+1]
        store_z : None | np.ndarray = None, #[2, N_waves]
        
        n_neighbour_bins : int = 3,
        skip_nan_flag : bool = True,
):

    if store_x is None:
        store_x = np.zeros((out.shape[0],), dtype=float)
    else:
        for i in range(store_x.shape[0]):
            store_x[i] = 0.0
    
    if store_y is None:
        store_y = np.zeros((2*n_neighbour_bins+1,), dtype=float)
    
    if store_z is None:
        store_z = np.zeros((2, out.shape[0],), dtype=float)
    else:
        for j in range(store_z.shape[1]):
            store_z[0,j] = 0.0
            store_z[1,j] = 0.0
    
    # get the index of first and last bins in `wn_grid`
    
    first_wn_idx = -1
    last_wn_idx = -1
    for i in range(wn_bin_centers.shape[0]):
        bin_min = wn_bin_centers[i] - wn_bin_widths[i] / 2.0
        bin_max = wn_bin_centers[i] + wn_bin_widths[i] / 2.0
        
        if first_wn_idx == -1 and bin_min <= wn_grid[0]:
            first_wn_idx = i
        if last_wn_idx == -1 and bin_max > wn_grid[-1]:
            last_wn_idx = i
        
        if first_wn_idx != -1 and last_wn_idx != -1:
            break
    
    if first_wn_idx == -1:
        first_wn_idx = wn_bin_centers.shape[0]
    if last_wn_idx == -1:
        last_wn_idx = wn_bin_centers.shape[0]
    
    
    for i in range(first_wn_idx, last_wn_idx):
    
        lineshape_sum : float = 0.0
        for delta_k in range(-n_neighbour_bins, n_neighbour_bins+1):
            k = n_neighbour_bins + delta_k
            ii = i + delta_k
            
            if 0 <= ii and ii < wn_bin_centers.shape[0]:
                store_y[k] = lineshape_fn(
                    wn_bin_centers[ii] - wn_bin_centers[i],
                    mean_alpha_d[i],
                    mean_gamma_l[i],
                )
                lineshape_sum += store_y[k]
            else:
                store_y[k] = 0
        
        for delta_k in range(-n_neighbour_bins, n_neighbour_bins+1):
            k = n_neighbour_bins + delta_k
            ii = i + delta_k
        
            if 0 <= ii and ii < wn_bin_centers.shape[0] and lineshape_sum != 0:
                store_x[ii] += strength_sum[i] * store_y[k] / lineshape_sum
    
    for i in range(wn_bin_centers.shape[0]):
        store_x[i] /= wn_bin_widths[i]
    
    # Interpolate result to `wn_grid`
    # NOTE: Here we are interpolating the pseudo-continuum into `wn_grid`. I am certain this can be more elegant
    j_min = wn_bin_centers.shape[0]
    j_max = 0
    j_start = 0
    for i in range(wn_bin_centers.shape[0]):
        for j in range(j_start, out.shape[0]):
            delta_wn_grid = (wn_grid[j] - wn_bin_centers[i]) / wn_bin_widths[i] # [-0.5, 0.5] if within bin width of bin center
            
            if delta_wn_grid < -0.5:
                j_start = j
                continue
            elif delta_wn_grid >= 0.5:
                break
            
            if j_min > j:
                j_min = j
            if j_max < j:
                j_max = j
            
            n = 1.0 - np.abs(delta_wn_grid)
            if delta_wn_grid < 0 and i > 0:
                store_z[0,j] += (1-n) * factor * store_x[i-1]
            elif delta_wn_grid > 0 and i < (wn_bin_centers.shape[0]-1):
                store_z[0,j] += (1-n) * factor * store_x[i+1]
            store_z[0,j] += n * factor * store_x[i]
            
            store_z[1,j] += 1.0
            
            
    for j in range(j_min, j_max):
        if store_z[1,j] == 0.0:
            continue
        out[j] += store_z[0,j] / store_z[1,j]
        
        
        
    return

@njit(parallel=False, cache=MODULE_NUMBA_CACHE)
def add_pseudo_continuum_monochromatic_absorption(
        wn_grid : np.ndarray, #[N_waves]
        lineshape_fn : Callable[[float,float,float], float],
        t_calc : float,
        t_ref : float,
        p_calc : float,
        p_ref : float,
        q_ratio : float,
        isotopic_abundance : float,
        isotopic_mass : float,
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        lsw_mean_broadening_params : np.ndarray, #[M,N] M = N_mol * 3, note: N_mol = 1 + number of ambient molecules (self counts as 1), M is arranged (gamma, n, delta) for each molecule
        wn_bin_centers : np.ndarray, #[N]
        wn_bin_widths : np.ndarray, #[N]
        sw_sum : np.ndarray, #[N]
        lsw_mean_e_lower : np.ndarray, #[N]
        
        out : np.ndarray, # Should all be zeros as result will be ADDED to this not overwritten
        
        store : np.ndarray | None = None, #[3,N]
        store_x : None | np.ndarray = None, #[N]
        store_y : None | np.ndarray = None, #[2*n_neighbour_bins+1]
        store_z : None | np.ndarray = None, #[2, N_waves]
        
        n_neighbour_bins : int = 3,
):
    if store is None:
        store = np.zeros((3,wn_bin_centers.shape[0]), dtype=float)
    if store_x is None:
        store_x = np.zeros((wn_bin_centers.shape[0],), dtype=float)
    if store_y is None:
        store_y = np.zeros((2*n_neighbour_bins+1,), dtype=float)
    if store_z is None:
        store_z = np.zeros((2, out.shape[0],), dtype=float)
    
    stimulated_emission(
        t_ref,
        wn_bin_centers,
        out = store[-1], # this is re-used later but should not matter
    )
    
    line_strength(
        t_calc,
        t_ref,
        q_ratio,
        wn_bin_centers,
        sw_sum,
        lsw_mean_e_lower,
        store[-1], # this is re-used later but should not matter
        
        out = store[0]
    )
    
    doppler_width( # alpha_d
        t_calc,
        isotopic_mass,
        wn_bin_centers,
        out = store[1]
    )
    
    lorentz_width(
        t_ref / t_calc,
        p_calc / p_ref,
        mol_mix_frac,
        lsw_mean_broadening_params,
        
        out = store[2]
    )
    
    add_pseudo_continuum_monochromatic_spectrum(
        wn_grid,
        lineshape_fn,
        wn_bin_centers,
        wn_bin_widths,
        store[0],
        store[1],
        store[2],
        factor = isotopic_abundance,
        
        out = out,
        
        store_x = store_x,
        store_y = store_y,
        store_z = store_z,
        
        n_neighbour_bins = n_neighbour_bins
    )
    
    return


@dc.dataclass(slots=True)
class CachePriorityDataHolder:
    n_max      : int                  = 1_000 # Need to set this to a large enough number that values don't constantly "fall off the back" of the cache
    identities : list[tuple[Any,...]] = dc.field(default_factory=list)
    values     : list[Any]            = dc.field(default_factory=list)
    
    def __contains__(self, identity : tuple[Any,...]) -> bool:
        return identity in self.identities
    
    def get(self, identity : tuple[Any,...], default : None | Any = None) -> Any:
        """
        Retrieve the value associated with `identity` if present else return `default`. 
        If present, move `identity` to the front of the data holder.
        """
        try:
            idx = self.identities.index(identity)
        except ValueError:
            return default
        
        # Move the requested value to the front of the list
        self.identities.pop(idx)
        result = self.values.pop(idx)
        self.identities.insert(0, identity)
        self.values.insert(0, result)
        
        return result
    
    def set(self, identity : tuple[Any,...], value : Any) -> Any:
        """
        Add `value` to the front of the data holder and associate it with `identity`. If `identity`
        is already present, overwrite `value` and move to front of data holder.
        """
        try:
            idx = self.identities.index(identity)
        except ValueError:
            pass
        else:
            # if no error `idx` is valid so use it
            self.identities.pop(idx)
            self.values.pop(idx)
        self.identities.insert(0, identity)
        self.values.insert(0, value)
        
        while len(self.identities) > self.n_max:
            self.identities.pop()
            self.values.pop
        
        return value

class Cache:
    def __init__(
            self, 
            n_max_per_bucket : int = 1_000, # Need to set this to a large enough number that values don't constantly "fall off the back" of the cache
    ):
        self.n_cache_hit : int = 0
        self.n_cache_miss : int = 0
        self._n_max_per_bucket = n_max_per_bucket
        self._buckets = dict()
    
    def cache_performance_str(self):
        n_cache_req = self.n_cache_hit + self.n_cache_miss
        return f'Cache Performance :: requests {n_cache_req} :: hit {self.n_cache_hit} [{100*self.n_cache_hit/n_cache_req:5.2f}%] :: miss {self.n_cache_miss} [{100*self.n_cache_miss/n_cache_req:5.2f}%]'
    
    def get(self, bucket : str, identity : tuple[Any,...], default : None | Any = None) -> Any:
        found_bucket = self._buckets.get(bucket, None)
        if found_bucket is None:
            self._buckets[bucket] = CachePriorityDataHolder(self._n_max_per_bucket)
            self.n_cache_miss += 1
            return None
        
        result = found_bucket.get(identity, default=default)
        if result is None:
            self.n_cache_miss += 1
        else:
            self.n_cache_hit += 1
            _lgr.debug(f'CACHE HIT:: {bucket=} {identity=}')
        
        return result
        
    def set(self, bucket : str, identity : tuple[Any,...], value : Any) -> Any:
        found_bucket = self._buckets.get(bucket, None)
        if found_bucket is None:
            self._buckets[bucket] = CachePriorityDataHolder(self._n_max_per_bucket)
        
        return found_bucket.set(identity, value)
        
_MODULE_CACHE : Cache = Cache()



@dc.dataclass(slots=True)
class LineSetSpecData:
    """
    Line set spectral data for a sinlge isotopologue
    """
    s_min : float
    t_ref : float
    p_ref : float
    rt_gas_desc : RadtranGasDescriptor
    broadening_molecule_ids : tuple[int,...]
    req_wn_range : tuple[float,float]
    
    # private attrs
    _molecular_mass : float
    _data : np.ndarray # [M, N_lines], M = (3[nu,sw,elower] + 1[stimulated_emission_at_t_ref] + 1[a] + 3[gamma_self, n_self, delta_self] + 3*n_ambient_gasses[gamma_amb, n_amb, delta_amb for each ambient gas])
    _result_cache : Cache = dc.field(default_factory=lambda : _MODULE_CACHE)
    _data_hash : int = 0
    
    @classmethod
    def create_from(
            cls, 
            mol_id : int, 
            iso_id : int, 
            ambient_gasses : tuple[ans.enum.AmbientGasEnum,...],
            single_iso_line_set_data : LineSetData,
            cache : None | Cache = None,
    ) -> Self:
        #print(f'{single_iso_line_set_data.nu=}')
        
        rt_gas_desc = RadtranGasDescriptor(mol_id, iso_id)
        
        instance = cls(
            single_iso_line_set_data.s_min,
            single_iso_line_set_data.t_ref,
            single_iso_line_set_data.p_ref,
            rt_gas_desc,
            (-1, *(int(x) for x in ambient_gasses)),
            single_iso_line_set_data.req_wn_range,
            _molecular_mass = rt_gas_desc.molecular_mass,
            _data = None
        )
        
        if cache is not None:
            instance._result_cache = cache
        
        instance._set_data_from_line_set_data(single_iso_line_set_data)
    
        return instance
    
    def _set_data_from_line_set_data(self, line_set_data : LineSetData):
        n = line_set_data.nu.shape[0]
        m = (
            3   # (nu, sw, elower)
            + 1 # (stimulated_emission_at_t_ref,)
            + 1 # (a,)
            + 3 # (gamma_self, n_self, delta_self)
            + line_set_data.gamma_amb.shape[1]*3 # (gamma_amb, n_amb, delta_amb) for m ambient gasses
        )
        
        if self._data is None or self._data.shape != (m,n):
            self._data = np.empty((m,n), dtype=float)
        self.NU[...] = line_set_data.nu
        self.SW[...] = line_set_data.sw
        self.A[...] = line_set_data.a
        self.ELOWER[...] = line_set_data.elower
        stimulated_emission(self.t_ref, self.NU, out=self.STIMULATED_EMISSION_REF)
        self.SELF_BROADENING_PARAMS[...] = np.stack(
            [
                line_set_data.gamma_self, 
                line_set_data.n_self, 
                np.zeros_like(line_set_data.n_self), #single_iso_line_set_data.delta_self
            ],
            axis=0,
        )
        self.FOREIGN_BROADENING_PARAMS[...] = np.stack(
            [
                *(line_set_data.gamma_amb.T), 
                *(line_set_data.n_amb.T), 
                *(line_set_data.delta_amb.T)
            ], 
            axis=0,
        )
        
        self._data.flags.writeable = False
        self._data_hash = hash(bytes(self._data))
    
    @property
    def n_lines(self):
        return self._data.shape[1]
    
    @property
    def n_fields(self):
        return self._data.shape[0]
    
    @property
    def n_broadeners(self):
        return len(self.broadening_molecule_ids)
    
    @property
    def has_data(self):
        return self.n_lines != 0
    
    @property
    def NU(self):
        return self._data[0]
    
    @property
    def SW(self):
        return self._data[1]
    
    @property
    def ELOWER(self):
        return self._data[2]
    
    @property
    def STIMULATED_EMISSION_REF(self):
        return self._data[3]
    
    @property
    def A(self):
        return self._data[4]
    
    @property
    def SELF_BROADENING_PARAMS(self):
        """
        gamma_self, n_self, delta_self
        """
        return self._data[5:8]
    
    @property
    def FOREIGN_BROADENING_PARAMS(self):
        """
        (gamma_amb, n_amb, delta_amb) for each ambient gas
        """
        return self._data[8:]
    
    @property
    def ALL_BROADENING_PARAMS(self):
        """
        (gamma, n, delta) for "self" and each ambient gas
        """
        return self._data[5:]
    
    def cache_identity(self):
        return (
            self.s_min,
            self.t_ref,
            self.p_ref,
            self.broadening_molecule_ids,
            self.req_wn_range,
            *self.rt_gas_desc,
            self._molecular_mass,
            self._data_hash,
        )
    
    def add_monochromatic_absorption(
        self,
        wn_grid : np.ndarray, #[N_waves] (cm^{-1})
        lineshape_fn : Callable[[float,float,float], float],
        t_calc : float, # Kelvin
        p_calc : float, # Atmospheres
        partition_function : Callable[[float], float],
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        isotopic_abundance : float = 1.0,
        
        out : None | np.ndarray = None, #[N_waves]
        store : None | np.ndarray = None, #[4,N_lines]
        
        s_floor : float = 0,
        wn_calc_window : float = 25.0, # (cm^{-1})
        wn_approx_window : float = 75.0, # (cm^{-1})
        wn_calc_range : None | tuple[float,float] = None,
        include_pressure_shift : bool = True,
        use_cache : bool = True,
    ) -> np.ndarray:
        if out is None:
            out = np.zeros_like(wn_grid, dtype=float)
        
        if not self.has_data: # Skip all calculation if no data present
            return out
        
        q_ratio = partition_function(self.t_ref) / partition_function(t_calc)
        
        if use_cache:
            wn_grid.flags.writeable = False
            mol_mix_frac.flags.writeable = False
            
            result_bucket = (self.__class__.__name__, 'add_monochromatic_absorption')
            result_identity = (
                self.cache_identity(),
                hash(bytes(wn_grid.data)),
                id(lineshape_fn),
                t_calc, 
                p_calc,
                q_ratio,
                hash(bytes(mol_mix_frac.data)),
                isotopic_abundance,
                s_floor,
                wn_calc_window,
                wn_approx_window,
                wn_calc_range,
                include_pressure_shift,
            )
            cache_result = self._result_cache.get(result_bucket, result_identity, None)
            
            if cache_result is not None:
                out[...] = cache_result
                return out
        
        
        if store is None:
            store = np.empty((4,self.n_lines), dtype=float)
        
        
        
        wn_mask = np.ones((self._data.shape[1],), dtype=bool) if wn_calc_range is None else ((wn_calc_range[0] <= self.NU) & (self.NU <= wn_calc_range[1]))
        
        
        broadening_data = self._data[5:, wn_mask]
        if not include_pressure_shift:
            # make a copy of the broadening data and set all pressure shifts to zero
            broadening_data = np.array(broadening_data) * np.array([0 if ((i+1)%3) == 0 else 1 for i in range(broadening_data.shape[0])], dtype=int)[:,None]
        
        add_line_set_monochromatic_absorption(
            wn_grid,
            lineshape_fn,
            t_calc,
            self.t_ref,
            p_calc,
            self.p_ref,
            q_ratio,
            isotopic_abundance,
            self._molecular_mass,
            mol_mix_frac,
            broadening_data,
            *self._data[:4, wn_mask],
            
            out = out,
            store = store,
            
            s_floor = s_floor,
            wn_calc_window = wn_calc_window,
            wn_approx_window = wn_approx_window,
        )
        
        if use_cache:
            self._result_cache.set(result_bucket, result_identity, out)
        
        return out
    
    def get_line_strength(
        self,
        t_calc,
        partition_function : Callable[[float,], float], # Accepts a temperature, returns a partition function value for that temperature for one isotopologue
        wn_calc_range : None | tuple[float,float] = None,
        use_cache = True
    ) -> np.ndarray:
        result = None
        
        if use_cache:
            result_bucket = (self.__class__.__name__, 'get_line_strength')
            result_identity = (wn_calc_range, t_calc, self.cache_identity())
            result = self._result_cache.get(result_bucket, result_identity, None)
        
        if result is None:
            result = np.empty((self._data.shape[1],))
            q_ratio = partition_function(self.t_ref) / partition_function(t_calc)
            
            if wn_calc_range is not None:
                wn_mask = wn_calc_range[0] <= self.NU & self.NU <= wn_calc_range[1]
                data = self._data[:4,wn_mask]
            else:
                data = self._data[:4]
            
            line_strength(
                t_calc,
                self.t_ref,
                q_ratio,
                *data,
                out = result
            )
            
            if use_cache:
                self._result_cache.set(result_bucket, result_identity, result)
        
        return result
    
    def get_doppler_width(
        self,
        t_calc,
        wn_calc_range : None | tuple[float,float] = None,
        use_cache = True
    ) -> np.ndarray:
        result = None
        
        if use_cache:
            result_bucket = (self.__class__.__name__, 'get_doppler_width')
            result_identity = ( wn_calc_range, t_calc, self.cache_identity())
            result = self._result_cache.get(result_bucket, result_identity, None)
        
        if result is None:
            result = np.empty((self._data.shape[1],))
            
            if wn_calc_range is not None:
                wn_mask = wn_calc_range[0] <= self.NU & self.NU <= wn_calc_range[1]
                NU = self.NU[wn_mask]
            else:
                NU = self.NU
            
            doppler_width(
                t_calc,
                self._molecular_mass,
                NU,
                out = result
            )
            
            if use_cache:
                self._result_cache.set(result_bucket, result_identity, result)
        
        return result
    
    def get_lorentz_width(
        self,
        t_calc,
        p_calc,
        mol_mix_frac : np.ndarray,
        wn_calc_range : None | tuple[float,float] = None,
        use_cache = True
    ) -> np.ndarray:
        result = None
        
        if use_cache:
            result_bucket = (self.__class__.__name__, 'get_lorentz_width')
            result_identity = ( wn_calc_range, t_calc, p_calc, *mol_mix_frac, self.cache_identity())
            result = self._result_cache.get(result_bucket, result_identity, None)
        
        if result is None:
            result = np.empty((self._data.shape[1],))
            
            if wn_calc_range is not None:
                wn_mask = wn_calc_range[0] <= self.NU & self.NU <= wn_calc_range[1]
                data = self.ALL_BROADENING_PARAMS[:,wn_mask]
            else:
                data = self.ALL_BROADENING_PARAMS
            
            lorentz_width(
                self.t_ref / t_calc,
                p_calc / self.p_ref,
                mol_mix_frac,
                data,
                out = result,
            )
            
            if use_cache:
                self._result_cache.set(result_bucket,result_identity, result)
        
        return result

    def remove_lines(
            self,
            mask : np.ndarray
    ):
        """
        Remove lines where `mask` is True
        """
        
        old_data = self._data
        self._data = np.empty((old_data.shape[0], np.count_nonzero(~mask)), dtype=float)
        self._data[...] = old_data[:,~mask]
        
        self._data.flags.writeable=False
        self._data_hash = hash(bytes(self._data))

@dc.dataclass
class CombinedLineSetSpecData:
    n_isos : int
    _id_data : np.ndarray
    _data : np.ndarray
    
    @classmethod
    def create_from(cls, ld_list : list[LineSetSpecData]) -> Self:
        n_total_lines = sum(x.n_lines for x in ld_list)
    
            
        if not all(x.n_fields == ld_list[0].n_fields for x in ld_list):
            raise RuntimeError('Cannot create combined line data as different isotopologues have a different number of fields')
        
        if not all(x.n_broadeners == ld_list[0].n_broadeners for x in ld_list):
            raise RuntimeError('Cannot create combined line data as different isotopologues have a different number of broadeners')
    
        id_data = np.empty((2,n_total_lines), dtype=int)
        data = np.empty((ld_list[0].n_fields, n_total_lines), dtype=float)
        
        i = 0
        j = 0
        for iso_line_data in ld_list:
            _lgr.info(f'{iso_line_data._data.shape=}')
            j = i + iso_line_data.n_lines
            id_data[0,i:j] = iso_line_data.rt_gas_desc.gas_id
            id_data[1,i:j] = iso_line_data.rt_gas_desc.iso_id
            data[:,i:j] = iso_line_data._data
            i = j
        
        instance = cls(len(ld_list), _id_data=id_data, _data = data)
        
        return instance
    
    @property
    def ID(self):
        return self._id_data[0]
    
    @property
    def ISO(self):
        return self._id_data[1]
    
    @property
    def RT_GAS_DESC(self):
        return self._id_data
    
    @property
    def NU(self):
        return self._data[0]
    
    @property
    def SW(self):
        return self._data[1]
    
    @property
    def ELOWER(self):
        return self._data[2]
    
    @property
    def STIMULATED_EMISSION_REF(self):
        return self._data[3]
    
    @property
    def A(self):
        return self._data[4]
    
    @property
    def SELF_BROADENING_PARAMS(self):
        """
        gamma_self, n_self, delta_self
        """
        return self._data[5:8]
    
    @property
    def FOREIGN_BROADENING_PARAMS(self):
        """
        (gamma_amb, n_amb, delta_amb) for each ambient gas
        """
        return self._data[8:]

@dc.dataclass
class PseudoContSpecData:
    s_min : float
    t_cont : float
    p_cont : float
    rt_gas_desc : RadtranGasDescriptor
    broadening_molecule_ids : tuple[int,...]
    req_wn_range : tuple[float,float]
    
    # private attrs
    _molecular_mass : float # mass of isotopologue (g)
    _data : np.ndarray # [X, N_cont_bins]
    _result_cache : Cache = dc.field(default_factory=lambda : _MODULE_CACHE)
    _data_hash : int = 0

    @classmethod
    def create_from(
            cls, 
            mol_id : int, 
            iso_id : int, 
            ambient_gasses : tuple[ans.enum.AmbientGasEnum,...],
            single_iso_pseudo_continuum_data : PseudoContinuumData,
            cache : None | Cache = None,
    ) -> Self:

        rt_gas_desc = RadtranGasDescriptor(mol_id, iso_id)
        
        instance = cls(
            single_iso_pseudo_continuum_data.s_max,
            single_iso_pseudo_continuum_data.t_cont,
            single_iso_pseudo_continuum_data.p_cont,
            rt_gas_desc,
            (-1, *(int(x) for x in ambient_gasses)),
            single_iso_pseudo_continuum_data.req_wn_range,
            _molecular_mass = rt_gas_desc.molecular_mass,
            _data = None
        )
        
        if cache is not None:
            instance._result_cache = cache
        
        instance._set_data_from_pseudo_continuum_data(single_iso_pseudo_continuum_data)
        
        return instance
    
    def _set_data_from_pseudo_continuum_data(self, pseudo_continuum_data : PseudoContinuumData):
        #print(f'{pseudo_continuum_data=}')
        n = pseudo_continuum_data.wn_bin_center.shape[0]
        m = (
            2   # (wn_bin_center, wn_bin_width,)
            + 2 # (line_strength_sum, lsw_mean_lower_state_energy,)
            + 3 # (lsw_gamma_self, lsw_n_self, lsw_delta_self)
            + pseudo_continuum_data.line_strength_weighted_gamma_amb.shape[1]*3 # (lsw_gamma_amb, lsw_n_amb, lsw_delta_amb) for x ambient gasses
        )
        if self._data is None or self._data.shape != (m,n):
            self._data = np.zeros((m,n), dtype=float)
        
        self.WN_BIN_CENTER[...] = pseudo_continuum_data.wn_bin_center
        self.WN_BIN_WIDTH[...] = pseudo_continuum_data.wn_bin_width
        self.LINE_STRENGTH_SUM[...] = pseudo_continuum_data.line_strength_sum
        self.LSW_MEAN_LOWER_STATE_ENERGY[...] = pseudo_continuum_data.line_strength_weighted_mean_lower_energy_state
        self.SELF_BROADENING_LSW_PARAMS[...] = np.stack(
            [
                pseudo_continuum_data.line_strength_weighted_gamma_self, 
                pseudo_continuum_data.line_strength_weighted_n_self,
                np.zeros_like(pseudo_continuum_data.line_strength_weighted_n_self), # lsw_delta_self
            ],
            axis=0
        )
        self.FOREIGN_BROADENING_LSW_PARAMS[...] = np.stack(
            [
                *(pseudo_continuum_data.line_strength_weighted_gamma_amb.T),
                *(pseudo_continuum_data.line_strength_weighted_n_amb.T),
                *(np.zeros_like(pseudo_continuum_data.line_strength_weighted_n_amb.T)) # lsw_delta_amb
            ],
            axis=0
        )
        
        self._data.flags.writeable = False
        self._data_hash = hash(bytes(self._data))
    
    @property
    def WN_BIN_CENTER(self):
        return self._data[0]
    
    @property
    def WN_BIN_WIDTH(self):
        return self._data[1]
    
    @property
    def LINE_STRENGTH_SUM(self):
        return self._data[2]
    
    @property
    def LSW_MEAN_LOWER_STATE_ENERGY(self):
        return self._data[3]
    
    @property
    def WAVE_AND_LINE_DATA(self):
        """
        (wn_bin_center, wn_bin_width, line_strength_sum, lsw_mean_lower_state_energy)
        """
        return self._data[:4]
    
    @property
    def SELF_BROADENING_LSW_PARAMS(self):
        return self._data[4:7]
    
    @property
    def FOREIGN_BROADENING_LSW_PARAMS(self):
        return self._data[7:]
    
    @property
    def ALL_BROADENING_LSW_PARAMS(self):
        """
            (gamma, n, delta) for "self" and ambient gasses
        """
        return self._data[4:]
    
    @property
    def has_data(self):
        """
        Return `True` if there is some data in the pseudo continuum that would produce some absorption.
        """
        return np.any(self._data[2] != 0)
    
    def cache_identity(self) -> tuple:
        return (
            self.s_min,
            self.t_cont,
            self.p_cont,
            self.broadening_molecule_ids,
            self.req_wn_range,
            *self.rt_gas_desc,
            self._molecular_mass,
            self._data_hash,
        )
    
    def add_lines(
            self,
            line_set_data : LineSetSpecData,
            partition_function : PFList,
            s_max : float, # maximum strength of line to add to continuum
    ):
        """
        Add lines with strength below `s_max` to the pseudo continuum from `line_set_data` 
        """
        self._data.flags.writeable = True
        
        self._data[3:] *= self._data[2:3] # multiply "line strength weighted means" by line strength sum
        
        # loop through `line_set_data` and add lower state energies and broadening params to correct wn_bins
        bin_edges = np.zeros((self.WN_BIN_CENTER.shape[0]+1,), dtype=float)
        bin_edges[:-1] = self.WN_BIN_CENTER - self.WN_BIN_WIDTH/2
        bin_edges[-1] = self.WN_BIN_CENTER[-1] + self.WN_BIN_WIDTH[-1]/2
        
        line_strength = line_set_data.get_line_strength(
            t_calc = self.t_cont,
            partition_function = partition_function
        )
        
        # NOTE: We need to use bin edges as otherwise there is a degeneracy between things in 0th bin and
        # things to the left of 0th bin. However, we must then subtract 1 from the index.
        ls_idxs = np.digitize(line_set_data.NU, bin_edges, right=False) - 1
        ls_mask = (line_strength <= s_max) & (ls_idxs > 0) & (ls_idxs < (self._data.shape[1]))
    
        for i in np.flatnonzero(ls_mask):
            self.LINE_STRENGTH_SUM[ls_idxs[i]] += line_strength[i]
            self.LSW_MEAN_LOWER_STATE_ENERGY[ls_idxs[i]] += (line_set_data.ELOWER[i]*line_strength[i])
            self.SELF_BROADENING_LSW_PARAMS[:,ls_idxs[i]] += (line_set_data.SELF_BROADENING_PARAMS[:,i]*line_strength[i])
            self.FOREIGN_BROADENING_LSW_PARAMS[:,ls_idxs[i]] += (line_set_data.FOREIGN_BROADENING_PARAMS[:,i]*line_strength[i])
        
        zero_mask = self._data[2]==0
        self._data[3:, ~zero_mask] /= self._data[2:3, ~zero_mask] # divide by line strength sum to get "line strength weighted means" again
        self._data[3:, zero_mask] = 0.0
        
        
        self.s_min = s_max
        
        self._data.flags.writeable=False
        self._data_hash = hash(bytes(self._data))
        
        line_set_data.remove_lines(ls_mask)
        line_set_data.s_min = s_max
        
        assert not np.any(np.isnan(self._data)), "No data should be NAN when adding lines to PseudoContSpecData"
    
    def add_monochromatic_absorption(
        self,
        wn_grid : np.ndarray, # [N_waves]
        lineshape_fn : Callable[[float,float,float], float],
        t_calc : float, # Kelvin
        p_calc : float, # Atmospheres
        partition_function : Callable[[float], float],
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        isotopic_abundance : float = 1.0, # abundance of the `self` isotopologue.
        
        out : None | np.ndarray = None, #[N_waves]
        
        store : None | np.ndarray = None, #[3,N_lines]
        store_x : None | np.ndarray = None, #[N_pc_bins]
        store_y : None | np.ndarray = None, #[2*n_neighbour_bins+1]
        store_z : None | np.ndarray = None, #[2, N_waves]
        
        n_neighbour_bins : int = 3,
        
        wn_calc_range : None | tuple[float,float] = None,
        use_cache : bool = True,
    ) -> np.ndarray:
        """
        Adds monochromatic absorption from PseudoContinuum. Will calculate at `self.WN_BIN_CENTER` and interpolate to `wn_grid`
        """
        
        if out is None:
            out = np.zeros_like(wn_grid, dtype=float)
        
        if not self.has_data: # Skip all calculation if no data present
            return out
        
        
        if use_cache:
            wn_grid.flags.writeable = False
            mol_mix_frac.flags.writeable = False
            
            result_bucket = (self.__class__.__name__, 'add_monochromatic_absorption')
            result_identity = (
                self.cache_identity(),
                hash(bytes(wn_grid.data)),
                id(lineshape_fn),
                t_calc, 
                p_calc,
                id(partition_function),
                hash(bytes(mol_mix_frac.data)),
                isotopic_abundance,
                n_neighbour_bins,
                wn_calc_range,
            )
            cache_result = self._result_cache.get(result_bucket, result_identity, None)
            
            if cache_result is not None:
                out[...] = cache_result
                return out
        
        
        if store is None:
            store = np.zeros((3,self.WN_BIN_WIDTH.shape[0]), dtype=float)
        if store_x is None:
            store_x = np.zeros((self.WN_BIN_WIDTH.shape[0],), dtype=float)
        if store_y is None:
            store_y = np.zeros((2*n_neighbour_bins+1,), dtype=float)
        if store_z is None:
            store_z = np.zeros((2, out.shape[0],), dtype=float)
    
        
        q_ratio = partition_function(self.t_cont) / partition_function(t_calc)
        
        wn_mask = np.ones((self._data.shape[1],), dtype=bool) if wn_calc_range is None else ((wn_calc_range[0] <= self.WN_BIN_CENTER) & (self.WN_BIN_CENTER <= wn_calc_range[1]))
        
        add_pseudo_continuum_monochromatic_absorption(
            wn_grid,
            lineshape_fn,
            t_calc,
            self.t_cont,
            p_calc,
            self.p_cont,
            q_ratio,
            isotopic_abundance,
            self._molecular_mass,
            mol_mix_frac,
            self.ALL_BROADENING_LSW_PARAMS[:, wn_mask], # self and non-self broadening
            *self.WAVE_AND_LINE_DATA[:, wn_mask],
            out = out,
            
            store = store,
            store_x = store_x,
            store_y = store_y,
            store_z = store_z,
            n_neighbour_bins = n_neighbour_bins
        )
        
        if use_cache:
            self._result_cache.set(result_bucket, result_identity, out)
        
        return out


@dc.dataclass(slots=True)
class AnsDatabase:
    """
    Handles getting data out of archNEMESIS spectral line database HDF5 files.
    """
    LINE_DATABASE : str
    PARTITION_FUNCTION_DATABASE : str
    CONTINUUM_DATABASE : str | None = None # If `None` will use the same file as `LINE_DATABASE`
    cache : None | Cache = _MODULE_CACHE
    
    # private attrs
    _ans_line_data_file : AnsLineDataFile = None
    _ans_partition_fn_file : AnsPartitionFunctionDataFile = None
    _ans_pseudo_continuum_file : AnsPseudoContinuumFile = None
    
    
    def __post_init__(self):
        if self.LINE_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a LINE_DATABASE')
        if self.PARTITION_FUNCTION_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a PARTITION_FUNCTION_DATABASE')
        
        ref_dbase_downloader = get_reference_database_downloader_for(self.LINE_DATABASE)
        if ref_dbase_downloader is not None:
            ref_dbase_downloader.action_download_reference_database()
            
        self._ans_line_data_file = AnsLineDataFile(self.LINE_DATABASE)
        self._ans_partition_fn_file = AnsPartitionFunctionDataFile(self.PARTITION_FUNCTION_DATABASE)
        self._ans_pseudo_continuum_file = AnsPseudoContinuumFile(self.LINE_DATABASE if self.CONTINUUM_DATABASE is None else self.CONTINUUM_DATABASE)
    
    def fetch_partition_fn(
            self,
            mol_id : int | tuple[int,...] | np.ndarray,
            iso_id : int | tuple[int,...] | tuple[tuple[int,...]] | np.ndarray,
            out : None | list | tuple[list,...] = None,
            refresh : bool = False,
    ) -> tuple[list[PFList],...] | list[PFList] | Self:
        """
        Fetches partition function for a single isotopologue, a set of isotopologues, or a set of molecules and isotopologues.
        
        ## ARGUMENTS ##
            out - If not `None` will place the result into this tuple-of-lists-like-object (by index of supplied `mol_id` and `iso_id`)
                  and return `self`. If not `None` will return a newly created tuple-of-lists that contains the requested data.
        """
        #print('AnsDatabase::fetch_partition_fn(...)')
        #print(f'{mol_id=}')
        #print(f'{iso_id=}')
        #print(f'{out=}')
        #print(f'{refresh=}')
        
        mol_id, iso_id = mol_id_and_iso_id_to_arrays(mol_id, iso_id)
            
        if out is None:
            result = get_container_for_iso_id_array(iso_id)
        else:
            result = out
        
        if isinstance(result, tuple):
            for i, a_mol_id in enumerate(mol_id):
                j = 0
                for a_iso_id in iso_id[i]:
                    if a_iso_id != INVALID_ISOTOPOLOGUE_ID:
                        result[i][j] = self._single_iso_fetch_partition_fn(a_mol_id, a_iso_id, refresh=refresh)
                        j += 1
        else:
            j = 0
            for a_iso_id in iso_id[0]:
                if a_iso_id != INVALID_ISOTOPOLOGUE_ID:
                    result[j] = self._single_iso_fetch_partition_fn(mol_id[0], a_iso_id, refresh=refresh)
                    j += 1
        
        if out is None:
            return result
        else:
            return self
        
    
    def _single_iso_fetch_partition_fn(
            self,
            mol_id : int,
            iso_id : int,
            refresh : bool = False,
    ) -> PFList:
        pf_instance = None
        if self.cache is not None:
            pf_cache_bucket = (self.PARTITION_FUNCTION_DATABASE, 'partition_function')
            pf_cache_identity = (mol_id, iso_id)
            
            pf_instance = self.cache.get(pf_cache_bucket, pf_cache_identity, None)
        #print(f'{pf_instance=}')
        
        if refresh or pf_instance is None:
            #print('Must get data')
            pf_instance = self._ans_partition_fn_file.get_data(RadtranGasDescriptor(mol_id,iso_id).gas_name, iso_id)
            
            if self.cache is not None:
                self.cache.set(pf_cache_bucket, pf_cache_identity, pf_instance)
        
        return pf_instance
    
    
    def fetch_line_data(
            self,
            mol_id : int | tuple[int,...] | np.ndarray,
            iso_id : int | tuple[int,...] | tuple[tuple[int,...]] | np.ndarray,
            wn_min : float, # Always wavenumber (cm^{-1})
            wn_max : float, # Always wavenumber (cm^{-1})
            s_min : float = -1,
            temperature : float = 0,
            ambient_gasses : tuple[ans.enum.AmbientGasEnum,...] = (ans.enum.AmbientGasEnum.AIR,),
            out : None | list | tuple[list,...] = None,
            refresh : bool = False,
    ) -> tuple[list[tuple[tuple[int, int], LineSetData, PseudoContinuumData]],...] | list[tuple[tuple[int,int], LineSetData, PseudoContinuumData]] | Self:
        """
        Fetches line data for a single isotopologue, a set of isotopologues, or a set of molecules and isotopologues.
        
        ## ARGUMENTS ##
            out - If not `None` will place the result into this tuple-of-lists-like-object (by index of supplied `mol_id` and `iso_id`)
                  and return `self`. If not `None` will return a newly created tuple-of-lists that contains the requested data.
        """
        mol_id, iso_id = mol_id_and_iso_id_to_arrays(mol_id, iso_id)
            
        if out is None:
            result = get_container_for_iso_id_array(iso_id)
        else:
            result = out
        
        #print(f'{type(result)=}')
        if isinstance(result, tuple):
            #print(f'{len(result)=} {len(result[0])=}')
            for i, a_mol_id in enumerate(mol_id):
                j = 0
                for a_iso_id in iso_id[i]:
                    if a_iso_id != INVALID_ISOTOPOLOGUE_ID:
                        result[i][j] = (
                            (a_mol_id, a_iso_id), 
                            *self._single_iso_fetch_line_data(
                                a_mol_id, 
                                a_iso_id, 
                                wn_min, 
                                wn_max, 
                                s_min=s_min, 
                                temperature=temperature, 
                                ambient_gasses=ambient_gasses, 
                                refresh=refresh
                            )
                        )
                        j += 1
        else:
            #print(f'{len(result)=}')
            j = 0
            for a_iso_id in iso_id[0]:
                if a_iso_id != INVALID_ISOTOPOLOGUE_ID:
                    result[j] = (
                            (mol_id[0], a_iso_id),
                            *self._single_iso_fetch_line_data(
                                mol_id[0], 
                                a_iso_id, 
                                wn_min, 
                                wn_max, 
                                s_min=s_min, 
                                temperature=temperature, 
                                ambient_gasses=ambient_gasses, 
                                refresh=refresh
                            )
                        )
                    j += 1
        
        if out is None:
            return result
        else:
            return self
    
    
    def _single_iso_fetch_line_data(
            self,
            mol_id : int,
            iso_id : int,
            wn_min : float, # Always wavenumber (cm^{-1})
            wn_max : float, # Always wavenumber (cm^{-1})
            s_min : float = -1, # 
            temperature : float = 0, # Kelvin
            ambient_gasses : tuple[ans.enum.AmbientGasEnum,...] = (ans.enum.AmbientGasEnum.AIR,),
            refresh : bool = False,
    ) -> tuple[LineSetData, PseudoContinuumData]:

        ld_instance = None
        
        if self.cache is not None:
            # build data group
            ld_cache_bucket = (self.LINE_DATABASE, 'line_data')
            ld_cache_identity = (mol_id, iso_id, s_min, temperature, *ambient_gasses)
            
            ld_instance = self.cache.get(ld_cache_bucket, ld_cache_identity, None)
        
        if refresh or ld_instance is None or not wn_range_is_within((wn_min, wn_max), ld_instance.req_wn_range):
            # Get line set data. Returned `ls_instance.s_min` should be less than or equal to `s_min`, but accept larger values if no smaller ones are available
            ld_instance = self._ans_line_data_file.get_data(
                mol_name = RadtranGasDescriptor(mol_id, iso_id).gas_name, 
                local_iso_id = iso_id, 
                s_min = s_min,
                ambient_gasses = ambient_gasses,
                temperature = temperature,
                requested_wn_range = (wn_min, wn_max),
            )
            
            if self.cache is not None:
                self.cache.set(ld_cache_bucket, ld_cache_identity, ld_instance)
    
        if wn_min == 0:
            wn_min = np.min(ld_instance.nu)
        if wn_max == np.inf:
            wn_max = np.max(ld_instance.nu)
    
        if self._ans_pseudo_continuum_file is None:
            pc_instance = AnsPseudoContinuumFile._get_null_data(
                ld_instance.s_min,
                ld_instance.t_ref,
                ld_instance.p_ref,
                (wn_min, wn_max),
                len(ambient_gasses),
            )
        else:
            pc_instance = None
            
            if self.cache is not None:
                pc_cache_bucket = (self.CONTINUUM_DATABASE, 'pseudo_continuum')
                pc_cache_identity = (mol_id, iso_id, ld_instance.s_min, ld_instance.t_ref, *ambient_gasses)
            
                pc_instance = self.cache.get(pc_cache_bucket, pc_cache_identity, None)
            
            if refresh or pc_instance is None or not wn_range_is_within((wn_min, wn_max), pc_instance.req_wn_range):
                # Get continuum data. `pc_instance.s_min` should be less than or equal to `s_min`, if not satisfied return null data
                pc_instance = self._ans_pseudo_continuum_file.get_data(
                        mol_name = RadtranGasDescriptor(mol_id, iso_id).gas_name,
                        local_iso_id = iso_id,
                        temperature = ld_instance.t_ref, # try to match temperature to `ld_instance` if possible
                        s_max = ld_instance.s_min if s_min <= 0 else s_min, # in special cases, match to `ld_instance`, otherwise use passed value
                        s_max_null = ld_instance.s_min, # always match to `ld_instance` when returning null data
                        ambient_gasses = ambient_gasses,
                        requested_wn_range = (wn_min, wn_max),
                    )
                _lgr.debug(f'{pc_instance=}')
                if self.cache is not None:
                    self.cache.set(pc_cache_bucket, pc_cache_identity, pc_instance)
        
        return ld_instance, pc_instance
    
    def fetch(
            self,
            mol_id : int | tuple[int,...] | np.ndarray,
            iso_id : int | tuple[int,...] | tuple[tuple[int,...]] | np.ndarray,
            wn_min : float, # Always wavenumber (cm^{-1})
            wn_max : float, # Always wavenumber (cm^{-1})
            s_min : float = -1,
            temperature : float = 0,
            ambient_gasses : tuple[ans.enum.AmbientGasEnum,...] = (ans.enum.AmbientGasEnum.AIR,),
            out : None | list | tuple[list,...] = None,
            refresh : bool = False,
    ) -> tuple[list[tuple[tuple[int,int], LineSetData, PseudoContinuumData, PFList]],...] | list[tuple[tuple[int,int], LineSetData, PseudoContinuumData, PFList]] | Self:
        mol_id, iso_id = mol_id_and_iso_id_to_arrays(mol_id, iso_id)
    
        if out is None:
            result = get_container_for_iso_id_array(iso_id)
        else:
            result = out
    
        if isinstance(result, tuple):
            for i, a_mol_id in enumerate(mol_id):
                j = 0
                for a_iso_id in iso_id[i]:
                    if a_iso_id != INVALID_ISOTOPOLOGUE_ID:
                        result[i][j] = (
                            (a_mol_id, a_iso_id),
                            *self._single_iso_fetch_line_data(
                                a_mol_id, 
                                a_iso_id, 
                                wn_min, 
                                wn_max, 
                                s_min=s_min, 
                                temperature=temperature, 
                                ambient_gasses=ambient_gasses, 
                                refresh=refresh
                            ),
                            self._single_iso_fetch_partition_fn(
                                a_mol_id,
                                a_iso_id,
                                refresh,
                            )
                        )
                        j += 1
        else:
            j = 0
            for a_iso_id in iso_id[i]:
                if a_iso_id != INVALID_ISOTOPOLOGUE_ID:
                    result[j] = (
                        (mol_id[0], a_iso_id),
                        *self._single_iso_fetch_line_data(
                            mol_id[0], 
                            a_iso_id, 
                            wn_min, 
                            wn_max, 
                            s_min=s_min, 
                            temperature=temperature, 
                            ambient_gasses=ambient_gasses, 
                            refresh=refresh
                        ),
                        self._single_iso_fetch_partition_fn(
                            a_mol_id,
                            a_iso_id,
                            refresh,
                        )
                    )
                    j += 1
        if out is None:
            return result
        else:
            return self
    
    
    def single_iso_fetch(
            self,
            mol_id : int,
            iso_id : int,
            wn_min : float, # Always wavenumber (cm^{-1})
            wn_max : float, # Always wavenumber (cm^{-1})
            s_min : float = -1,
            temperature : float = 0,
            ambient_gasses : tuple[ans.enum.AmbientGasEnum,...] = (ans.enum.AmbientGasEnum.AIR,),
            refresh : bool = False,
    ) -> tuple[LineSetData, PseudoContinuumData, PFList]:
        ld_instance, pc_instance = self._single_iso_fetch_linedata(
            mol_id,
            iso_id,
            wn_min,
            wn_max,
            s_min,
            temperature,
            ambient_gasses,
            refresh
        )
        pf_instance = self._single_iso_fetch_partition_fn(
            mol_id,
            iso_id,
            refresh,
        )
        
        return ld_instance, pc_instance, pf_instance


@dc.dataclass(slots=True)
class LineDataParams:
    ambient_gasses : tuple[ans.enum.AmbientGasEnum,...] = (ans.enum.AmbientGasEnum.AIR,)
    wn_min : float = 0.0
    wn_max : float = np.inf
    s_min : float = -1
    t_req : float = 296 # (Kelvin) temperature that we want to request data for
    p_req : float = 1 # (Atmospheres) pressure that we want to request data for
    default_continuum_wn_bin_width : float = 1.0 # (cm^{-1})
    
    def cache_identity(self) -> tuple:
        """
        Returns a tuple that can be used to identify a set of line and continuum data that was calculated previously
        """
        return (
            self.s_min,
            self.t_req,
            self.p_req,
            self.ambient_gasses,
        )


class LineData_0:
    def __init__(
            self,
            ID : int,
            ISO : int = 0,
            ambient_gasses : ans.enum.AmbientGasEnum | tuple[ans.enum.AmbientGasEnum,...] = (ans.enum.AmbientGasEnum.AIR,),
            LINE_DATABASE : None | str = None,
            PARTITION_FUNCTION_DATABASE : None | str = archnemesis_path()+'/archnemesis/Data/partition_functions/tips2025.h5',
            CONTINUUM_DATABASE : None | str = None, # If `None` will use the same file as `LINE_DATABASE`
            cache : None | Cache = _MODULE_CACHE,
    ):
        if LINE_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a LINE_DATABASE')
        if PARTITION_FUNCTION_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a PARTITION_FUNCTION_DATABASE')
        
        
        
        # Private attrs
        self._ID : int = INVALID_MOLECULE_ID
        self._ISO = None
        self._ans_database : AnsDatabase = AnsDatabase(LINE_DATABASE, PARTITION_FUNCTION_DATABASE, CONTINUUM_DATABASE, cache=cache)
        self._rt_gas_descs : None | tuple[RadtranGasDescriptor,...] = None 
        self._default_iso_abundances : None | np.ndarray = None,
        self._n_isos : None | int = None
        self._mol_ids : None | np.ndarray = None
        self._iso_ids : None | np.ndarray = None
        self._mol_id_tpl : None | tuple[int,...] = None
        self._iso_id_tpl : None | tuple[tuple[int,...],...] = None
        self._params : LineDataParams = LineDataParams(ambient_gasses=(ambient_gasses,) if isinstance(ambient_gasses, ans.enum.AmbientGasEnum) else ambient_gasses)
        self._params_fetched_lines_last = False
        self._params_fetched_partition_last = False
        self._combined_line_data = None
        
        # Public attrs from args
        self.ID = ID
        self.ISO = ISO
        self.cache = cache
        
        # Public attrs
        self.partition_fn_data : list[PFList]             = [None]*self.n_isos
        self.line_data         : list[LineSetSpecData]    = [None]*self.n_isos
        self.continuum_data    : list[PseudoContSpecData] = [None]*self.n_isos
    
    
    def __repr__(self):
        s = (
            'LineData_0('
            f'MEM={id(self)},'
            f'ID={self.ID},'
            f'ISO={self.ISO},'
            f'params={self._params}'
            ')'
        )
        return s
    
    def __eq__(self, other : Self) -> bool:
        """
        Tests if two LineData_0 instances have approximately
        the same setup.
        """
        _lgr.warn('Comparing LineData_0 instances for equality is not really a good idea. This equality function is here so that we can check that no major information is altered when converting from LEGACY to HDF5 format.')
        is_equal = isinstance(other, LineData_0)
        is_equal = is_equal and (self.ID == other.ID)
        is_equal = is_equal and (self.ISO == other.ISO)
        is_equal = is_equal and (self._params == other._params)
        return is_equal
    
    @property
    def ID(self) -> int:
        return self._ID
    
    @ID.setter
    def ID(self, value : int):
        assert isinstance(value, (int, np.integer)), "LineData_0.ID must be an integer"
        if value != self._ID:
            self._ID = value
            self._rt_gas_descs = None
            self._mol_ids = None
            self._iso_ids = None
            self._mol_id_tpl = None
            self._iso_id_tpl = None
            self._n_isos = None
            self._default_iso_abundances = None
    
    @property
    def ISO(self) -> int | tuple[int,...]:
        return self._ISO
    
    @ISO.setter
    def ISO(self, value : int | tuple[int,...]):
        assert isinstance(value, (int, np.integer)) if not isinstance(value, tuple) else all(isinstance(x,(int,np.integer)) for x in value), "LineData_0.ISO must be an integer or a tuple of integers"
        if value != self._ISO:
            self._ISO = value
            self._rt_gas_descs = None
            self._iso_ids = None
            self._iso_id_tpl = None
            self._n_isos = None
            self._default_iso_abundances = None
    
    @property
    def n_isos(self):
        if self._n_isos is None:
            self._n_isos = len(self.rt_gas_descs)
        return self._n_isos
    
    @property
    def rt_gas_descs(self):
        if self._rt_gas_descs is None:
            self._rt_gas_descs = tuple(GasIsotopes(self.ID, self.ISO).as_radtran_gasses())
        return self._rt_gas_descs
    
    @property
    def default_iso_abundances(self):
        if self._default_iso_abundances is None:
            self._default_iso_abundances = np.array([x.abundance for x in self.rt_gas_descs], dtype=float)
        return self._default_iso_abundances
    
    @property
    def mol_ids(self):
        if self._mol_ids is None:
            unique_mol_ids = []
            for x in self.rt_gas_descs:
                if x.gas_id not in unique_mol_ids:
                    unique_mol_ids.append(x.gas_id)
            self._mol_ids = np.array(unique_mol_ids, dtype=int)
        return self._mol_ids
    
    @property
    def iso_ids(self):
        if self._iso_ids is None:
            # get max number of iso ids for a mol id
            max_iso_id_slots = 0
            for mol_id in self.mol_ids:
                n = len(tuple(filter(lambda x: x.gas_id == mol_id, self.rt_gas_descs)))
                if n > max_iso_id_slots:
                    max_iso_id_slots = n
            
            self._iso_ids = np.ones((self.mol_ids.size, max_iso_id_slots), dtype=int) * INVALID_ISOTOPOLOGUE_ID
            for i, mol_id in enumerate(self.mol_ids):
                for j, rt_gas_desc in enumerate(filter(lambda x: x.gas_id == mol_id, self.rt_gas_descs)):
                    self._iso_ids[i,j] = rt_gas_desc.iso_id
                
        return self._iso_ids
    
    @property
    def mol_id_tpl(self):
        if self._mol_id_tpl is None:
            self._mol_id_tpl = (self.ID,)
        return self._mol_id_tpl
    
    @property
    def iso_id_tpl(self):
        if self._iso_id_tpl is None:
            self._iso_id_tpl = tuple(tuple(x.iso_id for x in self.rt_gas_descs if x.gas_id == mol_id) for mol_id in self.mol_id_tpl)
        return self._iso_id_tpl
    
    @property
    def combined_line_data(self) -> np.array:
        if self._combined_line_data is None and self.line_data is not None:
            self._combined_line_data = CombinedLineSetSpecData.create_from(self.line_data)
        return self._combined_line_data
    
    @property
    def max_lines_or_bins(self) -> int:
        return max(max(x.NU.shape[0] for x in self.line_data), max(x.WN_BIN_CENTER.shape[0] if x is not None else 0 for x in self.continuum_data))
    
    def cache_identity(self) -> tuple:
        """
        Returns a tuple that can be used to identify a set of line and continuum data that was calculated previously
        """
        return (self.mol_id_tpl, self.iso_id_tpl, self._params.cache_identity())
    
    def _set_params_direct(self, **kwargs):
        for k,v in kwargs.items():
            if v is not None and getattr(self._params, k) != v:
                setattr(self._params, k, v)
                self._params_fetched_lines_last = False
                self._params_fetched_partition_last = False
    
    def set_params(
            self,
            vmin : None | float = None,
            vmax : None | float = None,
            s_min : None | float = None,
            t_req : None | float = None, # Kelvin
            p_req : None | float = None, # Atmospheres
            default_continuum_bin_width : None | float = None,
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm,
    ) -> Self:
        if vmin is not None:
            vmin = WavePoint(vmin, wave_unit).as_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value
        
        if vmax is not None:
            vmax = WavePoint(vmax, wave_unit).as_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value
        
        if default_continuum_bin_width is not None:
            default_continuum_bin_width = WavePoint(default_continuum_bin_width, wave_unit).as_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value
        
        self._set_params_direct(
            wn_min = vmin, 
            wn_max = vmax, 
            s_min = s_min, 
            t_req = t_req, 
            p_req = p_req,
            default_continuum_wn_bin_width = default_continuum_bin_width,
        )
        
        return self
    
    def get_params(self) -> LineDataParams:
        return LineDataParams(**vars(self._params)) # return a copy not the original object
    
    @contextmanager
    def param_context(
            self,
            vmin : None | float = None,
            vmax : None | float = None,
            s_min : None | float = None,
            t_req : None | float = None, # Kelvin
            p_req : None | float = None, # Atmospheres
            default_continuum_bin_width : None | float = None,
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm,
    ) -> Self:
        prev_params = self.get_params()
        
        self.set_params(vmin, vmax, s_min, t_req, p_req, default_continuum_bin_width, wave_unit)
        
        yield self
        
        self._set_params_direct(**vars(prev_params))
        
        return

    def fetch_partition_fn(
            self,
            refresh : bool = False,
    ) -> Self:
        if self._params_fetched_partition_last and not refresh:
            return self
        
        self.partition_fn_data = self._ans_database.fetch_partition_fn(self.mol_ids, self.iso_ids, refresh=refresh)
        #print(f'{self.partition_fn_data=}')
        self._params_fetched_partition_last = True
        
        return self

    def fetch_linedata(
            self, 
            refresh : bool = False,
    ) -> Self:
    
        if self._params_fetched_lines_last and not refresh:
            return self
        
        self.fetch_partition_fn(refresh=refresh)
    
        _lgr.debug(f'{self._params=}')
        _lgr.debug(f'{self.mol_ids=}')
        _lgr.debug(f'{self.iso_ids=}')
        _lgr.debug(f'{self.cache=}')
        
        
        retrieved_from_cache = False # Flag
        
        if self.cache is not None:
            # build data group
            ld_pc_cache_bucket = ('line_and_continuum_data_pairs',id(self._ans_database))
            ld_pc_cache_identity = self.cache_identity()
            
            ld_pc_pairs = self.cache.get(ld_pc_cache_bucket, ld_pc_cache_identity, None)
            if (
                refresh 
                or ld_pc_pairs is None 
                or not all(wn_range_is_within((self._params.wn_min, self._params.wn_max), ld_pc_pair[0].req_wn_range) for ld_pc_pair in ld_pc_pairs)
            ): # retrieval from cache failed
                retrieved_from_cache = False
                
                # When getting data again, request the superset of all wavenumbers requested
                if ld_pc_pairs is not None:
                    for ld_pc_pair in ld_pc_pairs:
                        self._params.wn_min = self._params.wn_min if self._params.wn_min <= ld_pc_pair[0].req_wn_range[0] else ld_pc_pair[0].req_wn_range[0]
                        self._params.wn_max = self._params.wn_max if self._params.wn_max >= ld_pc_pair[0].req_wn_range[1] else ld_pc_pair[0].req_wn_range[1]
            else:
                retrieved_from_cache = True


        if not retrieved_from_cache:
            self.line_data         : list[LineSetSpecData]    = [None]*self.n_isos
            self.continuum_data    : list[PseudoContSpecData] = [None]*self.n_isos
            
            id_line_cont_triplet = get_container_for_iso_id_array(self.iso_ids)
            self._ans_database.fetch_line_data(
                self.mol_ids,
                self.iso_ids,
                self._params.wn_min,
                self._params.wn_max,
                self._params.s_min,
                self._params.t_req,
                self._params.ambient_gasses,
                out = id_line_cont_triplet
            )
            
            # At this point `id_line_cont_triplet` is 
            # a list of ((mol_id, iso_id), line_data, cont_data) entries
            # as any instance of LineData_0 only everh as one molecule
            
            for i, ((mol_id, iso_id), line_data, cont_data) in enumerate(id_line_cont_triplet):
                #print(f'DEBUG: {i=} {mol_id=} {iso_id=} {line_data.s_min=} {cont_data.s_max=}')
                
                self.line_data[i] = LineSetSpecData.create_from(mol_id, iso_id, self._params.ambient_gasses, line_data, cache=self.cache)
                self.continuum_data[i] = PseudoContSpecData.create_from(mol_id, iso_id, self._params.ambient_gasses, cont_data, cache=self.cache)
        
                #print(f'DEBUG: {self._params.s_min=} {self.line_data[i].s_min=} {self.continuum_data[i].s_min=}')
        
                # If we need to, add more lines into the continuum so that we always have the requested `s_min` for both the line set data and continuum data.
                if (self._params.s_min > self.line_data[i].s_min):
                    if (self._params.s_min > self.continuum_data[i].s_min):
                        #print('DEBUG: Adding lines')
                        
                        self.continuum_data[i].add_lines(
                            self.line_data[i],
                            self.partition_fn_data[i],
                            self._params.s_min
                        )
                    elif (self._params.s_min < self.continuum_data[i].s_min):
                        raise RuntimeError(f'Have retrieved continuum data that has a larger `s_min` ({self.continuum_data[i].s_min=}) than requested ({self._params.s_min=}). This is not allowed as it can lead to double-counting of absorption lines.')
        
        if self.cache is not None:
            # Store result in cache
            self.cache.set(ld_pc_cache_bucket, ld_pc_cache_identity, (self.line_data, self.continuum_data))
        
        # Remember that we used these parameters to fetch the last lot of linedata
        self._params_fetched_lines_last = True
        
        return self

    def calculate_doppler_width(
            self,
            t_calc : float,
            wave_calc_range : None | tuple[float,float] = None,
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm, # unit of `wave_calc_range`
            combined_output : bool = False, # if `True` will combine output into a single array instead of splitting by isotopologue
    ) -> tuple[np.ndarray,...]:
        if not self._params_fetched_lines_last:
            self.fetch_linedata()
        
        if wave_calc_range is not None and wave_unit != ans.enum.WaveUnitEnum.Wavenumber_cm:
            wn_calc_range = (WavePoint(wave_calc_range[0], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value, WavePoint(wave_calc_range[1], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value)
        else:
            wn_calc_range = wave_calc_range
        
        if combined_output:
            n_iso_lines = []
            for iso_line_data in self.line_data:
                if wn_calc_range is None:
                    n_iso_lines.append( iso_line_data.n_lines)
                else:
                    n_iso_lines.append(np.count_nonzero((wn_calc_range[0] <= iso_line_data.NU ) & (iso_line_data.NU < wn_calc_range[1])))
            
            result = np.empty((sum(n_iso_lines),), dtype=float)
            
            i_start = 0
            for i, iso_line_data in enumerate(self.line_data):
                i_end = i_start + n_iso_lines[i]
                result[i_start:i_end] = iso_line_data.get_doppler_width(t_calc, wn_calc_range)
                i_start = i_end
        else:
            result = tuple(x.get_doppler_width(t_calc, wn_calc_range) for x in self.line_data)
        
        return result
    
    def calculate_lorentz_width(
            self,
            t_calc : float,
            p_calc : float,
            amb_frac : float | np.ndarray = 0.5,
            wave_calc_range : None | tuple[float,float] = None,
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm, # unit of `wave_calc_range`
            combined_output : bool = False, # if `True` will combine output into a single array instead of splitting by isotopologue
    ) -> tuple[np.ndarray,...]:
        if not self._params_fetched_lines_last:
            self.fetch_linedata()
        
        # create array that holds mixing fractions for self as well as ambient gasses
        if isinstance(amb_frac, float):
            mol_mix_frac = np.array([1-amb_frac, amb_frac], dtype=float)
        else:
            mol_mix_frac = np.array([1 - sum(amb_frac), *amb_frac], dtype=float)
        
        if wave_calc_range is not None and wave_unit != ans.enum.WaveUnitEnum.Wavenumber_cm:
            wn_calc_range = (WavePoint(wave_calc_range[0], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value, WavePoint(wave_calc_range[1], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value)
        else:
            wn_calc_range = wave_calc_range
            
        assert mol_mix_frac.shape[0] == len(self._params.ambient_gasses)+1, "LineData_0::calculate_lorentz_width(...) `amb_frac` must have enough entries for each ambient gas"
        
        if combined_output:
            n_iso_lines = []
            for iso_line_data in self.line_data:
                if wn_calc_range is None:
                    n_iso_lines.append( iso_line_data.n_lines)
                else:
                    n_iso_lines.append(np.count_nonzero((wn_calc_range[0] <= iso_line_data.NU ) & (iso_line_data.NU < wn_calc_range[1])))
            
            result = np.empty((sum(n_iso_lines),), dtype=float)
            
            i_start = 0
            for i, iso_line_data in enumerate(self.line_data):
                i_end = i_start + n_iso_lines[i]
                result[i_start:i_end] = iso_line_data.get_lorentz_width(t_calc, p_calc, mol_mix_frac, wn_calc_range)
                i_start = i_end
        
        else:
            result = tuple(x.get_lorentz_width(t_calc, p_calc, mol_mix_frac, wn_calc_range) for x in self.line_data)
        
        return result

    def calculate_line_strength(
            self,
            t_calc : float,
            wave_calc_range : None | tuple[float,float] = None,
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm, # unit of `wave_calc_range`
            combined_output : bool = False, # if `True` will combine output into a single array instead of splitting by isotopologue
    ) -> np.ndarray | tuple[np.ndarray,...]:
        if not self._params_fetched_lines_last:
            self.fetch_linedata()
        if not self._params_fetched_partition_last:
            self.fetch_partition_fn()
        
        if wave_calc_range is not None and wave_unit != ans.enum.WaveUnitEnum.Wavenumber_cm:
            wn_calc_range = (WavePoint(wave_calc_range[0], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value, WavePoint(wave_calc_range[1], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value)
        else:
            wn_calc_range = wave_calc_range
        
        result = tuple(self.line_data[i].get_line_strength(t_calc, self.partition_fn_data[i], wn_calc_range) for i in range(len(self.line_data)))
        
        if combined_output:
            combined_result = np.empty((sum(x.shape[0] for x in result),), dtype=float)
            
            i_start = 0
            for i, x in enumerate(result):
                i_end = i_start + x.shape[0]
                combined_result[i_start:i_end] = x
                i_start = i_end
            
            result = combined_result
        
        return result

    def calculate_monochromatic_absorption(
            self,
            wave_grid : np.ndarray,
            
            t_calc : float,
            p_calc : float,

            amb_frac : float | np.ndarray = 0.5,
            wave_calc_range : None | tuple[float,float] = None,
            isotopic_abundance : None | float | np.ndarray = None,
            lineshape_fn : Callable[[float,float,float], float] = ans.lineshape.voigt,
            s_floor : float = 0.0,
            wn_calc_window : float = 25.0, # (cm^{-1})
            wn_approx_window : float = 75.0, # (cm^{-1})
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm, # unit of `wave_grid` and `wave_calc_range`
            
            include_lines : bool = True,
            include_continuum : bool = True,
            include_pressure_shift : bool = True,
            combined_output : bool = True,
            each_iso_output : bool = False,
            use_cache : bool = True,
    ) -> np.ndarray:
    
        out = np.zeros((2,self.n_isos, wave_grid.shape[0]), dtype=float)
    
        self.add_monochromatic_absorption(
            wave_grid = wave_grid,
            t_calc = t_calc,
            p_calc = p_calc,
            amb_frac = amb_frac,
            wave_calc_range = wave_calc_range,
            isotopic_abundance = isotopic_abundance,
            lineshape_fn = lineshape_fn,
            s_floor = s_floor,
            wn_calc_window = wn_calc_window,
            wn_approx_window = wn_approx_window,
            include_lines = include_lines,
            include_continuum = include_continuum,
            include_pressure_shift = include_pressure_shift,
            use_cache = use_cache,
            out = out,
        )
        
        if not each_iso_output:
            out = np.sum(out, axis=1)
        if combined_output:
            out = np.sum(out, axis=0)
        return out

    def add_monochromatic_absorption(
            self,
            wave_grid : np.ndarray,
            
            t_calc : float,
            p_calc : float,
            
            out : None | np.ndarray = None, # if 2-dimensions, will assume the 1st dimension is 2 and will fill with (line_set absorption, continuum absorption), if 3 dimensional will assume 1st dimension is 2 (line set, contiuum) then 2nd dimension is per isotopologue
            store : None | np.ndarray = None,
            
            amb_frac : float | np.ndarray = 0.5,
            wave_calc_range : None | tuple[float,float] = None,
            isotopic_abundance : None | float | np.ndarray = None,
            lineshape_fn : Callable[[float,float,float], float] = ans.lineshape.voigt,
            s_floor : float = 0.0, # Minimum line strength to include. NOTE: Is independent of the value in `self._params.s_min`
            wn_calc_window : float = 25.0, # (cm^{-1})
            wn_approx_window : float = 75.0, # (cm^{-1})
            wave_unit :ans.enum.WaveUnitEnum = ans.enum.WaveUnitEnum.Wavenumber_cm, # unit of `wave_grid` and `wave_calc_range`
            
            include_lines : bool = True,
            include_continuum : bool = True,
            include_pressure_shift : bool = True,
            use_cache : bool = True,
    ) -> np.ndarray:
        
        
        # Create default arrays
        if out is None:
            result = np.zeros_like(wave_grid, dtype=float)
            out = result[...]
        else:
            result = out[...]
        
        if store is None:
            store = np.zeros((4, self.max_lines_or_bins), dtype=float)
        
        
        #Converting spectral array to wavenumbers in cm-1
        if wave_unit != ans.enum.WaveUnitEnum.Wavenumber_cm:
            wn_grid = WavePoint(wave_grid, wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value
        else:
            wn_grid = wave_grid
        
        #Calculating the spectral range required to perform the calculations
        if wave_calc_range is None:
            wn_calc_range = (np.min(wn_grid) - 2*wn_approx_window, np.max(wn_grid) + 2*wn_approx_window)
        elif wave_unit != ans.enum.WaveUnitEnum.Wavenumber_cm:
            wn_calc_range = (WavePoint(wave_calc_range[0], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value, WavePoint(wave_calc_range[1], wave_unit).to_unit(ans.enum.WaveUnitEnum.Wavenumber_cm).value)
        
        # Ensure that `wn_grid` is ascending. Flip both `wn_grid` and `out` if `wn_grid` is decending.
        if np.all(wn_grid[:-1] < wn_grid[1:]):
            pass
        elif np.all(wn_grid[:-1] > wn_grid[1:]):
            out = out.flip(axis=-1)
        else:
            raise RuntimeError('`wave_grid` passed to LineData_0::monochromatic_absorption(...) must be either ascending or decending.')
        
        #Calculating the relative abundances of self and ambient gases
        if isinstance(amb_frac, float):
            mol_mix_frac = np.array([1-amb_frac, amb_frac], dtype=float)
        else:
            mol_mix_frac = np.array([1 - sum(amb_frac), *amb_frac], dtype=float)
        
        assert mol_mix_frac.shape[0] == len(self._params.ambient_gasses)+1, "LineData_0::add_monochromatic_absorption(...) `amb_frac` must have enough entries for each ambient gas"
        
        # Ensure isotopic abundances are arrays of correct length
        # Isotopic abundances are only applied if the gas is a mixture of isotopes (ISO=0)
        if self.ISO == 0:
            if isotopic_abundance is None:
                isotopic_abundance = self.default_iso_abundances
            elif isinstance(isotopic_abundance, float):
                assert self.n_isos == 1, "If provided, there must be an isotopic abundance for each isotopologue in the LineData_0 instance"
                isotopic_abundance = np.array([isotopic_abundance], dtype=float)
            else:
                assert self.n_isos == isotopic_abundance.shape[0], "If provided, there must be an isotopic abundance for each isotopologue in the LineData_0 instance"
        else:
            isotopic_abundance = [1.]

        #Debugging statements
        if _lgr.level <= logging.DEBUG:
            msg = '## ARGUMENTS ##' +'\n\t'.join((
                f'{wave_grid=}',
                f'{t_calc=}',
                f'{p_calc=}',
                f'{amb_frac=}',
                f'{wave_calc_range=}',
                f'{isotopic_abundance=}',
                f'{lineshape_fn=}',
                f'{s_floor=}',
                f'{wn_calc_window=}',
                f'{wn_approx_window=}',
                f'{wave_unit=}',
                f'{include_lines=}',
                f'{include_continuum=}',
                f'{use_cache=}',
                f'{out=}',
                f'{store=}',
            )) + '##-----------##'
            _lgr.debug(msg)
        
        #Defining the array where the results will be stored
        if out.ndim >= 2:
            out_line_set_abs = out[0]
            out_continuum_abs = out[1]
        else:
            out_line_set_abs = out
            out_continuum_abs = out
        
        # Loop over all line data and add monochromatic absorption to `out`
        for i, (iso_line_data, iso_continuum_data) in enumerate(zip(self.line_data, self.continuum_data)):


            _lgr.debug(f'{i=} {iso_line_data._data.shape=} iso_continuum_data._data.shape={iso_continuum_data._data.shape if iso_continuum_data is not None else "None"}')
            
            #Defining the characteristics of the output array
            #   If 1 dimension will add all absorption to same array.  (NWAVE)
            #   If 2 will split line absorption and continuum absorption.  (2,NWAVE)
            #   If 3 will split line absorption and continuum absorption, and split by isotopologue. (2,NISO,NWAVE)
            if out.ndim < 3:
                out_line_set_abs_i = out_line_set_abs
                out_continuum_abs_i = out_continuum_abs
            elif out.ndim == 3:
                assert out.shape[1] == len(self.line_data), f'`out` must have 1, 2 or 3 dimensions ({out.ndim=}). If 1 dimension will add all absorption to same array. If 2 will split line absorption and continuum absorption. If 3 will split line absorption and continuum absorption, and split by isotopologue and therefore must have out.shape[1] == number of isotopoluges ({out.shape=}) ({len(self.line_data)})'
                out_line_set_abs_i = out_line_set_abs[i]
                out_continuum_abs_i = out_continuum_abs[i]
            else:
                raise RuntimeError(f'`out` must have 1, 2 or 3 dimensions ({out.ndim=}). If 1 dimension will add all absorption to same array. If 2 will split line absorption and continuum absorption. If 3 will split line absorption and continuum absorption, and split by isotopologue.')
            
            #Performing the line-by-line spectroscopic calculations
            #This is the formulation from HITRAN
            if include_lines:
                iso_line_data.add_monochromatic_absorption(
                    wn_grid, #[N_waves]
                    lineshape_fn,
                    t_calc,
                    p_calc,
                    self.partition_fn_data[i],
                    mol_mix_frac,
                    isotopic_abundance[i],
                    
                    out = out_line_set_abs_i, #[N_waves]
                    store = store,
                    
                    s_floor = s_floor,
                    wn_calc_window = wn_calc_window,
                    wn_approx_window = wn_approx_window,
                    wn_calc_range = wn_calc_range,
                    include_pressure_shift = include_pressure_shift,
                    use_cache=use_cache,
                )
            
            #Including pseudo-continuum absorption from superposition of weak lines
            #This is the formulation from Irwin+19
            #Here, however, the pseudo-continuum is often pre-computed in the HDF5 database
            if include_continuum:
                _lgr.debug(f'{iso_continuum_data=}')
                iso_continuum_data.add_monochromatic_absorption(
                    wn_grid,
                    lineshape_fn,
                    t_calc,
                    p_calc,
                    self.partition_fn_data[i],
                    mol_mix_frac,
                    isotopic_abundance[i],
                    
                    out = out_continuum_abs_i, 
                    
                    store=store,
                    store_z = np.zeros((2,wn_grid.shape[0],), dtype=float),
                    n_neighbour_bins= 3,
                    
                    wn_calc_range = wn_calc_range,
                    use_cache=use_cache,
                )
            
        return result # This should be a view of `out`

    def plot_linedata(
            self, 
            logscale : bool = True, 
            scatter_style_kw : dict[str,Any] = {},
            line_style_kw : dict[str,Any] = {},
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
        
        if self.line_data is None:
            raise RuntimeError(f'No line data ready in {self}')
        
        scatter_style_defaults = dict(
            s = 2,
            marker='.',
            edgecolor = 'none',
            alpha=0.6,
        )
        scatter_style_defaults.update(scatter_style_kw)
        
        line_style_defaults = dict(
            linewidth=1,
            alpha=0.5,
        )
        line_style_defaults.update(line_style_kw)
        
        ax_style_defaults = dict(
            facecolor='#F8F8F8'
        )
        ax_style_defaults.update(ax_style_kw)
        
        legend_style_defaults = dict(
            fontsize = 10,
            title_fontsize=12
        )
        legend_style_defaults.update(legend_style_kw)
        
        f, ax_array = plt.subplots(
            self.n_isos+1,1, 
            figsize=(12,4*(self.n_isos+1)), 
            gridspec_kw={'hspace':0.3},
            squeeze=False
        )
        ax_array = ax_array.flatten()
        
        combined_ax = ax_array[0]
        combined_ax.set_title('Line data for '+ans.Data.gas_data.id_to_name(self.ID,self.ISO))
        
        line_strengths_max = 0.0
        line_strengths_min = np.inf
        
        for i, iso_line_data in enumerate(self.line_data):

            if iso_line_data.has_data:
                ls_max = iso_line_data.SW.max()
                ls_min = iso_line_data.SW[iso_line_data.SW>0].min()
                line_no_data_str = ''
            else:
                ls_max = 0
                ls_min = np.inf
                line_no_data_str = ''
                        
            line_strengths_max = ls_max if ls_max > line_strengths_max else line_strengths_max
            line_strengths_min = ls_min if ((ls_min < line_strengths_min) and (ls_min > 0)) else line_strengths_min
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(iso_line_data.rt_gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            # Combined plot, all isotopes on one figure, coloured by isotope
            combined_ax.scatter(
                iso_line_data.NU,
                iso_line_data.SW,
                label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){line_no_data_str}',
                zorder = i,
                **scatter_style_defaults
            )
                    
        lgnd = combined_ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1.05),  # Shift legend to the right
            title='Isotope', 
            **legend_style_defaults
        )
        for hdl in lgnd.legend_handles:
            if isinstance(hdl, mpl.collections.PathCollection):
                hdl.set_sizes([50.0])
            else:
                pass
        
        if np.isinf(line_strengths_min):
            line_strengths_min = 1E-25
        if line_strengths_max == 0:
            line_strengths_max = 1E-15
        
        if logscale:
            combined_ax.set_yscale('log')
            combined_ax.set_ylim(line_strengths_min, line_strengths_max * 10)
        
        combined_ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        combined_ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
        combined_ax.set(**ax_style_defaults)
    
        for i, iso_line_data in enumerate(self.line_data):
            ax = ax_array[i+1]
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(iso_line_data.rt_gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            
            if iso_line_data.has_data:
                line_no_data_str = ''
            else:
                line_no_data_str = ' [NO LINE DATA]'
            
            ax.set_title(
                f'Line data ${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){line_no_data_str}'
                )
            
            
            # Plots for specific isotopes, coloured by lower energy state
            
            p1 = ax.scatter(
                iso_line_data.NU,
                iso_line_data.SW,
                c = iso_line_data.ELOWER,
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
                ax.set_ylim(line_strengths_min, line_strengths_max * 10)
            ax.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
            ax.set(**ax_style_defaults)

        plt.tight_layout()

    def plot_continuumdata(
            self, 
            logscale : bool = True, 
            scatter_style_kw : dict[str,Any] = {},
            line_style_kw : dict[str,Any] = {},
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
        
        if self.line_data is None:
            raise RuntimeError(f'No line data ready in {self}')
        
        scatter_style_defaults = dict(
            s = 2,
            marker='.',
            edgecolor = 'none',
            alpha=0.6,
        )
        scatter_style_defaults.update(scatter_style_kw)
        
        line_style_defaults = dict(
            linewidth=1,
            alpha=0.5,
        )
        line_style_defaults.update(line_style_kw)
        
        ax_style_defaults = dict(
            facecolor='#F8F8F8'
        )
        ax_style_defaults.update(ax_style_kw)
        
        legend_style_defaults = dict(
            fontsize = 10,
            title_fontsize=12
        )
        legend_style_defaults.update(legend_style_kw)
        
        f, ax_array = plt.subplots(
            self.n_isos+1,1, 
            figsize=(12,4*(self.n_isos+1)), 
            gridspec_kw={'hspace':0.3},
            squeeze=False
        )
        ax_array = ax_array.flatten()
        
        combined_ax = ax_array[0]
        combined_ax.set_title('Line strength (scatter) and continuum strength (line) coloured by isotopologue')
        
        line_strengths_max = 0.0
        line_strengths_min = np.inf
        
        for i, (iso_line_data, iso_continuum_data) in enumerate(zip(self.line_data, self.continuum_data)):

            if iso_line_data.has_data:
                ls_max = iso_line_data.SW.max()
                ls_min = iso_line_data.SW[iso_line_data.SW>0].min()
                line_no_data_str = ''
            else:
                ls_max = 0
                ls_min = np.inf
                line_no_data_str = ' [NO DATA]'
            
            if iso_continuum_data.has_data:
                cont_ls_min = np.min(iso_continuum_data.LINE_STRENGTH_SUM[iso_continuum_data.LINE_STRENGTH_SUM > 0])
                cont_ls_max = iso_continuum_data.LINE_STRENGTH_SUM.max()
                ls_min = ls_min if ls_min < cont_ls_min else cont_ls_min
                ls_max = ls_max if ls_max > cont_ls_max else cont_ls_max
                cont_no_data_str = ''
            else:
                cont_no_data_str = ' [NO DATA]'
            
            line_strengths_max = ls_max if ls_max > line_strengths_max else line_strengths_max
            line_strengths_min = ls_min if ((ls_min < line_strengths_min) and (ls_min > 0)) else line_strengths_min
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(iso_line_data.rt_gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            # Combined plot, all isotopes on one figure, coloured by isotope
            combined_ax.scatter(
                iso_line_data.NU,
                iso_line_data.SW,
                label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){line_no_data_str}',
                zorder = i,
                **scatter_style_defaults
            )
            
            # Plot twice so we can have contrasting colour, remember to set z_order
            if iso_continuum_data is not None:
                combined_ax.plot(
                    iso_continuum_data.WN_BIN_CENTER,
                    iso_continuum_data.LINE_STRENGTH_SUM,
                    linewidth = line_style_defaults['linewidth']*2,
                    alpha = line_style_defaults['alpha'],
                    color=ax_style_defaults['facecolor'],
                    zorder = self.n_isos + i,
                )
                combined_ax.plot(
                    iso_continuum_data.WN_BIN_CENTER,
                    iso_continuum_data.LINE_STRENGTH_SUM,
                    label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){cont_no_data_str}',
                    zorder = 2*self.n_isos + i,
                    **line_style_defaults
                )
        
        lgnd = combined_ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1.05),  # Shift legend to the right
            title='Isotope', 
            **legend_style_defaults
        )
        for hdl in lgnd.legend_handles:
            if isinstance(hdl, mpl.collections.PathCollection):
                hdl.set_sizes([50.0])
            else:
                pass
        
        if np.isinf(line_strengths_min):
            line_strengths_min = 1E-25
        if line_strengths_max == 0:
            line_strengths_max = 1E-15
        
        if logscale:
            combined_ax.set_yscale('log')
            combined_ax.set_ylim(line_strengths_min, line_strengths_max * 10)
        
        combined_ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        combined_ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
        combined_ax.set(**ax_style_defaults)
    
        for i, (iso_line_data, iso_continuum_data) in enumerate(zip(self.line_data, self.continuum_data)):
            ax = ax_array[i+1]
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(iso_line_data.rt_gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            
            if iso_line_data.has_data:
                line_no_data_str = ''
            else:
                line_no_data_str = ' [NO LINE DATA]'
            if iso_continuum_data.has_data:
                cont_no_data_str = ''
            else:
                cont_no_data_str = ' [NO CONT DATA]'
            
            ax.set_title(
                '\n'.join((
                    f'Line data ${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){line_no_data_str}{cont_no_data_str}',
                    f'line: (t_ref={iso_line_data.t_ref}, p_ref={iso_line_data.p_ref}) cont: (t_cont={iso_continuum_data.t_cont}, p_cont={iso_continuum_data.p_cont})'
                ))
            )
            
            
            # Plots for specific isotopes, coloured by lower energy state
            
            p1 = ax.scatter(
                iso_line_data.NU,
                iso_line_data.SW,
                c = iso_line_data.ELOWER,
                cmap = 'turbo',
                vmin = 0,
                **scatter_style_defaults
            )
            
            if iso_continuum_data.has_data:
                ax.plot(
                    iso_continuum_data.WN_BIN_CENTER,
                    iso_continuum_data.LINE_STRENGTH_SUM,
                    color='#EE22EE',
                    label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id})',
                    **line_style_defaults
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
                ax.set_ylim(line_strengths_min, line_strengths_max * 10)
            ax.set_xlabel('Wavenumber (cm$^{-1}$)')
            ax.set_ylabel('Line strength (cm$^{-1}$ / (molec cm$^{-2}$))')
            ax.set(**ax_style_defaults)




