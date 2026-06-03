

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

from archnemesis.database.datatypes.line_set_data import LineSetData
from archnemesis.database.datatypes.pseudo_continuum_data import PseudoContinuumData
from archnemesis.database.datatypes.pf_list import PFList

from archnemesis.database.filetypes.ans_line_data_file import AnsLineDataFile
from archnemesis.database.filetypes.ans_partition_fn_data_file import AnsPartitionFunctionDataFile
from archnemesis.database.filetypes.ans_pseudo_continuum_file import AnsPseudoContinuumFile


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

@njit(parallel=False)
def stimulated_emission(
    t_calc : float,
    nu : np.ndarray, # [N] wavenumber
    out : np.ndarray, # [N] stimulated emission factor
):
    for i in prange(nu.shape[0]):
        out[i] = 1 - np.exp(-Data.constants.c2_cgs * nu[i] / t_calc)

@njit(parallel=False)
def boltzmann_factor(
    t_calc : float,
    t_ref : float,
    e_lower : np.ndarray, #[N]
    out : np.ndarray, #[N]
):
    boltz_const_factor = Data.constants.c2_cgs * (t_calc - t_ref)/(t_calc*t_ref)
    for i in prange(e_lower.shape[0]):
        out[i] = np.exp(boltz_const_factor*e_lower[i])

@njit(parallel=False)
def doppler_width(
        t_calc : float, 
        molecular_mass : float,
        nu : np.ndarray, #[N] wavenumber
        out : np.ndarray, #[N] doppler width
):
    doppler_width_const_cgs = (1.0 / Data.constants.c_light_cgs) * np.sqrt(2 * np.log(2) * Data.constants.N_avogadro * Data.constants.k_boltzmann_cgs)
    
    for i in prange(nu.shape[0]):
        out[i] = doppler_width_const_cgs * nu[i] * np.sqrt( t_calc / molecular_mass)

@njit(parallel=False)
def lorentz_width(
        t_ratio : float,
        p_ratio : float,
        mol_mix_frac : np.ndarray, #[M] fraction of mixture that consists of each molecule. Should sum to 1.0
        broadening_params : np.ndarray, #[M,N] M = N_mol * 3, note: N_mol = 1 + number of ambient molecules (self counts as 1), M is arranged (gamma, n, delta) for each molecule
        out : np.ndarray, #[N]
):
    for i in prange(broadening_params.shape[1]):
        n_combined = 0
        gamma_combined = 0
        for j in prange(mol_mix_frac.shape[0]):
            n_combined += broadening_params[3*j+1,i] * mol_mix_frac[j]
            gamma_combined += broadening_params[3*j,i] * mol_mix_frac[j]
        out[i] = (t_ratio**n_combined) * gamma_combined * p_ratio

@njit(parallel=False)
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

@njit(parallel=False)
def add_line_set_monochromatic_spectrum(
        wn_grid : np.ndarray, #[N_waves] (cm^{-1}) must be in ascending order
        lineshape_fn : Callable[[float,float,float], float],
        nu : np.ndarray, #[N_lines] (cm^{-1}) ideally sorted into ascending order
        strength : np.ndarray, #[N_lines]
        alpha_d : np.ndarray, #[N_lines]
        gamma_l : np.ndarray, #[N_lines]
        factor : float, # Factor to apply to final result (e.g. 1E20 when creating k-tables, a factor to account for isotopic abundance, or a combination)
        
        out : np.ndarray, #[N_waves] should be all zeros, as result will be ADDED to this not overwritten
        
        s_min : float = 1E-32,
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
        
        if strength[i] < s_min:
            continue
            
        line_approx_const = lineshape_fn(wn_calc_window_max, alpha_d[i], gamma_l[i])
        
        for j in range(wn_grid.shape[0]):
            wn_delta = wn_grid[j] - nu[i]
            
            if wn_delta >= wn_approx_window_max:
                    break
            if wn_delta < wn_approx_window_min:
                    continue
            
            if wn_calc_window_min <= wn_delta and wn_delta < wn_calc_window_max:
                # Lineshape calculation
                out[j] += factor * strength[i] * lineshape_fn(wn_delta, alpha_d[i], gamma_l[i])
            else:
                # 1/x**2 approximation
                out[j] += factor * strength[i] * line_approx_const / (wn_delta * wn_delta)
    
    return

@njit(parallel=False)
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
        
        store : np.ndarray | None = None, #[3,N]
        
        s_min : float = 1E-32,
        wn_calc_window : float = 25.0, # (cm^{-1})
        wn_approx_window : float = 75.0, # (cm^{-1})
):
    if store is None:
        store = np.empty((3, nu.shape[0]), dtype=float)
    
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
    
    add_line_set_monochromatic_spectrum(
        wn_grid,
        lineshape_fn,
        nu,
        store[0],
        store[1],
        store[2],
        factor = isotopic_abundance,
        out = out,
        s_min = s_min,
        wn_calc_window = wn_calc_window,
        wn_approx_window = wn_approx_window,
    )
    
    return

@njit(parallel=False)
def add_pseudo_continuum_monochromatic_spectrum(
        wn_grid : np.ndarray, #[N_waves] (cm^{-1}) must be in ascending order
        lineshape_fn : Callable[[float,float,float], float],
        wn_bin_centers : np.ndarray, #[N] (cm^{-1}) must be in ascending order
        wn_bin_widths : np.ndarray, #[N] (cm^{-1}) in same order as `wn_bin_centers`
        strength_sum : np.ndarray, #[N]
        mean_alpha_d : np.ndarray, #[N]
        mean_gamma_l : np.ndarray, #[N]
        factor : float, # Factor to apply to final result (e.g. 1E20 when creating k-tables, a factor to account for isotopic abundance, or a combination)
        
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
        
            if 0 <= ii and ii < wn_bin_centers.shape[0]:
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


@njit(parallel=False)
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
        store = np.empty((3,wn_bin_centers.shape[0]), dtype=float)
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
    n_max      : int                  = 10
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
    def __init__(self, n_max_per_bucket : int = 10):
        self._n_max_per_bucket = n_max_per_bucket
        self._buckets = dict()
    
    def get(self, bucket : str, identity : tuple[Any,...], default : None | Any = None) -> Any:
        return self._buckets.setdefault(bucket, CachePriorityDataHolder(self._n_max_per_bucket)).get(identity, default=default)
        
    def set(self, bucket : str, identity : tuple[Any,...], value : Any) -> Any:
        return self._buckets.setdefault(bucket, CachePriorityDataHolder(self._n_max_per_bucket)).set(identity, value)
        
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
    _data : np.ndarray
    _result_cache : Cache = dc.field(default_factory=lambda : Cache())
    
    @classmethod
    def create_from(
            cls, 
            mol_id : int, 
            iso_id : int, 
            ambient_gasses : tuple[ans.enums.AmbientGas,...],
            single_iso_line_set_data : LineSetData,
    ) -> Self:
        print(f'{single_iso_line_set_data.nu=}')
        
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
        
    
    @property
    def identity(self):
        return (self.s_min, self.t_ref, self.p_ref, self.rt_gas_desc.gas_id, self.rt_gas_desc.iso_id, *self.broadening_molecule_ids)
    
    @property
    def n_lines(self):
        return self._data.shape[1]
    
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
        store : None | np.ndarray = None, #[3,N_lines]
        
        s_min : float = 1E-32,
        wn_calc_window : float = 25.0, # (cm^{-1})
        wn_approx_window : float = 75.0, # (cm^{-1})
        wn_calc_range : None | tuple[float,float] = None,
        use_cache : bool = True,
    ) -> np.ndarray:
        if out is None:
            out = np.zeros_like(wn_grid, dtype=float)
        
        q_ratio = partition_function(self.t_ref) / partition_function(t_calc)
        
        wn_mask = np.ones((self._data.shape[1],), dtype=bool) if wn_calc_range is None else ((wn_calc_range[0] <= self.NU) & (self.NU <= wn_calc_range[1]))
        
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
            self._data[5:, wn_mask],
            *self._data[:4, wn_mask],
            
            out = out,
            store = store,
            
            s_min = s_min,
            wn_calc_window = wn_calc_window,
            wn_approx_window = wn_approx_window,
        )
        
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
            result_bucket = 'get_line_strength'
            result_identity = (wn_calc_range, t_calc,)
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
            result_bucket = 'get_doppler_width'
            result_identity = ( wn_calc_range, t_calc,)
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
            result_bucket = 'get_lorentz_width'
            result_identity = ( wn_calc_range, t_calc, p_calc, *mol_mix_frac)
            result = self._result_cache.get(result_bucket, result_identity, None)
        
        if result is None:
            result = np.empty((self._data.shape[1],))
            
            if wn_calc_range is not None:
                wn_mask = wn_calc_range[0] <= self.NU & self.NU <= wn_calc_range[1]
                data = self._data[5:,wn_mask]
            else:
                data = self._data[5:]
            
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
    _result_cache : Cache = dc.field(default_factory=lambda : Cache())

    @classmethod
    def create_from(
            cls, 
            mol_id : int, 
            iso_id : int, 
            ambient_gasses : tuple[ans.enums.AmbientGas,...],
            single_iso_pseudo_continuum_data : PseudoContinuumData,
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
        
        instance._set_data_from_pseudo_continuum_data(single_iso_pseudo_continuum_data)
        
        return instance
    
    def _set_data_from_pseudo_continuum_data(self, pseudo_continuum_data : PseudoContinuumData):
        print(f'{pseudo_continuum_data=}')
        n = pseudo_continuum_data.wn_bin_center.shape[0]
        m = (
            2   # (wn_bin_center, wn_bin_width,)
            + 2 # (line_strength_sum, lsw_mean_lower_state_energy,)
            + 3 # (lsw_gamma_self, lsw_n_self, lsw_delta_self)
            + pseudo_continuum_data.line_strength_weighted_gamma_amb.shape[1]*3 # (lsw_gamma_amb, lsw_n_amb, lsw_delta_amb) for x ambient gasses
        )
        if self._data is None or self._data.shape != (m,n):
            self._data = np.empty((m,n), dtype=float)
        
        self.WN_BIN_CENTER[...] = pseudo_continuum_data.wn_bin_center
        self.WN_BIN_WIDTH[...] = pseudo_continuum_data.wn_bin_width
        self.LINE_STRENGTH_SUM[...] = pseudo_continuum_data.line_strength_sum
        self.LSW_MEAN_LOWER_STATE_ENERGY[...] = pseudo_continuum_data.line_strength_weighted_mean_lower_energy_state
        self.SELF_BROADENING_LSW_PARAMS[...] = np.stack(
            [
                pseudo_continuum_data.line_strength_weighted_gamma_self,
                pseudo_continuum_data.line_strength_weighted_n_self,
                np.zeros_like(pseudo_continuum_data.line_strength_weighted_n_self),
            ],
            axis=0
        )
        self.FOREIGN_BROADENING_LSW_PARAMS[...] = np.stack(
            [
                *(pseudo_continuum_data.line_strength_weighted_gamma_amb.T),
                *(pseudo_continuum_data.line_strength_weighted_n_amb.T),
                *(np.zeros_like(pseudo_continuum_data.line_strength_weighted_n_amb.T))
            ],
            axis=0
        )
    
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
    def SELF_BROADENING_LSW_PARAMS(self):
        return self._data[4:7]
    
    @property
    def FOREIGN_BROADENING_LSW_PARAMS(self):
        return self._data[7:]
    
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
        if store is None:
            store = np.empty((3,self.WN_BIN_WIDTH.shape[0]), dtype=float)
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
            self._data[4:, wn_mask], # self and non-self broadening
            *self._data[:4, wn_mask],
            out = out,
            
            store = store,
            store_x = store_x,
            store_y = store_y,
            store_z = store_z,
            n_neighbour_bins = n_neighbour_bins
        )
        
        return out
    


@dc.dataclass(slots=True)
class AnsDatabase:
    """
    Handles getting data out of archNEMESIS spectral line database HDF5 files.
    """
    LINE_DATABASE : str
    PARTITION_FUNCTION_DATABASE : str
    CONTINUUM_DATABASE : str | None = None
    cache : Cache = _MODULE_CACHE
    
    # private attrs
    _ans_line_data_file : AnsLineDataFile = None
    _ans_partition_fn_file : AnsPartitionFunctionDataFile = None
    _ans_pseudo_continuum_file : AnsPseudoContinuumFile = None
    
    
    def __post_init__(self):
        if self.LINE_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a LINE_DATABASE')
        if self.PARTITION_FUNCTION_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a PARTITION_FUNCTION_DATABASE')
            
        self._ans_line_data_file = AnsLineDataFile(self.LINE_DATABASE)
        self._ans_partition_fn_file = AnsPartitionFunctionDataFile(self.PARTITION_FUNCTION_DATABASE)
        self._ans_pseudo_continuum_file = None if self.CONTINUUM_DATABASE is None else AnsPseudoContinuumFile(self.CONTINUUM_DATABASE)
    
    
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
        pf_cache_bucket = self.PARTITION_FUNCTION_DATABASE
        pf_cache_identity = (mol_id, iso_id)
        
        pf_instance = self.cache.get(pf_cache_bucket, pf_cache_identity, None)
        #print(f'{pf_instance=}')
        if refresh or pf_instance is None:
            #print('Must get data')
            pf_instance = self._ans_partition_fn_file.get_data(RadtranGasDescriptor(mol_id,iso_id).gas_name, iso_id)
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
            ambient_gasses : tuple[ans.enums.AmbientGas,...] = (ans.enums.AmbientGas.AIR,),
            out : None | list | tuple[list,...] = None,
            refresh : bool = False,
    ) -> tuple[list[tuple[tuple[int, int], LineSetData, None | PseudoContinuumData]],...] | list[tuple[tuple[int,int], LineSetData, None | PseudoContinuumData]] | Self:
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
            ambient_gasses : tuple[ans.enums.AmbientGas,...] = (ans.enums.AmbientGas.AIR,),
            refresh : bool = False,
    ) -> tuple[LineSetData, None | PseudoContinuumData]:

        # build data group
        ld_cache_bucket = self.LINE_DATABASE
        ld_cache_identity = (mol_id, iso_id, s_min, temperature, *ambient_gasses)
        
        ld_instance = self.cache.get(ld_cache_bucket, ld_cache_identity, None)
        if refresh or ld_instance is None or not wn_range_is_within((wn_min, wn_max), ld_instance.req_wn_range):
            ld_instance = self._ans_line_data_file.get_data(
                mol_name = RadtranGasDescriptor(mol_id, iso_id).gas_name, 
                local_iso_id = iso_id, 
                ambient_gasses = ambient_gasses,
                temperature = temperature,
                requested_wn_range = (wn_min, wn_max),
            )
            self.cache.set(ld_cache_bucket, ld_cache_identity, ld_instance)
    
        if self._ans_pseudo_continuum_file is None:
            pc_instance = None
        else:
            pc_cache_bucket = self.CONTINUUM_DATABASE
            pc_cache_identity = (mol_id, iso_id, ld_instance.s_min, ld_instance.t_ref, *ambient_gasses)
        
            pc_instance = self.cache.get(pc_cache_bucket, pc_cache_identity, None)
            if refresh or pc_instance is None or not wn_range_is_within((wn_min, wn_max), pc_instance.req_wn_range):
                pc_instance = self._ans_pseudo_continuum_file.get_data(
                        mol_name = RadtranGasDescriptor(mol_id, iso_id).gas_name,
                        local_iso_id = iso_id,
                        temperature = ld_instance.t_ref,
                        s_max = ld_instance.s_min,
                        ambient_gasses = ambient_gasses,
                        requested_wn_range = (wn_min, wn_max),
                    )
                print(f'{pc_instance=}')
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
            ambient_gasses : tuple[ans.enums.AmbientGas,...] = (ans.enums.AmbientGas.AIR,),
            out : None | list | tuple[list,...] = None,
            refresh : bool = False,
    ) -> tuple[list[tuple[tuple[int,int], LineSetData, None | PseudoContinuumData, PFList]],...] | list[tuple[tuple[int,int], LineSetData, None | PseudoContinuumData, PFList]] | Self:
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
            ambient_gasses : tuple[ans.enums.AmbientGas,...] = (ans.enums.AmbientGas.AIR,),
            refresh : bool = False,
    ) -> tuple[LineSetData, None | PseudoContinuumData, PFList]:
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
    ambient_gasses : tuple[ans.enums.AmbientGas,...] = (ans.enums.AmbientGas.AIR,)
    wn_min : float = 0
    wn_max : float = 0
    s_min : float = -1
    temp_requested : float = 0
    press_requested : float = 0

    

class LineData_0:
    def __init__(
            self,
            ID : int,
            ISO : int = 0,
            ambient_gasses : ans.enums.AmbientGas | tuple[ans.enums.AmbientGas,...] = (ans.enums.AmbientGas.AIR,),
            LINE_DATABASE : None | str = None,
            PARTITION_FUNCTION_DATABASE : None | str = None,
            CONTINUUM_DATABASE : None | str = None,
            cache : None | Cache = _MODULE_CACHE,
    ):
        if LINE_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a LINE_DATABASE')
        if PARTITION_FUNCTION_DATABASE is None:
            raise RuntimeError('LineData_0 instance must have a PARTITION_FUNCTION_DATABASE')
        
        
        # Private attrs
        self._ID : int = INVALID_MOLECULE_ID
        self._ISO = None
        self._ans_database : AnsDatabase = AnsDatabase(LINE_DATABASE, PARTITION_FUNCTION_DATABASE, CONTINUUM_DATABASE)
        self._rt_gas_descs : None | tuple[RadtranGasDescriptor,...] = None 
        self._default_iso_abundances : None | np.ndarray = None,
        self._n_isos : None | int = None
        self._mol_ids : None | np.ndarray = None
        self._iso_ids : None | np.ndarray = None
        self._mol_id_tpl : None | tuple[int,...] = None
        self._iso_id_tpl : None | tuple[tuple[int,...],...] = None
        self._params : LineDataParams = LineDataParams(ambient_gasses=(ambient_gasses,) if isinstance(ambient_gasses, ans.enums.AmbientGas) else ambient_gasses)
        self._params_fetched_lines_last = False
        self._params_fetched_partition_last = False
        
        # Public attrs from args
        self.ID = ID
        self.ISO = ISO
        self.cache = cache
        
        # Public attrs
        self.partition_fn_data : list[PFList]             = [None]*self.n_isos
        self.line_data         : list[LineSetSpecData]    = [None]*self.n_isos
        self.continuum_data    : list[PseudoContSpecData] = [None]*self.n_isos
    
    
    @property
    def ID(self) -> int:
        return self._ID
    
    @ID.setter
    def ID(self, value : int):
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
            temp_requested : None | float = None, # Kelvin
            press_requested : None | float = None, # Atmospheres
            wave_unit :ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,
    ) -> Self:
        if vmin is not None:
            vmin = WavePoint(vmin, wave_unit).as_unit(ans.enums.WaveUnit.Wavenumber_cm).value
        
        if vmax is not None:
            vmax = WavePoint(vmax, wave_unit).as_unit(ans.enums.WaveUnit.Wavenumber_cm).value
            
        self._set_params_direct(
            wn_min = vmin, 
            wn_max = vmax, 
            s_min = s_min, 
            temp_requested = temp_requested, 
            press_requested = press_requested
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
            temp_requested : None | float = None, # Kelvin
            press_requested : None | float = None, # Atmospheres
            wave_unit :ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,
    ) -> Self:
        prev_params = self.get_params()
        
        self.set_params(vmin, vmax, s_min, temp_requested, press_requested, wave_unit)
        
        yield self
        
        self._set_params_direct(**vars(prev_params))
        
        return

    def fetch_partition_fn(
            self,
            refresh : bool = False,
    ):
        print('LineData_0::fetch_partition_fn()')
        if self._params_fetched_partition_last and not refresh:
            return
        
        print('Actually getting the partition function')
        self.partition_fn_data = self._ans_database.fetch_partition_fn(self.mol_ids, self.iso_ids, refresh=refresh)
        print(f'{self.partition_fn_data=}')
        self._params_fetched_partition_last = True

    def fetch_linedata(
            self, 
            refresh : bool = False,
    ) -> None:
    
        if self._params_fetched_lines_last and not refresh:
            return
    
        #print(f'{self._params=}')
        #print(f'{self.mol_ids=}')
        #print(f'{self.iso_ids=}')

        retrieved_from_cache = False # Flag
        
        if self.cache is not None:
            # build data group
            ld_pc_cache_bucket = ('line_and_continuum_data_pairs',id(self._ans_database))
            ld_pc_cache_identity = (self.mol_id_tpl, self.iso_id_tpl, self._params.s_min, self._params.temp_requested, *self._params.ambient_gasses)
            
            ld_pc_pairs = self.cache.get(ld_pc_cache_bucket, ld_pc_cache_identity, None)
            if refresh or ld_pc_pairs is None or not all(wn_range_is_within((self._params.wn_min, self._params.wn_max), ld_pc_pair[0].req_wn_range) for ld_pc_pair in ld_pc_pairs):
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
                self._params.temp_requested,
                self._params.ambient_gasses,
                out = id_line_cont_triplet
            )
            
            #print(f'{type(id_line_cont_triplet)=} {len(id_line_cont_triplet)=}')
            #print(f'{type(id_line_cont_triplet[0])=} {len(id_line_cont_triplet[0])=}')
            #print(f'{type(id_line_cont_triplet[0][0])=} {len(id_line_cont_triplet[0][0])=}')
            
            for i, ((mol_id, iso_id), line_data, cont_data) in enumerate(id_line_cont_triplet):
                self.line_data[i] = LineSetSpecData.create_from(mol_id, iso_id, self._params.ambient_gasses, line_data)
                self.continuum_data[i] = None if cont_data is None else PseudoContSpecData.create_from(mol_id, iso_id, self._params.ambient_gasses, cont_data)
        
        if self.cache is not None:
            # Store result in cache
            self.cache.set(ld_pc_cache_bucket, ld_pc_cache_identity, (self.line_data, self.continuum_data))
        
        # Remember that we used these parameters to fetch the last lot of linedata
        self._params_fetched_lines_last = True
        
        return

    def calculate_line_strength(
            self,
            t_calc : float,
            wn_calc_range : None | tuple[float,float] = None,
    ) -> tuple[np.ndarray,...]:
        if not self._params_fetched_lines_last:
            self.fetch_linedata()
        if not self._params_fetched_partition_last:
            self.fetch_partition_fn()
        
        return tuple(self.line_data[i].get_line_strength(t_calc, self.partition_fn_data[i], wn_calc_range) for i in range(len(self.line_data)))

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
            lineshape_fn : Callable[[float,float,float], float] = Data.lineshapes.voigt,
            s_min : float = 1E-32,
            wn_calc_window : float = 25.0, # (cm^{-1})
            wn_approx_window : float = 75.0, # (cm^{-1})
            wave_unit :ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm, # unit of `wave_grid` and `wave_calc_range`
            
            include_lines : bool = True,
            include_continuum : bool = True,
    ) -> np.ndarray:
        
        # Create default arrays
        if out is None:
            result = np.zeros_like(wave_grid, dtype=float)
            out = result[...]
        else:
            result = out[...]
        
        if store is None:
            store = np.empty((3, max(max(x.NU.shape[0] for x in self.line_data), max(x.WN_BIN_CENTER.shape[0] if x is not None else 0 for x in self.continuum_data))), dtype=float)
        
        
        # Handle wave units
        if wave_unit != ans.enums.WaveUnit.Wavenumber_cm:
            wn_grid = WavePoint(wave_grid, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value
        else:
            wn_grid = wave_grid
        
        if wave_calc_range is None:
            wn_calc_range = (np.min(wn_grid) - 2*wn_approx_window, np.max(wn_grid) + 2*wn_approx_window)
        elif wave_unit != ans.enums.WaveUnit.Wavenumber_cm:
            wn_calc_range = (WavePoint(wave_calc_range[0], wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value, WavePoint(wave_calc_range[1], wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value)
        
        # Ensure that `wn_grid` is ascending. Flip both `wn_grid` and `out` if `wn_grid` is decending.
        if np.all(wn_grid[:-1] < wn_grid[1:]):
            pass
        elif np.all(wn_grid[:-1] > wn_grid[1:]):
            out = out.flip(axis=-1)
        else:
            raise RuntimeError('`wave_grid` passed to LineData_0::monochromatic_absorption(...) must be either ascending or decending.')
        
        # create array that holds mixing fractions for self as well as ambient gasses
        if isinstance(amb_frac, float):
            mol_mix_frac = np.array([1-amb_frac, amb_frac], dtype=float)
        else:
            mol_mix_frac = np.array([1 - sum(amb_frac), *amb_frac], dtype=float)
        
        assert mol_mix_frac.shape[0] == len(self._params.ambient_gasses)+1, "LineData_0::add_monochromatic_absorption(...) `amb_frac` must have enough entries for each ambient gas"
        
        # Ensure isotopic abundances are arrays of correct length
        if isotopic_abundance is None:
            isotopic_abundance = self.default_iso_abundances
        elif isinstance(isotopic_abundance, float):
            assert self.n_isos == 1, "If provided, there must be an isotopic abundance for each isotopologue in the LineData_0 instance"
            isotopic_abundance = np.array([isotopic_abundance], dtype=float)
        else:
            assert self.n_isos == isotopic_abundance.shape[0], "If provided, there must be an isotopic abundance for each isotopologue in the LineData_0 instance"
        
        if out.ndim >= 2:
            out_line_set_abs = out[0]
            out_continuum_abs = out[1]
        else:
            out_line_set_abs = out
            out_continuum_abs = out
        
        # Loop over all line data and add monochromatic absorption to `out`
        for i, (iso_line_data, iso_continuum_data) in enumerate(zip(self.line_data, self.continuum_data)):
            print(f'{i=} {iso_line_data._data.shape=} {(iso_continuum_data._data.shape if iso_continuum_data is not None else "None")=}')
            
            if out.ndim < 3:
                out_line_set_abs_i = out_line_set_abs
                out_continuum_abs_i = out_continuum_abs
            elif out.ndim == 3:
                assert out.shape[2] == len(self.line_data), f'`out` must have 1, 2 or 3 dimensions ({out.ndim=}). If 1 dimension will add all absorption to same array. If 2 will split line absorption and continuum absorption. If 3 will split line absorption and continuum absorption, and split by isotopologue and therefore must have out.shape[1] == number of isotopoluges ({out.shape[1]=}) ({len(self.line_data)})'
                out_line_set_abs_i = out_line_set_abs[i]
                out_continuum_abs_i = out_continuum_abs[i]
            else:
                raise RuntimeError(f'`out` must have 1, 2 or 3 dimensions ({out.ndim=}). If 1 dimension will add all absorption to same array. If 2 will split line absorption and continuum absorption. If 3 will split line absorption and continuum absorption, and split by isotopologue.')
            
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
                    #store = store[:,:iso_line_data.n_lines], #[3,N_lines]
                    store = store,
                    
                    s_min = s_min,
                    wn_calc_window = wn_calc_window,
                    wn_approx_window = wn_approx_window,
                    wn_calc_range = wn_calc_range,
                )
            
            if include_continuum and iso_continuum_data is not None:
                print(f'{iso_continuum_data=}')
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
                )
            else:
                print('No continuum data')
            
        
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
            #s = 15,
            #edgecolor='black',
            #linewidth=-.2
            s = 2,
            marker='.',
            edgecolor = 'none',
            alpha=0.6,
        )
        scatter_style_defaults.update(scatter_style_kw)
        
        line_style_defaults = dict(
            linewidth=1,
        )
        line_style_defaults.update(line_style_kw)
        
        ax_style_defaults = dict(
            facecolor='#EEEEEE'
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

            if iso_line_data.n_lines == 0:
                ls_max = 0
                ls_min = 0
                no_data_str = ' [NO DATA]'
            else:
                ls_max = iso_line_data.SW.max()
                ls_min = min(iso_line_data.SW.min(), iso_continuum_data.LINE_STRENGTH_SUM.min())
                no_data_str = ''
            line_strengths_max = ls_max if ls_max > line_strengths_max else line_strengths_max
            line_strengths_min = ls_min if ls_min < line_strengths_min else line_strengths_min
            
            try:
                gas_name_latex = ans.Data.gas_data.molecule_to_latex(iso_line_data.rt_gas_desc.isotope_name)
            except KeyError:
                gas_name_latex = r'\text{UNKNOWN GAS ISOTOPE}'
            
            # Combined plot, all isotopes on one figure, coloured by isotope
            combined_ax.scatter(
                iso_line_data.NU,
                iso_line_data.SW,
                label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){no_data_str}',
                zorder = i,
                **scatter_style_defaults
            )
            
            # Plot twice so we can have contrasting colour, remember to set z_order
            combined_ax.plot(
                iso_continuum_data.WN_BIN_CENTER,
                iso_continuum_data.LINE_STRENGTH_SUM,
                linewidth = line_style_defaults['linewidth']*2,
                color=ax_style_defaults['facecolor'],
                zorder = self.n_isos + i,
            )
            combined_ax.plot(
                iso_continuum_data.WN_BIN_CENTER,
                iso_continuum_data.LINE_STRENGTH_SUM,
                label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){no_data_str}',
                zorder = 2*self.n_isos + i,
                **line_style_defaults
            )
        
        combined_ax.legend(
            loc='upper left', 
            bbox_to_anchor=(1.01, 1.05),  # Shift legend to the right
            title='Isotope', 
            **legend_style_defaults
        )
        
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
            
            
            if iso_line_data.n_lines == 0:
                no_data_str = ' [NO DATA]'
            else:
                no_data_str = ''
            
            ax.set_title(f'Line data for ${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){no_data_str}')
            
            
            # Plots for specific isotopes, coloured by lower energy state
            
            p1 = ax.scatter(
                iso_line_data.NU,
                iso_line_data.SW,
                c = iso_line_data.ELOWER,
                cmap = 'turbo',
                vmin = 0,
                **scatter_style_defaults
            )
            ax.plot(
                iso_continuum_data.WN_BIN_CENTER,
                iso_continuum_data.LINE_STRENGTH_SUM,
                color='#EE22EE',
                label=f'${gas_name_latex}$ (ID={int(iso_line_data.rt_gas_desc.gas_id)}, ISO={iso_line_data.rt_gas_desc.iso_id}){no_data_str}',
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



class LineData_0_OLD:
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
            CONTINUUM_DATABASE : None | str = None,
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
        self.CONTINUUM_DATABASE = CONTINUUM_DATABASE
        
        self.t_ref : float = 296.0
        self.s_min : float = 0.0
        self.line_data = None
        self.continuum_data = None
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
            s_min : float = -1,
            temperature : float = 0,
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
        already_contain_wave_range = ((self._total_line_data_wave_lims[0] <= vmin) and (vmax <= self.total_line_data_wave_lims[1]))
            
        
        if already_contain_wave_range and self.is_line_data_ready() and not refresh:
            _lgr.debug(f'Line data already loaded. {already_contain_wave_range=} {self.is_line_data_ready()=} {refresh=}')
            return
        
        # retrieve the superset of the existing interval and the desired interval
        self._total_line_data_wave_lims[0] = min(self._total_line_data_wave_lims[0], vmin)
        self._total_line_data_wave_lims[1] = max(self._total_line_data_wave_lims[1], vmax)
        
        vmin, vmax = self._total_line_data_wave_lims
        
        # get line database file
        ans_line_data_file = AnsLineDataFile(self.LINE_DATABASE)
        ans_pseudo_continuum_file = None if self.CONTINUUM_DATABASE is None else AnsPseudoContinuumFile(self.CONTINUUM_DATABASE)
        
        mol_id_list = []
        local_iso_id_list = []
        nu_list = []
        sw_list = []
        a_list = []
        elower_list = []
        t_ref_list = []
        p_ref_list = []
        gamma_self_list = []
        n_self_list = []
        gamma_amb_list = []
        n_amb_list = []
        delta_amb_list = []
        
        continuum_list : list[PseudoContinuumData] = []

        # NOTE: abundances are accounted for in `self.calculate_monochromatic_absorption(...)`
        #scale_strength_by_abundance = self.ISO == 0
            

        for gas_desc in self.gas_isotopes.as_radtran_gasses():
            print(f'{gas_desc=} {vmin=} {vmax=}')
        
            line_set_data : LineSetData = ans_line_data_file.get_data(
                mol_name = gas_desc.gas_name, 
                local_iso_id = gas_desc.iso_id, 
                ambient_gasses = (self.ambient_gas,),
                temperature = temperature,
                wn_mask_fn = lambda nu: ((vmin < nu) & (nu <= vmax))
            )
            
            line_set_spec_data = LineSetSpecData.create_from(
                gas_desc.gas_id, 
                gas_desc.iso_id, 
                (self.ambient_gas,),
                line_set_data
            )
            print(f'{line_set_spec_data=}')
            
            mol_id_list.append(line_set_data.mol_id)
            local_iso_id_list.append(line_set_data.local_iso_id)
            nu_list.append(line_set_data.nu)
            sw_list.append(line_set_data.sw) #sw_list.append((sw*gas_desc.abundance) if scale_strength_by_abundance else sw) # NOTE: abundances are accounted for in `self.calculate_monochromatic_absorption(...)`
            a_list.append(line_set_data.a)
            elower_list.append(line_set_data.elower)
            t_ref_list.append(np.ones_like(line_set_data.nu)*line_set_data.t_ref)
            p_ref_list.append(np.ones_like(line_set_data.nu)*line_set_data.p_ref)
            gamma_self_list.append(line_set_data.gamma_self)
            n_self_list.append(line_set_data.n_self)
            gamma_amb_list.append(line_set_data.gamma_amb)
            n_amb_list.append(line_set_data.n_amb)
            delta_amb_list.append(line_set_data.delta_amb)
            
            if line_set_data.s_min > 0:
                # we should have corresponding continuum data to go with the line set
                continuum_list.append(
                    ans_pseudo_continuum_file.get_data(
                        mol_name = gas_desc.gas_name,
                        local_iso_id = gas_desc.iso_id,
                        temperature = line_set_data.t_ref,
                        s_max = line_set_data.s_min,
                        ambient_gasses = (self.ambient_gas,),
                        wn_mask_fn = lambda nu: ((vmin < nu) & (nu <= vmax)),
                    )
                )

        dtype = [
            ('RT_GAS_DESC', int, (2,)),
            ('NU', float),
            ('SW', float),
            ('A', float),
            ('ELOWER', float),
            ('T_REF', float),
            ('P_REF', float),
            ('GAMMA_SELF', float),
            ('N_SELF', float),
            ('GAMMA_AMB', float, (line_set_data.gamma_amb.shape[1],)),
            ('N_AMB', float, (line_set_data.n_amb.shape[1],)),
            ('DELTA_AMB', float, (line_set_data.delta_amb.shape[1],)),
        ]
        
        self.line_data = np.rec.fromarrays(
            (
                np.array((np.concatenate(mol_id_list),np.concatenate(local_iso_id_list)), dtype=int).T,
                np.concatenate(nu_list),
                np.concatenate(sw_list),
                np.concatenate(a_list),
                np.concatenate(elower_list),
                np.concatenate(t_ref_list),
                np.concatenate(p_ref_list),
                np.concatenate(gamma_self_list),
                np.concatenate(n_self_list),
                np.concatenate(gamma_amb_list),
                np.concatenate(n_amb_list),
                np.concatenate(delta_amb_list),
            ),
            dtype=dtype
        )
        
        self.continuum_data = None if len(continuum_list) == 0 else continuum_list


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
            self.partition_data_dict[gas_desc] = ans_pf_file.get_data(gas_desc.gas_name, gas_desc.iso_id)
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
            amb_frac: np.ndarray, # fraction of ambient gas
            tref : float = 296,
    ) -> np.ndarray[['N_LINES_OF_GAS'], float]:
        """
        Calculate pressure-broadened width HWHM (half-width-half-maximum) of cauchy-lorentz distribution.
        """
        _lgr.debug(f'{press=} {temp=} {amb_frac=} {tref=}')
        
        tratio = tref/temp

        lws = np.sum(
            (
                (tratio**self.line_data.N_AMB) 
                * ( self.line_data.GAMMA_AMB * amb_frac + self.line_data.GAMMA_SELF[:,None] * (1-amb_frac) )
            ),
            axis=1
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
    
    
    @classmethod
    def calculate_monochromatic_line_absorption(
            cls,
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
    
    @classmethod 
    def calculate_monochromatic_spectrum(
            cls,
            nu : np.ndarray, #1D array with the wavenumber of the lines
            strength : np.ndarray, #1D array with the intensity of the lines
            alpha_d : np.ndarray, #1D array with the Doppler HWHM of the lines
            gamma_l : np.ndarray, #1D array with the Lorentzian HWHM of the lines
            waves : np.ndarray, # 1D array with shape [NWAVE]
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,  # unit of `waves` argument
            lineshape_fn : Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, # lineshape function to use
            line_calculation_wavenumber_window: float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
    ) -> np.ndarray:
        """
        Calculates the combined spectrum from many absorption lines considering their relative broadening.
        This function differs from calculate_monochromatic_absorption() because that one is intended to be used for
        the calculation of absorption coefficients. In this one, we explicitly provide the "line strengths" and the
        broadening parameters, so that it can be used for other types of spectra, not necessarily absorption
        (e.g., emission profiles or g-factors)
        """

        CONT_CALC_MSG_ONCE_FLAG = False
        n_line_progress = 1000
        
        # Convert waves to wavenumber (cm^{-1})
        wavenumbers = np.array(WavePoint(waves, wave_unit).to_unit(ans.enums.WaveUnit.Wavenumber_cm).value)
        
        # Remember the ordering we got as input
        wav_sort_idxs = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[wav_sort_idxs]
        
        # Define arrays here to re-use memory
        delta_wn = np.zeros_like(wavenumbers, dtype=float)
        scratch = np.zeros_like(wavenumbers, dtype=float)
        mask_leq = np.zeros_like(wavenumbers, dtype=bool)
        mask_geq = np.zeros_like(wavenumbers, dtype=bool)
        mask = np.zeros_like(wavenumbers, dtype=bool)
        wide_mask_leq = np.zeros_like(wavenumbers, dtype=bool)
        wide_mask_geq = np.zeros_like(wavenumbers, dtype=bool)
        wide_mask = np.zeros_like(wavenumbers, dtype=bool)
        k_total = np.zeros_like(wavenumbers, dtype=float)
        spectrum= np.zeros_like(wavenumbers, dtype=float) # Final output array

        with SimpleProgressTracker("Computing line contributions. ", (len(nu)-1), display_interval_n=n_line_progress, output_target=_lgr) as line_contrib_prog:
            for nu_val, strength_val, alpha_d_val, gamma_l_val in zip(nu,strength,alpha_d,gamma_l):
                
                line_contrib_prog.display()
                
                scratch.fill(0.0)
                
                np.subtract(wavenumbers, nu_val, out=delta_wn)
                
                np.less_equal(delta_wn, line_calculation_wavenumber_window, out=mask_leq)
                np.greater_equal(delta_wn, -line_calculation_wavenumber_window, out=mask_geq)
                np.logical_and(mask_leq, mask_geq, out=mask)
                
                np.less_equal(delta_wn, line_calculation_wavenumber_window, out=wide_mask_leq)
                np.greater_equal(delta_wn, -line_calculation_wavenumber_window, out=wide_mask_geq)
                np.logical_and(wide_mask_leq, wide_mask_geq, out=wide_mask)
                wide_mask[:] = False

                cls.calculate_monochromatic_line_absorption(
                    delta_wn,
                    mask, 
                    wide_mask,
                    strength_val,
                    alpha_d_val,
                    gamma_l_val,
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
                spectrum[wav_sort_idxs] = k_total*1E20
            
        #The units of the spectrum are "per wavenumber" (i.e., (cm-1)-1). Change to "per wavelength" (um-1) if required 
        if wave_unit == ans.enums.WaveUnit.Wavelength_um:
            spectrum = spectrum / (waves)**2. / 1.0e-4

        return spectrum
    
    def calculate_monochromatic_absorption(
            self,
            waves : np.ndarray, # 1D array with shape [NWAVE]
            temp : float, # kelvin
            press : float, # Atmospheres
            amb_frac : float | np.ndarray = 1.0, # fraction of broadening due to ambient gas
            wave_unit : ans.enums.WaveUnit = ans.enums.WaveUnit.Wavenumber_cm,  # unit of `waves` argument
            lineshape_fn : Callable[[np.ndarray, float, float], np.ndarray] = Data.lineshapes.voigt, # lineshape function to use
            line_calculation_wavenumber_window: float = 25.0, # cm^{-1}, contribution from lines outside this region should be modelled as continuum absorption (see page 29 of RADTRANS manual).
            tref : float = 296, # Reference temperature (Kelvin). TODO: This should be set by the database used
            s_min : float = 1E-32, # Strength below which a line is ignored.
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
        
        if not isinstance(amb_frac, np.ndarray):
            amb_frac = np.array([amb_frac], dtype=float)
        
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
        
        if self.continuum_data is not None:
            raise NotImplementedError('Continuum calculation is not implemented yet')

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

        linestrength_mask = strength >= s_min

        return self.calculate_monochromatic_spectrum(
            self.line_data.NU[linestrength_mask],
            strength[linestrength_mask],
            alpha_d[linestrength_mask],
            gamma_l[linestrength_mask],
            waves,
            wave_unit,
            lineshape_fn,
            line_calculation_wavenumber_window,
        )
    
    
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











