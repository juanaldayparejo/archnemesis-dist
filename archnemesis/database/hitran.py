from __future__ import annotations #  for 3.9 compatability

import os
import os.path
import pickle

import numpy as np
import hapi

import archnemesis as ans
import archnemesis.enums
from .line_database_protocol import LineDatabaseProtocol, LineDataProtocol, PartitionFunctionDataProtocol
from .datatypes.wave_range import WaveRange
from .datatypes.gas_isotopes import GasIsotopes
from .datatypes.gas_descriptor import RadtranGasDescriptor, HitranGasDescriptor

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

class HITRAN(LineDatabaseProtocol):
    """
    Class that implements the "LineProviderProtocol" for HITRAN datasets.
    
    NOTE: In the background are using one local table per gas, therefore we can use 'local_iso_id' to uniquely determine
    which isotopologue we are talking about when working from local tables. If this changes, we need to re-think how this
    is done.
    
    NOTE: HAPI does not do databases properly so we cannot get a handle to the actual database connection, we need to 
    ensure that `hapi.db_begin(...)` is only called once so we need class variables to make sure the class can remember
    if it has initialised the database or not, and for instances to pass information between eachother.
    
    NOTE: There is not way to get HAPI to update a table it will always overwrite it, therefore we want to have a single
    table for each isotope.
    
    TODO: HAPI does not store the last request, therefore we cannot know how many wavelengths were requested. At the moment
    I have set it to request 10% more than required so when the wavenumber interval is calculated it only downloads again if
    more wavenumbers are requested. However, this fails when there is not enough wavelengths to cover the whole range. Therefore
    I should save the last wavelength range requested to disk so I do not re-request when I do not need to.
    """
    
    
    local_storage_dir : str = os.path.normpath('local_line_database')
    
    downloaded_gas_wavenumber_interval_cache_file : str = 'downloaded_gas_wavenumber_interval.pkl'
    
    db_init_flag : bool = False
    
    class_downloaded_gas_wavenumber_interval : dict[RadtranGasDescriptor, WaveRange] = dict() 
        
    class_gas_wavenumber_interval_to_download : dict[RadtranGasDescriptor, WaveRange] = dict() 
    
    
    @classmethod
    def set_local_storage_dir(cls, local_storage_dir : str):
        if cls.db_init_flag:
            raise RuntimeError(f'For now, cannot change location of HITRAN database after it has been initialised')
        else:
            cls.local_storage_dir = os.normpath(local_storage_dir)
    
    
    @classmethod
    def set_db_init_flag(cls, v : bool):
        cls.db_init_flag = v
    
    
    def __init__(
            self, 
        ):
        
        self._init_database()
    
    
    def get_line_data(
            self, 
            gas_descs : tuple[RadtranGasDescriptor,...], 
            wave_range : WaveRange, 
            ambient_gas : ans.enums.AmbientGas
        ) -> dict[RadtranGasDescriptor, LineDataProtocol]:
        gd = tuple(gas_descs)
        self._check_available_data(gd, wave_range)
        self._fetch_line_data()
        
        return self._read_line_data(gd, wave_range, ambient_gas)
    
    
    def get_partition_function_data(
            self, 
            gas_descs : tuple[RadtranGasDescriptor,...]
        ) -> dict[RadtranGasDescriptor, PartitionFunctionDataProtocol]:
        return self._read_partition_function_data(tuple(gas_descs))
    
    
    @property
    def _downloaded_gas_wavenumber_interval(self) -> dict[RadtranGasDescriptor, WaveRange]:
        return self.class_downloaded_gas_wavenumber_interval
    
    
    @property
    def _gas_wavenumber_interval_to_download(self) -> dict[RadtranGasDescriptor, WaveRange]:
        return self.class_gas_wavenumber_interval_to_download
    
    
    @property
    def _downloaded_gas_wavenumber_interval_cache_file(self):
        return os.path.join(self.local_storage_dir, self.downloaded_gas_wavenumber_interval_cache_file)
    
    
    def retrieve_downloaded_gas_wavenumber_interval_from_cache(self):
        cache_file_path = self._downloaded_gas_wavenumber_interval_cache_file
        
        if os.path.exists(cache_file_path):
            _lgr.debug(f'Loading `self._downloaded_gas_wavenumber_interval` from {cache_file_path=}')
            loaded = dict()
            with open(cache_file_path, 'rb') as f:
                try:
                    loaded = pickle.load(f)
                except Exception as e:
                    _lgr.warning(f'Something went wrong when unpickling `self._downloaded_gas_wavenumber_interval` from "{cache_file_path}", assuming no cached data. Error: {str(e)}')
                else:
                    self._downloaded_gas_wavenumber_interval.update(loaded)
            _lgr.debug(f'Loaded cached {self._downloaded_gas_wavenumber_interval=}')
        else:
            _lgr.info(f'Cache file for `self._downloaded_gas_wavenumber_interval` not found at {cache_file_path=}')

    def store_downloaded_gas_wavenumber_interval_to_cache(self):
        cache_file_path = self._downloaded_gas_wavenumber_interval_cache_file
        
        _lgr.debug(f'Caching {self._downloaded_gas_wavenumber_interval=} to {cache_file_path=}')
        with open(cache_file_path, 'wb') as f:
            try:
                pickle.dump(self._downloaded_gas_wavenumber_interval, f)
            except Exception as e:
                _lgr.warning(f'Something went wrong when pickling `self._downloaded_gas_wavenumber_interval` to "{cache_file_path}". Data will not be cached. Error: {str(e)}')


    def _init_database(self):
        if not self.db_init_flag:
            # Create directory to store database in
            os.makedirs(self.local_storage_dir, exist_ok=True)
            
            # Read downloaded_gas_wavenumber_interval_cache_file if it exists
            self.retrieve_downloaded_gas_wavenumber_interval_from_cache()
            
            #Starting the HAPI database
            hapi.db_begin(self.local_storage_dir)
            self.set_db_init_flag(True)
            
            tablenames = list(hapi.tableList())
            
            _lgr.debug(f'{tablenames=}')
            
            
            for tablename in tablenames:
                if _lgr.level <= logging.DEBUG:
                    hapi.describeTable(tablename)
                
                if hasattr(ans.enums.Gas, tablename.split('_',1)[0]):
                    hapi.select(tablename, ParameterNames=('molec_id', 'local_iso_id', 'nu'), Conditions=None, DestinationTableName = 'temp')
                    found_molec_ids, found_iso_local_ids, found_v = hapi.getColumns('temp', ('molec_id', 'local_iso_id', 'nu'))
                    
                    found_iso_local_ids = np.array(found_iso_local_ids, dtype=int)
                    found_v = np.array(found_v, dtype=float)
                    
                    molec_id_set = tuple(set(found_molec_ids))
                    local_iso_set = set(found_iso_local_ids)
                    
                    _lgr.debug(f'{found_iso_local_ids=}')
                    _lgr.debug(f'{molec_id_set=}')
                    _lgr.debug(f'{local_iso_set=}')
                    
                    
                    assert len(molec_id_set) == 1, "Should only have a single gas per HITRAN database table"

                    for iso in local_iso_set:
                        iso_mask = found_iso_local_ids == iso
                        _lgr.debug(f'{iso_mask=}')
                        iso_v = found_v[iso_mask]
                        vmin = np.min(iso_v)
                        vmax = np.max(iso_v)
                        
                        gas_desc = HitranGasDescriptor.from_gas_and_iso_id(molec_id_set[0], iso).to_radtran()
                        if gas_desc not in self._downloaded_gas_wavenumber_interval:
                            self._downloaded_gas_wavenumber_interval[gas_desc] = WaveRange(vmin, vmax, ans.enums.WaveUnit.Wavenumber_cm)
                            _lgr.info(f'No cache of downloaded wave range for {gas_desc}, using wave range {self._downloaded_gas_wavenumber_interval[gas_desc]=} found in tables {tablename} which will underestimate the requested range.')
                        
                    hapi.dropTable('temp')
        return


    def _check_available_data(self, gas_descs : tuple[RadtranGasDescriptor,...], wave_range : WaveRange):
        self._gasses_to_download = set()
        for gas_desc in gas_descs:
            if gas_desc not in self._gas_wavenumber_interval_to_download:
                self._gas_wavenumber_interval_to_download[gas_desc] = wave_range.as_unit(ans.enums.WaveUnit.Wavenumber_cm)
            else:
                if not self._downloaded_gas_wavenumber_interval[gas_desc].contains(wave_range):
                    self._gas_wavenumber_interval_to_download[gas_desc] = self._downloaded_gas_wavenumber_interval[gas_desc].union(wave_range)
    
    
    def _fetch_line_data(self):
        
        for gas_desc, wave_range in tuple(self._gas_wavenumber_interval_to_download.items()):
            if gas_desc not in self._gas_wavenumber_interval_to_download:
                continue
            saved_wave_range = self._downloaded_gas_wavenumber_interval.get(gas_desc, None)
            if saved_wave_range is not None and saved_wave_range.contains(wave_range):
                _lgr.debug(f'Downloaded gas data {gas_desc} CONTAINS desired wave range {saved_wave_range} vs {wave_range}')
                continue
            else:
                _lgr.debug(f'Downloaded gas data {gas_desc} DOES NOT CONTAIN desired wave range {saved_wave_range} vs {wave_range}')
        
            ht_gas = gas_desc.to_hitran()
            
            vmin, vmax = wave_range.as_unit(ans.enums.WaveUnit.Wavenumber_cm).values()
            try:
                hapi.fetch(
                    f'{gas_desc.gas_name}_{gas_desc.iso_id}',
                    ht_gas.gas_id,
                    ht_gas.iso_id,
                    vmin,
                    vmax,
                )
            except Exception as e:
                raise RuntimeError('Something went wrong when attempting to download data from HITRAN servers.') from e
            
            hapi.db_commit()
            
            self._downloaded_gas_wavenumber_interval[gas_desc] = wave_range
            if gas_desc in self._gas_wavenumber_interval_to_download:
                del self._gas_wavenumber_interval_to_download[gas_desc]
            
        # Cache to downloaded_gas_wavenumber_interval_cache_file for future
        self.store_downloaded_gas_wavenumber_interval_to_cache()
        
        return


    def _get_ambient_gas_parameter_strings(self, ambient_gas : ans.enums.AmbientGas) -> tuple[str,str,str]:
        if ambient_gas == ans.enums.AmbientGas.AIR:
            gamma_str = 'gamma_air'
            n_str = 'n_air'
            delta_str = 'delta_air'
        elif ambient_gas == ans.enums.AmbientGas.CO2:
            gamma_str = 'gamma_co2'
            n_str = 'n_co2'
            delta_str = 'delta_co2'
        else:
            raise ValueError(f'Unrecognised ambient gas {ambient_gas}')
        
        return gamma_str, n_str, delta_str


    def _read_line_data(self, gas_descs : tuple[RadtranGasDescriptor,...], wave_range : WaveRange, ambient_gas : ans.enums.AmbientGas):
        # Assume we have downloaded all the files we need at this point
        temp_line_data_table_name = 'temp_line_data'
        _lgr.debug(f'{temp_line_data_table_name=}')
        
        line_data = dict()
        
        _lgr.debug(f'{gas_descs=}')
        
        for gas_desc in gas_descs:
            _lgr.debug(f'{gas_desc=}')
            ht_gas_desc = gas_desc.to_hitran()
            if ht_gas_desc is None:
                continue
            
            vmin, vmax = wave_range.values()
            Conditions = ('and',('between', 'nu', vmin, vmax),('equal','local_iso_id',ht_gas_desc.iso_id))
            
            hapi.select(f'{gas_desc.gas_name}_{gas_desc.iso_id}', Conditions=Conditions, DestinationTableName=temp_line_data_table_name)
        
            gamma_str, n_str, delta_str = self._get_ambient_gas_parameter_strings(ambient_gas)
            
            cols = hapi.getColumns(
                temp_line_data_table_name,
                [
                    'nu',
                    'sw',
                    'a',
                    gamma_str,
                    n_str,
                    delta_str,
                    'gamma_self',
                    'elower'
                ]
            )
            
            _lgr.debug(f'{len(cols)=} {[len(c) for c in cols]=}')
            
            line_data[gas_desc] = np.array(
                list(zip(*cols)),
                dtype = [
                    ('NU', float), # Transition wavenumber (cm^{-1})
                    ('SW', float), # transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                    ('A', float), # einstein-A coeifficient (s^{-1})
                    ('GAMMA_AMB', float), # ambient gas broadening coefficient (cm^{-1} atm^{-1})
                    ('N_AMB', float), # temperature dependent exponent for `gamma_amb` (NUMBER)
                    ('DELTA_AMB', float), # ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                    ('GAMMA_SELF', float), # self broadening coefficient (cm^{-1} atm^{-1})
                    ('ELOWER', float), # lower state energy (cm^{-1})
                ]
            ).view(np.recarray)
            
            hapi.dropTable(temp_line_data_table_name)
        
        return line_data
    
    
    def _read_partition_function_data(self, gas_descs : tuple[RadtranGasDescriptor,...]):
        partition_function_data = dict()
        
        for gas_desc in gas_descs:
            ht_gas = gas_desc.to_hitran()
            if ht_gas is None:
                continue
            
            temps = hapi.TIPS_2021_ISOT_HASH[(ht_gas.gas_id,ht_gas.iso_id)]
            qs = hapi.TIPS_2021_ISOQ_HASH[(ht_gas.gas_id,ht_gas.iso_id)]
            partition_function_data[gas_desc] = np.array(
                list(zip(
                    temps,
                    qs
                )),
                dtype=[
                    ('TEMP', float), 
                    ('Q', float)
                ]
            ).view(np.recarray)
        return partition_function_data
    
