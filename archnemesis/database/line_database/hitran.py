

import os
import os.path
import pickle
import warnings

import numpy as np
import numpy.ma
import numpy.lib.recfunctions


import archnemesis as ans
import archnemesis.enums
import archnemesis.database.wrappers.hapi as hapi
from ..protocols import (
    LineDatabaseProtocol, 
    LineDataProtocol, 
)
from ..datatypes.wave_range import WaveRange
#from ..datatypes.gas_isotopes import GasIsotopes
from ..datatypes.gas_descriptor import RadtranGasDescriptor
from ..datatypes.hitran.gas_descriptor import HitranGasDescriptor

import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.INFO)
_lgr.setLevel(logging.DEBUG)

# NOTE: HAPI does not differentiate between an actual failure to retrieve data because of a problem vs
#       not retrieving any data because there is no data in a wavelength range. Until I can handle those
#       cases differently, there will always be some annoying exceptions when no lines are present in a 
#       requested wavelength range.

class HITRAN(LineDatabaseProtocol):
    """
    Class that implements the "LineProviderProtocol" for HITRAN datasets.
    
    NOTE: In the background are using one local table per gas, therefore we can use 'local_iso_id' to uniquely determine
    which isotopologue we are talking about when working from local tables. If this changes, we need to re-think how this
    is done.
    
    NOTE: HAPI does not do databases properly so we cannot get a handle to the actual database connection, we need to 
    ensure that `hapi.db_begin(...)` is only called once so we need class variables to make sure the class can remember
    if it has initialised the database or not, and for instances to pass information between eachother. In the background
    there is **only ever one database** so if we want to move to a different database while running we have to somehow
    "forget" the old one, and the re-initialise on the new one (including the time taken to parse the large column text
    files that act as the "database" on-disk storage).
    
    NOTE: There is not way to get HAPI to update a table it will always overwrite it, therefore we want to have a single
    table for each isotope.
    
    TODO: HAPI does not store the last request, therefore we cannot know how many wavelengths were requested. We therefore
    keep track of the limits of the wave range requested for each (gas, ambient-gas) pair. We always request the 
    wave interval that encompasses the cached interval and the requested wave range. That means if you ask for 
    (100,200) cm^{-1} and the cached range is (9000, 10000) cm^{-1}, then the re-requested range will be the encompassing
    interval (100, 10000) cm^{-1}. We do this becase HAPI always overwrites the previously downloaded data, it cannot
    update it.
    """
    
    
    _class_local_storage_dir : str = os.path.abspath('local_line_database')
    
    _class_downloaded_gas_wavenumber_interval_cache_file : str = 'downloaded_gas_wavenumber_interval.pkl'
    
    _class_db_init_flag : bool = False
    
    _class_downloaded_gas_wavenumber_interval : dict[tuple[RadtranGasDescriptor, ans.enums.AmbientGas], WaveRange] = dict() 
        
    _class_gas_wavenumber_interval_to_download : dict[tuple[RadtranGasDescriptor, ans.enums.AmbientGas], WaveRange] = dict() 
    
    
    @classmethod
    def set_class_local_storage_dir(cls, local_storage_dir : str):
        if cls._class_db_init_flag:
            raise RuntimeError('Cannot change location of HITRAN database after it has been initialised as HAPI does not have a way of using multiple databases at once.')
        else:
            cls._class_local_storage_dir = os.path.abspath(local_storage_dir)
    
    
    @classmethod
    def set_class_db_init_flag(cls, v : bool):
        cls._class_db_init_flag = v
    
    
    @property
    def local_storage_dir(self) -> str:
        """
        Required by LineDatabaseProtocol. Gets the directory the local database is in.
        """
        return self._class_local_storage_dir
    
    @local_storage_dir.setter
    def local_storage_dir(self, value : str) -> None:
        """
        Required by LineDatabaseProtocol. Sets the directory the local database is in.
        """
        self.set_class_local_storage_dir(value)
    
    @property
    def db_init_flag(self) -> bool:
        """
        Gets the value of the flag that tell us if the local database has been initialised
        """
        return self._class_db_init_flag
    
    @db_init_flag.setter
    def db_init_flag(self, value : bool) -> None:
        """
        Sets the value of the flag that tell us if the local database has been initialised
        """
        self.set_class_db_init_flag(value)
    
    @property
    def _downloaded_gas_wavenumber_interval(self) -> dict[RadtranGasDescriptor, WaveRange]:
        return self._class_downloaded_gas_wavenumber_interval
    
    @property
    def _gas_wavenumber_interval_to_download(self) -> dict[RadtranGasDescriptor, WaveRange]:
        return self._class_gas_wavenumber_interval_to_download
    
    @property
    def _downloaded_gas_wavenumber_interval_cache_file(self):
        return os.path.join(self._class_local_storage_dir, self._class_downloaded_gas_wavenumber_interval_cache_file)
    
    
    @staticmethod
    def get_tablename_from_props(gas_desc : RadtranGasDescriptor, ambient_gas : ans.enums.AmbientGas) -> str:
        return f'{gas_desc.gas_name}_{gas_desc.iso_id}_ambient_{ambient_gas.name}'
    
    @staticmethod
    def get_props_from_tablename(tablename : str) -> tuple[RadtranGasDescriptor, ans.enums.AmbientGas]:
        gas_name, iso_id, _, amb_gas = tablename.split('_')
        return (
            RadtranGasDescriptor(
                getattr(ans.enums.Gas, gas_name),
                int(iso_id)
            ),
            getattr(ans.enums.AmbientGas, amb_gas),
        )
    
    @staticmethod
    def get_ambient_gas_parameter_name_strings(ambient_gas : ans.enums.AmbientGas) -> tuple[str,str,str]:
        """
        HITRAN paramter names have gas in UPPERCASE (except for 'air')
        """
        if ambient_gas == ans.enums.AmbientGas.AIR:
            gamma_str = 'gamma_air'
            n_str = 'n_air'
            delta_str = 'delta_air'
        elif ambient_gas == ans.enums.AmbientGas.CO2:
            gamma_str = 'gamma_CO2'
            n_str = 'n_CO2'
            delta_str = 'delta_CO2'
        else:
            gamma_str = f'gamma_{ambient_gas.name.upper()}'
            n_str = f'n_{ambient_gas.name.upper()}'
            delta_str = f'delta_{ambient_gas.name.upper()}'
            #raise ValueError(f'Unrecognised ambient gas {ambient_gas}')
        
        return gamma_str, n_str, delta_str
    
    @staticmethod
    def get_ambient_gas_column_name_strings(ambient_gas : ans.enums.AmbientGas) -> tuple[str,str,str]:
        """
        HAPI column names have gas in LOWERCASE (including 'air')
        """
        if ambient_gas == ans.enums.AmbientGas.AIR:
            gamma_str = 'gamma_air'
            n_str = 'n_air'
            delta_str = 'delta_air'
        elif ambient_gas == ans.enums.AmbientGas.CO2:
            gamma_str = 'gamma_co2'
            n_str = 'n_co2'
            delta_str = 'delta_co2'
        else:
            gamma_str = f'gamma_{ambient_gas.name.lower()}'
            n_str = f'n_{ambient_gas.name.lower()}'
            delta_str = f'delta_{ambient_gas.name.lower()}'
            #raise ValueError(f'Unrecognised ambient gas {ambient_gas}')
        
        return gamma_str, n_str, delta_str
    
    def __init__(
            self,
            local_storage_dir : None | str = None,
        ):
        self.init_database(local_storage_dir)
    
    @property
    def ready(self) -> bool:
        """
        Required by LineDatabaseProtocol. Returns True if the database is ready to use, False otherwise
        """
        return self.db_init_flag
    
    def purge(self) -> None:
        """
        Required by LineDatabaseProtocol. Remove all local data and make it so the database must be reinitalised
        """
        if not self.db_init_flag:
            _lgr.warning('HITRAN database is not initialised yet, so cannot purge it.')
            return
        
        tablenames = list(hapi.tableList())
        
        
        # drop all tables
        for tablename in tablenames:
            hapi.dropTable(tablename)
        hapi.db_commit()
        
        # Remove any table files remaining
        for tablename in tablenames:
            table_header_path = os.path.join(self.local_storage_dir, f'{tablename}.header')
            table_data_path = os.path.join(self.local_storage_dir, f'{tablename}.data')
            table_fpaths = (
                table_header_path, 
                table_data_path
            )
            for fpath in table_fpaths:
                if os.path.exists(fpath):
                    os.remove(fpath)
                
        # delete all cache files
        cache_fpaths = (
            self._downloaded_gas_wavenumber_interval_cache_file,
        )
        for fpath in cache_fpaths:
            if os.path.exists(fpath):
                os.remove(fpath)
        
        # Delete local storage of data
        for k in tuple(self._downloaded_gas_wavenumber_interval.keys()):
            del self._downloaded_gas_wavenumber_interval[k]
        
        for k in tuple(self._gas_wavenumber_interval_to_download.keys()):
            del self._gas_wavenumber_interval_to_download[k]
        
        # Need this hack so tables are not held on to by `hapi`
        hapi.LOCAL_TABLE_CACHE = dict()
        
        # set flag so that database can be reinitalised
        self.db_init_flag = False
        _lgr.info(f'Database {self} has been purged. It no longer holds any local data and must be re-initialised using `self.init_database(...)`.')
        
        return


    def get_line_data(
            self, 
            gas_descs : tuple[RadtranGasDescriptor,...], 
            wave_range : WaveRange, 
            ambient_gas : ans.enums.AmbientGas
        ) -> LineDataProtocol:
        """
        Required by LineDatabaseProtocol. Retrieves line data from the database.
        
        Checks if database has been initialised, checks if we have all requested data, download if required, reads
        requested data from database and returns it in LineDataProtocol format
        
        NOTE: Should always return data in the same units it was requested in
        """
        
        gd = tuple(gas_descs)
        if not self.db_init_flag:
            self._init_database()
        
        self._check_available_data(gd, wave_range, ambient_gas)
        self._fetch_line_data()
        
        ld = self._read_line_data(
            gd, 
            wave_range, 
            ambient_gas
        )
        
        
        
        return ld
        
        
        
    
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
            _lgr.info(f'Cache file for `self._downloaded_gas_wavenumber_interval` not found at {cache_file_path=}. This is not a problem if the database is being built from scratch.')

    def store_downloaded_gas_wavenumber_interval_to_cache(self):
        cache_file_path = self._downloaded_gas_wavenumber_interval_cache_file
        
        _lgr.debug(f'Caching {self._downloaded_gas_wavenumber_interval=} to {cache_file_path=}')
        with open(cache_file_path, 'wb') as f:
            try:
                pickle.dump(self._downloaded_gas_wavenumber_interval, f)
            except Exception as e:
                _lgr.warning(f'Something went wrong when pickling `self._downloaded_gas_wavenumber_interval` to "{cache_file_path}". Data will not be cached. Error: {str(e)}')


    def init_database(
            self,
            local_storage_dir : None | str = None,
        ):
        
        
        if self.db_init_flag:
            _lgr.warning(f'Database {self} is already initialised, cannot initialise it again unless it is purged using `self.purge()`.')
            return
        
        if local_storage_dir is not None:
            self.local_storage_dir = local_storage_dir
        
        # Create directory to store database in
        os.makedirs(self.local_storage_dir, exist_ok=True)
        
        # Read downloaded_gas_wavenumber_interval_cache_file if it exists
        self.retrieve_downloaded_gas_wavenumber_interval_from_cache()
        

        #Starting the HAPI database
        hapi.db_begin(self.local_storage_dir)
        tablenames = list(hapi.tableList())
        
        _lgr.debug(f'{tablenames=}')
        
        # Read all of the data we have stored locally
        for tablename in tablenames:
            #if _lgr.level <= logging.DEBUG:
            #    hapi.describeTable(tablename)
            
            # Assume tables conform to our naming format
            try:
                gas_desc, ambient_gas = self.get_props_from_tablename(tablename)
            except Exception:
                _lgr.warning(f'HITRAN database found table "{tablename}" that does not conform to our naming format. Skipping...')
                continue
            
            # Should only ever have a single gas isotope per file
            hapi.select(tablename, ParameterNames=('molec_id', 'local_iso_id', 'nu'), Conditions=None, DestinationTableName = 'temp')
            found_molec_ids, found_iso_local_ids, found_v = hapi.getColumns('temp', ('molec_id', 'local_iso_id', 'nu'))
            
            assert len(set(found_molec_ids)) <= 1, "Should have at most one HITRAN gas per table"
            assert len(set(found_iso_local_ids)) <= 1, "Should have at most one HITRAN isotope per table"
            
            
            gda_pair = (gas_desc, ambient_gas)
            
            if gda_pair not in self._downloaded_gas_wavenumber_interval:
                found_v = np.array(found_v, dtype=float)
                if found_v.size != 0:
                    vmin = np.min(found_v)
                    vmax = np.max(found_v)
                    self._downloaded_gas_wavenumber_interval[gda_pair] = WaveRange(vmin, vmax, ans.enums.WaveUnit.Wavenumber_cm)
                    _lgr.info(f'No cache of downloaded wave range for {gda_pair}, using wave range {self._downloaded_gas_wavenumber_interval[gda_pair]=} found in tables {tablename} which will underestimate the requested range.')
                else:
                    _lgr.info(f'No cache of downloaded wave range for {gda_pair}, and no data found in table {tablename}, assuming no data has been downloaded yet for this gas/ambient-gas pair.')
                
            hapi.dropTable('temp')
        
        self.db_init_flag = True
        _lgr.info(f'Database {self} initialised.')
        
        return

    def _check_available_data(self, gas_descs : tuple[RadtranGasDescriptor,...], wave_range : WaveRange, ambient_gas : ans.enums.AmbientGas):
        # Always download 0.5% more than asked for to avoid annoying downloads due to edge effects
        expanded_wave_range = WaveRange(
            wave_range.min - 0.005*wave_range.size, 
            wave_range.max + 0.005*wave_range.size,
            wave_range.unit
        ).to_unit(ans.enums.WaveUnit.Wavenumber_cm)
        
        for gas_desc in gas_descs:
            gda_pair = (gas_desc, ambient_gas)
            if gda_pair not in self._downloaded_gas_wavenumber_interval:
                self._gas_wavenumber_interval_to_download[gda_pair] = expanded_wave_range
            elif not self._downloaded_gas_wavenumber_interval[gda_pair].contains(expanded_wave_range):
                self._gas_wavenumber_interval_to_download[gda_pair] = self._downloaded_gas_wavenumber_interval[gda_pair].union(expanded_wave_range)
    
    
    def _fetch_line_data(self) -> dict[bool,...]:
        if not self.db_init_flag:
            raise RuntimeError(f'Cannot fetch line data as database {self} is not initalised yet.')
        
        gas_exists_in_db = [False]*len(self._gas_wavenumber_interval_to_download)
        
        for gas_idx, (gda_pair, wave_range) in enumerate(tuple(self._gas_wavenumber_interval_to_download.items())):
            
            # Check that the gas exists in the HITRAN database
            gas_desc, ambient_gas = gda_pair
            ht_gas = HitranGasDescriptor.from_radtran(gas_desc)
            if ht_gas is None:
                gas_exists_in_db[gas_idx] = False
                _lgr.warning(f'Cannot download data for {gas_desc}, as that gas is not present in the HITRAN database')
                continue # cannot download data for a gas that is not present in HITRAN database
            else:
                gas_exists_in_db[gas_idx] = True
            
            # Work out if we have to download anything, or if we can just use the data we already have
            if gda_pair not in self._gas_wavenumber_interval_to_download:
                continue
            saved_wave_range = self._downloaded_gas_wavenumber_interval.get(gda_pair, None)
            if saved_wave_range is not None and saved_wave_range.contains(wave_range):
                _lgr.debug(f'Downloaded gas data {gda_pair} CONTAINS desired wave range {saved_wave_range} vs {wave_range}')
                continue
            else:
                _lgr.debug(f'Downloaded gas data {gda_pair} DOES NOT CONTAIN desired wave range {saved_wave_range} vs {wave_range}')
        
            
            # If we must download data then perform the download here.
            
            _lgr.info(f'Downloading data for {gda_pair} where {wave_range=} ({saved_wave_range=})...')
            
            vmin, vmax = wave_range.as_unit(ans.enums.WaveUnit.Wavenumber_cm).values()
            
            gamma_str, n_str, delta_str = self.get_ambient_gas_parameter_name_strings(ambient_gas)
            parameters = (
                'nu',
                'sw',
                'a',
                gamma_str,
                n_str,
                delta_str,
                'gamma_self',
                'n_self', # NOTE: This may be missing, in that case it is assumed to be equal to n_air
                'elower',
                'n_air', # NOTE: here so we always fetch this value as we do not know if 'n_self' will be different or not
            )
            _lgr.debug(f'Fetching the following parameters from HITRAN: {parameters}')
            try:
                hapi.fetch(
                    self.get_tablename_from_props(gas_desc, ambient_gas),
                    ht_gas.gas_id,
                    ht_gas.iso_id,
                    vmin,
                    vmax,
                    Parameters=parameters
                )
            except Exception as e:
                msg = '\n'.join((
                    'HAPI failed to retrieve data from HITRAN servers, possible reasons are listed below:',
                    '',
                    f'    1) There may be no transitions (lines) in your chosen wavelength range, try again with a different range. {wave_range=}',
                    '',
                    f'    2) One of the parameter names may be misspelled (unlikely), the parameter names used were: {parameters}',
                    '',
                    f'    3) There may be no gas with the requested gas_id {ht_gas.gas_id} and iso_id {ht_gas.iso_id} numbers, see "https://hitran.org/docs/iso-meta/" for accepted hitran gas codes (NOTE: these are different to Radtran gas codes, but they should be converted internally)',
                    '',
                    '    4) The internet connection may have dropped',
                    '',
                    'For more detailed troubleshooting change line 22 of archnemesis/database/wrappers/hapi.py to "_lgr.setLevel(logging.DEBUG)" and re-run. More detailed output from HAPI will be displayed.'
                ))
                raise RuntimeError(msg) from e
            else:
                hapi.db_commit()
            
            _lgr.info('Data downloaded.')
            
            self._downloaded_gas_wavenumber_interval[gda_pair] = wave_range
            if gda_pair in self._gas_wavenumber_interval_to_download:
                del self._gas_wavenumber_interval_to_download[gda_pair]
            
        # Cache to downloaded_gas_wavenumber_interval_cache_file for future
        self.store_downloaded_gas_wavenumber_interval_to_cache()
        
        return gas_exists_in_db


    def _read_line_data(
            self, 
            gas_descs : tuple[RadtranGasDescriptor,...], 
            wave_range : WaveRange, ambient_gas : ans.enums.AmbientGas
        ) -> LineDataProtocol:
        if not self.db_init_flag:
            raise RuntimeError(f'Cannot read line data as database {self} is not initalised yet.')
        
        temp_line_data_table_name = 'temp_line_data'
        _lgr.debug(f'{temp_line_data_table_name=}')
        
        #line_data = dict()
        
        _lgr.debug(f'{gas_descs=}')
        
        line_data = np.array(
            [], 
            dtype = [
                #('RT_GLOBAL_ISO_ID', int), # Radtran global isotope id
                ('RT_GAS_DESC', int, (2,)), # Radtran gas descriptor
                ('NU', float), # Transition wavenumber (cm^{-1})
                ('SW', float), # transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                ('A', float), # einstein-A coeifficient (s^{-1})
                ('GAMMA_AMB', float), # ambient gas broadening coefficient (cm^{-1} atm^{-1})
                ('N_AMB', float), # temperature dependent exponent for `gamma_amb` (NUMBER)
                ('DELTA_AMB', float), # ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                ('GAMMA_SELF', float), # self broadening coefficient (cm^{-1} atm^{-1})
                ('N_SELF', float), # temperature dependent exponent for `gamma_self` (NUMBER)
                ('ELOWER', float), # lower state energy (cm^{-1})
            ]
        ).view(np.recarray)
        
        vmin, vmax = wave_range.values()
        gamma_str, n_str, delta_str = self.get_ambient_gas_column_name_strings(ambient_gas)
        col_names = (
            'nu',
            'sw',
            'a',
            gamma_str,
            n_str,
            delta_str,
            'gamma_self',
            'n_self', # NOTE: This may be missing, in that case it is assumed to be equal to n_air
            'elower',
            'n_air', # NOTE: here as we always fetch this value as we do not know if 'n_self' will be different or not
        )
        
        for gas_desc in gas_descs:
            _lgr.debug(f'{gas_desc=}')
            ht_gas_desc = HitranGasDescriptor.from_radtran(gas_desc)
            if ht_gas_desc is None:
                continue
            
            #if _lgr.level <= logging.DEBUG:
            #    hapi.describeTable(self.get_tablename_from_props(gas_desc, ambient_gas))
            
            
            Conditions = ('and',('between', 'nu', vmin, vmax),('equal','local_iso_id',ht_gas_desc.iso_id))
            
            try:
                hapi.select(
                    self.get_tablename_from_props(gas_desc, ambient_gas), 
                    Conditions=Conditions, 
                    DestinationTableName=temp_line_data_table_name
                )
            except Exception as e:
                raise RuntimeError(f'Failure when reading from database {self}.') from e
            
            n_rows = hapi.getTableHeader(temp_line_data_table_name)['number_of_rows']
        
        
            
            if n_rows == 0:
                _lgr.info(f'{gas_desc=} has no line data in the requested wave range {wave_range=}, returning empty dataset.')
                continue

            else:
                cols = [
                    #[gas_desc.global_iso_id]*n_rows, 
                    [(gas_desc.gas_id, gas_desc.iso_id)]*n_rows, 
                    *hapi.getColumns(
                        temp_line_data_table_name,
                        col_names
                    )
                ]
                
                with warnings.catch_warnings():
                    warnings.simplefilter('default', UserWarning)
                    
                    # If we don't have any 'n_self' values, replace 'n_self' with final 'n_air' column
                    # Then, drop final 'n_air' column as we don't need it
                    n_self_tmp = np.ma.array(cols[-3], dtype=float)
                    n_self = np.array(cols[-1], dtype=float)
                    n_self[~n_self_tmp.mask] = n_self_tmp[~n_self_tmp.mask]
                    
                    cols[-3] = n_self
                    cols = cols[:-1]
                    
                    _lgr.debug(f'{len(cols)=} {[len(c) for c in cols]=}')
                    #_lgr.debug(f'{cols[0]=}')
                    #_lgr.debug(f'{cols[1]=}')
                    #_lgr.debug(f'{cols[2]=}')
                    
                    for i, c in enumerate(cols[1:]):
                        missing_col_names = []
                        if isinstance(c[0], np.ma.core.MaskedConstant):
                            missing_col_names.append(col_names[i])
                        if len(missing_col_names) > 0:
                            raise RuntimeError(f'Gas {gas_desc.gas_name} isotope {gas_desc.iso_id} "{gas_desc.isotope_name}" with ambient gas "{ambient_gas.name}" has NULL entries in the HITRAN database for the following columns: {missing_col_names}')
                
                    line_data = np.lib.recfunctions.stack_arrays(
                        (
                            line_data, 
                            np.array(
                                list(zip(*cols)),
                                dtype = [
                                    ('RT_GAS_DESC', int, (2,)), # Radtran gas descriptor
                                    ('NU', float), # Transition wavenumber (cm^{-1})
                                    ('SW', float), # transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                                    ('A', float), # einstein-A coeifficient (s^{-1})
                                    ('GAMMA_AMB', float), # ambient gas broadening coefficient (cm^{-1} atm^{-1})
                                    ('N_AMB', float), # temperature dependent exponent for `gamma_amb` (NUMBER)
                                    ('DELTA_AMB', float), # ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                                    ('GAMMA_SELF', float), # self broadening coefficient (cm^{-1} atm^{-1})
                                    ('N_SELF', float), # temperature dependent exponent for `gamma_self` (NUMBER)
                                    ('ELOWER', float), # lower state energy (cm^{-1})
                                ]
                            ).view(np.recarray)
                        ),
                        defaults=None,
                        usemask = False,
                        asrecarray=True,
                        autoconvert=False
                    )
            
            hapi.dropTable(temp_line_data_table_name)
        
        return line_data
    
    
    
    
