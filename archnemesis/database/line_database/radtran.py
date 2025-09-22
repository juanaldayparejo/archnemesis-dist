#from __future__ import annotations #  for 3.9 compatability

import os
import os.path

import numpy as np
import numpy.ma


import archnemesis as ans
import archnemesis.enums
from ..protocols import (
    LineDatabaseProtocol, 
    LineDataProtocol, 
)
from ..datatypes.wave_range import WaveRange
#from ..datatypes.gas_isotopes import GasIsotopes
from ..datatypes.gas_descriptor import RadtranGasDescriptor
import archnemesis.database.datatypes.fixed_width.hitran

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


from typing import NamedTuple
from enum import IntEnum

class LineRecordFormat(IntEnum):
    HITRAN = 0


class RadtranGasIsotopeData(NamedTuple):
    gas_id : int
    iso_id : int
    global_id : int
    n_isotopes : int
    abundance : float
    mol_mass : float
    name : str
    partition_fn_polynomial_coeffs : np.ndarray


def load_gas_isotope_data(fpath : str):
    gasses = []
    with open(fpath, 'r') as f:
        
        
        while True:
            a = f.readline();  #_lgr.debug(f'{a=}')
            if a.strip().startswith('#'):
                continue
            break 
        
        # there is always a line after the header that we should ignore
        # which `a` now contains, therefore start reading the records
        # and discard the current value of `a`
        
        while True:
            
            # read each record
            a = f.readline(); #_lgr.debug(f'{a=}') # skip record separator line
            
            if len(a) == 0:
                break
            a = f.readline(); #_lgr.debug(f'{a=}')
            gas_id = int(a.strip().split()[0])
            
            a = f.readline(); #_lgr.debug(f'{a=}')
            gas_name = a.strip()
            
            a = f.readline();# _lgr.debug(f'{a=}')
            n_isotopes = int(a.strip().split()[0])
            
            for i in range(0, n_isotopes):
                a = f.readline(); #_lgr.debug(f'{a=}')
                # this line is formatted as:
                # `<iso_id> <abundance> <mol_mas> (<global_id>) <REST_OF_LINE>
                # unfortunately the brackets around <global_id> can have spaces after them so I can't
                # just use 'a.strip().split()`. I must accout for this case.
                z = a.strip().split(maxsplit=3) # split the first 3 fields, therefore z[3]="(<global_id) <REST_OF_LINE>"
                z[3] = z[3][z[3].index('(')+1:z[3].index(')')] # assuming <gloab_id> is just digits, only keep the string after the first opening bracket and before the first closing bracket. NOTE: This does NOT BALANCE BRACKETS
                
                iso_id, abundance, mol_mass, global_id = (fn(x) for fn,x in zip((int, float, float, int), z))
                
                a = f.readline(); #_lgr.debug(f'{a=}')
                partition_fn_polynomial_coeffs = np.array(list(map(float, a.strip().split())), dtype=float)
                
                gasses.append(
                    RadtranGasIsotopeData(
                        gas_id,
                        iso_id,
                        global_id,
                        n_isotopes,
                        abundance,
                        mol_mass,
                        gas_name,
                        partition_fn_polynomial_coeffs
                    )
                )
    
    # all gasses read in, so now create a dictionary of them
    gas_info = dict(
        [(RadtranGasDescriptor(x.gas_id, x.iso_id), x) for x in gasses]
    )
    return gas_info


class RADTRAN(LineDatabaseProtocol):
    FORTRAN_RECORD_LENGTH : int = 4 # bytes
    
    def __init__(
            self, 
            fpath : str, 
            ambient_gas = ans.enums.AmbientGas.AIR,
            
    ):
        _lgr.debug(f'Creating {self.__class__.__name__} with {fpath=} {ambient_gas=}')
        
        if os.path.isdir(fpath):
            # .key file should be present, and there should only be one
            keyfiles = [x for x in os.listdir(path=fpath) if x.endswith('.key')]
            if len(keyfiles) > 1:
                raise ValueError(f'Cannot create {self.__class__.__name__} from path "{fpath}". Requires a *.key file or directory containing a single *.key file')
            else:
                self.keyfile = os.path.abspath(os.path.join(fpath,keyfiles[0]))
            
        elif os.path.isfile(fpath):
            # it should be the .key file
            if not fpath.endswith('.key'):
                if os.path.isfile(fpath+'.key'):
                    self.keyfile = os.path.abspath(fpath+'.key')
                raise ValueError(f'Cannot create {self.__class__.__name__} from path "{fpath}". Requires a *.key file or directory containing a single *.key file')
            else:
                self.keyfile = os.path.abspath(fpath)
        else:
            raise ValueError(f'Cannot create {self.__class__.__name__} from path "{fpath}". Requires a *.key file or directory containing a single *.key file')
        
        
        self.ambient_gas = ambient_gas
        self._local_storage_dir = os.path.dirname(self.keyfile)
        
        with open(self.keyfile, 'r') as f:
            self.name = f.readline().strip()
            self.keyfile_location = f.readline().strip()
            self.original_data_file = f.readline().strip()
            self.database_file = f.readline().strip()
            self.index_file = f.readline().strip()
            self.gas_data_file = f.readline().strip()
            self.isotope_translation_file = f.readline().strip()
            self.n_records = int(f.readline().strip().split()[0])
            self.record_format = LineRecordFormat(int(f.readline().strip().split()[0]))
            self.index_record_length = int(f.readline().strip().split()[0])*self.FORTRAN_RECORD_LENGTH #  bytes
            self.database_record_length = int(f.readline().strip().split()[0]) # bytes
        
        _lgr.debug(f'Read {self.keyfile}')
        _lgr.debug(f'{self.name=}')
        _lgr.debug(f'{self.keyfile_location=}')
        _lgr.debug(f'{self.original_data_file=}')
        _lgr.debug(f'{self.database_file=}')
        _lgr.debug(f'{self.index_file=}')
        _lgr.debug(f'{self.gas_data_file=}')
        _lgr.debug(f'{self.isotope_translation_file=}')
        _lgr.debug(f'{self.n_records=}')
        _lgr.debug(f'{self.record_format=}')
        _lgr.debug(f'{self.index_record_length=}')
        _lgr.debug(f'{self.database_record_length=}')
        
        # NOTE: the below is not recommended as it loads all the data into memory at once.
        
        _lgr.debug(f'Reading gas information from "{self.gas_data_file}"')
        self.gas_info = load_gas_isotope_data(self.gas_data_file)
        
        _lgr.info(f'Finding format class for {self.record_format=} {self.database_record_length=}')
        format_class = None
        if self.record_format == LineRecordFormat.HITRAN:
            format_class_name = f'FormatHitran{self.database_record_length}'
            format_class = getattr(ans.database.datatypes.fixed_width.hitran, format_class_name, None)
        
            if format_class is None:
                _lgr.error(f'Systematic format class name "{format_class_name}" not present in "archnemesis.database.datatypes.fixed_width.hitran"')
            else:
                _lgr.warning(f'Format class found from systematic format class name "{format_class_name}", NOTE: HITRAN legacy format is not used at the moment so be careful if using "einstein_a_coeff" attribute of records.')
        else:
            _lgr.error(f'No LineRecordFormat matching {self.record_format}. Possible values are {list(LineRecordFormat)}')
        
        if format_class is None:
            raise RuntimeError(f'Could not find format class for {self.record_format=} {self.database_record_length=}')
        
        _lgr.debug(format_class.to_string())
        
        _lgr.debug(f'Reading line records (format={self.record_format}, record_length={self.database_record_length}) from "{self.database_file}"')
        records = format_class.read_records(self.database_file)
        
        
        _lgr.debug('Loading all records into memory...')
        
        # NOTE: Here is where I would put in something that swaps from legacy values to 'einstein_a_coeff'
        #       or put it in the `format_class` classes that perform the reading.
        
        attrs = ( # which attributes of "format_class" are associated with the attributes of LineDataProtocol
            'line_wavenumber',
            'line_strength',
            'einstein_a_coeff',
            'gamma_amb',
            'n_amb',
            'delta_amb',
            'gamma_self',
            'n_amb',
            'e_lower'
        )
        
        self.data = dict()
        
        for gas_desc in self.gas_info.keys():
            key = (gas_desc.gas_id, gas_desc.iso_id, self.ambient_gas)
            self.data[key] = np.rec.fromrecords(
                [tuple(getattr(x,attr) for attr in attrs) for x in records if ((x.gas_id == gas_desc.gas_id) and (x.iso_id == gas_desc.iso_id))],
                dtype=[
                    ('NU', float), # Transition wavenumber (cm^{-1})
                    ('SW', float), # transition intensity per molecule (weighted by terrestrial isotopologue abundance) (cm^{-1} molecule^{-1} cm^{-2}) at standard temperature and pressure (STP)
                    ('A', float), # einstein-A coefficient for spontaneous emission (s^{-1})
                    ('GAMMA_AMB', float), # ambient gas broadening coefficient (cm^{-1} atm^{-1})
                    ('N_AMB', float), # temperature dependent exponent for `GAMMA_AMB` (NUMBER)
                    ('DELTA_AMB', float), # ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                    ('GAMMA_SELF', float), # self broadening coefficient (cm^{-1} atm^{-1})
                    ('N_SELF', float), # temperature dependent exponent for `GAMMA_SELF` (NUMBER)
                    ('ELOWER', float), # lower state energy (cm^{-1})
                ]
            )
            
            # if we have no data for a gas, delete the entry from the dictionary
            if len(self.data[key]) == 0:
                del self.data[key]
        
        _lgr.debug(f'Initialisation of {self} complete.')
        return
        
        
    def __repr__(self):
        """
        Returns a string that represents the current state of the class
        """
        return f'{self.__class__.__name__}(instance_id={id(self)}, local_storage_dir={self.local_storage_dir}, name={self.name})'
    
    @property
    def ready(self) -> bool:
        """
        Returns True if the database is ready to use, False otherwise.
        """
        raise NotImplementedError
    
    @property
    def local_storage_dir(self) -> str:
        """
        Returns the directory the local database is stored in
        """
        return self._local_storage_dir
    
    @local_storage_dir.setter
    def local_storage_dir(self, value : str) -> None:
        """
        Sets the directory the local database is stored in
        """
        raise NotImplementedError
    
    def purge(self):
        """
        Remove all cached data from this database and make it so the database must be reinitalised
        """
        raise NotImplementedError
    
    def get_line_data(
            self, 
            gas_descriptors : tuple[RadtranGasDescriptor,...], 
            wave_range : WaveRange, 
            ambient_gas : ans.enums.AmbientGas
        ) -> dict[RadtranGasDescriptor, None | LineDataProtocol]:
        """
        """
        result = dict()
        
        vmin, vmax = wave_range.to_unit(ans.enums.WaveUnit.Wavenumber_cm).values()
        
        for gas_desc in gas_descriptors:
            key = (gas_desc.gas_id, gas_desc.iso_id, ambient_gas)
            
            if key not in self.data:
                _lgr.warning(f'Cannot retrieve data for {gas_desc} and {ambient_gas=} as there is no entry in database that matches. Entries are {list(self.data.keys())}')
                result[gas_desc] = None
                continue
            
            x = self.data[key]
            result[gas_desc] = x[(vmin <= x.NU) & (x.NU <= vmax)]
        return result