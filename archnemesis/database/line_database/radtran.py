from __future__ import annotations #  for 3.9 compatability

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

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


from typing import NamedTuple
from enum import IntEnum

class LineRecordFormat(IntEnum):
    HITRAN=0



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


def in_chunks_of(a, chunk_sizes):
    i = 0
    for s in chunk_sizes:
        yield a[i:i+s]
        i+=s


class RecordHitran160(NamedTuple):
    gas_id : int
    iso_id : int
    line_wavenumber : float
    line_strength : float
    einstein_a_coeff : float
    gamma_air : float
    gamma_self : float
    e_lower : float
    n_air : float
    delta_air : float
    global_upper_quanta : str
    global_lower_quanta : str
    local_upper_quanta : str
    local_lower_quanta : str
    ierr : None | int
    iref : int
    line_mixing_flag : str
    gp : float
    gpp : float


def read_records_hitran_160(fpath : str, fixed_width=True):
    i=0
    records = []
    with open(fpath, 'r') as f:
        # skip anytihng starting with a '#'
        while True:
            if (i%10000 == 0):
                _lgr.info(f'read {i} records...')
            i+=1
            
            a = f.readline() if not fixed_width else f.read(160)
            if a.strip().startswith('#'):
                continue
            
            if len(a) == 0:
                break # end of file
            
            
            records.append(
                RecordHitran160(
                    *(fn(x) for fn,x in zip((
                        int,#lambda x: (_lgr.debug(f'{x=}'), int(x))[1],
                        int,#lambda x: (_lgr.debug(f'{x=}'), int(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        str,#lambda x: (_lgr.debug(f'{x=}'), str(x))[1],
                        str,#lambda x: (_lgr.debug(f'{x=}'), str(x))[1],
                        str,#lambda x: (_lgr.debug(f'{x=}'), str(x))[1],
                        str,#lambda x: (_lgr.debug(f'{x=}'), str(x))[1],
                        lambda x: tuple(int(z) for z in x),#lambda x: (_lgr.debug(f'{x=}'), tuple(int(z) for z in x))[1],
                        lambda x: tuple(x[2*i:2*i+2] for i in range(6)),#lambda x: (_lgr.debug(f'{x=}'), tuple(x[2*i:2*i+2] for i in range(6)))[1],
                        str,#lambda x: (_lgr.debug(f'{x=}'), str(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                        float,#lambda x: (_lgr.debug(f'{x=}'), float(x))[1],
                    ),
                    in_chunks_of(a, (
                        2,
                        1,
                        12,
                        10,
                        10,
                        5,
                        5,
                        10,
                        4,
                        8,
                        15,
                        15,
                        15,
                        15,
                        6,
                        12,
                        1,
                        7,
                        7,
                    ))
                ))
            ))
    
    return records

class RADTRAN(LineDatabaseProtocol):
    FORTRAN_RECORD_LENGTH : int = 4 # bytes
    
    def __init__(self, keyfile : str, ambient_gas = ans.enums.AmbientGas.AIR):
        _lgr.debug(f'Creating {self.__class__.__name__} with {keyfile=} {ambient_gas=}')
        
        self.keyfile = os.path.abspath(keyfile)
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
        
        _lgr.debug(f'Reading line records (format={self.record_format}, record_length={self.database_record_length}) from "{self.database_file}"')
        if (self.record_format == LineRecordFormat.HITRAN) and (self.database_record_length == 160):
            records = read_records_hitran_160(self.database_file)
        else:
            raise RuntimeError(f'No known reader for combination of record format {self.record_format} and record length {self.database_record_length}')
        
        
        _lgr.debug('Loading all records into memory...')
        
        attrs = ( # which attributes of "RecordHitran160" are associated with the attributes of LineDataProtocol
            'line_wavenumber',
            'line_strength',
            'einstein_a_coeff',
            'gamma_air',
            'n_air',
            'delta_air',
            'gamma_self',
            'n_air',
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
                _lgr.warning(f'Cannot retrieve data for {gas_desc} and {ambient_gas} as there is no entry in database that matches')
                result[gas_desc] = None
                continue
            
            x = self.data[key]
            result[gas_desc] = x[(vmin <= x.NU) & (x.NU <= vmax)]
        return result