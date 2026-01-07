import os.path
import dataclasses as dc
import bz2
from typing import Self
import json

import numpy as np

from ...utils import fetch
#from .isotope_def import ExomolIsotopeDef
#from ...datatypes.wave_range import WaveRange
#from archnemesis.enums import WaveUnit


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


EXOMOL_API_URL : str = "https://exomol.com/api/?molecule="

@dc.dataclass()
class ExomolIsotopeTrans:
    mol_formula : str
    iso_formula : str
    dataset : str
    
    data : np.ndarray
    
    @property
    def upper_state_id(self):
        return self.data['upper_state_id']
    
    @property
    def lower_state_id(self):
        return self.data['lower_state_id']
    
    @property
    def einstein_a_coeff(self):
        return self.data['einstein_a_coeff']
    
    @property
    def wavenumber(self):
        return self.data['wavenumber']
    
    @classmethod
    def from_attrs(cls,
            mol_formula : str,
            iso_formula : str,
            dataset : str, 
            dest_dir : str,
            n_file_limit = 10000, # max number of files to download
    ) -> Self:
        
        url = f'{EXOMOL_API_URL}{mol_formula}'
        _lgr.info(f'Fetching ExoMol molecule trans file lists from "{url}"')
        spec = json.loads(fetch.file(url, encoding='ascii'))
        
        trans_files = [x['url'] for x in spec[iso_formula]['linelist'][dataset]['files'] if x['url'].endswith('.trans.bz2')]
        _lgr.info(f'Found {len(trans_files)} transition files for molecule="{mol_formula}", isotope="{iso_formula}", dataset="{dataset}". Only using up to {n_file_limit} files.')
        
        for trans_file_url in trans_files:
            if n_file_limit <= 0:
                break
                
            fpath = cls.download_file(f'https://{trans_file_url}', dest_dir)
            _lgr.info(f'"{trans_file_url}" downloaded to "{fpath}"')

            data = cls.get_data_from_file_bz2(fpath)

            yield cls(mol_formula, iso_formula, dataset, data)
            n_file_limit -= 1
    
    @staticmethod
    def download_file(
            url : str, 
            dest_dir : str, 
            overwrite : bool = False
    ) -> str:
        filename = url.rsplit('/',1)[1]
        fpath = os.path.join(dest_dir, filename)

        if os.path.exists(fpath) and (not overwrite):
            found_file_size = os.path.getsize(fpath)
            if found_file_size == 0:
                _lgr.warn(f'WARNING: File "{fpath}" already exists but is {found_file_size} bytes in size, therefore will download again despite request to not overwrite file')
            else:
                return fpath

        fetch.file(url, to_fpath=fpath, encoding=None)
        return fpath
    
    @classmethod
    def get_data_from_file_bz2(cls, fpath):
    
        # Read in the first line to see 1) how long is it; 2) how many colums does it have
        
        line_size = None
        n_col = None
        col_start_idxs = []
        with bz2.open(fpath, 'rb') as f:
            next_line = f.readline().decode('ascii')
            _lgr.debug(f'{next_line=}')
            
            line_size = len(next_line)
            _lgr.debug(f'{line_size=}')

            _lgr.debug(f'{next_line.split()=}')
            n_col = len(next_line.split())
            _lgr.debug(f'{n_col=}')

            # Assume at least a single space between columns
            # and columns are right aligned
            pos = 0
            new_col = True
            prev_c = ' '
            for c in next_line:
                if new_col:
                    col_start_idxs.append(pos)
                    new_col = False
                
                if (prev_c != ' ') and (c == ' ') and (not new_col):
                    new_col = True
                prev_c = c       
                pos += 1
        
        col_start_idxs.append(line_size)
        col_slices = [slice(i,j) for i,j in zip(col_start_idxs[:-1],col_start_idxs[1:])]
        col_types = (int, int, float, float)
        
        if next_line[col_slices[-1]] == '\n':
            col_slices = col_slices[:3]
        
        del col_start_idxs
        
        
        n_data_rows = 10000
        dtype_list = [
            ('upper_state_id', np.int32),
            ('lower_state_id', np.int32),
            ('einstein_a_coeff', np.float64),
            ('wavenumber', np.float64),
        ]
        fill_value=(-1,-1,np.nan,np.nan)
        data = np.empty((n_data_rows,), dtype=dtype_list)
        data[:] = fill_value
        
        i=0
        with bz2.open(fpath, 'rb') as f:
            while len(row := f.read(line_size)) > 0:
                if (i % 100000 == 0):
                    _lgr.info(f'reading in row {i}')
                    
                if i >= n_data_rows:
                    # Expand the array as we need to, grow exponentially
                    new_n_data_rows = 2*n_data_rows
                    data_new = np.empty((new_n_data_rows,), dtype=dtype_list)
                    data_new[:] = fill_value
                    data_new[:n_data_rows] = data
                    del data
                    data = data_new
                    n_data_rows = new_n_data_rows
                    
                for j, (col_slice, col_type, col_name) in enumerate(zip(col_slices, col_types, data.dtype.names)):
                    data[i][col_name] = col_type(row[col_slice])
                i += 1
        
        return np.array(data[:i])








