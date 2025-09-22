import os.path
import dataclasses as dc
import bz2
from typing import Self, Iterator
import textwrap

import numpy as np
import matplotlib.pyplot as plt

from ...utils import fetch
from .isotope_def import ExomolIsotopeDef


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)



@dc.dataclass(repr=False)
class ExomolIsotopeStates():
    iso_slug : str
    dataset : str
    data : np.ndarray # structured array
    col_descriptions : dict[str,str]

    @staticmethod
    def get_initial_structured_dtype_list(
            iso_def : None | ExomolIsotopeDef = None
    ) -> tuple[list[tuple[str,type,int],...], dict[str,str]]:
        """
        ## EXOMOL ".states" file ##
        Often these have been zipped, should be able to read without unzipping via a standard library function thing (`import bz2; bz2.open(...)`)
        
        NOTE: We have to use an `ExomolIsotopeDef` instance to make sure all of the columns are correctly named,
        if not given an `ExomolIsotopeDef` instance will only label the first 4 columns (id, energy, degeneracy, J), the rest will
        be labelled as `col_i` where `i` is the index of the column (starting from zero).
        """        
        column_descriptions = [
            "Unique identification number for a state",
            "Energy (cm^-1) of state (above ground state)",
            "total degeneracy of state",
            "total angular momentum quantum number"
        ]
        column_types = [
            np.int32,
            np.float64,
            np.int32,
            np.float64
        ]
        column_names = [
            'id', # state ID number
            'energy', # state energy in cm^-1
            'degeneracy', # state total degeneracy
            'J', # total angular momentum quantum number
        ] 

        if iso_def is not None:
            column_descriptions += (["lifetime of state in seconds (INF if ground state)"] if iso_def.lifetime_available == 1 else []) # lifetime of state (seconds)
            column_descriptions += (["Lande g-factor of state"] if iso_def.lande_g_factor_available == 1 else []) # Lande g-factor
            
            column_types += ([np.float64] if iso_def.lifetime_available == 1 else []) # lifetime of state (seconds)
            column_types += ([np.float64] if iso_def.lande_g_factor_available == 1 else []) # Lande g-factor
            
            column_names += (['lifetime'] if iso_def.lifetime_available == 1 else []) # lifetime of state (seconds)
            column_names += (['lande_g'] if iso_def.lande_g_factor_available == 1 else []) # Lande g-factor
            
    
            # Add each quanta case to the column names
            for qc in iso_def.quanta_case:
                for q in qc.quanta:
                    column_names.append(q.label)
                    column_descriptions.append(q.description)
    
                    # Translate FORTRAN format strings into types
                    if 'A' in q.format:
                        str_size = int(q.format[q.format.index('A'):].split()[0][1:])
                        col_type = (np.ubyte, str_size)
                    elif 'F' in q.format:
                        col_type = np.float64
                    elif 'I' in q.format:
                        col_type = np.int32
                    else:
                        col_type = (np.ubyte, 128) # most general case, 128 characters
                    column_types.append(col_type)
    
            # Add each auxiliary to the column names
            for aux in iso_def.auxiliary_list.entry[::-1]: # NOTE: for some reason these seem to be listed back-to-front
                column_names.append(aux.label)
                column_descriptions.append(aux.description)
    
                # Translate FORTRAN format strings into types
                if 'A' in aux.format:
                    str_size = int(q.format[q.format.index('A'):].split()[0][1:])
                    col_type = (np.ubyte, str_size)
                elif 'F' in aux.format:
                    col_type = np.float64
                elif 'I' in aux.format:
                    col_type = np.int32
                else:
                    col_type = (np.ubyte, 128) # most general case, 128 characters
                column_types.append(col_type)
        
        dtype_list : list[tuple[str, type, int],...] = [(name, *(t if (type(t) is tuple) else (t,1))) for name, t in zip(column_names, column_types)] 
        col_descriptions = dict((name, desc) for name, desc in zip(column_names, column_descriptions))
        return dtype_list, col_descriptions


    
    @classmethod
    def from_isotope_def(cls, 
            iso_def : ExomolIsotopeDef, 
            database_url : str, 
            dest_dir : str
    ) -> Self:
        iso_slug = iso_def.iso_slug
        dataset = iso_def.dataset
        
        url = iso_def.get_parent_url(database_url) + f'/{iso_slug}__{dataset}.states.bz2'
        dtype_list, col_descriptions = cls.get_initial_structured_dtype_list(iso_def)
        
        fpath = cls.download_file(url, dest_dir)
        _lgr.debug(f'"{url}" downloaded to "{fpath}"')

        data = cls.get_data_from_file_bz2(fpath, iso_def.n_states, dtype_list, col_descriptions)

        return cls(iso_slug, dataset, data, col_descriptions) 
    
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
                _lgr.debug(f'WARNING: File "{fpath}" already exists but is {found_file_size} bytes in size, therefore will download again despite request to not overwrite file')
            else:
                return fpath

        fetch.file(url, to_path=fpath, encoding=None)
        return fpath
    
    @classmethod
    def get_data_from_file_bz2(cls,
            fpath : str, 
            n_rows : int,
            dtype_list : None | list[tuple[str,type,int],...] = None, 
            col_descriptions : None | dict[str,str] = None,
    ):
        # Read first line of file to ensure we have the correct number of columns
        # if we need more columns add to `dtype_list` and update `col_descriptions`
        # then create a structured numpy array to hold everything and fill it from
        # the file.
        

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

        assert len(col_start_idxs) == n_col, f"Must have same number of columns from getting column widths {len(col_start_idxs)} and counting column entries {n_col}"

        # Increase `dtype_list` so it contains text fields for columns we have not managed to get information for yet
        col_edge_idxs = []
        for i, csi in enumerate(col_start_idxs):
            col_edge_idxs.append((csi, (col_start_idxs[i+1] - 1) if i+1 < n_col else (line_size-1)))
        
        col_idx = len(dtype_list)
        while (len(dtype_list) < n_col):
            col_name = f'col_{col_idx}'
            col_size = col_edge_idxs[col_idx][1] - col_edge_idxs[col_idx][0]
            
            dtype_list.append((col_name, np.ubyte, col_size))
            col_descriptions[col_name] = f"No description found for column {col_idx} with width {col_size}"
            
            col_idx += 1
        
        data = np.empty((n_rows,), dtype=dtype_list)

        row_idx = 0
        with bz2.open(fpath, 'r') as f:
            while (a_row := f.read(line_size)):
                if (row_idx % 100000 == 0):
                    _lgr.debug(f'reading in row {row_idx} / {n_rows}')
                
                if len(a_row) != line_size:
                    raise RuntimeError(f'While reading "{fpath}", expected rows of size {line_size}, but row {row_idx} has size {len(a_row)}')

                data[row_idx] = tuple(dtype_list[i][1](a_row[col_edge_idxs[i][0]:col_edge_idxs[i][1]]) for i, col in enumerate(data.dtype.names))
                row_idx += 1

        _lgr.debug(f'{cls.__name__} size of loaded data : {data.nbytes / (1024*1024)} Mb')
        return data


    def plot(self, max_points : int = 1000000):
        n_points_step = int(max(1, np.ceil(self.data.shape[0] / max_points)))
        
        plt.plot(
            self.data['id'][::n_points_step],
            self.data['energy'][::n_points_step],
            ',',
            alpha=0.3
        )
        plt.title(f'Energy of states of {self.iso_slug}__{self.dataset}')
        plt.xlabel('State id (number)')
        plt.ylabel('State energy (cm^{-1})')
    
    def column_info(self) -> Iterator[tuple[str,str]]:
        """
        Returns an iterator of column (name, description) pairs
        """
        for name in self.data.dtype.names:
            yield (name, self.col_descriptions[name])
    
    def print_column_info(self):
        """
        Print column information to screen
        """
        for name, description in self.column_info:
            print(f'{name}\n{textwrap.indent(description,"    ")}')














