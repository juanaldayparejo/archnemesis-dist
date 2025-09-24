

import os
import os.path

import numpy as np
import numpy.ma
import sqlite3
import tempfile
import json
from urllib.error import HTTPError

from archnemesis.Data import gas_data
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

from ..datatypes.exomol.gas_descriptor import ExomolGasDescriptor
from ..filetypes.exomol.root import ExomolRootFormat
from ..filetypes.exomol.isotope_def import ExomolIsotopeDef
from ..filetypes.exomol.states import ExomolIsotopeStates
from ..filetypes.exomol.trans import ExomolIsotopeTrans
from ..utils import fetch

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


EXOMOL_DATABASE_URL : str = "https://www.exomol.com/db"

EXOMOL_API_URL_FMT : str = "https://exomol.com/api/?molecule={}"
EXOMOL_API_DATA_CACHE = dict()

def get_exomol_api_data_path(directory : str, mol_formula : str) -> str:
    return os.path.join(directory, f'exomol_api_data_{mol_formula}.json')

def iter_exomol_api_data(exomol_root, directory : str):
    for mi in exomol_root.molecule_info:
        mol_formula = mi.formula
        if mol_formula.endswith('+'):
            mol_formula = mol_formula[:-1] + '_p'
        if mol_formula.endswith('-'):
            mol_formula = mol_formula[:-1] + '_n'
        
        
        # First check the cache
        api_data = EXOMOL_API_DATA_CACHE.get(mol_formula, None)
        if api_data is not None:
            return api_data
        
        
        api_data_path = get_exomol_api_data_path(directory, mol_formula)
        
        # If not on disk, download from web
        if not os.path.exists(api_data_path):
            fetch.file(EXOMOL_API_URL_FMT.format(mol_formula), encoding=None, to_fpath=api_data_path)
        
        
        # By now, should be on disk, so open it, read it, store in cache, and return the contents
        with open(api_data_path, 'r') as f:
            api_data = json.load(f)
        
        # store in cache
        EXOMOL_API_DATA_CACHE[mol_formula] = api_data
        
        yield api_data

def iter_exomol_def_files_from_api_data(api_data):
    for iso_formula, iso_info in api_data.items():
        mol_formula = iso_info['molecule']
        for dataset_name in [k for k in iso_info['linelist'].keys() if k!='data type']:
            iso_slug = iso_formula.replace(')(', '-').replace('(','').replace(')','')
            yield iso_formula, f'{mol_formula}/{iso_slug}/{dataset_name}/{iso_slug}__{dataset_name}.def'

def get_exomol_api_data(exomol_def, directory : str):
    mol_formula = exomol_def.get_mol_formula()
    
    # First check the cache
    api_data = EXOMOL_API_DATA_CACHE.get(mol_formula, None)
    if api_data is not None:
        return api_data
    
    
    api_data_path = get_exomol_api_data_path(directory, mol_formula)
    
    # If not on disk, download from web
    if not os.path.exists(api_data_path):
        api_data_path.file(EXOMOL_API_URL_FMT.format(mol_formula), encoding=None, to_fpath=api_data_path)
    
    
    # By now, should be on disk, so open it, read it, store in cache, and return the contents
    with open(api_data_path, 'r') as f:
        api_data = json.load(f)
    
    # store in cache
    EXOMOL_API_DATA_CACHE[mol_formula] = api_data
    
    return api_data


def get_exomol_trans_urls(exomol_def, api_data):
    trans_files = [x['url'] for x in api_data[exomol_def.iso_formula]['linelist'][exomol_def.dataset]['files'] if x['url'].endswith('.trans.bz2')]
    return trans_files

def get_exomol_states_urls(exomol_def, api_data):
    states_files = [x['url'] for x in api_data[exomol_def.iso_formula]['linelist'][exomol_def.dataset]['files'] if x['url'].endswith('.states.bz2')]
    return states_files

def get_exomol_broad_urls(exomol_def, database_url : str) -> tuple[str,...]:
    isotope_url = database_url + f'/{exomol_def.get_mol_formula()}/{exomol_def.iso_slug}'
    return tuple((isotope_url + '/' + x.filename) for x in exomol_def.broadener_file_info)

class EXOMOL:
    database_url : str = EXOMOL_DATABASE_URL
    all_file : str = EXOMOL_DATABASE_URL + '/exomol.all'
    proxy : None | dict[str, str] = None # None or dictionary mapping protocol names to URLs of proxies
    
    
    def __init__(self,
            local_storage_dir : str, # Directory in which we should store the database
    ):
        # 0) Create/validate the database backend
        # 1) Download parts of the exomol database incrementally (only the bits we need)
        # 2) Calculate the values required by the `LineDataProtocol` and store them in the a database
        # 3) provide ways to pull wanted values out of the database on demand.
        
        self.local_storage_dir = local_storage_dir
        self.database_fpath = os.path.join(local_storage_dir, 'exomol.db')
        
        
        if not os.path.exists(self.database_fpath):
            exomol_root = ExomolRootFormat.from_url(self.all_file)
            self.build_exomol_database(self.database_fpath, exomol_root)
        
    
    @classmethod
    def build_exomol_database(cls, fpath, exomol_root):
        """
        NOTE: Not everything is in the "exomol.all" file, also need to use "https://exomol.com/api/?molecule={MOLECULE_NAME}" and find all 
        the `linelist` entries for each isotope.
        """
        # NOTE: Need to use RADTRAN names for everything
        _lgr.info(f'Building database in "{fpath}"')
        
        #logging.getLogger("archnemesis.database.datatypes.exomol.tagged_format").setLevel(logging.DEBUG)
        
        db = sqlite3.connect(fpath)
        cur = db.cursor()
        
        cur.execute(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                gas_id INTEGER, 
                isotope_id INTEGER,
                dataset_name VARCHAR(255)
            )
            """
        )
        
        
        exomol_defs = dict()
        
        exomol_def_urls = set(exomol_root.get_def_urls())
        for api_data in iter_exomol_api_data(exomol_root, os.path.dirname(fpath)):
            exomol_def_urls = exomol_def_urls.union(set(iter_exomol_def_files_from_api_data(api_data)))
        
        n_urls = len(exomol_def_urls)
        for i, (def_iso_formula, def_url) in enumerate(exomol_def_urls):
            _lgr.info(f'{i=} {n_urls=} {def_iso_formula=} {def_url=}')
            gas_id_iso_id = gas_data.isotope_name_to_id(def_iso_formula)
            
            if gas_id_iso_id is None:
                _lgr.info(f'No corresponding RADTRAN gas for "{def_iso_formula}"')
                continue
            
            rt_gas_desc = RadtranGasDescriptor(*gas_id_iso_id)
            try:
                exomol_def = ExomolIsotopeDef.from_url(cls.database_url+'/'+def_url)
            except HTTPError as e:
                # NOTE: If we cannot get a *.def file that probably means there is no broadening information for that molecule.
                _lgr.warn(f'Could not get *.def file for {def_url}. Error: {str(e)}')
                continue
            exomol_defs[rt_gas_desc] = exomol_defs.get(rt_gas_desc,[]) + [exomol_def]
            
            cur.execute(
                """
                INSERT INTO datasets (gas_id, isotope_id, dataset_name) VALUES (?, ?, ?)
                """,
                (rt_gas_desc.gas_id, rt_gas_desc.iso_id, exomol_def.dataset)
            )
            id = cur.execute("SELECT last_insert_rowid()")
            _lgr.debug(f'{rt_gas_desc=} {id=}')
        
        if _lgr.level == logging.DEBUG:
            row_itr = cur.execute("SELECT id, gas_id, isotope_id, dataset_name FROM datasets")
            for row in row_itr:
                _lgr.debug(f'{row=}')
            
            
            
    
    