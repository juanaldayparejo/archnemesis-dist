

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
    db : None | sqlite3.Connection = None
    cur : None | sqlite3.Cursor = None
    
    
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
    
    
    
    
    
    @staticmethod
    def db_load_gasses(cur):
        
        cur.executescript(
            """
            BEGIN;
            
            -- Remove all existing tables and views
            
            DROP TABLE IF EXISTS molecules;
            DROP TABLE IF EXISTS isotopologues;
            DROP TABLE IF EXISTS _gasses;
            DROP TABLE IF EXISTS gas_components;
            DROP VIEW IF EXISTS gasses;
            
            
            -- List of known molecules
            CREATE TABLE molecules (
                id INTEGER PRIMARY KEY,
                formula VARCHAR(255) UNIQUE NOT NULL
            );

            -- List of all isotopes associated with their molecule
            CREATE TABLE isotopologues (
                id INTEGER PRIMARY KEY,
                molecule_id INTEGER REFERENCES molecules (id) ON UPDATE CASCADE ON DELETE CASCADE,
                iso_id INTEGER NOT NULL,
                formula VARCHAR(255) UNIQUE NOT NULL,
                terrestrial_abundance FLOAT NOT NULL,
                mass FLOAT NOT NULL,
                mass_unit VARCHAR(32) DEFAULT "g/mol",
                CONSTRAINT unique_mol_iso_id_pair UNIQUE (molecule_id, iso_id)
            );

            -- List of all known gasses, they are made up of one or more isotopologues
            CREATE TABLE _gasses (
                id INTEGER PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL
            );

            -- Components of each gas, each gas_id has one or more isotopologues with associated volume mixing ratios
            CREATE TABLE gas_components (
                gas_id INTEGER REFERENCES _gasses (id) ON UPDATE CASCADE ON DELETE CASCADE,
                gas_name VARCHAR(255) REFERENCES _gasses (name) ON UPDATE CASCADE ON DELETE CASCADE,
                global_iso_id INTEGER NOT NULL REFERENCES isotopologues (id) ON UPDATE CASCADE ON DELETE CASCADE,
                molecule_id INTEGER NOT NULL,
                iso_id INTEGER NOT NULL,
                volume_mixing_ratio FLOAT NOT NULL,
                FOREIGN KEY (molecule_id, iso_id) REFERENCES isotopologues (molecule_id, iso_id) ON UPDATE CASCADE ON DELETE CASCADE
            );

            CREATE VIEW gasses (
                id,
                name,
                n_components,
                volume_mixing_ratio_sum,
                mean_molecular_mass
            ) AS
                SELECT
                    id,
                    name,
                    (SELECT count() FROM gas_components WHERE gas_id = id),
                    (SELECT sum(volume_mixing_ratio) FROM gas_components WHERE gas_id = id),
                    ( 
                        WITH 
                        t1 AS (
                            SELECT global_iso_id, volume_mixing_ratio AS vmr
                                FROM gas_components 
                                WHERE gas_id = id
                        ),
                        t2 AS (
                            SELECT mass, id AS global_iso_id FROM isotopologues
                        )
                        SELECT sum(mass*vmr)/sum(vmr) 
                            FROM t2 INNER JOIN t1 
                                ON t1.global_iso_id = t2.global_iso_id
                    )
                FROM _gasses
            ;
            
            COMMIT;
            """
        )
        
        # Fill tables
        for mol_id, gas_props in gas_data.gas_info.items():
            if 'name' not in gas_props:
                    continue
            # Add a molecule entry
            EXOMOL.db_add_molecule(cur, gas_props['name'])
            
            # Add a gas entry for each molecule
            EXOMOL.db_add_gas(cur, gas_props['name'])
            
            for iso_id, iso_props in gas_props['isotope'].items():
                
                iso_name = iso_props.get('name', gas_props['name'] + f'_{iso_id}')
                if iso_name == gas_props['name']:
                    iso_name = gas_props['name'] + f'_{iso_id}'
                
                #v = (int(mol_id), int(iso_id), iso_name, iso_props['abun'], iso_props["mass"])
                EXOMOL.db_add_isotopologue(
                    cur,
                    int(mol_id),
                    int(iso_id),
                    iso_name,
                    iso_props['abun'],
                    iso_props["mass"],
                )
                
                # Add a gas entry for every isotopologue
                EXOMOL.db_add_gas(
                    cur,
                    iso_name,
                )
                
                # Single isotopologue gasses are only composed of themselves
                EXOMOL.db_add_single_gas_component(
                    cur,
                    iso_name,
                    int(mol_id),
                    int(iso_id),
                )
            
            # Add molecular gasses that are made up of multiple isotopologues
            EXOMOL.db_add_molecular_gas_components(
                cur,
                gas_props['name'],
                [(gas_props['name'], 1.0)],
                True
            )
        
        
        # Add non-present gasses that we need
        extra_gasses = {
            "air" : {
                "components" : [
                    {
                        "mol" : "N2",
                        "vmr" : 0.7808,
                    },
                    {
                        "mol" : "O2",
                        "vmr" : 0.2095,
                    },
                    {
                        "mol" : "Ar",
                        "vmr" : 0.00934,
                    },
                    {
                        "mol" : "CO2",
                        "vmr" : 0.000412,
                    },
                ]
            }
        }
        
        for extra_gas_name, extra_gas_info in extra_gasses.items():
            EXOMOL.db_add_gas(cur, extra_gas_name)
            EXOMOL.db_add_molecular_gas_components(
                cur,
                extra_gas_name,
                [(x['mol'], x['vmr']) for x in extra_gas_info['components']],
                True,
            )
        
        return
    
    @staticmethod
    def db_add_isotopologue(
            cur,
            mol_id : int,
            iso_id : int,
            formula : str,
            abun : float,
            mass : float,
            mass_unit : str = "g/mol",
    ):
        cur.execute(
            """
            INSERT INTO isotopologues (molecule_id, iso_id, formula, terrestrial_abundance, mass, mass_unit) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (mol_id, iso_id, formula, abun, mass, mass_unit)
        )
        return
    
    @staticmethod
    def db_add_gas(
            cur,
            name : str,
    ):
        cur.execute(
            """
            INSERT INTO _gasses (name) 
            VALUES (?)
            """,
            (name,)
        )
        return
    
    @staticmethod
    def db_add_molecule(
            cur,
            formula : str,
    ):
        cur.execute(
            """
            INSERT INTO molecules (formula) 
            VALUES (?)
            """,
            (formula,)
        )
        return
    
    @staticmethod
    def db_add_single_gas_component(
            cur,
            gas_name : str,
            mol_id : int,
            iso_id : int,
            vmr : float = 1.0,
    ):
        global_iso_id = cur.execute(
            "SELECT id FROM isotopologues WHERE iso_id = ? AND molecule_id = ?",
            (iso_id, mol_id)
        ).fetchone()[0]
        
        gas_id = cur.execute(
            "SELECT id FROM _gasses WHERE name = ?",
            (gas_name,)
        ).fetchone()[0]
        
        cur.execute(
            """
            INSERT INTO gas_components (gas_id, gas_name, global_iso_id, molecule_id, iso_id, volume_mixing_ratio) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (gas_id, gas_name, global_iso_id, mol_id, iso_id, vmr)
        )
        return
    
    @staticmethod
    def db_add_molecular_gas_components(
            cur,
            gas_name : str,
            components : list[tuple[str, float],...], # list of (molecule_formula, volume_mixing_ratio)
            modify_isotope_vmr_by_abundance : bool = True, # If True, multiply the supplied vmr by the terrestrial abundance of each isotope. If False, divide the supplied vmr equally among all isotopes.
    ):
        gas_id = cur.execute(
            "SELECT id FROM _gasses WHERE name = ?",
            (gas_name,)
        ).fetchone()[0]
        
        
        
        for i, (formula, vmr) in enumerate(components):
            mol_id = cur.execute(
                "SELECT id FROM molecules WHERE formula = ?",
                (formula,)
            ).fetchone()[0]
        
            iso_info = cur.execute(
                "SELECT id, iso_id, terrestrial_abundance FROM isotopologues WHERE molecule_id = ?",
                (mol_id,)
            ).fetchall()
            
            if not modify_isotope_vmr_by_abundance:
                vmr /= len(iso_info)
            
            for global_iso_id, iso_id, abun in iso_info:
                if modify_isotope_vmr_by_abundance:
                    vmr *= abun
                
                cur.execute(
                    """
                    INSERT INTO gas_components (gas_id, gas_name, global_iso_id, molecule_id, iso_id, volume_mixing_ratio) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (gas_id, gas_name, global_iso_id, mol_id, iso_id, vmr)
                )
        return
    
    
    
    
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
        
        cls.db_load_gasses(cur)
        db.commit()
        
        cur.executescript(
            """
            BEGIN;
            
            DROP TABLE IF EXISTS datasets;
            DROP TABLE IF EXISTS broadeners;
            
            -- Holds information about EXOMOL datasets
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY,
                source VARCHAR(255) NOT NULL,
                global_iso_id INTEGER NOT NULL REFERENCES isotopologues (id) ON UPDATE CASCADE ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                version INTEGER NOT NULL,
                
                CONSTRAINT unique_source_gas_isotope_dataset UNIQUE (source, global_iso_id, name, version)
            );
            
            -- Holds information about which gasses are used as broadeners for each dataset
            CREATE TABLE broadeners (
                dataset_id INTEGER REFERENCES datasets (id) ON UPDATE CASCADE ON DELETE CASCADE,
                gas_id INTEGER REFERENCES _gasses (id) ON UPDATE CASCADE ON DELETE CASCADE
            );
            
            COMMIT;
            """
        )
        
        
        exomol_defs = dict()
        
        # Get all the URLs for the *.def files we need to process, including those not in exomol.all
        exomol_def_urls = set(exomol_root.get_def_urls())
        
        for api_data in iter_exomol_api_data(exomol_root, os.path.dirname(fpath)):
            exomol_def_urls = exomol_def_urls.union(set(iter_exomol_def_files_from_api_data(api_data)))
        
        n_urls = len(exomol_def_urls)
        
        # Loop over the URLs, download and parse each *.def file, and store the relevant information in the database
        for i, (def_iso_formula, def_url) in enumerate(exomol_def_urls):
        
            """
            if def_iso_formula != "(75As)(1H)3":
                #_lgr.error('TESTING')
                continue
            else:
                _lgr.error('TESTING')
            """
            
            
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
            
            global_iso_id = cur.execute(
                "SELECT id FROM isotopologues WHERE molecule_id = ? AND iso_id = ?",
                (rt_gas_desc.gas_id, rt_gas_desc.iso_id)
            ).fetchone()[0]
            
            cur.execute(
                """
                INSERT INTO datasets (source, global_iso_id, name, version) VALUES (?, ?, ?, ?)
                """,
                ("EXOMOL", global_iso_id, exomol_def.dataset, int(exomol_def.version))
            )
            id = cur.execute("SELECT last_insert_rowid()")
            _lgr.debug(f'{rt_gas_desc=} {id=}')
        
        db.commit() # Remember to commit any transactions, otherwise they will not be written to disk.
        
        if _lgr.level == logging.DEBUG:
            row_itr = cur.execute("SELECT source, global_iso_id, name FROM datasets")
            for row in row_itr:
                _lgr.debug(f'{row=}')
































