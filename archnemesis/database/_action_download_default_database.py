import os
from pathlib import Path
import tempfile
import shutil
from typing import Callable

from archnemesis.Data.path_data import archnemesis_path, archnemesis_resolve_path
from .utils import fetch

import archnemesis.cfg.logs as logger
_lgr = logger.getLogger(__name__)
_lgr.setLevel(logger.DEBUG)

DEFAULT_DBASE_URL  = "https://digital.csic.es/bitstream/10261/437343/3/hitran24.h5"
DEFAULT_DBASE_PATH = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases/hitran_2024.h5'))


def safe_download(url : str, path : Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        download_path = Path(temp_dir) / path.name
        
        _lgr.info(f'ARCHNEMESIS :: Setup :: Downloading to temporary file {download_path!s}')
        
        fetch.file(
            url,
            to_fpath = download_path,
            chunk_size = 1024*1024*10,
        )
        
        path.unlink()
        shutil.move(download_path, path)
        _lgr.info(f'ARCHNEMESIS :: Setup :: Moved temporary download {download_path!s} to {path!s}')

def download_default_database():
    # Check if the database path exists. 
    # If it does not exist, then do not download the file as the user has deleted it on purpose.
    # If it does exist, and its contents are "PLACEHOLDER" then DO download the file
    # If it does exist, and its contents are not "PLACEHOLDER" then do nothing as the file is already downloaded properly.
    
    if os.environ.get('PYTEST_VERSION') is not None: # ENSURE THAT PYTEST DOES NOT DOWNLOAD STUFF.
        _lgr.info( 'ARCHNEMESIS :: Setup :: Running within a `pytest` run. Will not perform any downloads.')
        return
    
    content_options : dict[str,tuple[bytes,str,Callable]] = {
        "PLACEHOLDER" : (
            "PLACEHOLDER".encode('utf-8'), # byte content of file to check agains
            "Files with content 'PLACEHOLDER' will be downloaded from the web.", # Description of task
            lambda path: safe_download(DEFAULT_DBASE_URL, path), # callable that performs the task
        )
,    }
    
    max_content_option_byte_length = max((len(v[0])+1) for k,v in content_options.items()) # plus one for possible new-line or not.
    
    _lgr.info( 'ARCHNEMESIS :: Setup :: Checking default database...')

    if not DEFAULT_DBASE_PATH.exists():
        _lgr.info(f'ARCHNEMESIS :: Setup :: Default database file is not present at {DEFAULT_DBASE_PATH}. Therefore will not download.')
        _lgr.info(f'ARCHNEMESIS :: Setup :: To enable download, create a dummy file at {DEFAULT_DBASE_PATH} with the contents "PLACEHOLDER" (without the quotes).')
    elif DEFAULT_DBASE_PATH.is_symlink():
        _lgr.info(f'ARCHNEMESIS :: Setup :: Default database file at {DEFAULT_DBASE_PATH} is a symlink. Therefore will not download.')
        _lgr.info(f'ARCHNEMESIS :: Setup :: To enable download, create a dummy file at {DEFAULT_DBASE_PATH} with the contents "PLACEHOLDER" (without the quotes).')
    else:
        _lgr.info(f'ARCHNEMESIS :: Setup :: Default database file found at {DEFAULT_DBASE_PATH!s}')
        _lgr.info( 'ARCHNEMESIS :: Setup :: Checking size of file...')

        if DEFAULT_DBASE_PATH.stat().st_size > max_content_option_byte_length:
            _lgr.info('ARCHNEMESIS :: Setup :: Size of file is too large to be a placeholder. Therefore, assume file has already been downloaded correctly.')
        else:
            _lgr.info('ARCHNEMESIS :: Setup :: Size of file is small enough that it could be a placeholder.')
            _lgr.info('ARCHNEMESIS :: Setup :: Checking contents of file...')
            with open(DEFAULT_DBASE_PATH, 'rb') as f:
                content_bytes = f.read(max_content_option_byte_length)
            
            # Go through possible options one by one.
            any_task_match : bool = False
            for content_str,(to_match_content_bytes, task_description, task_callable) in content_options.items():
                if content_bytes.startswith(to_match_content_bytes):
                    any_task_match = True
                    _lgr.info(f'ARCHNEMESIS :: Setup :: Content of file is "{content_str}".')
                    _lgr.info(f'ARCHNEMESIS :: Setup :: {task_description}')
                    
                    task_callable(DEFAULT_DBASE_PATH)
            if not any_task_match:
                _lgr.info(f'ARCHNEMESIS :: Setup :: Content of file at {DEFAULT_DBASE_PATH!s} does not match any of the possible options')
                _lgr.info(f'  File is {DEFAULT_DBASE_PATH.stat().st_size} bytes')
                _lgr.info(f'  First {max_content_option_byte_length} bytes are: {content_bytes}')
                _lgr.info( '  Options are:')
                for content_str,(to_match_content_bytes, task_description, task_callable) in content_options.items():
                    _lgr.info( '    ------------------------------------------------------------------------------------')
                    _lgr.info(f'    Name: {content_str}')
                    _lgr.info(f'    File Contents: {to_match_content_bytes}')
                    _lgr.info(f'    Task Description: {task_description}')
                _lgr.info( '    ------------------------------------------------------------------------------------')


# Peform action
download_default_database()