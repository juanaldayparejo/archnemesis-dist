import os
from pathlib import Path
import tempfile
import shutil

from archnemesis.Data.path_data import archnemesis_path, archnemesis_resolve_path
from .utils import fetch

import archnemesis.cfg.logs as logger
_lgr = logger.getLogger(__name__)
_lgr.setLevel(logger.DEBUG)

# NOTE: Unsure which to use here. This will probably run before any modifications to logging occur.
# therefore may not want to use the logging system.
display = lambda *args, **kwargs: print(*args, **kwargs)
#display = lambda *args, **kwargs: _lgr.info(*args, **kwargs)


DEFAULT_DBASE_URL         = "https://digital.csic.es/bitstream/10261/437343/3/hitran24.h5"
DEFAULT_DBASE_PATH        = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases/hitran_2024.h5'))
DEFAULT_DBASE_SENTRY_PATH = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases/hitran_2024.h5.sentry'))


def safe_download(url : str, path : Path):
    with tempfile.TemporaryDirectory() as temp_dir:
        download_path = Path(temp_dir) / path.name
        
        display(f'ARCHNEMESIS :: Setup :: Downloading to temporary file {download_path!s}')
        
        fetch.file(
            url,
            to_fpath = download_path,
            chunk_size = 1024*1024*10,
        )
        
        if path.exists():
            path.unlink()
        shutil.move(download_path, path)
        display(f'ARCHNEMESIS :: Setup :: Moved temporary download {download_path!s} to {path!s}')

def download_default_database():
    # Check if the database path exists. 
    # If it does not exist, then do not download the file as the user has deleted it on purpose.
    # If it does exist, and its contents are "PLACEHOLDER" then DO download the file
    # If it does exist, and its contents are not "PLACEHOLDER" then do nothing as the file is already downloaded properly.
    
    if os.environ.get('PYTEST_VERSION') is not None: # ENSURE THAT PYTEST DOES NOT DOWNLOAD STUFF.
        display( 'ARCHNEMESIS :: Setup :: Running within a `pytest` run. Will not perform any downloads.')
        return
    
    
    display( 'ARCHNEMESIS :: Setup :: Checking default database...')

    if not DEFAULT_DBASE_SENTRY_PATH.exists():
        display(f'ARCHNEMESIS :: Setup :: Default database sentry file is not present at {DEFAULT_DBASE_SENTRY_PATH!s}. Therefore will not download.')
        display(f'ARCHNEMESIS :: Setup :: To enable download, create a dummy file at {DEFAULT_DBASE_SENTRY_PATH!s}.')
    elif DEFAULT_DBASE_PATH.exists():
        display(f'ARCHNEMESIS :: Setup :: Default database file at {DEFAULT_DBASE_PATH!s} exists. Therefore will not download again.')
        display(f'ARCHNEMESIS :: Setup :: To enable download, delete the database file at {DEFAULT_DBASE_PATH!s} and create a dummy file at {DEFAULT_DBASE_SENTRY_PATH!s}.')
    else:
        display(f'ARCHNEMESIS :: Setup :: Default database at {DEFAULT_DBASE_PATH!s} has not been downloaded yet and sentry file is present at {DEFAULT_DBASE_SENTRY_PATH!s}. Therefore, downloading file....')
        

        safe_download(DEFAULT_DBASE_URL, DEFAULT_DBASE_PATH)
        
        # If file downloaded successfully, delete the sentry
        if DEFAULT_DBASE_PATH.exists():
            display( 'ARCHNEMESIS :: Setup :: Download completed successfully, deleting sentry file...')
            DEFAULT_DBASE_SENTRY_PATH.unlink()
            display( 'ARCHNEMESIS :: Setup :: Sentry file deleted.')

# Peform action
download_default_database()