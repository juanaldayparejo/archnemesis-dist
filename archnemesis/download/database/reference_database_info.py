import os
from typing import NamedTuple
from pathlib import Path
from archnemesis.Data.path_data import archnemesis_path, archnemesis_resolve_path

REFERENCE_DATABASE_ENV_VAR     : str  = "ANS_REF_DBASE_DIR"
REFERENCE_DATABASE_DIR_DEFAULT : Path = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases'))

if len(x := os.environ.get(REFERENCE_DATABASE_ENV_VAR, '')) > 0:
    REFERENCE_DATABASE_DIR        : Path = Path(x)
    REFERENCE_DATABASE_DIR_SOURCE : str  = f'from environment variable `{REFERENCE_DATABASE_ENV_VAR}`'
else:
    REFERENCE_DATABASE_DIR        : Path = REFERENCE_DATABASE_DIR_DEFAULT
    REFERENCE_DATABASE_DIR_SOURCE : str  = 'default location'


class HITRAN2024_RefDBaseInfo(NamedTuple):
    DBASE_NAME                    : str  = "hitran_2024"
    DBASE_URL                     : str  = "https://digital.csic.es/bitstream/10261/437343/3/hitran24.h5"
    DBASE_PATH                    : Path = REFERENCE_DATABASE_DIR / 'hitran_2024.h5'
    DBASE_DOWNLOAD_SENTRY_FILE    : Path = REFERENCE_DATABASE_DIR / 'hitran_2024.h5.download'
    DBASE_NO_DOWNLOAD_SENTRY_FILE : Path = REFERENCE_DATABASE_DIR / 'hitran_2024.h5.no_download'

