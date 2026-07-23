from typing import NamedTuple
from pathlib import Path
from archnemesis.Data.path_data import archnemesis_path, archnemesis_resolve_path

class HITRAN2024_RefDBaseInfo(NamedTuple):
    DBASE_NAME                    : str  = "hitran_2024"
    DBASE_URL                     : str  = "https://digital.csic.es/bitstream/10261/437343/3/hitran24.h5"
    DBASE_PATH                    : Path = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases/hitran_2024.h5'))
    DBASE_DOWNLOAD_SENTRY_FILE    : Path = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases/hitran_2024.h5.download'))
    DBASE_NO_DOWNLOAD_SENTRY_FILE : Path = Path(archnemesis_resolve_path(archnemesis_path()+'/archnemesis/Data/reference_databases/hitran_2024.h5.no_download'))

