# flake8: noqa

from pathlib import Path

from archnemesis.Data.path_data import archnemesis_path


from .reference_database_info import (
    HITRAN2024_RefDBaseInfo,
)

all_ref_dbase_info = (
    HITRAN2024_RefDBaseInfo()
)

def get_reference_database_downloader_for(test_path : str | Path):
    ans_path = Path(archnemesis_path()).resolve()
    if isinstance(test_path, str):
        test_path = path(test_path).resolve()
    
    if ans_path in test_path.parents:
        
        for ref_dbase_info in all_ref_dbase_info:
            if test_path == ref_dbase_info.DBASE_PATH:
                from .base_database_downloader import BaseDatabaseDownloader
                ref_dbase_downloader = BaseDatabaseDownloader(*ref_dbase_info)
                return ref_dbase_downloader
    return None