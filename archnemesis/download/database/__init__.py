# flake8: noqa

from pathlib import Path

from archnemesis.Data.path_data import archnemesis_path


from .reference_database_info import (
    HITRAN2024_RefDBaseInfo,
)

all_ref_dbase_info = (
    HITRAN2024_RefDBaseInfo(),
)

all_ref_dbase_names = tuple(x.DBASE_NAME for x in all_ref_dbase_info)


def get_reference_database_downloader_from_info(ref_dbase_info):
    from .base_database_downloader import BaseDatabaseDownloader
    ref_dbase_downloader = BaseDatabaseDownloader(*ref_dbase_info)
    return ref_dbase_downloader

def get_reference_database_downloader_for(test_path : str | Path):
    ans_path = Path(archnemesis_path()).resolve()
    if isinstance(test_path, str):
        test_path = Path(test_path).resolve()
    
    if ans_path in test_path.parents:
        
        for ref_dbase_info in all_ref_dbase_info:
            if test_path == ref_dbase_info.DBASE_PATH:
                return get_reference_database_downloader_from_info(ref_dbase_info)
    return None


def get_reference_database_downloader_by_name(ref_dbase_name):
    for ref_dbase_info in all_ref_dbase_info:
        if ref_dbase_info.DBASE_NAME == ref_dbase_name:
            return get_reference_database_downloader_from_info(ref_dbase_info)
    return None