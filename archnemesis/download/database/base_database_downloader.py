
import os
from pathlib import Path


from ...database.utils import fetch

from ...ui.terminal import ui_show, ui_ask_yn


class BaseDatabaseDownloader:
    def __init__(self,
        DBASE_NAME                    : str,
        DBASE_URL                     : str,
        DBASE_PATH                    : Path,
        DBASE_DOWNLOAD_SENTRY_FILE    : Path,
        DBASE_NO_DOWNLOAD_SENTRY_FILE : Path,
    ):
        self.DBASE_NAME = DBASE_NAME
        self.DBASE_URL = DBASE_URL
        self.DBASE_PATH = DBASE_PATH
        self.DBASE_DOWNLOAD_SENTRY_FILE = DBASE_DOWNLOAD_SENTRY_FILE
        self.DBASE_NO_DOWNLOAD_SENTRY_FILE = DBASE_NO_DOWNLOAD_SENTRY_FILE
        
        
    def is_database_present(self) -> bool:
        return self.DBASE_PATH.exists()

    def get_sentry_file_state(self) -> bool | None:
        """
        Returns `None` if none or both "<database_name>.(no_)download" files are present, otherwise True (or False) if download should (or not) be performed.
        """
        ui_show('Checking sentry files...')
        
        
        a = self.DBASE_DOWNLOAD_SENTRY_FILE.exists()
        ui_show(f'  [{"+" if a else "-"}] Download sentry file {"exists" if a else "does not exist"} at {self.DBASE_DOWNLOAD_SENTRY_FILE}')
        b = self.DBASE_NO_DOWNLOAD_SENTRY_FILE.exists()
        ui_show(f'  [{"+" if b else "-"}] No-Download sentry file {"exists" if b else "does not exist"} at {self.DBASE_DOWNLOAD_SENTRY_FILE}')
        
        if (a and b):
            ui_show('  Both "download" and "no_download" sentry files exist, asking user...')
            return None
        if not (a or b):
            ui_show('  Neither "download" or "no_download" sentry files exist, asking user...')
            return None
        if a:
            ui_show('  Only "download" sentry file exists, **will** perform download.')
            return True
        if b:
            ui_show('  Only "no_download" sentry file exists, **will not** perform download.')
            return False
        raise RuntimeError('Could not get sentry file state, this should never happen.')

    
    @staticmethod
    def is_in_pytest_environment() -> bool:
        if os.environ.get('PYTEST_VERSION') is not None: # ENSURE THAT PYTEST DOES NOT DOWNLOAD STUFF.
            return True
        return False

    

    def should_do_download(self, refresh : bool = False) -> bool:
        if self.is_in_pytest_environment():
            ui_show('In `pytest` environment, no downloads will be performed.')
            return False
        
        if self.is_database_present():
            ui_show(f'Database is present at "{self.DBASE_PATH}".')
            if not refresh:
                return False
            else:
                ui_show('Refresh has been requested...')
        
        if (sentry_file_state := self.get_sentry_file_state()) is not None:
            return sentry_file_state
        
        if ui_ask_yn(f'Download spectral database {self.DBASE_NAME}?', default=True):
            ui_show('User requested download...')
            return True
        else:
            ui_show('User declined download.')
            return False
        
        raise RuntimeError('Cannot work out whether download should continue. This should never happen.')

    def action_check_and_download_reference_database(self, refresh : bool = False) -> None:
        if self.should_do_download():
            self.action_download_reference_database()

    def action_download_reference_database(self) -> None:
        fetch.safe_download(self.DBASE_URL, self.DBASE_PATH, ui_show = ui_show)