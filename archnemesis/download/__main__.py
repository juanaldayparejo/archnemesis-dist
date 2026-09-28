
import sys
import argparse as ap

from .database import (
    REFERENCE_DATABASE_ENV_VAR,
    REFERENCE_DATABASE_DIR,
    REFERENCE_DATABASE_DIR_SOURCE,
    all_ref_dbase_names, 
    get_reference_database_downloader_by_name
)

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


def create_parser():
    
    parser = ap.ArgumentParser(
        prog = 'python -m archnemesis.download',
        description = f'Download reference databases of spectral data to "{REFERENCE_DATABASE_DIR!s}" ({REFERENCE_DATABASE_DIR_SOURCE})',
        epilog = f'Choose download directory by setting the environment variable `{REFERENCE_DATABASE_ENV_VAR}`. If not set (or set to an empty string) will use the default location.'
    )
    
    parser.add_argument('ref_dbase_names', nargs='+', choices = all_ref_dbase_names, help='Name of reference databases to download', default=[])
    parser.add_argument('--force', '-f', action='store_true', help='If present, force download the databases even if they are already present and/or "no_download" sentry files are present.', default=False)

    return parser

if __name__ == '__main__':
    
    
    parser = create_parser()
    
    args = vars(parser.parse_args(sys.argv[1:]))
    
    _lgr.info('ARGUMENTS\n' + '\n'.join(f'\t{k} : {v}' for k,v in args.items())+'\nEND ARGUMENTS')
    
    
    ref_dbase_downloaders = dict()
    
    for x in args['ref_dbase_names']:
        if x not in ref_dbase_downloaders:
            ref_dbase_downloaders[x] = get_reference_database_downloader_by_name(x)
            if ref_dbase_downloaders[x] is None:
                raise RuntimeError(f'Failed to get reference database downloader for {x}')
        else:
            _lgr.info(f'Have already found a downloader for reference database "{x}"')
    
    for x in ref_dbase_downloaders.values():
        x.action_non_interactive_check_and_download_reference_database(
            refresh = args['force'],
            msg_indeterminate = 'Download requested via command-line, assuming download should go ahead in indeterminate case.',
        )
