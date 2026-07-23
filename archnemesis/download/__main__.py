
import sys
import argparse as ap

from .database import all_ref_dbase_names, get_reference_database_downloader_by_name

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


def create_parser():
    parser = ap.ArgumentParser()
    
    parser.add_argument('ref_dbase_names', nargs='+', choices = all_ref_dbase_names, help='Name of reference database to download', default=[])

    return parser

if __name__ == '__main__':
    
    
    parser = create_parser()
    
    args = vars(parser.parse_args(sys.argv[1:]))
    
    _lgr.info('ARGUMENTS\n' + '\n'.join(f'\t{k} : {v}' for k,v in args.items())+'\nEND ARGUMENTS')
    
    
    ref_dbase_downloaders = []
    
    for x in args['ref_dbase_names']:
        ref_dbase_downloaders.append(get_reference_database_downloader_by_name(x))
        if ref_dbase_downloaders[-1] is None:
            raise RuntimeError(f'Failed to get reference database downloader for {x}')
    
    for x in ref_dbase_downloaders:
        x.action_download_reference_database()