"""
Main entry point for archnemesis can run a retrieval via `python3 -m archnemesis <path_to_retrieval>`.


"""
import sys
from pathlib import Path

import argparse as ap

from archnemesis.Retrieval import Retrieval

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)



def create_parser():

	parser = ap.ArgumentParser()
	parser.add_argument('paths', type=str, nargs='+', help='Path to a file (<runname>.inp or <runname>.h5 file) or directory. If a directory, will try and guess runname and HDF5(preferred) or LEGACY input.')
	parser.add_argument('--log_at', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRIT'), help='Level to set all logging to', default=None)
	parser.add_argument('--log_stream', action='append', choices=('package', 'progress'), help='Logging stream to set level of (applied in order after `log_at`)', default=[])
	parser.add_argument('--log_stream_level', action='append', choices=('DEBUG', 'INFO', 'WARN', 'ERROR', 'CRIT'), help='Level to set logging stream to (applied in order after `log_at`)', default=[])
	#parser.add_argument('--redirect_path', type=redirect_file_access.Redirector.from_string, action='append', help='Creates a redirect from as string formatted as "{old_path}->{new_path}" any paths relative to {old_path} are redirected to be relative to {new_path}', default=[])
	#parser.add_argument('--no_show_plots', action='store_true', help='if present, will not show plots (but will still write them to disk)', default=False)
	#parser.add_argument('--plot_path_fmt', type=PathFormat.from_argument, help=PathFormat.get_description(extra_keywords=('retrieval_dir', 'plot_name',)) + ' (default = "{retrieval_dir}/{plot_name}.png")', default=PathFormat('{retrieval_dir}/{plot_name}.png'))


	retrieval_engine_subparsers = parser.add_subparsers(
		title = 'Retrieval Engine',
		description = 'Method to use when retrieving parameters (default=OptimalEstimation)',
		required = False,
		help = None, #'Choose retrieval engine to use (default=OptimalEstimation)',
		dest='retrieval_engine',
	)

	optimal_estimation_parser = retrieval_engine_subparsers.add_parser('OptimalEstimation', help='PLACEHOLDER')
	optimal_estimation_parser.add_argument('-n', '--n_cores', type=int, help='Number of cores to run parallelised processes on (default is to use the number specified in the input files)', default=None)
	optimal_estimation_parser.add_argument('--iter', dest='n_iter', type=int, help='Number of iterations to perform if running optimal estimation (default is to use number specified in the input files)', default=None)
	optimal_estimation_parser.add_argument('--write_itr', action='store_true', help='if running optimal estimation, should we write a *.itr file?', default=False)

	retrieval_engine_subparsers._choices_actions[-1].help = ' '.join(optimal_estimation_parser._get_formatter()._get_actions_usage_parts(optimal_estimation_parser._actions, optimal_estimation_parser._mutually_exclusive_groups))
	#optimal_estimation_parser.help = optimal_estimation_parser.usage

	nested_sampling_parser = retrieval_engine_subparsers.add_parser('NestedSampling', help='PLACEHOLDER')
	nested_sampling_parser.add_argument('-n', '--n_cores', type=int, help='Number of cores to run parallelised processes on (default is to use the number specified in the input files)', default=None)

	retrieval_engine_subparsers._choices_actions[-1].help = ' '.join(nested_sampling_parser._get_formatter()._get_actions_usage_parts(nested_sampling_parser._actions, nested_sampling_parser._mutually_exclusive_groups))

	return parser


def set_packagewide_log_level(log_stream : str, log_level : None | str):
	if log_level is None:
		return
	else:
		log_level = logging.getLevelName(log_level)
	
	if log_stream == 'all':
		log_stream = ('package', 'progress')
	else:
		log_stream = (log_stream,)
	
	if 'package' in log_stream:
		logging.pkg_lgr.setLevel(log_level)
	if 'progress' in log_stream:
		logging.progress_lgr.setLevel(log_level)



if __name__ == '__main__':

	parser = create_parser()
	# Use dictionary interface for args
	args = vars(parser.parse_args(sys.argv[1:]))

	# Set logging level
	both_log_stream_level = args.get('log_at', None)
	if both_log_stream_level is not None:
		set_packagewide_log_level('all', both_log_stream_level)
	
	log_streams = args.get('log_stream',[])
	log_stream_levels = args.get('log_stream_level', [])
	if len(log_streams) != len(log_stream_levels):
		print('Must have same number of arguments to `log_stream` as to `log_stream_level`')
		parser.print_usage()
		exit(1)
	
	for log_stream, log_stream_level in zip(log_streams, log_stream_levels):
		set_packagewide_log_level(log_stream, log_stream_level)

	_lgr.info('ARGUMENTS\n' + '\n'.join(f'\t{k} : {v}' for k,v in args.items())+'\nEND ARGUMENTS')


	run_retrieval_actions = {
		'OptimalEstimation' : lambda retrieval_instance, **kwargs: retrieval_instance.run_optimal_estimation(**kwargs),
		'NestedSampling' : lambda retrieval_instance, **kwargs: retrieval_instance.run_nested_sampling(**kwargs),
	}

	# Unpack non-retrieval engine arguments
	paths = args.pop('paths')
	_ = args.pop('log_at', None)
	_ = args.pop('log_stream', None)
	_ = args.pop('log_stream_level', None)
	retrieval_engine = args.pop('retrieval_engine', 'OptimalEstimation')

	# Create and run retrieval with specified engine.
	n_paths = len(paths)
	if n_paths == 1:
		path = Path(paths[0])
		
		if not path.exists(): # Assume ends in a runname
			retrieval = Retrieval.from_runname(str(path))
		elif path.is_file():
			retrieval = Retrieval.from_file(str(path))
		elif path.is_dir():
			retrieval = Retrieval.from_dir(str(path))
		else:
			raise RuntimeError('Could not create retrieval instance')
		
		run_retrieval_actions[retrieval_engine](retrieval, **args)
		
	else:
		raise NotImplementedError('Multiple archnemesis runs at once are not implemented yet.')
		









