import sys, os
import dataclasses as dc
from typing import IO

_default_out_width = 80
_set_out_width = []

def get_out_width(f : IO = sys.stdout) -> int:
	"""
	Get the widest string an output file descriptor can cope with.
	
	`f` is "stdout" by default.
	"""
	global _set_out_width
	global _default_out_width
	if f.isatty():
		tty_cols = os.get_terminal_size().columns
		if len(_set_out_width) != 0 and _set_out_width[-1] < tty_cols:
			return _set_out_width
		else:
			return tty_cols
	else:
		return _default_out_width if len(_set_out_width) == 0  else _set_out_width[-1]


def set_out_width(width=None):
	global _set_out_width
	
	if width == None:
		_set_out_width = []
	else:
		if len(_set_out_width) != 0:
			_set_out_width[-1] = width
		else:
			_set_out_width = [width]

def push_out_width(width):
	global _set_out_width
	_set_out_width.append(width)

def pop_out_width():
	global _set_out_width
	if len(_set_out_width) != 0:
		_set_out_width = _set_out_width[:-1]
	
	