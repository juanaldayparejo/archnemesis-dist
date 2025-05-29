from __future__ import annotations #  for 3.9 compatability

import sys, os
import dataclasses as dc
from typing import IO

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

_default_out_width = 80
_set_out_width = []


class OutWidth:
	_default_out_width = 80
	_set_out_width = []

	@classmethod
	def get(cls, f : IO = sys.stdout) -> int:
		"""
		Get the widest string an output file descriptor can cope with.
		
		`f` is "stdout" by default.
		"""
		if f.isatty():
			tty_cols = os.get_terminal_size().columns
			if len(cls._set_out_width) != 0 and cls._set_out_width[-1] < tty_cols:
				return cls._set_out_width[-1]
			else:
				return tty_cols
		else:
			return cls._default_out_width if len(cls._set_out_width) == 0  else cls._set_out_width[-1]

	@classmethod
	def set(cls, width=None):
		
		if width == None:
			cls._set_out_width = []
		else:
			if len(cls._set_out_width) != 0:
				cls._set_out_width[-1] = width
			else:
				cls._set_out_width = [width]
	
	@classmethod
	def push(cls,width):
		cls._set_out_width.append(width)

	@classmethod
	def pop(cls):
		if len(cls._set_out_width) != 0:
			cls._set_out_width = cls._set_out_width[:-1]
	
	