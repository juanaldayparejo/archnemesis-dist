"""
Perform compatability setup for python version 3.10
"""

import os
import typing, typing_extensions
import contextlib

typing.Self = typing_extensions.Self

@contextlib.contextmanager
def __chdir(self, path):
	prev_path = os.getcwd()
	os.chdir(path)
	yield
	os.chdir(prev_path)
	return

contextlib.chdir = __chdir
