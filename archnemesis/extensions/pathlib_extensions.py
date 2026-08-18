"""
Contains extensions for `pathlib` module.
"""

import sys, os
from typing import Self, Any, Iterable
from pathlib import Path, PurePath

_section_descriptions = dict(
	drive = 'drive letter or name (usually empty unless windows)',
	root = 'local or global root',
	anchor = 'concatenation of drive letter and root',
	parents = 'tuple of parent path parts',
	parent = 'logical parent of the path',
	name = 'last section of path including the suffix/extension',
	suffix = 'part of the last section of the path after and including the final dot ("."), but not including a starting dot',
	ext = 'part of the last section of the path after and including the final dot ("."), but not including a starting dot',
	suffixes = 'a list of all suffixes of the path (i.e. repeated application of `suffix`)',
	exts = 'a list of all the extensions of the path (i.e. repeated application of `ext`)',
	stem = 'last section of path excluding the suffix/extension',
)
	

def _PurePath_get_sections(
		self : PurePath,
	) -> dict[str,Any]:
	"""
	Enables a way to get named sections of a PurePath as a dictionary
	"""
	return dict(
		drive = self.drive,
		root = self.root,
		anchor = self.anchor,
		parents = self.parents,
		parent = self.parent,
		name = self.name, # last section of path including suffix
		suffix = self.suffix,
		ext = self.suffix,
		suffixes = self.suffixes,
		exts = self.suffixes,
		stem = self.stem, # last section of path excluding suffix
	)

def _PurePath_format(
		self : PurePath, 
		format_path : PurePath | str | Iterable[PurePath | str], 
		*args : tuple[Any], 
		**kwargs : dict[str,Any]
	) -> PurePath:
	"""
	Enables the formatting of PurePath objects in much the same way as strings.
	"""
	
	sections = _PurePath_get_sections(self)
	sections.update(kwargs)
	
	if isinstance(format_path, PurePath):
		parts = format_path.parts
	elif isinstance(format_path, str):
		parts = tuple(format_path.split(os.sep))
	elif isinstance(format_path, Iterable):
		parts = Path(*format_path).parts
	else:
		raise TypeError('Argument `format_path` must be a PurePath or a string, or an iterable of these.')
	
	return Path(*(part.format(*args, **sections) for part in parts))


PurePath.format = _PurePath_format
PurePath.get_sections = _PurePath_get_sections


# Hack because before 3.12 inheriting from `Path` broke things
if sys.version_info[0] <= 3 and sys.version_info[1] <= 11:
	from pathlib import WindowsPath, PosixPath
	ParentPathClass = WindowsPath if os.name == 'nt' else PosixPath
else:
	ParentPathClass = Path


class PathFormat(ParentPathClass):
	"""
	A path-like to be used as a template create paths formatted from other paths.
	
	Usage:
		pfmt = PathFormat('{parent}/new_{stem}.tbl')
		table_path = pfmt.format('/some/path/to/a/file.txt')
		print(str(table_path)) # prints "/some/path/to/a/new_file.tbl"
	"""
	@classmethod
	def get_description(cls, extra_keywords : tuple[str,...] = tuple()) -> str:
		return f"A path-like used to format other paths in the same way as format strings for 'string.format'. Recognises the following keywords by default {(*tuple(_section_descriptions.keys()), *extra_keywords)}."
	
	@classmethod
	def from_argument(cls, arg_str) -> Self | None:
		if arg_str.to_lower() == 'none':
			return None
		return cls(arg_str)
	
	def __new__(cls, *args, **kwargs):
		instance = super().__new__(cls,*args, **kwargs)
		return instance
	
	def __init__(self, *args, **kwargs):
		if sys.version_info[0] <= 3 and sys.version_info[1] <= 11:
			super().__init__()#*args, **kwargs)
		else:
			super().__init__(*args, **kwargs)
	
	def format(
		self, 
		other : PurePath | str,
		*args : tuple[Any], 
		**kwargs : dict[str,Any],
	) -> PurePath:
		return _PurePath_format(
			other if isinstance(other, PurePath) else Path(other), 
			self, 
			*args, 
			**kwargs
		)
