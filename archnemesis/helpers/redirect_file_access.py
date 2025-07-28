"""
This module overwrites the default 'open' implementation to enable us to
redirect a file path to a different target. Used when transferring code
between machines which store reference files in different locations.

Redirects are checked in order, so a later redirect can apply to the
result of an earilier redirect.

The redirects are callables with the following signature

```
	def redirector(path : Path) -> Path | None
```

## ARGUMENTS ##
	path : Path
		The path to the file or directory to be opened

## RETURNS ##
	
	new_path : Path | None
		Path that the argument `path` is redirected to or `None` if the path was not redirected
		
"""

import builtins
from pathlib import Path
from typing import Callable, IO, Self
from contextlib import contextmanager

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)

# Save the original open function so we can access it later
_builtin_open = builtins.open

_redirects : dict[int, Callable[[Path], Path|None],...] = dict()

@property
def redirects() -> tuple[tuple[int, Callable[[Path], Path|None]], ...]:
	"""
	Get a tuple (read-only) of the current handles and redirectors
	"""
	global _redirects
	return tuple(_redirects.items())

class Redirector:

	@classmethod
	def from_string(cls, redirect_spec : str) -> Self:
		"""
		Accepts a `redirect_spec`, a string formatted as "{old_path}->{new_path}",
		constructs a Recirector that redirects any path that is relative to "old_path" to
		a path that is now relative to "new_path"
		"""
		old_path, new_path = map(Path,redirect_spec.split('->', 1))
		return cls(old_path, new_path)
	
	@classmethod
	def __init__(self, old_path : Path | str, new_path : Path | str) -> Self:
		self.old_path = Path(old_path)
		self.new_path = Path(new_path)
	
	
	def __call__(self, path : Path) -> None | Path:
		if self.old_path in path.parents:
			return self.new_path / path.relative_to(self.old_path)
		else:
			return None
	
	
	def __repr__(self):
		return f'Redirect("{self.old_path}" TO "{self.new_path}")'
	

def add_redirect(redirector : Callable[[Path], Path|None]) -> int:
	"""
	Add a callable to the redirects, returns a handle that can be used to remove the redirect
	"""
	global _redirects
	handle = hash(redirector)
	_redirects[handle] = redirector
	return handle

def remove_redirect(handle : int) -> bool:
	"""
	Removes the redirect with `handle`. Returns True if handle was vaid, False if not.
	"""
	global _redirects
	if handle in _redirects:
		del _redirects[handle]
		return True
	return False


def _open_redirector(path : Path | str, *args, **kwargs) -> IO:
	global _redirects
	global _open_fn_stack
	global _builtin_open
	
	target_path = Path(path)
	
	for hdl, redirector in _redirects.items():
		new_path = redirector(target_path)
		if new_path is not None:
			target_path = new_path
	
	return _builtin_open(target_path, *args, **kwargs)


def apply():
	builtins.open = _open_redirector

def release():
	global _builtin_open
	builtins.open = _builtin_open

@contextmanager
def using(*redirectors : Callable[[Path], Path|None]):
	_lgr.debug(f'{redirectors=}')
	global _redirects
	
	hdls = []
	for redirector in redirectors:
		hdls.append(add_redirect(redirector))
	
	was_applied = False
	if len(_redirects) > 0:
		was_applied = True
		apply()
	
	_lgr.debug(f'REDIRECT SET: {hdls=} {was_applied=}')
	
	yield
	
	if was_applied:
		release()
		was_applied=False
	
	for hdl in hdls:
		remove_redirect(hdl)
	
	_lgr.debug(f'REDIRECT UNSET: {hdls=} {was_applied=}')
	
	return
	

