

class PathRedirectList:
	
	def __init__(self, values : list[str|tuple[str,...]] = [], redirects : tuple[tuple[str,str],...] = tuple()):
		self._path_redirects = dict(redirects)
		self._raw_paths = [x for x in values]
	
	def _get_redirected_path(self, path : str | tuple[str,...]) -> str | tuple[str,...]:
		if isinstance(path, tuple):
			return tuple(self._get_redirected_path(p) for p in path)
		else:
			for k in self._path_redirects.keys():
				if path.startswith(k):
					return self._path_redirects[k]+path[len(k):]
			return path
	
	def append(self, value : str | tuple[str,...]):
		self._raw_paths.append(value)
	
	def __getitem__(self, i):
		return self._get_redirected_path(self._raw_paths[i])
	
	# Special methods cannot be intercepted with `__getattr__` so must define them
	def __len__(self):
		return len(self._raw_paths)
	
	def __getattr__(self, name):
		return getattr(super().__getattribute__('_raw_paths'), name) # have to do it this way to allow copy.deepcopy to work
	
	def __repr__(self):
		return f'PathRedirectList({repr(self._raw_paths)}, redirects = {self._path_redirects})'
	
	def __str__(self):
		return str([self._get_redirected_path(x) for x in self._raw_paths])