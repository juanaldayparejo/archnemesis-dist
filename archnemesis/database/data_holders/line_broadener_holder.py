

import dataclasses as dc

import numpy as np




@dc.dataclass
class LineBroadenerHolder:
	name : str
	gamma_amb : None | np.ndarray = None
	n_amb : None | np.ndarray = None
	delta_amb : None | np.ndarray = None
	
	def __post_init__(self):
		if all(x is None for x in (self.gamma_amb, self.n_amb, self.delta_amb)):
			raise ValueError('LineBroadenerHolder cannot be instantiated with all ("gamma_amb", "n_amb", "delta_amb") = `None`')
		
		shape = None
		for x in (self.gamma_amb, self.n_amb, self.delta_amb):
			if x is None:
				continue
			if shape is None:
				shape = x.shape
			assert len(shape) == len(x.shape), \
				'LineBroadenerHolder must have same number of dimensions for "gamma_amb", "n_amb", "delta_amb" if they are not None'
			
			assert all(s1==s2 for s1,s2 in zip(shape,x.shape)), \
				'LineBroadenerHolder must have same shape for "gamma_amb", "n_amb", "delta_amb" if they are not None'
		
		if self.gamma_amb is None:
			self.gamma_amb = np.ones(shape, dtype=float)
		
		if self.n_amb is None:
			self.n_amb = np.zeros(shape, dtype=float)
			
		if self.delta_amb is None:
			self.delta_amb = np.zeros(shape, dtype=float)