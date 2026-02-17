import dataclasses as dc

import numpy as np




@dc.dataclass(slots=True)
class PFData:
	
	def __post_init__(self):
		self.init_domain()
	
	def init_domain(self):
		"""
		Overload this to initialise `self.domain` if it is more complicated to construct it than using a default value.
		"""
		pass
	
	def in_domain(self, T : float | np.ndarray) -> bool | np.ndarray:
		"""
		Returns a bool or boolean array where True means `T` is within `self.domain`, and False means `T` is not within `self.domain`
		"""
		return ((self.domain[0] <= T) & (T <= self.domain[1]))
	
	def as_structured_array(self) -> np.ndarray:
		"""
		Returns a structured array representation of the instance
		"""
		raise NotImplementedError
	
	def __call__(self, T: float | np.ndarray) -> float | np.ndarray:
		"""
		Returns partition function calculated at temperature `T`
		"""
		raise NotImplementedError
	
	def as_table(self, t_array : None | np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Returns tuple of arrays that can be used as input to TabulatedPFData constructor
		"""
		raise NotImplementedError
	
	def as_poly(self, t_array : None | np.ndarray = None, n : int = 5) -> tuple[np.ndarray, np.ndarray]:
		"""
		Returns tuple of arrays that can be used as input to PolynomialPFData constructor
		"""
		raise NotImplementedError