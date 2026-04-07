
import dataclasses as dc

import numpy as np

from archnemesis.database.datatypes.pf_data.pf_data import PFData


@dc.dataclass(slots=True)
class TabulatedPFData(PFData):
	temp : np.ndarray
	q : np.ndarray
	domain : np.ndarray = dc.field(default_factory=lambda : np.array([0,np.inf],dtype=float))
	
	
	def init_domain(self):
		min, max = (np.min(self.temp), np.max(self.temp))
		if self.domain[0] < min:
			self.domain[0] = min
		if self.domain[1] > max:
			self.domain[1] = max
		return
	
	def as_structured_array(self) -> np.ndarray:
		return np.array(
			[
				self.temp, 
				self.q
			], 
			dtype=[('temp',float), ('q',float)]
		)
	
	def __call__(self, T : float | np.ndarray) -> float | np.ndarray:
		return np.interp(T, self.temp, self.q)
	
	def as_table(self, t_array : None | np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		if t_array is None:
			return (self.temp, self.q, self.domain) 
		else:
			return (t_array, np.interp(t_array, self.temp, self.q), np.array([np.min(t_array), np.max(t_array)], dtype=float))
		
	
	def as_poly(self, t_array : None | np.ndarray = None, n : int = 5) -> tuple[np.ndarray, np.ndarray]:
		temp, q, domain = self.as_table(t_array)
		return (np.polynomial.Polynomial.fit(temp, q, deg=n).coef, domain)