
import dataclasses as dc

import numpy as np

from archnemesis.database.datatypes.pf_data.pf_data import PFData



@dc.dataclass(slots=True)
class PolynomialPFData(PFData):
	coeffs : np.ndarray
	domain : np.ndarray = dc.field(default_factory=lambda : np.array([-1,1],dtype=float))
	
	_poly : np.polynomial.Polynomial = dc.field(init=False, repr=False, default=None)
	
	def __post_init__(self):
		self._poly = np.polynomial.Polynomial(self.coeffs, domain=self.domain, window=self.domain, symbol='T')
	
	def as_structured_array(self) -> np.ndarray:
		return np.array(
			[
				self._poly.coef
			], 
			dtype=[('coeffs',float, self.poly.coef.shape)]
		)
	
	def __call__(self, T : float | np.ndarray) -> float | np.ndarray:
		return self._poly(T)
	
	def as_table(self, t_array : np.ndarray = None) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
		if t_array is None:
			t_array = np.linspace(50, 500, 1000)
		return ((t_array, self._poly(t_array)), np.array([np.min(t_array), np.max(t_array)], dtype=float))
	
	def as_poly(self, t_array : None | np.ndarray = None, n : None | int = None) -> tuple[np.ndarray, np.ndarray]:
		return (self._poly.coef, self._poly.domain)