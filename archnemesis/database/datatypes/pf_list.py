
import dataclasses as dc
from typing import Self

import numpy as np

from archnemesis.database.datatypes.pf_data.pf_data import PFData


@dc.dataclass(slots=True)
class PFList:
	pf_data_list : list[PFData,...] = dc.field(default_factory=list)
	
	def append(self, pf_data : PFData) -> Self:
		self.pf_data_list.append(pf_data)
		return self
	
	def __call__(self, T : float | np.ndarray) -> float | np.ndarray:
		if len(self.pf_data_list) == 0:
			raise RuntimeError(f'{self} has no partition data.')
	
		if isinstance(T, np.ndarray):
			acc = np.zeros_like(T, float)
			n = np.zeros_like(T, int)
			for pf_data in self.pf_data_list:
				mask = pf_data.in_domain(T)
				acc[mask] += pf_data(T[mask])
				n[mask] += 1
			
			if np.any(~np.nonzero(n)):
				raise RuntimeError(f'Partition functions do not completely cover the range of temperatures provided. Missing data for T={T[~np.nonzero(n)]}')
		
		else:
			acc = 0
			n = 0
			for pf_data in self.pf_data_list:
				#print(f'DEBUG : {pf_data=} {acc=} {n=} {T=}')
				if pf_data.in_domain(T):
					acc += pf_data(T)
					n += 1
			
			if n == 0:
				raise RuntimeError(f'Partition functions do not cover the requested temperature {T=}.')
		
		return acc/n