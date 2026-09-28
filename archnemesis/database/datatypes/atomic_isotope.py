


import re
from typing import NamedTuple, Self

from archnemesis.database.datatypes.atomic_element import AtomicElement


atomic_isotope_regex        = re.compile(r'\((?P<nucleon_number>\d+)(?P<name>[A-Z][a-z]*)\)')

class AtomicIsotope(NamedTuple):
	name : str
	nucleon_number : int 

	@classmethod
	def regex_match(cls, formula : str) -> tuple[None | Self, str]:
		match = atomic_isotope_regex.match(formula) # Could use `search` instead
		
		if match is not None:
			return cls(match["name"],int(match["nucleon_number"])), formula[match.end():]
		else:
			return None, formula
	
	@classmethod
	def from_str(cls, formula, multiple=False) -> Self | tuple[Self,...]:
		# (1H), (16Na), (32Cl), ...
		
		instance, formula = cls.regex_match(formula)
		
		if instance is None:
			raise ValueError(f'{formula=} does not denote an AtomicIsotope')
		
		if not multiple:
			return instance
		else:
		
			instances = []
			while instance is not None:
				instances.append(instance)
				instance, formula = cls.regex_match(formula)
			
			return tuple(instances)
	
	def __repr__(self):
		return f'({self.nucleon_number}{self.name})'
	
	@property
	def AtomicElement(self):
		return AtomicElement(self.name)