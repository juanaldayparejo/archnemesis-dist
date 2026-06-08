

import re
from typing import NamedTuple, Self

atomic_element_regex        = re.compile(r'(?P<name>[A-Z][a-z]*)')

class AtomicElement(NamedTuple):
	name : str
	
	@classmethod
	def regex_match(cls, formula) -> tuple[None | Self, str]:
		match = atomic_element_regex.match(formula) # Could use `search` instead
		
		if match is not None:
			return cls(match["name"]), formula[match.end():]
		else:
			return None, formula
		
	@classmethod
	def from_str(cls, formula, multiple=False) -> Self | tuple[Self,...]:
		# H, Na, Cl, ...
		
		instance, formula = cls.regex_match(formula)
		
		if instance is None:
			raise ValueError(f'{formula=} does not denote an AtomicElement')
		
		if not multiple:
			return instance
		else:
		
			instances = []
			while instance is not None:
				instances.append(instance)
				instance, formula = cls.regex_match(formula)
			
			return tuple(instances)
	
	def __repr__(self):
		return f'{self.name}'
