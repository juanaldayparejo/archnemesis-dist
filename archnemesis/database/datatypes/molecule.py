



import re
from typing import NamedTuple, Self

from archnemesis.database.datatypes.atomic_element import AtomicElement

plural_atomic_element_regex = re.compile(r'(?P<name>[A-Z][a-z]*)(?P<plurality>\d*)')
molecule_regex              = re.compile(r'(?P<elements>(' +plural_atomic_element_regex.pattern+ r')+)(?P<charge>[+-]\d*)?')

class Molecule(NamedTuple):
	elements : tuple[AtomicElement,...] # always sort when assigning
	charge : int
	
	@classmethod
	def regex_match(cls, formula) -> tuple[None | Self, str]:
		match = molecule_regex.match(formula)
		
		if match is not None:
			ep = tuple((AtomicElement(pm["name"]), (int(pm["plurality"]) if len(pm["plurality"]) > 0 else 1)) for pm in plural_atomic_element_regex.finditer(match["elements"]))
			
			elements = []
			for e,p in ep:
				while p>0:
					elements.append(e)
					p -= 1
			charge_str = match["charge"]
			if charge_str is None or len(charge_str) == 0:
				charge = 0
			elif len(charge_str) == 1:
				charge = int(charge_str+'1')
			else:
				charge = int(charge_str)
			return cls(tuple(sorted(elements)), charge), formula[match.end():]
		else:
			return None, formula
	
	@classmethod
	def from_str(cls, formula : str, multiple=False) -> Self | tuple[Self,...]:
		# Example "H2O", "CH4", "H2SO4", "NaCl+"
		instance, formula = cls.regex_match(formula)
		
		if instance is None:
			raise ValueError(f'{formula=} does not denote a Molecule')
		
		if not multiple:
			return instance
		else:
		
			instances = []
			while instance is not None:
				instances.append(instance)
				instance, formula = cls.regex_match(formula)
			
			return tuple(instances)
	
	def _get_charge_str(self):
		if self.charge == 0:
			return ''
		elif self.charge == 1:
			return '+'
		elif self.charge == -1:
			return '-'
		else:
			return str(self.charge)
	
	def __repr__(self):
		a_list = tuple(sorted(set(self.elements)))
		return ''.join((str(a) if ((n := self.elements.count(a)) == 1) else f'{a}{n}' for a in a_list)) + self._get_charge_str()
