


import re
from typing import NamedTuple, Self

from archnemesis.database.datatypes.atomic_isotope import AtomicIsotope
from archnemesis.database.datatypes.molecule import Molecule


plural_atomic_isotope_regex = re.compile(r'\((?P<nucleon_number>\d+)(?P<name>[A-Z][a-z]*)\)(?P<plurality>\d*)')
isotopologue_regex          = re.compile(r'(?P<isotopes>(' +plural_atomic_isotope_regex.pattern+ r')+)(?P<charge>[+-]\d*)?')

class Isotopologue(NamedTuple):
	isotopes : tuple[AtomicIsotope,...]
	charge : int
	
	@classmethod
	def regex_match(cls, formula) -> tuple[None | Self, str]:
		match = isotopologue_regex.match(formula)
		
		if match is not None:
			item_counts = tuple((AtomicIsotope(pm["name"], int(pm["nucleon_number"])), (int(pm["plurality"]) if len(pm["plurality"]) > 0 else 1)) for pm in plural_atomic_isotope_regex.finditer(match["isotopes"]))
			items = []
			for item, count in item_counts:
				while count > 0:
					items.append(item)
					count -= 1
			
			charge_str = match["charge"]
			if charge_str is None or len(charge_str) == 0:
				charge = 0
			elif len(charge_str) == 1:
				charge = int(charge_str+'1')
			else:
				charge = int(charge_str)
			return cls(tuple(items), charge), formula[match.end():]
		else:
			return None, formula
	
	@classmethod
	def from_str(cls, formula : str, multiple=False) -> Self | tuple[Self,...]:
		# Example "(1H)2(16O)", "(16O)(17O)(16O)"
		
		instance, formula = cls.regex_match(formula)
		
		if instance is None:
			raise ValueError(f'{formula=} does not denote an Isotopologue')
		
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
		s = []
		n = []
		prev_iso = None
		for iso in self.isotopes:
			if prev_iso is not None and prev_iso == iso:
				n[-1] += 1
			else:
				s.append(str(iso))
				n.append(1)
			prev_iso = iso
		return ''.join((a if x == 1 else f'{a}{x}' for a,x in zip(s,n)))+self._get_charge_str()
	
	@property
	def Molecule(self):
		return Molecule(tuple(sorted(isotope.AtomicElement for isotope in self.isotopes)), self.charge)
