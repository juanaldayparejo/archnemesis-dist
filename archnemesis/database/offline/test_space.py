

from typing import NamedTuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

import archnemesis.database.utils.fetch as fetch

from archnemesis.database.utils.html_readers import html_to_molecular_formula, html_to_isotopologue_formula, parse_html_float

from archnemesis.database.datatypes.molecule import Molecule
from archnemesis.database.datatypes.isotopologue import Isotopologue

url_base = "https://hitran.org/docs/iso-meta/"
soup = BeautifulSoup(fetch.file(url_base), 'html.parser')


mols_div = soup.find_all(class_='www_content')[0]



mols = dict()
isos = dict()

field_match = {
	'global_id' : 'global id',
	'iso_id' : 'local id',
	'isotopologue' : 'formula',
	'AFGL' : 'afgl code',
	'terrestrial_abundance' : 'abundance',
	'mol_mass' : 'molar mass',
	'q_ref': 'q(296',
	'q_file' : 'q (full',
	'gi' : 'gi'
}

field_match_rev = dict((v,k) for k,v in field_match.items())


field_readers = {
	'global_id' : lambda x: int(x.text),
	'iso_id' : lambda x: int(x.text),
	'isotopologue' : lambda x: Isotopologue.from_str(html_to_isotopologue_formula(x.contents)),
	'AFGL' : lambda x: int(x.text),
	'terrestrial_abundance' : lambda x: parse_html_float(x.contents),
	'mol_mass' : lambda x: parse_html_float(x.contents),
	'q_ref': lambda x: parse_html_float(x.contents),
	'q_file' : lambda x: urljoin(url_base,x.a["href"]),
	'gi' : lambda x: int(x.text)
}

class IsoInfo(NamedTuple):
	global_id : int
	iso_id : int
	isotopologue : Isotopologue
	AFGL : int
	terrestrial_abundance : float
	mol_mass : float
	q_ref : float
	q_file : str
	gi : int


def read_mol(h4):
	print('MOL', h4)
	print(f'{h4.contents=}')
	id_and_mol_formula_start, mol_formula_parts = h4.contents[0], h4.contents[1:]
	mol_id, mol_formula_start = id_and_mol_formula_start.split(':')
	mol_formula_start = mol_formula_start.lstrip()
	
	mol_formula = [mol_formula_start, *mol_formula_parts]
	
	return mol_id, Molecule.from_str(html_to_molecular_formula(mol_formula))
	

def read_iso_table(table):
	print('## ISO ##')
	
	field_idx_to_name = dict()
	i=0
	for tag in table.thead.tr.children:
		if tag.name != "th":
			continue
		print(f'{tag.text=}')
		for k in field_match_rev.keys():
			if tag.text.lower()[:len(k)] == k:
				print(f'{k=} {tag.text.lower()[:len(k)]=}')
				field_idx_to_name[i] = field_match_rev[k]
				i+= 1
				break
			
	
	print(field_idx_to_name)
	
	for tr in table.tbody.children:
		if tr.name != "tr":
			continue
		fields = dict()
		i=0
		for tag in tr.children:
			if tag.name != 'td':
				continue
			print(f'{i=} {field_idx_to_name[i]=} {tag=}')
			fields[field_idx_to_name[i]] = field_readers[field_idx_to_name[i]](tag)
			i+=1
		
		iso_info = IsoInfo(**fields)
		isos.setdefault(iso_info.isotopologue.Molecule,[]).append(iso_info)
	
	

for tag in mols_div.children:
	if tag.name == 'h4':
		mol_id, molecule = read_mol(tag)
		mols[molecule] = mol_id
	elif tag.name == 'table':
		read_iso_table(tag)
	


for k, v in mols.items():
	print(k)
	print(f'\t{v}')

for k,v in isos.items():
	print(k)
	print(f'\t{v}')
	
for mol_id,mol in mols.items():
	for x in isos[mol_id]:
		print(f'{mol_id} {str(mol)} {x.global_id} {x.iso_id} {str(x.isotopologue)} {x.terrestrial_abundance:09.4E} {x.mol_mass:09.4E} {x.q_ref:09.4E}')

for mol in mols.keys():
	print(mol)





