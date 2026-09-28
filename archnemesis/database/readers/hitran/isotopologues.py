
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urljoin
import textwrap

from bs4 import BeautifulSoup

import archnemesis.database.utils.fetch as fetch

from archnemesis.database.utils.html_readers import html_to_molecular_formula, html_to_isotopologue_formula, parse_html_float

from archnemesis.database.datatypes.molecule import Molecule
from archnemesis.database.datatypes.isotopologue import Isotopologue

url_base = "https://hitran.org/docs/iso-meta/"


# https://hitran.org/lbl/api?iso_ids_list=13&numin=6300&numax=6400&head=False&fixwidth=0&sep=[comma]&request_params=par_line,n_self,delta_self
hitran_api_fmt = "https://hitran.org/lbl/api?iso_ids_list={global_id}&head=False&fixwidth=0&sep=[comma]&request_params={par_list}"

hitran_pars = (
	"molec_id",
	"local_iso_id",
	"nu",
	"sw",
	"a",
	"elower",
	"gamma_self",
	"n_self",
	"delta_self"
)

hitran_broadeners = (
	'air',
	'H2',
	'CO2',
	'H2O',
)

hitran_broadener_pars = (
	"gamma_",
	"n_",
	"delta_"
)

def get_hitran_broadener_par_list(broadener):
	return tuple(['nu'] + [x+broadener for x in hitran_broadener_pars])





DOWNLOAD_LOCATION = Path(__file__).parent / "../../offline/HITRAN_TEST"

mol_ids = dict()
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

	def __repr__(self):
		return ', '.join((
			f'{self.global_id: 8d}', 
			f'{self.iso_id: 8d}', 
			f'{self.mol_mass: 12.6E}', 
			f'{self.terrestrial_abundance: 12.6E}', 
			f'{self.q_ref: 12.6E}', 
			f'"{self.isotopologue.Molecule!r}"', 
			f'"{self.isotopologue!r}"', 
		))

def read_mol(h4):
	print('MOL', h4)
	print(f'{h4.contents=}')
	id_and_mol_formula_start, mol_formula_parts = h4.contents[0], h4.contents[1:]
	mol_id, mol_formula_start = id_and_mol_formula_start.split(':')
	mol_formula_start = mol_formula_start.lstrip()
	
	mol_formula = [mol_formula_start, *mol_formula_parts]
	
	return int(mol_id), Molecule.from_str(html_to_molecular_formula(mol_formula))
	

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



def download_hitran_isotope_data(dir = DOWNLOAD_LOCATION):
	dir.mkdir(parents=True, exist_ok=True)

	soup = BeautifulSoup(fetch.file(url_base), 'html.parser')
	mols_div = soup.find_all(class_='www_content')[0]
	
	for tag in mols_div.children:
		if tag.name == 'h4':
			mol_id, molecule = read_mol(tag)
			mols[molecule] = mol_id
			mol_ids[mol_id] = molecule
		elif tag.name == 'table':
			read_iso_table(tag)
	
	assert len(mols) == len(mol_ids), "Should have a 1:1 relation between molecules and molecule ids"
	
	with open(dir / 'isotope_data.py', 'w') as f:
		f.write(textwrap.dedent(
		'''
		from typing import Any, Annotated, NamedTuple, Self, Iterable, Callable
		
		class HitranIsotope(NamedTuple):
			mol_id : Annotated[int, "HITRAN molecular id"]
			global_id : Annotated[int, "HITRAN global isotopologue id"]
			iso_id : Annotated[int, "HITRAN local isotopologue id"]
			mol_mass : Annotated[float, ("g mol^{-1}", "Molar mass of isotope")]
			abundance : Annotated[float, ("NUMBER", "Terrestrial abundance of isotope")]
			q_ref : Annotated[float, ("NUMBER", "Partition function at 296 Kelvin")]
			mol_formula : Annotated[str, "Molecular formula"]
			iso_formula : Annotated[str, "Isotopologue formula"]
			
			@staticmethod
			def iter() -> Iterable[Self]:
				"""
				Return an iterator through all hitran isotope instances
				"""
				return (HitranIsotope(*x) for x in hitran_isotope_instance_data)
			
			@staticmethod
			def load() -> tuple[Self,...]:
				"""
				Load all instance data into a tuple of HitranIsotope instances
				"""
				return tuple(HitranIsotope.iter())
			
			@staticmethod
			def dict(key : str | Callable[[Self],Any] = lambda x: (x.mol_id, x.iso_id)) -> dict[Any, Self]:
				"""
				Load all instance data into a dictionary indexed by `key`
				"""
				if isinstance(key, str):
					return dict((getattr(x, key), x) for x in HitranIsotope.iter())
				elif callable(key):
					return dict((key(x), x) for x in HitranIsotope.iter())
				else:
					raise TypeError(f'Cannot use {key=} to get key for dictionary')
		
		
		'''.replace('\t','    ')
		))
		f.write('hitran_isotope_instance_data = (\n')
		for mol, iso_data_list in isos.items():
			for iso_data in iso_data_list:
				f.write(f'    ({mols[mol]: 8d}, {iso_data!r}),\n')
		
		f.write(')\n')
	
def main(skip_downloads = False):

	dir = DOWNLOAD_LOCATION
	dir.mkdir(parents=True, exist_ok=True)
	
	download_hitran_isotope_data(dir)
	
	from archnemesis.database.offline.HITRAN_TEST.isotope_data import HitranIsotope
	
	hitran_isotopes : tuple[HitranIsotope] = HitranIsotope.load()
	print(f'{hitran_isotopes=}')
	
	
	pf_dir = (dir / "partition_function_data")
	pf_dir.mkdir(parents=True, exist_ok=True)
	
	line_dir = (dir / 'line_data')
	broadener_dir = (dir / 'broadeners')
	
	line_dir.mkdir(parents=True, exist_ok=True)
	
	if not skip_downloads:

		for mol, iso_data_list in isos.items():
			for iso_data in iso_data_list:
				pf_file = pf_dir / f'{iso_data.isotopologue}.txt'
				
				if not pf_file.exists():
					fetch.file(
						iso_data.q_file, 
						to_fpath = pf_file,
						use_working_file=True
					)
				
				line_file = line_dir / f'{iso_data.isotopologue}.txt'
				if not line_file.exists():
					fetch.file(
						hitran_api_fmt.format(global_id = iso_data.global_id, par_list=','.join(hitran_pars)),
						to_fpath = line_file,
						prefix = ','.join(hitran_pars),
						use_working_file=True
					)
				
				for broadener in hitran_broadeners:
					this_broad_dir = broadener_dir / broadener
					this_broad_dir.mkdir(parents=True, exist_ok=True)
					broad_file = this_broad_dir / f'{iso_data.isotopologue}.txt'
					
					broad_pars = get_hitran_broadener_par_list(broadener)
					
					if not broad_file.exists():
						fetch.file(
							hitran_api_fmt.format(global_id = iso_data.global_id, par_list=','.join(broad_pars)),
							to_fpath = broad_file,
							prefix = ','.join(broad_pars),
							error_code_action = {500 : 'warning'},
							use_working_file=True
						)
							
	
		
	
if __name__=='__main__':
	main()




