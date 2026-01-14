import dataclasses as dc
from typing import ClassVar

from ...datatypes.exomol.line_format import ExomolLineFileFormat
from ...utils.tree_printer import TreePrinter
from ...utils import fetch



@dc.dataclass(repr=False)
class ExomolIsoDatasetInformation(TreePrinter, ExomolLineFileFormat):
    inchi_key : str
    iso_slug : str
    formula : str
    dataset : str
    version : str

  

@dc.dataclass(repr=False)
class ExomolMoleculeInformation(TreePrinter, ExomolLineFileFormat):
    _field_for_n_entries : ClassVar = {
        'names' : 'n_names',
        'iso_dataset' : 'n_datasets',
        
    }
    
    n_names : int
    names : list[str]
    formula : str
    n_datasets : int
    iso_dataset : list[ExomolIsoDatasetInformation,...]

   

@dc.dataclass(repr=False)
class ExomolRootFormat(TreePrinter, ExomolLineFileFormat):
    _field_for_n_entries : ClassVar[dict[str,str]] = {
        'molecule_info' : 'n_mol',
    }
    
    id : str
    version : str
    n_mol : int
    n_iso : int
    n_dataset : int
    n_species : int

    # Molecule information
    molecule_info : tuple[ExomolMoleculeInformation,...]

    @classmethod
    def from_url(cls, url):
        line_itr = (x for x in fetch.file(url, encoding='ascii').splitlines())
        return cls.from_line_itr(None, line_itr)[1]
 
    def get_isotopes(self, mol_formula : str):
        for mi in self.molecule_info:
            if mi.formula == mol_formula:
                return mi
        return None

    def get_urls_of_molecule_defs(self, mol_formula : str):
        # mol/iso/dataset/iso__dataset.def
        for mi in self.molecule_info:
            if mi.formula == mol_formula:
                return tuple(f'{mi.formula}/{idi.iso_slug}/{idi.dataset}/{idi.iso_slug}__{idi.dataset}.def' for idi in mi.iso_dataset)
        return None
    
    
    def iter_def_urls(self) -> tuple[str, str]:
        for mi in self.molecule_info:
            for idi in mi.iso_dataset:
                yield (idi.formula, f'{mi.formula}/{idi.iso_slug}/{idi.dataset}/{idi.iso_slug}__{idi.dataset}.def')
    
    def get_api_urls(self):
        return tuple("https://exomol.com/api/?molecule={}".format(mi.formula) for mi in self.molecule_info)
    
    
    def get_def_urls(self):
        return tuple(self.iter_def_urls())
    
    




