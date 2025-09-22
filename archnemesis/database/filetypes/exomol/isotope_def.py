import dataclasses as dc
from typing import ClassVar, Any, Iterator

from ...datatypes.exomol.line_format import ExomolLineFileFormat
from ...datatypes.exomol.col_format import ExomolColFileFormat
from ...utils.tree_printer import TreePrinter
from ...utils import fetch



def get_number_of_atom_entries(arg : dict[str,Any])->int:
    return arg['iso_formula'].count('(')


@dc.dataclass(repr=False)
class ExomolQuantumNumberSet(TreePrinter, ExomolLineFileFormat):
    _field_for_n_entries : ClassVar[dict[str,str]] = {
        'quantum_number': 'n_quantum_numbers',
    }
    
    code : str
    n_lines : int
    n_quantum_numbers : int
    quantum_number : tuple[str,...]

@dc.dataclass(repr=False)
class ExomolBroadenerFileInfo(TreePrinter, ExomolLineFileFormat):
    _field_for_n_entries : ClassVar[dict[str,str]] = {
        'qn_set' : 'n_qn_sets',
    }
    
    label : str
    filename : str
    max_J_qn : str # maximum J for which pressure broadening parameters provided
    lorentz_hwhm_broad_default : float
    lorentz_temp_exp_broad_default : float
    n_qn_sets : int # number of quantum number sets
    qn_set : tuple[ExomolQuantumNumberSet,...]


@dc.dataclass(repr=False)
class ExomolAuxiliaryInfo(TreePrinter, ExomolLineFileFormat):
    title : str
    format : str
    description : str

@dc.dataclass(repr=False)
class ExomolAuxiliaryList(TreePrinter, ExomolLineFileFormat):
    # NOTE: This only exists in some files and there does not seem to be
    # a values in any *.def file that tells us how many ExomolAuxiliaryEntry
    # values are present. Therefore need to override the `from_line_itr`
    # class method in this case.
    n_entries : str
    entry : tuple[ExomolAuxiliaryInfo,...]

    @classmethod
    def from_line_itr(cls, next_line : None | str, line_itr : Iterator[str]):
        if next_line is None:
            next_line = next(line_itr)
        
        n_entries = 0
        entry = []
        value, comment = next_line.split('#', 1)
        while 'Auxiliary title' in comment:
            next_line, instance = ExomolAuxiliaryInfo.from_line_itr(next_line, line_itr)
            entry.append(instance)
            n_entries += 1
            if next_line is None:
                next_line = next(line_itr)
            value, comment = next_line.split('#', 1)

        return next_line, cls(n_entries, tuple(entry))

@dc.dataclass(repr=False)
class ExomolQuanta(TreePrinter, ExomolLineFileFormat):
    label : str
    format : str
    description : str

@dc.dataclass(repr=False)
class ExomolQuantaCase(TreePrinter, ExomolLineFileFormat):
    _field_for_n_entries : ClassVar[dict[str,str]] = {
        'quanta' : 'n_quanta',
    }
    
    label : str
    n_quanta : int
    quanta : tuple[ExomolQuanta,...]

@dc.dataclass(repr=False)
class ExomolIrreducibleRep(TreePrinter, ExomolLineFileFormat):
    id : int
    label : str
    nuclear_spin_degeneracy : int

@dc.dataclass(repr=False)
class ExomolMass(TreePrinter, ExomolColFileFormat):
    mass_da : float
    mass_kg : float

@dc.dataclass(repr=False)
class ExomolAtom(TreePrinter, ExomolLineFileFormat):
    iso_number : int
    element_symbol : str

@dc.dataclass(repr=False)
class ExomolCoolingFnInfo(TreePrinter, ExomolLineFileFormat):
    max_temp : float
    temp_step : float

@dc.dataclass(repr=False)
class ExomolIsotopeDef(TreePrinter, ExomolLineFileFormat):
    """
    Definition for each isotope. NOTE there is an *actual* API available for EXOMOL (e.g. "https://exomol.com/api/?molecule=CH4")
    but I do not know where it is documented.
    """
    
    _field_for_n_entries : ClassVar[dict[str,str]] = {
        'atom' : get_number_of_atom_entries,
        'irreducible_rep': 'n_irreducible_reps',
        'quanta_case': 'n_quanta_cases',
        'cooling_fn_info': 'cooling_fn_avail',
        'broadener_file_info': 'n_broadeners',
    }
    _skip_if_comment_predicate : ClassVar = {
        'specific_heat_avail' : (0,lambda comment: 'Specific heat availability' not in comment)
    }
    
    id : str
    iso_formula : str
    iso_slug : str
    dataset : str
    version : str
    inchi_key : str
    n_atoms : int
    atom : tuple[ExomolAtom,...]
    mass : ExomolMass
    symmetry_group : str
    n_irreducible_reps : int
    irreducible_rep : tuple[ExomolIrreducibleRep,...]
    max_temp : float
    n_broadeners : int
    dipole_available : int
    n_cross_section_files : int
    n_kcoef_files : int
    lifetime_available : int
    lande_g_factor_available : int
    n_states : int
    n_quanta_cases: int
    quanta_case : tuple[ExomolQuantaCase,...]
    auxiliary_list : ExomolAuxiliaryList # only present in some files, has a custom `from_line_itr` class-method
    n_transitions : int
    n_transition_files : int
    max_wavenumber : float
    highest_complete_state_energy : float
    part_fn_max_temp : float
    temp_step : float
    cooling_fn_avail : int
    cooling_fn_info : tuple[ExomolCoolingFnInfo]
    specific_heat_avail : int
    lorentz_hwhm_default : float
    lorentz_temp_exp_default : float
    broadener_file_info : tuple[ExomolBroadenerFileInfo,...]
    
    @classmethod
    def from_url(cls, url):
        line_itr = (x for x in fetch.file(url, encoding='ascii').splitlines())
        return cls.from_line_itr(None, line_itr)[1]

    def get_molecule_name(self) -> str:
        # iso_formula = "(12C)(1H)4
        # want to ignore brackets and numbers inside brackets, but keep letters and numbers outside brackets
        molecule_name = ''
        bracket_depth = 0
        for c in self.iso_formula: 
            if c == '(':
                bracket_depth += 1
            elif c == ')':
                bracket_depth -= 1
            elif c in "0123456789" and bracket_depth > 0:
                pass
            else:
                molecule_name += c
        return molecule_name

    def get_parent_url(self, database_url : str):
        while database_url.endswith('/'):
            database_url = database_url[:-1]
        return database_url + f'/{self.get_molecule_name()}/{self.iso_slug}/{self.dataset}'
    
    