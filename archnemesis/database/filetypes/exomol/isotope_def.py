"""
TODO:

Alter this file to work by reading a line, then processing the line based on the contents (probably the comment) of that line.

I am having the problem that the *.def files are not really formatted the way the EXOMOL paper says they should be,
so I cannot use the layout presented there.

Idea is to have similar classes like I do now, but have the (next_line, line_itr) pair passed to them depending upon the
content of `next_line`. That way, hopefully some of the structure will still be present but the parser will be more adaptable
to the many deviations from the expected format.
"""

import dataclasses as dc
from typing import ClassVar, Type

from ...datatypes.exomol.tagged_format import ExomolTaggedFileFormat, PeekableIterator
from ...datatypes.exomol.col_format import ExomolColFileFormat
from ...utils.tree_printer import TreePrinter
from ...utils import fetch

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)


EXOMOL_API_URL_FMT : str = "https://exomol.com/api/?molecule={}"



@dc.dataclass(repr=False)
class ExomolQuantumNumberSet(TreePrinter, ExomolTaggedFileFormat):
    _field_dispatch_map : ClassVar[dict[str,str]] = {
        'code' : 'A code that defines this set of quantum numbers',
        'n_lines' : 'No. of lines in the broad that contain this code',
        'n_quantum_numbers' : 'No. of quantum numbers defined',
        'quantum_number' : 'Defined quantum number',
    }
    
    code : str = None
    n_lines : int = 0
    n_quantum_numbers : int = 0
    quantum_number : tuple[str,...] = tuple()

@dc.dataclass(repr=False)
class ExomolBroadenerFileInfo(TreePrinter, ExomolTaggedFileFormat):
    _field_dispatch_map : ClassVar[dict[str,str]] = {
        'label' : 'Label for a particular broadener',
        'filename' : 'Filename of particular broadener',
        'max_J_qn' : 'Maximum J for which pressure broadening parameters provided',
        'lorentz_hwhm_broad_default' : 'Value of Lorentzian half-width for J" > Jmax',
        'lorentz_temp_exp_broad_default' : 'Value of temperature exponent for lines with J" > Jmax',
        'n_qn_sets' : 'Number of defined quantum number sets',
        'qn_set' : 'A code that defines this set of quantum numbers',
    }
    
    label : str = None
    filename : str = None
    max_J_qn : float = None # maximum J for which pressure broadening parameters provided
    lorentz_hwhm_broad_default : float = None
    lorentz_temp_exp_broad_default : float = None
    n_qn_sets : int  = 0# number of quantum number sets
    qn_set : tuple[ExomolQuantumNumberSet,...] = tuple()


@dc.dataclass(repr=False)
class ExomolAuxiliaryInfo(TreePrinter):
    index : int = None
    label : str = None
    format : str = None
    description : str = None

@dc.dataclass(repr=False)
class ExomolQuantaInfo(TreePrinter):
    index : int = None
    label : str = None
    format : str = None
    description : str = None

@dc.dataclass(repr=False)
class ExomolIrreducibleRep(TreePrinter, ExomolTaggedFileFormat):
    _field_dispatch_map = {
        'id' : 'Irreducible representation ID',
        'label' : 'Irreducible representation label',
        'nuclear_spin_degeneracy' : 'Nuclear spin degeneracy',
    }
    
    id : int = None
    label : str = None
    nuclear_spin_degeneracy : int = None

@dc.dataclass(repr=False)
class ExomolMass(TreePrinter, ExomolColFileFormat):
    mass_da : float = None
    mass_kg : float = None

@dc.dataclass(repr=False)
class ExomolAtom(TreePrinter, ExomolTaggedFileFormat):
    _field_dispatch_map = {
        'iso_number' : 'Isotope number',
        'element_symbol' : 'Element symbol',
    }
    
    iso_number : int = None
    element_symbol : str = None

@dc.dataclass(repr=False)
class ExomolCoolingFnInfo(TreePrinter, ExomolTaggedFileFormat):
    _field_dispatch_map = {
        'max_temp' : 'Maximum temperature of cooling function',
        'temp_step' : 'Step size of temperature'
    }
    
    max_temp : float = None
    temp_step : float = None

@dc.dataclass(repr=False)
class ExomolPartitionFnInfo(TreePrinter, ExomolTaggedFileFormat):
    _field_dispatch_map = {
        'max_temp' : 'Maximum temperature of partition function',
        'temp_step' : 'Step size of temperature'
    }
    
    max_temp : float = None
    temp_step : float = None


def field_quanta_reader(
        type : Type, 
        peek_itr : PeekableIterator, 
        current_instance : None | tuple[None|ExomolQuantaInfo,...]
) -> tuple[ExomolQuantaInfo,...]:
    # Quanta have their ordering in their comment
    # so we need to pull that data out and ensure that the ordering
    # is respected.
    
    # Assume that we have label, format, description for each Quanta. Missing elements will be set to `None`
    
    current_instance = current_instance if current_instance is not None else tuple() # Start with empty tuple
    
    #_lgr.debug(f'{current_instance=}')
    
    idx = None
    label = None
    format = None
    description = None
    
    lbl_found_idx : None | int = None
    fmt_found_idx : None | int = None
    dsc_found_idx : None | int = None
    
    v,c = peek_itr.value.split('#',1)
    if c.strip().startswith('Quantum label'):
        try:
            lbl_found_idx = int(c.strip().split()[-1])
        except ValueError:
            lbl_found_idx = None
        label = v.strip()
        peek_itr.next()
    
    v,c = peek_itr.value.split('#',1)
    if c.strip().startswith('Format quantum label'):
        try:
            fmt_found_idx = int(c.strip().split()[-1])
        except ValueError:
            fmt_found_idx = None
        format = v.strip()
        peek_itr.next()
        
    v,c = peek_itr.value.split('#',1)
    if c.strip().startswith('Description quantum label'):
        try:
            dsc_found_idx = int(c.strip().split()[-1])
        except ValueError:
            dsc_found_idx = None
        description = v.strip()
        peek_itr.next()
    
    # Ensure we have the same idx for each found field
    for i in (lbl_found_idx, fmt_found_idx, dsc_found_idx):
        for j in (lbl_found_idx, fmt_found_idx, dsc_found_idx):
            if i is not None and j is not None:
                if i!=j:
                    _lgr.warn(f"Found index of a Quanta IS NOT the same accross label, format, and description: {(lbl_found_idx, fmt_found_idx, dsc_found_idx)}. Will use majority case if present or else will use lowest number.")
                
    
    # if all found idx are None, assume we have the index of the number of already existing element
    if all([i is None for i in (lbl_found_idx, fmt_found_idx, dsc_found_idx)]):
        idx = len(current_instance)+1 # NOTE: these are 1-indexed
    else:
        found_idx_values = [x for x in (lbl_found_idx, fmt_found_idx, dsc_found_idx) if x is not None]
        # find majority case
        idx = int(sorted([(x, found_idx_values.count(x)) for x in found_idx_values], key=lambda x: (x[1]+1 - 1E-6*x[0]), reverse=True)[0][0])
        #idx = [i for i in (lbl_found_idx, fmt_found_idx, dsc_found_idx) if i is not None][0]
    
    
    # Look throught the current instances and if we share an index, choose the next free index
    if len(current_instance) > 0:
        current_idxs = [x.index for x in current_instance if x is not None]
    else:
        current_idxs = []
    
    while idx in current_idxs:
        idx += 1
    
    
    item = ExomolQuantaInfo(idx, label, format, description)
    _lgr.debug(f'{item=}')
    
    old_max = max(0, 0, *current_idxs)
    max_idx = max(idx, old_max)

    new_list = []
    for i in range(1, max_idx+1): # NOTE: index starts at 1
        # get item in `current_instance` that has the index `i`
        # if cannot, fill with None
        # insert `item` into correct place
        # throw error if `item.index` collides with an existing items index.
        
        insert_item = None
        for old_item in current_instance:
            if old_item is not None and old_item.index == i:
                insert_item = old_item
                break
        
        if insert_item is None and item.index == i:
            insert_item = item
        
        new_list.append(insert_item)
    
    return tuple(new_list)


@dc.dataclass(repr=False)
class ExomolQuantumCaseInfo(TreePrinter, ExomolTaggedFileFormat):
    _complete_when_fields_filled = False
    _expect_contiguous_fields = True
    _repeated_fields_overwrite = False
    _field_dispatch_map = {
        'label' : 'Quantum case label',
        'n_quanta' : 'No. of quanta defined',
        'quanta' : 'Quantum label',
    }
    
    _alternate_field_readers = {
        'quanta' : field_quanta_reader,
    }
    
    label : str = None
    n_quanta : int = 0
    quanta : tuple[ExomolQuantaInfo,...] = None


def field_auxiliary_reader(
        type : Type, 
        peek_itr : PeekableIterator, 
        current_instance : None | tuple[ExomolAuxiliaryInfo,...]
) -> tuple[ExomolAuxiliaryInfo,...]:
    # Auxiliary columns have their ordering in their comment
    # so we need to pull that data out and ensure that the ordering
    # is respected.
    
    # Assume that we have label, format, description for each Auxiliary. Missing elements will be set to `None`
    
    current_instance = current_instance if current_instance is not None else tuple() # Start with empty tuple
    
    idx = None
    label = None
    format = None
    description = None
    
    lbl_found_idx : None | int = None
    fmt_found_idx : None | int = None
    dsc_found_idx : None | int = None
    
    v,c = peek_itr.value.split('#',1)
    if c.strip().startswith('Auxiliary'):
        try:
            lbl_found_idx = int(c.strip().split()[-1])
        except ValueError:
            lbl_found_idx = None
        label = v.strip()
        peek_itr.next()
    
    v,c = peek_itr.value.split('#',1)
    if c.strip().startswith('Format'):
        try:
            fmt_found_idx = int(c.strip().split()[-1])
        except ValueError:
            fmt_found_idx = None
        format = v.strip()
        peek_itr.next()
        
    v,c = peek_itr.value.split('#',1)
    if c.strip().startswith('Description'):
        try:
            dsc_found_idx = int(c.strip().split()[-1])
        except ValueError:
            dsc_found_idx = None
        description = v.strip()
        peek_itr.next()
    
    # Ensure we have the same idx for each found field
    for i in (lbl_found_idx, fmt_found_idx, dsc_found_idx):
        for j in (lbl_found_idx, fmt_found_idx, dsc_found_idx):
            if i is not None and j is not None:
                assert i==j, "Require that found index of an auxiliary_column is the same accross label, format, and description"
    
    # if all found idx are None, assume we have the index of the number of already existing element
    if all([i is None for i in (lbl_found_idx, fmt_found_idx, dsc_found_idx)]):
        idx = len(current_instance) + 1 # NOTE: indices are one based
    else:
        idx = [i for i in (lbl_found_idx, fmt_found_idx, dsc_found_idx) if i is not None][0]
    
    # Look throught the current instances and if we share an index, choose the next free index
    if len(current_instance) > 0:
        current_idxs = [x.index for x in current_instance if x is not None]
    else:
        current_idxs = []
    
    while idx in current_idxs:
        idx += 1
    
    
    item = ExomolQuantaInfo(idx, label, format, description)
    _lgr.debug(f'{item=}')
    
    old_max = max(0, 0, *current_idxs)
    max_idx = max(idx, old_max)
    
    new_list = []
    for i in range(1, max_idx+1): # NOTE: index starts at 1
        # get item in `current_instance` that has the index `i`
        # if cannot, fill with None
        # insert `item` into correct place
        # throw error if `item.index` collides with an existing items index.
        
        insert_item = None
        for old_item in current_instance:
            if old_item is not None and old_item.index == i:
                insert_item = old_item
                break
        
        if insert_item is None and item.index == i:
            insert_item = item
        
        new_list.append(insert_item)
    
    return tuple(new_list)



@dc.dataclass(repr=False)
class ExomolIsotopeDef(TreePrinter, ExomolTaggedFileFormat):
    """
    Definition for each isotope. NOTE there is an *actual* API available for EXOMOL (e.g. "https://exomol.com/api/?molecule=CH4")
    but I do not know where it is documented.
    """
    _complete_when_fields_filled = False
    _expect_contiguous_fields = False
    _self_dispatch_check = lambda next_line : next_line.split('#',1)[0].strip() == 'EXOMOL.def'
    _alternate_field_readers = {
        'quanta' : field_quanta_reader,
        'auxiliary_info' : field_auxiliary_reader,
    }
    _field_dispatch_map = {
        'id' : 'ID',
        'iso_formula' : 'IsoFormula',
        'iso_slug' : 'Iso-slug',
        'dataset' : 'Isotopologue dataset name',
        'version': 'Version number',
        'inchi_key' : 'Inchi key of molecule',
        'n_atoms' : 'Number of atoms',
        'atom' : 'Isotope number',
        'mass' : 'Isotopologue mass (Da) and (kg)',
        'symmetry_group' : 'Symmetry group',
        'n_irreducible_reps' : 'Number of irreducible representations',
        'irreducible_rep' : 'Irreducible representation ID',
        'max_temp' : 'Maximum temperature of linelist',
        'n_broadeners' : 'No. of pressure broadeners available',
        'dipole_available' : 'Dipole availability',
        'n_cross_section_files' : 'No. of cross section files available',
        'n_kcoef_files' : 'No. of k-coefficient files available',
        'lifetime_available' : 'Lifetime availability',
        'lande_g_factor_available' : 'Lande g-factor availability',
        'n_states' : 'No. of states',
        'n_quanta_cases' : 'No. of quanta cases',
        'quantum_cases' : 'Quantum case label',
        'auxiliary_info' : 'Auxiliary title',
        'n_transitions' : 'Total number of transitions',
        'n_transition_files' : 'No. of transition files',
        'max_wavenumber' : 'Maximum wavenumber',
        'highest_complete_state_energy' : 'Higher energy with complete set of transitions',
        'part_fn_info' : 'Maximum temperature of partition function',
        'cooling_fn_avail' : 'Cooling function availability',
        'cooling_fn_info' : 'Maximum temperature of cooling function',
        'specific_heat_avail' : 'Specific heat availability',
        'lorentz_hwhm_default' : 'Default value of Lorentzian half-width for all lines',
        'lorentz_temp_exp_default' : 'Default value of temperature exponent for all lines',
        'broadener_file_info' : 'Label for a particular broadener',
        'uncertainty_available' : 'Uncertainty availability',
    }
    
    
    
    id : str = None
    iso_formula : str = None
    iso_slug : str = None
    dataset : str = None
    version : str = None
    inchi_key : str = None
    n_atoms : int = 0
    atom : tuple[ExomolAtom,...] = tuple()
    mass : ExomolMass = None
    symmetry_group : str = None
    n_irreducible_reps : int = 0
    irreducible_rep : tuple[ExomolIrreducibleRep,...] = tuple()
    max_temp : float = None
    n_broadeners : int = 0
    dipole_available : int = 0
    n_cross_section_files : int = 0
    n_kcoef_files : int = 0
    lifetime_available : int = 0
    lande_g_factor_available : int = 0
    n_states : int = 0
    n_quanta_cases: int = 0
    quantum_cases : tuple[ExomolQuantumCaseInfo,...] = tuple()
    auxiliary_info : tuple[ExomolAuxiliaryInfo,...] = tuple()
    n_transitions : int = None
    n_transition_files : int = None
    max_wavenumber : float = None
    highest_complete_state_energy : float = None
    part_fn_info : ExomolPartitionFnInfo = None
    cooling_fn_avail : int = 0
    cooling_fn_info : ExomolCoolingFnInfo = None
    specific_heat_avail : int = 0
    lorentz_hwhm_default : float = None
    lorentz_temp_exp_default : float = None
    broadener_file_info : tuple[ExomolBroadenerFileInfo,...] = tuple()
    uncertainty_available : int = 0
    
    @classmethod
    def from_url(cls, url):
        text = fetch.file(url, encoding='utf-8')
        line_itr = (x for x in text.splitlines())
        return cls.from_line_itr(None, line_itr)[1].perform_fixes()

    def perform_fixes(self):
        if self.iso_formula.count('(') != self.iso_formula.count(')'):
            # brackets do not balance, ensure they do
            a = ''
            c_last = ' '
            for c in (self.iso_formula+' '): # add a space so we always consider the final character
                if not (c.isalpha() or c==')') and c_last.isalpha():
                    a += ')'
                a += c
            self.iso_formula = a.strip()
        
        
        return self

    def get_mol_formula(self) -> str:
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
        return database_url + f'/{self.get_mol_formula()}/{self.iso_slug}/{self.dataset}'










