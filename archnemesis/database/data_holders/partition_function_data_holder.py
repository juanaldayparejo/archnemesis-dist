
import dataclasses as dc

from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
from archnemesis.database.datatypes.pf_data.pf_data import PFData



@dc.dataclass(slots=True)
class PartitionFunctionDataHolder:
	# source information
	name : str # Name of source, will result in a "/sources/X" group
	description : str # Description of source, will be the "description" attribute of the "/sources/X" group
	
	# A dictionary of partition function data. Keys are `RadtranGasDescriptor` instances, 
	# values are a list of `PFData` instances. `PFData` sub-classes define a callable class that
	# returns the partition function of an isotopologue at a specified temperature for a specified
	# temperature domain.
	# Will define the "/sources/X/partition_function/<mol_name>/<iso_id>/pf_data_0000"
	# groups that hold data for each of the `PFData` instances in the list.
	data : dict[RadtranGasDescriptor, list[PFData,...]] = dc.field(default_factory=dict) 
	
	def add(self, mol_id, local_iso_id, pf_data : PFData):
		self.data.setdefault(RadtranGasDescriptor(mol_id, local_iso_id), []).append(pf_data)
	
	def items(self):
		yield from self.data.items()
