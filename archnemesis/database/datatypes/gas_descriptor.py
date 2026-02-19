

from typing import NamedTuple



from archnemesis import Data

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class RadtranGasDescriptor(NamedTuple):
    gas_id : int
    iso_id : int
    
    @classmethod
    def from_global_iso_id(cls, global_iso_id):
        gas_id = (global_iso_id >> 16)
        iso_id = (global_iso_id - (gas_id << 16))
        return cls(gas_id, iso_id)
    
    @property
    def gas_name(self):
        return Data.gas_info[str(self.gas_id)]['name']
    
    @property
    def isotope_name(self):
        if self.iso_id == 0:
            return "ALL"
        name = Data.gas_info[str(self.gas_id)]['isotope'][str(self.iso_id)].get('name',None)
        if name is None or len(name) == 0:
            return f'UNKNOWN_{self.global_iso_id}'
        return name

    @property
    def label(self):
        return f'Gas{{{self.gas_id} : {self.gas_name}, {self.iso_id} : {self.isotope_name}}}'

    @property
    def molecular_mass(self):
        """
        in grams / mol
        """
        return float(Data.gas_info[str(self.gas_id)]['isotope'][str(self.iso_id)]['mass'])
    
    @property
    def abundance(self):
        return float(Data.gas_info[str(self.gas_id)]['isotope'][str(self.iso_id)]['abun'])
    
    @property
    def global_iso_id(self) -> int:
        """
        Yields a unique value for each gas isotope
        """
        assert self.iso_id < 65536 # NOTE: 2^{16} = 65536
        return ((self.gas_id << 16) + self.iso_id)
