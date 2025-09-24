

from typing import NamedTuple



from archnemesis import Data

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class RadtranGasDescriptor(NamedTuple):
    gas_id : int
    iso_id : int
    
    
    @property
    def gas_name(self):
        return Data.gas_info[str(self.gas_id)]['name']
    
    @property
    def isotope_name(self):
        if self.iso_id == 0:
            return "(all isotopes in terrestrial abundance)"
        return Data.gas_info[str(self.gas_id)]['isotope'][str(self.iso_id)]['name']

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
    def global_id(self):
        return int(Data.gas_info[str(self.gas_id)]['isotope'][str(self.iso_id)]['id'])




