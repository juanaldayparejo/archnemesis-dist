#from __future__ import annotations #  for 3.9 compatability

from typing import NamedTuple

import archnemesis.database.wrappers.hapi as hapi

from ..mappings.hitran import radtran_to_hitran, hitran_to_radtran
from archnemesis import Data

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class GasDescriptor(NamedTuple):
    gas_id : int
    iso_id : int


class RadtranGasDescriptor(NamedTuple):
    gas_id : int
    iso_id : int
    
    def to_hitran(self):
        result = radtran_to_hitran.get((self.gas_id,self.iso_id), None)
        if result is None:
            _lgr.warning(f'Could not convert {self} to HITRAN.')
        return result if result is None else HitranGasDescriptor.from_gas_and_iso_id(*result)
    
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


class HitranGasDescriptor(NamedTuple):
    gas_id : int
    iso_id : int
    global_id : int # ID of the (gas_id, iso_id) pair
    
    @classmethod
    def from_gas_and_iso_id(cls, gas_id, iso_id):
        return cls(gas_id, iso_id, hapi.ISO[(gas_id, iso_id)][0])
    
    def to_radtran(self):
        result = hitran_to_radtran.get((self.gas_id, self.iso_id), None)
        if result is None:
            _lgr.warning(f'Could not convert {self} to RADTRAN')
        return result if result is None else RadtranGasDescriptor(*result)