

from typing import Self, Iterator

from .gas_descriptor import RadtranGasDescriptor
from archnemesis import Data

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

class GasIsotopes:
    def __init__(self, rt_gas_id : int = None, rt_iso_ids : tuple[int,...] = tuple()):
        """
        Stored using radtran ID mappings internally
        """
        if rt_gas_id is not None:
            if type(rt_iso_ids) is int:
                if rt_iso_ids == 0:
                    rt_iso_ids = tuple(i+1 for i in range(Data.gas_data.count_isotopes(rt_gas_id)))
                else:
                    rt_iso_ids = (rt_iso_ids,)
            else:
                rt_iso_ids = tuple(x for x in rt_iso_ids)
                
        for x in rt_iso_ids:
            if x == 0:
                raise ValueError('Gas isotope id should not be 0, should be split up into all isotopes e.g (1,2,3,4,...)')
    
        self.rt_gas_id : None | int = rt_gas_id
        self.rt_iso_ids : tuple[int] = rt_iso_ids
    
    def __repr__(self):
        return f'GasIsotopes({self.rt_gas_id=}, {self.rt_iso_ids=})'
    
    
    @classmethod
    def from_radtran(cls, rt_gas_id : int, rt_iso_ids : int | tuple[int,...]) -> Self:
        return cls(rt_gas_id, rt_iso_ids)
    
    @classmethod
    def from_radtran_gasses(cls, gas_descs : tuple[RadtranGasDescriptor,...]) -> Self:
        gas_ids = tuple(set([x.gas_id for x in gas_descs]))
        assert len(gas_ids) == 1, "When creating GasIsotopes instance, must supply multiple isotopes of the same gas"
        iso_ids = tuple(sorted([x.iso_id for x in gas_descs]))
        return cls(gas_ids[0], iso_ids)
    
    @property
    def n_isotopes(self) -> int:
        return len(self.rt_iso_ids)
    
    @property
    def gas_name(self) -> str:
        return Data.gas_info[str(self.rt_gas_id)]['name']
    
    @property
    def iso_names(self) -> Iterator[str]:
        return (Data.gas_info[str(self.rt_gas_id)]['isotope'][str(iso_id)]['name'] for iso_id in self.rt_iso_ids)
    
    @property
    def is_valid(self) -> bool:
        return self.rt_gas_id is not None and len(self.rt_iso_ids) > 0
    
    def as_radtran_gasses(self) -> Iterator[RadtranGasDescriptor]:
        return (RadtranGasDescriptor(self.rt_gas_id, iso_id) for iso_id in self.rt_iso_ids)
    
    def contains(self, other : Self) -> bool:
        """
        Does this set of gas isotopes contain `other`
        """
        if self.rt_gas_id != other.rt_gas_id:
            return False
        for iso_id in other.rt_iso_ids:
            if iso_id not in self.rt_iso_ids:
                return False
        
        return True