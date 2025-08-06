from __future__ import annotations #  for 3.9 compatability

import archnemesis.database.wrappers.hapi as hapi

from .gas_descriptor import GasDescriptor, RadtranGasDescriptor, HitranGasDescriptor
from archnemesis import Data
import archnemesis.Data.gas_data

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
    
    @classmethod
    def from_hitran(cls, ht_gas_id : int, ht_iso_id : int | tuple[int,...]) -> Self:
        if type(ht_gas_id ) is int:
            if ht_iso_id == 0:
                n_hitran_isotopes = len(True for x in hapi.ISO.keys() if x[0]==ht_gas_id)
                ht_iso_id = tuple(i+1 for i in range(n_hitran_isotopes))
            else:
                ht_iso_id = tuple(ht_iso_id,)
        else:
            ht_iso_id = tuple(x for x in ht_iso_id)
        
        rt_gas_id_isos = []
        for iso_id in ht_iso_id:
            id, iso = hitran_to_radtran.get((ht_gas_id, iso_id),(None,None))
            if id is not None:
                rt_gas_id_isos.append((id, iso))
        
        rt_gas_id = rt_gas_id_isos[0][0]
        rt_iso_id = tuple(x[1] for x in rt_gas_id_isos)
        
        assert all((x[0] == rt_gas_id for x in rt_gas_id_isos)), "When creating a GasIsotopesDescriptor from HITRAN ID values, all gasses must map to the same RADTRAN ID values"
        
        return cls(rt_gas_id, rt_iso_id)
    
    @property
    def n_isotopes(self) -> int:
        return len(self.rt_iso_ids)
    
    @property
    def gas_name(self) -> str:
        return Data.gas_info[str(self.rt_gas_id)]['name']
    
    @property
    def iso_names(self) -> Iterator[str,...]:
        return (Data.gas_info[str(self.rt_gas_id)]['isotope'][str(iso_id)]['name'] for iso_id in self.rt_iso_ids)
    
    @property
    def is_valid(self) -> bool:
        return self.rt_gas_id is not None and len(self.rt_iso_ids) > 0
    
    def as_hitran(self) -> tuple[int, tuple[int,...], tuple[int,...]]:
        g = tuple(self.as_hitran_gasses())
        ht_gas_id = [x.gas_id for x in g if x is not None][0]
        ht_iso_ids = [x.iso_id if x is not None else None for x in g]
        ht_global_ids = [x.global_id if x is not None else None for x in g]
        return ht_gas_id, ht_iso_ids, ht_global_ids
    
    def as_radtran_gasses(self) -> Iterator[RadtranGasDescriptor, ...]:
        return (RadtranGasDescriptor(self.rt_gas_id, iso_id) for iso_id in self.rt_iso_ids)

    def as_hitran_gasses(self, filter_missing=True) -> Iterator[None|HitranGasDescriptor, ...]:
        if filter_missing:
            return (x.to_hitran() for x in self.as_radtran_gasses() if x is not None)
        else:
            return (x.to_hitran() for x in self.as_radtran_gasses())
        