from typing import NamedTuple

from ..gas_descriptor import RadtranGasDescriptor
from ...mappings.hitran import radtran_to_hitran, hitran_to_radtran

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class ExomolGasDescriptor(NamedTuple):
    gas_id : int
    iso_id : int
    
    @classmethod
    def from_gas_and_iso_id(cls, gas_id, iso_id):
        return cls(gas_id, iso_id)
    
    @classmethod
    def from_radtran(cls, rt_gas_desc : RadtranGasDescriptor):
        result = radtran_to_hitran.get((rt_gas_desc.gas_id,rt_gas_desc.iso_id), None) # NOTE: Assume EXOMOL uses same gas numbering as HITRAN
        if result is None:
            _lgr.warning(f'Could not convert {rt_gas_desc} to EXOMOL.')
        return result if result is None else cls.from_gas_and_iso_id(*result)
    
    def to_radtran(self):
        result = hitran_to_radtran.get((self.gas_id, self.iso_id), None) # NOTE: Assume EXOMOL uses same gas numbering as HITRAN
        if result is None:
            _lgr.warning(f'Could not convert {self} to RADTRAN')
        return result if result is None else RadtranGasDescriptor(*result)