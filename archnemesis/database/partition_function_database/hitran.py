#from __future__ import annotations #  for 3.9 compatability

import numpy as np
import numpy.ma

import archnemesis.database.wrappers.hapi as hapi
from ..protocols import (
    PartitionFunctionDatabaseProtocol, 
    PartitionFunctionDataProtocol
)

from ..datatypes.gas_descriptor import RadtranGasDescriptor

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


class HITRAN(PartitionFunctionDatabaseProtocol):
    """
    Class that implements the "PartitionFunctionDatabaseProtocol" for HITRAN datasets.
    """
    
    
    @property
    def local_storage_dir(self) -> str:
        """
        Required by PartitionFunctionDatabaseProtocol. Gets the directory the local database is in.
        """
        return None
    
    @local_storage_dir.setter
    def local_storage_dir(self, value : str) -> None:
        """
        Required by PartitionFunctionDatabaseProtocol. Sets the directory the local database is in.
        """
        raise ValueError('HITRAN partition function database is stored in RAM and not in a file.')
    
    def __init__(
            self,
            local_storage_dir : None | str = None,
        ):
        return
    
    @property
    def ready(self) -> bool:
        """
        Required by PartitionFunctionDatabaseProtocol. Returns True if the database is ready to use, False otherwise
        """
        return self.db_init_flag
    
    def purge(self) -> None:
        """
        Required by PartitionFunctionDatabaseProtocol. Remove all local data and make it so the database must be reinitalised
        """
        return
    
    def get_partition_function_data(
            self, 
            gas_descs : tuple[RadtranGasDescriptor,...]
        ) -> dict[RadtranGasDescriptor, PartitionFunctionDataProtocol]:
        """
        Required by PartitionFunctionDatabaseProtocol. Retrieves partition function data from the database.
        
        HITRAN stores partition function data in the `hapi` module, so data is always local and available,
        therefore just need to get it into the PartitionFunctionDataProtocol format.
        """
        return self._read_partition_function_data(tuple(gas_descs))
    
    
    def _read_partition_function_data(self, gas_descs : tuple[RadtranGasDescriptor,...]) -> dict[RadtranGasDescriptor, PartitionFunctionDataProtocol]:
        partition_function_data = dict()
        
        for gas_desc in gas_descs:
            ht_gas = gas_desc.to_hitran()
            if ht_gas is None:
                partition_function_data[gas_desc] = None
                continue
            
            temps = hapi.TIPS_2021_ISOT_HASH[(ht_gas.gas_id,ht_gas.iso_id)]
            qs = hapi.TIPS_2021_ISOQ_HASH[(ht_gas.gas_id,ht_gas.iso_id)]
            partition_function_data[gas_desc] = np.array(
                list(zip(
                    temps,
                    qs
                )),
                dtype=[
                    ('TEMP', float), 
                    ('Q', float)
                ]
            ).view(np.recarray)
        return partition_function_data
    
