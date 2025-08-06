from __future__ import annotations #  for 3.9 compatability

import os.path
from typing import Self, Protocol, ClassVar, TYPE_CHECKING
import dataclasses as dc

import numpy as np

import archnemesis as ans
import archnemesis.enums
from .datatypes.wave_range import WaveRange
from .datatypes.gas_isotopes import GasIsotopes
from .datatypes.gas_descriptor import RadtranGasDescriptor

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


if TYPE_CHECKING:
    N_LINES_OF_GAS = 'Number of lines for a gas isotopologue'
    N_TEMPS_OF_GAS = 'Number of temperature points for a gas isotopologue'
    

class LineDataProtocol(Protocol):
    """
    Protocol for something with the same structure as a record array with the following description:
    
        np.recarray[
            ['N_LINES_OF_GAS'],
            [
                ('NU', float), # Transition wavenumber (cm^{-1})
                ('SW', float), # transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                ('A', float), # einstein-A coeifficient (s^{-1})
                ('GAMMA_AMB', float), # ambient gas broadening coefficient (cm^{-1} atm^{-1})
                ('N_AMB', float), # temperature dependent exponent for `GAMMA_AMB` (NUMBER)
                ('DELTA_AMB', float), # ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                ('GAMMA_SELF', float), # self broadening coefficient (cm^{-1} atm^{-1})
                ('ELOWER', float), # lower state energy (cm^{-1})
            ]
        ]
    """
    NU : np.ndarray[['N_LINES_OF_GAS'],float] # Transition wavenumber (cm^{-1})
    SW : np.ndarray[['N_LINES_OF_GAS'],float] # transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
    A : np.ndarray[['N_LINES_OF_GAS'],float] # einstein-A coeifficient (s^{-1})
    GAMMA_AMB : np.ndarray[['N_LINES_OF_GAS'],float] # ambient gas broadening coefficient (cm^{-1} atm^{-1})
    N_AMB : np.ndarray[['N_LINES_OF_GAS'],float] # temperature dependent exponent for `GAMMA_AMB` (NUMBER)
    DELTA_AMB : np.ndarray[['N_LINES_OF_GAS'],float] # ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
    GAMMA_SELF : np.ndarray[['N_LINES_OF_GAS'],float] # self broadening coefficient (cm^{-1} atm^{-1})
    ELOWER : np.ndarray[['N_LINES_OF_GAS'],float] # lower state energy (cm^{-1})


class PartitionFunctionDataProtocol(Protocol):
    """
    Protocol for something with the same structure as a record array with the following description:
    
        np.recarray[
            ['N_TEMPS_OF_GAS'],
            [
                ('TEMP', float), # Temperature of tablulated partition function (Kelvin)
                ('Q', float), # Tabulated partition function value
            ]
        ]
    """
    TEMP : np.ndarray[['N_TEMPS_OF_GAS'],float] # Temperature of tablulated partition function (Kelvin)
    Q : np.ndarray[['N_TEMPS_OF_GAS'],float] # Tabulated partition function value



class LineDatabaseProtocol(Protocol):
    local_storage_dir : ClassVar[str] = os.path.normpath("local_line_database")
    
    @classmethod
    def set_local_storage_dir(cls, local_storage_dir : str):
        cls.local_storage_dir = os.normpath(local_storage_dir)
    
    def __repr__(self):
        """
        Returns a string that represents the current state of the class
        """
        return f'{self.__class__.__name__}(instance_id={id(self)}, local_storage_dir={self.local_storage_dir})'
    
    def purge_cache(self):
        raise NotImplementedError
    
    def get_line_data(
            self, 
            gas_descriptors : tuple[RadtranGasDescriptor,...], 
            wave_range : WaveRange, 
            ambient_gas : ans.enums.AmbientGas
        ) -> dict[RadtranGasDescriptor, LineDataProtocol]:
        raise NotImplementedError
    
    def get_partition_function_data(
            self, 
            gas_descriptors : tuple[RadtranGasDescriptor,...]
        ) -> dict[RadtranGasDescriptor, PartitionFunctionDataProtocol]:
        raise NotImplementedError
    
