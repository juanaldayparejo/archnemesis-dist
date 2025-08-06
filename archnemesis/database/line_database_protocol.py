from __future__ import annotations #  for 3.9 compatability

import os.path
from typing import Self, Protocol, ClassVar
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
    
    def get_line_data(self, gas_descriptors : tuple[RadtranGasDescriptor,...], wave_range : WaveRange, ambient_gas : ans.enums.AmbientGas) -> dict[RadtranGasDescriptor, np.ndarray[
            ['N_LINES_OF_GAS'],
            [
                ('nu', float), # Transition wavenumber (cm^{-1})
                ('sw', float), # transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                ('a', float), # einstein-A coeifficient (s^{-1})
                ('gamma_air', float), # air broadening coefficient (cm^{-1} atm^{-1})
                ('n_air', float), # temperature dependent exponent for `gamma_air` (NUMBER)
                ('delta_air', float), # air pressure induced line-shift (cm^{-1} atm^{-1})
                ('gamma_self', float), # self broadening coefficient (cm^{-1} atm^{-1})
                ('elower', float), # lower state energy (cm^{-1})
            ]
        ]]:
        raise NotImplementedError
    
    def get_partition_function_data(self, gas_descriptors : tuple[RadtranGasDescriptor,...]) -> dict[RadtranGasDescriptor, np.ndarray[
            ['N_TEMPS_OF_GAS'],
            [
                ('T', float), # Temperature of tablulated partition function
                ('Q', float), # Tabulated partition function value
            ]
        ]]:
        raise NotImplementedError
    
