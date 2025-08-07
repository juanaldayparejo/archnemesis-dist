from __future__ import annotations #  for 3.9 compatability

import os.path
from typing import Self, Protocol, ClassVar, TYPE_CHECKING

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
    def __repr__(self):
        """
        Returns a string that represents the current state of the class
        """
        return f'{self.__class__.__name__}(instance_id={id(self)}, local_storage_dir={self.local_storage_dir})'
    
    @property
    def ready(self) -> bool:
        """
        Returns True if the database is ready to use, False otherwise.
        """
        raise NotImplementedError
    
    @property
    def local_storage_dir(self) -> str:
        """
        Returns the directory the local database is stored in
        """
        raise NotImplementedError
    
    @local_storage_dir.setter
    def local_storage_dir(self, value : str) -> None:
        """
        Sets the directory the local database is stored in
        """
        raise NotImplementedError
    
    def purge(self):
        """
        Remove all cached data from this database and make it so the database must be reinitalised
        """
        raise NotImplementedError
    
    def get_line_data(
            self, 
            gas_descriptors : tuple[RadtranGasDescriptor,...], 
            wave_range : WaveRange, 
            ambient_gas : ans.enums.AmbientGas
        ) -> dict[RadtranGasDescriptor, LineDataProtocol]:
        """
        Returns a dictionary where the keys are the `gas_descriptors`, and the values
        are the line data for the `gas_descriptor` in the specified `wave_range` with
        the specified `ambient_gas`.
        
        ## ARGUMENTS ##
        
            gas_descriptors : tuple[RadtranGasDescriptor,...]
                A tuple of `RadtranGasDescriptor` instances. RadtranGasDescriptor is a class wrapper around
                a pair of (gas_id, iso_id) values that use radtran id numbers. Determines which gasses
                the data is retrieved for.
                
            wave_range : WaveRange
                The range of wavelengths/wavenumbers to retrieve data for.
                
            ambient_gas : ans.enums.AmbientGas
                The ambient gas to use when retrieving data.
        
        ## RETURNS  ##
            
            line_data : dict[RadtranGasDescriptor, LineDataProtocol]
                A dictionary where the keys are the passed `gas_descriptors` and the values are objects that follow
                the `LineDataProtocol` i.e. an object that has the following attributes:
                
                    NU : np.ndarray[['N_LINES_OF_GAS'],float]
                        Transition wavenumber (cm^{-1})
                    
                    SW : np.ndarray[['N_LINES_OF_GAS'],float]
                        Transition intensity (weighted by isotopologue abundance) (cm^{-1} / molec_cm^{-2})
                    
                    A : np.ndarray[['N_LINES_OF_GAS'],float]
                        Einstein-A coeifficient (s^{-1})
                    
                    GAMMA_AMB : np.ndarray[['N_LINES_OF_GAS'],float]
                        Ambient gas broadening coefficient (cm^{-1} atm^{-1})
                    
                    N_AMB : np.ndarray[['N_LINES_OF_GAS'],float]
                        Temperature dependent exponent for `GAMMA_AMB` (NUMBER)
                    
                    DELTA_AMB : np.ndarray[['N_LINES_OF_GAS'],float]
                        Ambient gas pressure induced line-shift (cm^{-1} atm^{-1})
                    
                    GAMMA_SELF : np.ndarray[['N_LINES_OF_GAS'],float]
                        Self broadening coefficient (cm^{-1} atm^{-1})
                    
                    ELOWER : np.ndarray[['N_LINES_OF_GAS'],float]
                        Lower state energy (cm^{-1})
            
        """
        raise NotImplementedError
    
    def get_partition_function_data(
            self, 
            gas_descriptors : tuple[RadtranGasDescriptor,...]
        ) -> dict[RadtranGasDescriptor, PartitionFunctionDataProtocol]:
        """
        Returns a dictionary where the keys are the `gas_descriptors`, and the values
        are the partition function data for the `gas_descriptor`.
        
        ## ARGUMENTS ##
        
            gas_descriptors : tuple[RadtranGasDescriptor,...]
                A tuple of `RadtranGasDescriptor` instances. RadtranGasDescriptor is a class wrapper around
                a pair of (gas_id, iso_id) values that use radtran id numbers. Determines which gasses
                the data is retrieved for.
        
        ## RETURNS  ##
            
            line_data : dict[RadtranGasDescriptor, PartitionFunctionDataProtocol]
                A dictionary where the keys are the passed `gas_descriptors` and the values are objects that follow
                the `PartitionFunctionDataProtocol` i.e. an object that has the following attributes:
                
                    TEMP : np.ndarray[['N_TEMPS_OF_GAS'],float]
                        Temperature of tablulated partition function (Kelvin)
                    
                    Q : np.ndarray[['N_TEMPS_OF_GAS'],float]
                        Tabulated partition function value
            
        """
        raise NotImplementedError
    
