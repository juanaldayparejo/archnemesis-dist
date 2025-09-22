#from __future__ import annotations #  for 3.9 compatability

import os
import os.path

import numpy as np
import numpy.ma


import archnemesis as ans
import archnemesis.enums
import archnemesis.database.wrappers.hapi as hapi
from ..protocols import (
    LineDatabaseProtocol, 
    LineDataProtocol, 
)
from ..datatypes.wave_range import WaveRange
#from ..datatypes.gas_isotopes import GasIsotopes
from ..datatypes.gas_descriptor import RadtranGasDescriptor

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


EXOMOL_DATABASE_URL : str = "https://www.exomol.com/db"

class EXOMOL:
    database_url : str = EXOMOL_DATABASE_URL
    all_file : str = EXOMOL_DATABASE_URL + '/exomol.all'
    proxy : None | dict[str, str] = None # None or dictionary mapping protocol names to URLs of proxies