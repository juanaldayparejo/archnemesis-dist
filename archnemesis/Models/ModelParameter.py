from __future__ import annotations #  for 3.9 compatability
import dataclasses as dc

import numpy as np

@dc.dataclass
class ModelParameter:
    """
    Class that defines a model parameter. For use with models that inherit from archnemesis.Models.ModelBase.ModelBase.
    """
    name : str
    slice : slice
    description : str
    unit : str = 'UNDEFINED' # default is to have an undefined unit