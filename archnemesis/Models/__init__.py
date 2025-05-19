from __future__ import annotations #  for 3.9 compatability

from . import AtmosphericModels
from . import NonAtmosphericModels
from . import SpectralModels
from .ModelParameterEntry import ModelParameterEntry
from .ModelBase import ModelBase

import inspect


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

#Models=tuple()
Models = (
    *tuple(
        x[1] for x in inspect.getmembers(
            AtmosphericModels,
            lambda x: inspect.isclass(x) and issubclass(x, ModelBase) and x.id is not None
        )
    ),
    *tuple(
        x[1] for x in inspect.getmembers(
            NonAtmosphericModels,
            lambda x: inspect.isclass(x) and issubclass(x, ModelBase) and x.id is not None
        )
    ),
    *tuple(
        x[1] for x in inspect.getmembers(
            SpectralModels,
            lambda x: inspect.isclass(x) and issubclass(x, ModelBase) and x.id is not None
        )
    ),
)
