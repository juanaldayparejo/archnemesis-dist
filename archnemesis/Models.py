from __future__ import annotations #  for 3.9 compatability
import archnemesis.ModelClass
import inspect


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

#Models=tuple()
Models = tuple(
    x[1] for x in inspect.getmembers(
        archnemesis.ModelClass, 
        lambda x: inspect.isclass(x) and issubclass(x, archnemesis.ModelClass.ModelBase) and x.id is not None
    )
)


if __name__=='__main__':
    for model in Models:
        pass