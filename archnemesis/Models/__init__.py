from __future__ import annotations #  for 3.9 compatability

from typing import Iterator

from . import AtmosphericModels
from . import NonAtmosphericModels
from . import SpectralModels
from .ModelParameterEntry import ModelParameterEntry
from .ModelBase import ModelBase

import inspect


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class ModelPrinterMixin(type):
    def __repr__(cls) -> str:
        return '\n\n'.join((x.to_string() for x in cls._model_classes))
    
    def __str__(cls) -> str:
        return '\n\n'.join((x.to_string() for x in cls._model_classes))
    
    def __iter__(cls) -> Iterator[ModelBase]:
        yield from cls._model_classes
    
    def __getitem__(cls, id) -> ModelBase:
        try:
            return next(filter(lambda x: x.id == id, cls._model_classes))
        except StopIteration:
            raise KeyError(f'No model with id {id} found.')
    
    def __contains__(cls, id) -> bool:
        for x in cls._model_classes:
            if x.id == id:
                return True
        return False
    
    @property
    def ids(cls) -> tuple[int,...]:
        return tuple(x.id for x in cls._model_classes)
    
    @property
    def names(cls) -> tuple[str,...]:
        return tuple(x.__name__ for x in cls._model_classes)
    
    def info(cls, id=None) -> str:
        if id is None:
            return repr(cls)
        else:
            for x in cls._model_classes:
                if x.id == id:
                    return x.to_string()
            raise KeyError(f'No model with id {id} found.')
            


class Models(metaclass=ModelPrinterMixin):
    """
    Holds all of the model parameterisation classes indexed by ID number
    
    ## PROPERTIES ##
    
        ids -> tuple[int,...]
            Returns a tuple of all the ids for all the model classes stored by this class
        
        names -> tuple[str,...]
            Returns a tuple of the class names of all the model classes stored by this class
            
    ## METHODS ##
        
        info([id : int]) -> str
            Returns a string with a summary of information for each model class stored by this class,
            if passed an `id` will only return information string for the model class with that id.
        
        __str__() -> str
        __repr__() -> str
            Returns the same information as `info()`, but enables you to write `print(Models)`
            as syntactic sugar
        
        __iter__() -> Iterator[ModelBase]
            Returns an iterator that loops over each model class stored by this class. 
            Example:
                ```
                for model_class in Models:
                    print(model_class.parameters)
                ```
        
        __getitem__(id) -> ModelBase
            Returns the model class with `id` stored by this model class, or raises `KeyError` if
            no class with `id` exists.
        
        __contains__(id) -> bool
            Returns True if a model class with `id` is stored, else returns False.
        
        display(id) -> None
            Prints the `info()` entry for stored model class with `id` to stdout.
        
        
        
    """
    
    _model_classes = (
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
    
    @classmethod
    def display(cls, id) -> None:
        print(cls[id].to_string())
    
    @classmethod
    def as_string(cls, id) -> str:
        return cls[id].to_string()
    

