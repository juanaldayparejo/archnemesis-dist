import pytest  
import archnemesis as ans
import numpy as np

from archnemesis.Models import _get_all_model_classes

def test_model_classes_have_unique_id_numbers():
    """
    All model classes should have a unique ID number.
    """
    seen_id_numbers = dict()
    for model_class in _get_all_model_classes():
        seen_id_numbers[model_class.id] = (*seen_id_numbers.get(model_class.id, tuple()), model_class)
    
    for id, model_classes in seen_id_numbers.items():
        assert len(model_classes) == 1,f"Model id {id} is not unique, is used by all of these models: {tuple(m.__module__+'.'+m.__qualname__ for m in model_classes)}"


def test_model_classes_have_no_abstract_methods():
    """
    The aim is to make sure model classes can be instantiated, and that we didn't miss any abstract methods when
    writing a model class.
    """
    
    model_instances = []
    
    for model_class in _get_all_model_classes():
        assert len(model_class.__abstractmethods__) == 0, f'Model id {model_class.id} must not have any abstract methods left (must be a concrete class). Has abstract methods: {tuple(model_class.__abstractmethods__)}'


