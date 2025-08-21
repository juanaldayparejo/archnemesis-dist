from __future__ import annotations #  for 3.9 compatability

import h5py
from typing import Callable, Any

import numpy as np

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)

def retrieve_data(
        h5py_file : h5py.File | h5py.Group,
        item_path : str,
        mutator : Callable[[Any], Any] = lambda x: x, # default is identity function
        default : Any = None,
    ) -> Any:
    """
    Retrieves `item_path` data from `h5py_file`, passing it through the `mutator` callable as it does so.
    Makes it easier to ensure we return a certain type from this function but also enables the
    setting of a `default` value for cases where `item_path` is not present in `h5py_file`.
    """
    if item_path in h5py_file and h5py_file[item_path].shape is not None:
        return mutator(h5py_file[item_path])
    else:
        _lgr.warning(f'When reading file "{h5py_file.filename}", could not find element "{item_path}" setting returned value to "{default}"', stacklevel=2)
        return default


def store_data(
        h5py_file : h5py.File | h5py.Group,
        item_path : str,
        data : Any,
        dtype = None, # will guess data type
    ) -> None:
    r"""
    Stores `data` at `item_path` in `h5py_file`. Values of "None" create an empty dataset
    
    Regex replacement for previous version "(\w*?)\.create_dataset\(('.*?'),\s*data\s*=\s*(.*)\)" -> "h5py_helper.store_data($1, $2, $3)"
    """
    #f.create_dataset('Retrieval/Output/OptimalEstimation/NX',data=self.NX)
    
    if dtype is None:
        dtype = float
        if issubclass(type(data), np.ndarray):
            dtype = data.dtype
        elif type(data) is int:
            dtype = int
    
    
    if data is not None:
        return h5py_file.create_dataset(item_path, data=data, dtype=dtype)
    else:
        return h5py_file.create_dataset(item_path, shape=None, dtype=dtype)
    
    
    

