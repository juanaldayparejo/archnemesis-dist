from __future__ import annotations #  for 3.9 compatability

import h5py
from typing import Callable, Any

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)

def retrieve_data(
        h5py_file : h5py.File,
        item_path : str,
        mutator : Callable[[Any], Any] = lambda x: x, # default is identity function
        default : Any = None,
    ) -> Any:
    """
    Retrieves `item_path` data from `h5py_file`, passing it through the `mutator` callable as it does so.
    Makes it easier to ensure we return a certain type from this function but also enables the
    setting of a `default` value for cases where `item_path` is not present in `h5py_file`.
    """
    if item_path in h5py_file:
        return mutator(h5py_file[item_path])
    else:
        _lgr.warning(f'When reading file "{h5py_file.filename}", could not find element "{item_path}" setting returned value to "{default}"', stacklevel=2)
        return default

