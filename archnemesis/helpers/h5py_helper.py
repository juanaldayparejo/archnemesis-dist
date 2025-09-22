#from __future__ import annotations #  for 3.9 compatability

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
    
    if item_path not in h5py_file:
        
        if data is not None:
            return h5py_file.create_dataset(item_path, data=data, dtype=dtype)
        else:
            return h5py_file.create_dataset(item_path, shape=None, dtype=dtype)
    
    if data is not None:
        dset = h5py_file[item_path]
        dset[...] = data
        return dset
    else:
        del h5py_file[item_path]
        return h5py_file.create_dataset(item_path, shape=None, dtype=dtype)


def write(
        h5py_file : h5py.File | h5py.Group,
        obj : Any,
        item_path : str,
        *,
        attrs : None | tuple[str,...] = None,
        metadata : dict[str, dict[str,Any]] = dict(), # Any keys in this that are not in `attrs` that has a 'default' entry in `metadata` will use that value, if they do not have a 'default' entry will throw an error
    ):
    """
    Writes `obj` to `h5py_file` by writing all non-callable attributes of `obj` to the `h5py_file` at `item_path`
    
    
    ## Arguments ##
    
        attrs : None | tuple[str,...] = None
            A tuple of attributes of `obj` to be written to the file. If `None` will infer `attrs` from `obj`.
    
        metadata : dict[str, dict[str,Any]] = dict()
            A dictionary of metadata for each attribute, if the 'default' key is present will use that value if `attr` is not present in `attrs`
            otherwise an `attr` that is not in `attrs` will throw an error. Other keys will be passed to the HDF5 file as attributes for the `attr`
            being saved.
            
            Common keys:
                
                * 'default' - If `attr` is not present in `attrs`, use this value
                * 'unit' - Unit of `attr`
                * 'title' - A short descriptive title for `attr`
                * 'type' - A description of the type of object `attr` represents
    
    ## Example ##
        import h5py_helper
        from typing import NamedTuple
        
        class Point(NamedTuple):
            x : float
            y : float
            description : str
        
        origin = Point(0,0)
        
        h5py_helper.write('origin.h5', origin, '/origin', defaults={'description' : origin or a coord system})
    """
    
    if attrs is None: # Try and get attributes of `obj` if we are not given them
        if hasattr(obj, '_fields'):
            # Assume it is like a NamedTuple
            attrs = obj._fields
        elif hasattr(obj, '__dataclass_fields__'):
            # Assume it is like a dataclass
            attrs = tuple(x.name for x in obj.__dataclass_fields__)
        elif hasattr(obj, '__slots__'):
            # Assume the `__slots__` have the attributes we want
            attrs = obj.__slots__
        else:
            # Finally, just try and rip the values out via `vars`
            try:
                attrs = tuple(vars(obj).keys())
            except Exception as e:
                raise AttributeError('Cannot get attribute of object for writing to HDF5 file') from e
    
    for attr in attrs:
        attr_path = f'{item_path}/{attr}'
        meta = metadata.get(attr, dict())
        
        dset = store_data(h5py_file, attr_path, getattr(obj, attr))
        for k,v in meta.items():
            if k =='default':
                continue
            dset.attrs[k] = v
        
    for attr, meta in metadata.items():
        if attr not in attrs:
            if 'default' in v:
                attr_path = f'{item_path}/{attr}'
                dset = store_data(h5py_file, attr_path, meta['default'])
                for k,v in meta.items():
                    if k =='default':
                        continue
                    dset.attrs[k] = v
            else:
                raise AttributeError(f'Expected attribute "{attr}" when writing HDF5 file, but {obj=} has no such attribute and no default value provided.')
    


