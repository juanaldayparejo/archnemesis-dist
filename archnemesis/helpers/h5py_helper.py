

import h5py
from typing import Callable, Any, Literal, Type

import numpy as np

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)


def ensure_grp(
        grp : h5py.Group, 
        name: str, 
        attrs : None | dict[str,Any] = None, 
        **kwargs
    ) -> h5py.Group:
    """
    Return `name` sub-group of `grp`, create `name` sub-group if it does not already exist
    """
    create_group_flag = True
    if name in grp.keys():
        group = grp[name]
        if not isinstance(group, h5py.Group):
            del grp[name]
        else:
            create_group_flag = False
    
    if create_group_flag:
        group = grp.create_group(name, **kwargs)
    
    if attrs is not None:
        for attr, value in attrs.items():
            if attr in group.attrs and group.attrs[attr] == value:
                continue
            group.attrs[attr] = value
    return group

def get_dataset(
        grp : h5py.Group,
        name : str,
        defaults : dict[str, Any] = {},
        on_is_not_dataset : Literal['ignore', 'warn','error'] = 'error',
        on_missing : Literal['ignore', 'warn','error'] = 'ignore',
    ) -> h5py.Dataset:
    """
    Return `name` dataset of `grp` if dataset does not exist, create it with passed arguments
    """
    if name in grp.keys():
        dset = grp[name]
        if isinstance(dset, h5py.Dataset):
            return dset
        else:
            match on_is_not_dataset:
                case 'ignore':
                    _lgr.debug(f'Item "{name}" of group "{grp.name}" in HDF5 file "{grp.file.filename}" is type "{type(dset)}" not "{h5py.Dataset}". Returning object anyway.')
                    return dset
                case 'warn':
                    _lgr.warning(f'Item "{name}" of group "{grp.name}" in HDF5 file "{grp.file.filename}" is type "{type(dset)}" not "{h5py.Dataset}". Returning object anyway.')
                    return dset
                case _:
                    raise TypeError(f'Item "{name}" of group "{grp.name}" in HDF5 file "{grp.file.filename}" is type "{type(dset)}" not "{h5py.Dataset}".')
    
    match on_missing:
        case 'ignore':
            _lgr.debug(f'No item "{name}" of group "{grp.name}" in HDF5 file "{grp.file.filename}", creating and returning a default dataset')
            return grp.create_dataset(name, **defaults)
        case 'warn':
            _lgr.warning(f'No item "{name}" of group "{grp.name}" in HDF5 file "{grp.file.filename}", creating and returning a default dataset')
            return grp.create_dataset(name, **defaults)
        case _:
            raise KeyError(f'No item "{name}" of group "{grp.name}" in HDF5 file "{grp.file.filename}".')

def ensure_dataset(
        grp : h5py.Group, 
        name : str, 
        attrs : None | dict[str,Any] = None, 
        extend : None | Literal['stack'] | int = None, 
        **kwargs
    ) -> h5py.Dataset:
    """
    Return `name` dataset of `grp`, if dataset already exists remove it and re-create it with passed arguments. If `extend` is not None, either stack or extend the data in the dataset.
    
    ## ARGUMENTS ##
        extend : None | Literal['stack'] | int = None
            Should we extend the dataset instead of overwriting it? 
            If `extend` == 'stack', will stack along a new 0th axis if existing data and new data 
            are the same shape, otherwise will assume that the 0th axis is the axis to stack along,
            and other axes must be the same between old data and new data.
            If `extend` is an integer, will extend along that axis, shape of old and new data must
            be the same along other axes.
    """
    old_data = None
    del_flag = False
    
    data = kwargs.pop('data', None)
    
    
    if extend:
        if extend != 'stack' or not isinstance(extend, int):
            raise ValueError(f'h5py_helper.ensure_dataset(...) `extend` must be one of {{{None}, "stack", {int} instance}}, not "{extend}".')
    
    if name in grp.keys():
        del_flag = True
        if extend:
            old_data = grp[name][tuple()]
        
    
    
    
    if extend:
        if data is None: # if no new data, just keep old data
            data = old_data
        else: # otherwise, must stack or extend.
            if extend == 'stack':
                if old_data.ndim == data.ndim:
                    assert all(s0==s1 for s0,s1 in zip(old_data.shape, data.shape)), \
                        f"When extending via 'stack', if old data and new data have the same number of dimensions, they must also must have the same shape but have {old_data.shape=} {data.shape=}"
                    data = np.stack((old_data, data),axis=0)
                elif old_data.ndim == (data.ndim+1):
                    assert all(s0==s1 for s0,s1 in zip(old_data.shape[1:], data.shape)), \
                        f"When extending via 'stack', if old data has one more dimension than new data, the 0th dimension is assumed to be stacked along so the last dimensions of old data must have the same shape as new data but have {old_data.shape=} {data.shape=}"
                    data = np.stack((*old_data, data), axis=0)
                else:
                    raise ValueError(f"When extending via 'stack', old data must have the same or one more dimensions than new data but have {old_data.shape=} {data.shape=}")
                
            elif isinstance(extend, int):
                assert old_data.ndim == data.ndim, \
                    f"When `extend` is an integer, old data and new data must have same number of dimensions but have {old_data.ndim=} {data.ndim=}"
                assert all(s0 == s1 for i, (s0,s1) in enumerate(zip(old_data.shape, data.shape)) if i!=extend), \
                    f"When `extend` is an integer, old data and new data must have the same shape along all dimensions except the specified one but have {extend=} {old_data.shape=} {data.shape=}"
                
                new_shape = tuple(s0 if i != extend else (s0+s1) for i, (s0,s1) in enumerate(zip(old_data.shape, data.shape)))
                new_data = np.empty(new_shape, dtype=np.promote_types(old_data.dtype, data.dtype))
                
                new_data[tuple(slice(0,s) for s in old_data.shape)] = old_data
                new_data[tuple(slice(0,s1) if i != extend else slice(s0,s0+s1) for i, (s0,s1) in enumerate(zip(old_data.shape, data.shape)))] = data
                
                data = new_data
            else:
                raise ValueError(f'h5py_helper.ensure_dataset(...) `extend` must be one of {{{None}, "stack", {int} instance}}, not "{extend}".')
                
    if del_flag:
        del grp[name]
    
    dset = grp.create_dataset(name, data=data, **kwargs)
    
    if attrs is not None:
        for attr, value in attrs.items():
            if attr in dset.attrs and dset.attrs[attr] == value:
                continue
            dset.attrs[attr] = value
    
    return dset

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
    


