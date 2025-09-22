from __future__ import annotations #  for 3.9 compatability

from typing import get_args, Type, Self, Any, Iterator
from collections import namedtuple

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


def flat_iter(a) -> Iterator:
    """
    Iterate over `a` if possible. If an element of `a` is itself
    iterable, then iterate over that element of `a` before moving to
    the next element of `a`

    ## Example ##
    a = [1, 2, 3, [4, 5, 6], 7, [8, [9, 10]]]
    for x in flat_iter(a):
        print(x)

    Results in:
    ```
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    ```
    """
    try:
        for x in a:
            yield from flat_iter(x)
    except TypeError:
        yield a

def str_to_type(s) -> Type:
    if s == 'int':
        return int
    elif s=='float':
        return float
    elif s=='tuple':
        return tuple
    elif s.startswith('tuple'):
        #return tuple[*(str_to_type(x) for x in s[6:-1].split(', '))]
        return tuple((str_to_type(x.strip()) for x in s[6:-1].split(', ')))
    elif s=='str':
        return str

class FixedWidthFormatMeta(type):
    """
    Automatically add attributes needed to support being a FixedWidthFormat class, which is a
    class that describes a FixedWidthFormat and can read/write records in that format.

    Attributes added automatically:

        _attrs : tuple[str,...] 
            The names of attributes of the record the FixedWidthFormat class describes.

        _sizes : tuple[int,...]
            The sizes (in bytes, or characters, etc.) of the attributes of the record the FixedWidthFormat class describes.

        _types : tuple[Type | tuple[Type,...], ...]
            The types (which can be nested tuples of types) of the attributes of the record the FixedWidthFormat class describes.

        _record_length : int
            The total length of the record (in bytes, or characters, etc.) of the record the FixedWidthFormat class describes.

        _record_type : NamedTuple
            The record type the FixedWidthFormat class describes. It will be a `NamedTuple` and will have a systematic name
            based upon the name of the FixedWidthFormat class, and the attributes in `_attrs` (in the same order).
            
            The three cases for naming are:

                1) If "Format" is in the name of the FixedWidthFormat class, it will be replaced with "Record" 
                   (i.e. "FixedWidthFormat" -> "FixedWidthRecord")

                2) If "format" is in the name of the FixedWidthFormat class, it will be replaced with "record" 
                   (i.e. "FixedWidth_format" -> "FixedWidth_record")

                3) Otherwise, "_record" will be appended to the name of the FixedWidthFormat class (i.e. 
                   "FixedWidthType" -> "FixedWidthType_record")
    """
    
    def __new__(meta, name, bases, ctx):
        
        annotations = ctx.get('__annotations__',dict())
        default_attrs = tuple(k for k,v in ctx.items() if (not k.startswith('__')) and (not hasattr(v, '__func__')))
        anno_attrs = tuple(k for k in annotations)

        for attr in anno_attrs:
            if attr not in default_attrs:
                raise AttributeError(f'Attribte "{attr}" has a type annotation but no default value')

        for attr in default_attrs:
            if attr not in anno_attrs:
                raise AttributeError(f'Attribte "{attr}" has a default but no type annotation')

        attr_names = anno_attrs

        if 'Format' in name:
            record_type_name = name.replace('Format', 'Record')
        elif 'format' in name:
            record_type_name = name.replace('format', 'record')
        else:
            record_type_name = name + '_record'
        ctx['_attrs'] = attr_names
        ctx['_sizes'] = tuple(ctx[attr] for attr in attr_names)
        ctx['_types'] = tuple(annotations[attr] if (not type(annotations[attr]) is str) else str_to_type(annotations[attr]) for attr in attr_names)
        ctx['_record_length'] = sum(flat_iter(ctx['_sizes']))
        ctx['_record_type'] = namedtuple(record_type_name, attr_names)
        
        x = super().__new__(meta, name, bases, ctx)
        return x


class AsciiFixedWidthFormat(metaclass = FixedWidthFormatMeta):
    """
    A FixedWidthFormat class for records that are stored as ascii characters. Enables us to specify
    a format by laying out class-variables like a table with the following columns:

    attribute_name                     : type               = width           # in ascii characters

    ## Example ##
    
    ```
    class PointFixedWidthFormat(AsciiFixedWidthFormat):
        x : float = 5
        y : float = 8
    ```
    
    Then we could read a file like:
    ``` example_file.txt
    0000100003450
    000430005.500
    005E300300300
    ```

    via

    ```
    points = PointFixedWidthFormat.read_records("example_file.txt")
    for point in points:
        print(point)
    ```

    would output
    ```
    PointFixedWidthRecord(x=1, y=3450)
    PointFixedWidthRecord(x=43, y=5.5)
    PointFixedWidthRecord(x=5E3, y=3.003E5)
    ```
    
    """
    def __init__(self):
        raise RuntimeError(f'Instances of class "{self.__class__.__name__}" cannot be created.')
    
    @classmethod
    def to_string(cls):
        a = f'{cls.__name__}\n\ttotal_record_length = {cls._record_length}\n\trecord_type = {cls._record_type}\n'
        col_sizes = (
            max((len(a) for a in cls._attrs)),
            max((len(str(t)) for t in cls._types)),
            max((len(str(s)) for s in cls._sizes)),
        )
        col_fmts = (
            '{: <'+f'{col_sizes[0]}'+'}',
            '{: <'+f'{col_sizes[1]}'+'}',
            '{: <'+f'{col_sizes[2]}'+'}',
        )
        
        for attr, typ, size in zip(cls._attrs, cls._types, cls._sizes):
            a += '\t'+' | '.join([fmt.format(v) for fmt, v in zip(col_fmts, (attr,str(typ),str(size)))]) + '\n'
        return a
    
    @staticmethod
    def get_value_from_str(typ : Type, size : int | tuple[int,...], s : str) -> Any | tuple[Any,...]:
        """
        Given a type `typ`, a size `size` and a string `s`. Return the `typ` interpretation of `s[:size]`.

        `size` may be a tuple of integers, in which case the `typ` must be either type or a tuple of types.
        The tuple elements are recursed into this function to get the values.
        """
        if isinstance(size, int):
            # we have been given a type with no structure so assume it can be used directly
            # assume empty string is None
            a = s[:size]
            if typ == str:
                return a
            else:
                return typ(a) if not (a.isspace() or len(a) == 0) else None
        elif isinstance(size, tuple):
            # Type has some structure so must determine what that is
            argt = get_args(typ)
            size_edges = tuple((sum(flat_iter(size[:i])), sum(flat_iter(size[:i+1]))) for i in range(len(size)))
            if argt == ():
                # Assume they are all the same type
                return tuple(AsciiFixedWidthFormat.get_value_from_str(typ, x, s[y[0]:y[1]]) for x, y in zip(size, size_edges))
            else:
                # Assume argument types line up correctly
                return tuple(AsciiFixedWidthFormat.get_value_from_str(t, x, s[y[0]:y[1]]) for t, x, y in zip(argt, size, size_edges))
        else:
            raise TypeError(f'Cannot get value from string when `size` is not an integer or tuple of integers. {type(size)=}')
        return

    @staticmethod
    def chunk_iter(a : str, chunks : tuple[int, tuple[int,...], ...]) -> tuple[str,...]:
        """
        Given a string `a` and `chunks` ,a tuple of sizes (that can themselves be tuples of sizes), will
        return the string split up into `len(chunks)` elements such that each element has the same length as
        the chunk. Where a chunk is itself a tuple, will sum the tuple to get the size of the element.
        """
        i = 0
        for chunk in chunks:
            if isinstance(chunk, int):
                yield a[i:i+chunk]
                i += chunk
            else:
                size = sum(flat_iter(chunk))
                yield a[i:i+size]
                i += size
        return

    @classmethod
    def get_record_type(cls) -> Type:
        """
        Returns the record type that the FixedWidthFormat class returns from `read_records` and `get_record_from_str`
        """
        return cls._record_type

    @classmethod
    def get_record_length(cls) -> int:
        """
        Returns the length of a record in characters
        """
        return cls._record_length

    @classmethod
    def get_format_sizes(cls) -> tuple[int|tuple[int,...], ...]:
        """
        Returns the length of each record attribute in characters
        """
        return cls._sizes

    @classmethod
    def get_format_types(cls) -> tuple[Type,...]:
        """
        Returns the type of each record attribute
        """
        return cls._types

    @classmethod
    def get_format_attrs(cls) -> tuple[str,...]:
        """
        Returns the name of each record attribute
        """
        return cls._attrs
    
    @classmethod
    def get_record_from_str(cls, s : str) -> "Self._record_type":
        """
        Given a string `s`, will return a record created from that string
        """
        return cls._record_type(*(cls.get_value_from_str(typ, size, x) for typ, size, x in zip(cls._types, cls._sizes, cls.chunk_iter(s, cls._sizes))))

    @classmethod
    def read_records(cls, fpath : str, fixed_width = True) -> list["Self._record_type"]:
        """
        Given a path to a file `fpath`, will return a list of records found in that file
        """
        i = 0
        records = []
        _lgr.info(f'Starting to read records using "{cls.__name__}" from "{fpath}"')
        with open(fpath, 'r') as f:
            while True:
                a = (f.readline()[:cls._record_length]) if not fixed_width else (f.read(cls._record_length))
                
                if len(a) == 0:
                    break
                
                if a.strip().startswith('#'):
                    continue
                
                if a.isspace():
                    continue
                
                records.append(
                    cls.get_record_from_str(a)
                )
                
                if ((i % 10000) == 0):
                    _lgr.info(f'read record {i} ...')
                i += 1
        _lgr.info(f'Completed reading {i} records using "{cls.__name__}" from "{fpath}"')

        return records