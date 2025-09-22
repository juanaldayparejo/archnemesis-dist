import dataclasses as dc
from typing import get_origin, get_args, ClassVar, Iterator, Self


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)



@dc.dataclass(repr=False)
class ExomolColFileFormat:
    """
    Classes that inherit from this class are stored in a table-like format. With one row per class instance,
    and each column in the row denoting a class attribute.
    
    This class provides the methods `from_col_iter` to create an instance from an iterator over a single row (i.e.,
    that provides each column in turn). And `from_line_itr` to create an instance from the next value of a line iterator (that provides
    each row).
    """
    _field_for_n_entries : ClassVar[dict[str,str]] = dict()

    @classmethod
    def from_col_itr(cls, 
            col_itr : Iterator[str]
    ) -> Self:
        """
        Create an instance of the class (or subclass) from a `col_itr`
        
        ## ARGUMENTS ##
            col_itr : Iterator[str]
                An object that iterates through the columns of a table, returning each entry in a single row as a string.
        
        ## RETURNS ##
            A newly created instance.
        """
        args = dict()

        for i, field in enumerate(dc.fields(cls)):
            _lgr.debug(f'{field}')
            
            t_origin = get_origin(field.type)
            t_args = get_args(field.type)
            
            if t_origin in (list, tuple):
                assert len(t_args) == 1 or ((len(t_args) == 2) and (t_args[1]==...)) , "Only supports tuples or lists of a single type at the moment"
                
                entries = []
                t_arg = t_args[0]
                
                if dc.is_dataclass(t_arg):
                    for j in range(args[cls._field_for_n_entries[field.name]]):
                        entries.append(t_arg.from_col_itr(col_itr))
                else:
                    for j in range(args[cls._field_for_n_entries[field.name]]):
                        c = next(col_itr)
                        _lgr.debug(c)
                        entries.append(t_arg(c.strip()))
                
                args[field.name] = entries if t_origin == list else tuple(entries)
            
            else:
                c = next(col_itr)
                _lgr.debug(c)
                args[field.name] = field.type(c.strip())

        return cls(**args)
    
    @classmethod
    def from_line_itr(cls, 
            next_line : None | str, 
            line_itr : Iterator[str]
    ) -> Self:
        """
        Create an instance of the class (or subclass) from a `line_itr` object.
        
        ## ARGUMENTS ##
            next_line : None | str
                The next line to process, if `None` will get the next line from `line_itr`
            line_itr : Iterator[str]
                An interator that returns each row in a table.
        
        ## RETURNS ##
            tuple[None, Instance[Self]]
                A tuple of `None` (the `next_line` to be processed by a line iterator) and
                the instance from the row.
        """

        if next_line is None:
            next_line = next(line_itr)
        col_itr = (x for x in next_line.split())

        return None, cls.from_col_itr(col_itr)