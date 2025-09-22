import dataclasses as dc
from typing import get_origin, get_args, Any, ClassVar, Callable, Iterator, Self

from .col_format import ExomolColFileFormat

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)



@dc.dataclass(repr=False)
class ExomolLineFileFormat:
    """
    Dataclasses that inherit from this class are assumed to be stored in a "one line per attribute" format, 
    generally with one class instance per file.
    
    It is assumed that each line has the following format:
    
    ```
    value # comment
    ```
    
    where `value` is the value of the attribute and `comment` is an explanatory string.
    
    The exact format of a file is given by the ordering of the attributes in the (sub)class, subject to the
    modifications dictated by the value of this class's class attributes (see below).
    
    This class provides the method `from_line_itr` to create an instance from a `line_itr` object (something that
    iterates through the lines that contain the attributes of an instance).
    
    ## CLASS ATTRIBUTES ##
    
        _field_for_n_entries : ClassVar[dict[str,str]] = dict()
            A class variable that maps the names of attributes that hold a list of other class instances
            to a field that holds how many class instances are held by that attribute.
            
            E.g. {'molecule_info' : 'n_mol'} maps the attribute name 'molecule_info' (which would hold
            a list of instances of the class `ExomolMoleculeInformation`) to the attribute name 'n_mol'
            (which would be an integer that holds the number of molecules in the 'molecule_info' list).
            
        _skip_if_comment_predicate : ClassVar[dict[str, tuple[Any,Callable[[str],bool]]]] = dict()
            A class variable that maps the names of attributes to a tuple. The tuple consists of a
            value (of `Any` type) and a Callable[[str],bool] which are called the `default` and 
            `predicate` respectively.
            
            When populating the attributes, if the attribute name is in this dictionary, the `comment` 
            of a `value` is passed to the `predicate`. If the `predicate` returns `True` the `default`
            is assigned to the attribute (and normal processing is "skipped"), if the `predicate` 
            returns `False` the attribute is processed in the normal way.
            
            NOTE: This is checked before `_field_for_n_entries`
    
    ## EXAMPLE ##
    
        The following example subclass,
    
        ```
        import dataclasses as dc
        
        dc.dataclass()
        class ContactInfoExample(ExomolLineFileFormat):
            
            _field_for_n_entries : ClassVar[dict[str,str]]= {
                "address" : 
                "n_address_lines"
            }
            
            _skip_if_comment_predicate : ClassVar[dict[str,tuple[Any,Callable[[str],bool]]]] = {
                "landline_number" : (None, lambda comment: "Landline" not in comment)
            }
            
            
            version : int
            mobile_number : int
            landline_number : int
            n_address_lines : int
            address : tuple[str,...]
        ```
        
        would be able to read files like the following examples:
        
            ```contact_info_example_file_1.txt
            0                   # version number
            01234567890         # number
            5                   # number of lines in address
            0 example street    # address
            example town        # address
            example city        # address
            example county      # address
            EM01 2PL            # postcode
            ```
            
            ```contact_info_example_file_2.txt
            0                   # version number
            01234567890         # number
            09876543210         # Landline number
            3                   # number of lines in address
            0 example street    # address
            example city        # address
            EM01 2PL            # postcode
            ```
        
    """
    _field_for_n_entries : ClassVar[dict[str,str]] = dict()

    _skip_if_comment_predicate : ClassVar[dict[str, tuple[Any,Callable[[str],bool]]]] = dict()

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
                An interator that returns each line to be processed.
        
        ## RETURNS ##
            tuple[None, Instance[Self]]
                A tuple of `None` (the `next_line` to be processed by a line iterator) and
                the instance created.
        """
        args = dict()

        for i, field in enumerate(dc.fields(cls)):
            
            if next_line is None:
                try:
                    next_line = next(line_itr)
                except StopIteration:
                    args[field.name] = field.type()
                    break
                    

            _lgr.debug(field.name)
            

            value, comment = next_line.split('#', 1)

            skip_default, skip_predicate = cls._skip_if_comment_predicate.get(field.name, (None, lambda x: False))
            if skip_predicate(comment):
                args[field.name] = skip_default
                continue
            
            t_origin = get_origin(field.type)
            t_args = get_args(field.type)
            _lgr.debug(f'{t_origin=}')
            _lgr.debug(f'{t_args=}')
            
            if t_origin in (list, tuple):
                assert len(t_args) == 1 or ((len(t_args) == 2) and (t_args[1]==...)) , "Only supports tuples or lists of a single type at the moment"
                
                entries = []
                t_arg = t_args[0]
                n_entry_getter = cls._field_for_n_entries[field.name]
                if callable(n_entry_getter):
                    a = n_entry_getter(args)
                    if type(a) is str:
                        n_entries = args[a]
                    elif type(a) is int:
                        n_entries = a
                elif type(n_entry_getter) is str:
                    n_entries = args[n_entry_getter]
                elif type(n_entry_getter) is int:
                    n_entries = n_entry_getter
                else:
                    raise ValueError(f'Unknown method for getting number of entries for {field=}')

                _lgr.debug(f'{n_entries=}')
                
                if hasattr(t_arg, 'from_line_itr'):
                    for j in range(n_entries):
                        next_line, instance = t_arg.from_line_itr(next_line, line_itr)
                        entries.append(instance)
                else:
                    for j in range(n_entries):
                        if next_line is None:
                            next_line = next(line_itr)
                        _lgr.debug(next_line)
                        value, comment = next_line.split('#', 1)
                        entries.append(t_arg(value.strip()))
                        next_line = None
                
                args[field.name] = entries if t_origin == list else tuple(entries)
            
            elif isinstance(field.type, type) and issubclass(field.type, (ExomolColFileFormat, ExomolLineFileFormat)):
                next_line, instance = field.type.from_line_itr(next_line, line_itr)
                args[field.name] = instance
            
            elif type(field.type) is str and any([x in field.type for x in ("ExomolColFileFormat", "ExomolLineFileFormat")]):
                if "ExomolColFileFormat" in field.type:
                    next_line, instance = ExomolColFileFormat.from_line_itr(next_line, line_itr)
                    args[field.name] = instance
                elif "ExomolLineFileFormat" in field.type:
                    next_line, instance = ExomolLineFileFormat.from_line_itr(next_line, line_itr)
                    args[field.name] = instance
                else:
                    raise RuntimeError(f'Unknown type {field.type} when dispatching {cls.__name__} reader to attribute {field.name} reader.')
            
            else:
                if next_line is None:
                    next_line = next(line_itr)
                _lgr.debug(next_line)
                args[field.name] = field.type(value.strip())
                next_line = None

        return (next_line, cls(**args))
