
from typing import Iterable, ClassVar, Any, Type, get_origin
import inspect
import dataclasses as dc
import textwrap
import enum

import numpy as np

from .specialisable import Specialisable
from .formatting import format_float
from archnemesis.helpers.io_helper import OutWidth

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)


_variable_wrapper = textwrap.TextWrapper(
    width = OutWidth.get(),
    expand_tabs=True,
    tabsize=2,
    replace_whitespace=False,
    initial_indent='PLACEHOLDER_FIRST',
    subsequent_indent='PLACEHOLDER_AFTER'
)

def get_simple_attributes(obj):
    attrs = {}
    
    slots = []
    

    # Class-level attributes, including inherited ones.
    # reversed MRO means subclass values override base-class values.
    for cls in reversed(type(obj).__mro__):
        for name, value in cls.__dict__.items():
            if name=='__slots__':
                if type(value) is str:
                    slots.append(value)
                elif type(value) is dict:
                    slots.extend(value.keys())
                else:
                    slots.extend(value)
                continue
            
            if name.startswith("_"):
                continue

            if isinstance(value, property):
                continue

            if inspect.isroutine(value):
                continue

            # Optional: exclude other descriptors
            if hasattr(value, "__get__"):
                continue

            attrs[name] = value

    # Instance-level attributes.
    if (x := getattr(obj, "__dict__", None)) is not None:
        for name, value in x.items():
            if name.startswith("_"):
                continue

            if inspect.ismethod(value) or inspect.isfunction(value) or inspect.isroutine(value):
                continue

            attrs[name] = value
    
    if (x := getattr(obj, "__slots__", None)) is not None:
        if type(x) is str:
            slots.append(x)
        elif type(x) is dict:
            slots.extend(x.keys())
        else:
            slots.extend(x)
    

    for name in slots:
        #print(f'{name=} {value=}')
        value = getattr(obj, name)
        if name.startswith("_"):
            continue

        if inspect.ismethod(value) or inspect.isfunction(value) or inspect.isroutine(value):
            continue
        
        if isinstance(value, (StateParam, ConstParam, VarParam)):
            continue

        attrs[name] = value
    
    # Finally grab all values form the instance if we can
    for k in attrs.keys():
        if hasattr(obj, k):
            attrs[k] = getattr(obj,k)
    

    return attrs

def get_from_decendents(cls, item, default=dc.MISSING):
    for x in cls.mro():
        v = x.__dict__.get(item, dc.MISSING)
        if v is not dc.MISSING:
            return v
    
    return default
    


@dc.dataclass
class Param:
    @classmethod
    def iter_class_attrs(cls) -> Iterable[tuple[str,Any]]:
        #for k,t in cls.__annotations__.items():
        
        annos = get_from_decendents(cls, '__annotations__')
        if annos is dc.MISSING:
            return
        
        for k,t in annos.items():
            if get_origin(t) is ClassVar:
                yield (k, getattr(cls,k))
    
    def iter_fields(self) -> Iterable[tuple[str,Any]]:
        for field in dc.fields(self):
            yield (field.name, getattr(self, field.name))
    
    
    
    @staticmethod
    def format_variable(k : str, v : Any, indent : str = '') -> str:
        label = f'|- {k}: '
        width = OutWidth.get() - (len(indent) + len(label))
        
        _variable_wrapper.width = width
        _variable_wrapper.initial_indent = indent + label
        _variable_wrapper.subsequent_indent = indent + '|'+(' '*(len(label)-1))
        
        if isinstance(v, enum.Enum):
            f_str = _variable_wrapper.fill(f'{v!r}')
        elif isinstance(v, np.ndarray):
            array_str = np.array2string(
                v, 
                max_line_width=width, 
                prefix='', 
                threshold=1_000_000, 
                sign=' ', 
                separator=' ', 
                formatter = {'float_kind' : format_float  }
            )
            
            if array_str.count('\n') > 0:
                array_str = array_str[1:-1].replace('\n ', '\n')
                array_str = '[\n' + array_str + '\n]'
        
            f_str = (
                indent 
                + label 
                + array_str.replace('\n', f'\n{_variable_wrapper.subsequent_indent}')
            )
            
        else:
            f_str = _variable_wrapper.fill(f'{v}')
        
        return f_str
        
        
        
        
    def attrs_to_tree(self, indent : str = '') -> str:
        return '\n'.join((
            *(self.format_variable(k,v,indent=indent) for k, v in self.iter_class_attrs()),
            *(self.format_variable(k,v,indent=indent) for k, v in self.iter_fields())
        ))

@dc.dataclass(slots=True)
class ConstParam(Specialisable, Param):
    type : ClassVar[Type]
    description : ClassVar[str]
    unit : ClassVar[str] = 'Unit Not Specified'
    
    v : Any
    
    """
    def __init__(self, v : T):
        if not isinstance(v, self.__expected_types__[0]):
            _lgr.warn(f'ConstParam expected type {self.__expected_types__[0]}, but was set with type {type(v)}. Attempting to coerce...', stacklevel=1)
            try:
                self.v = self.__specialised_types__[0](v)
            except:
                _lgr.error(f'Cannot coerce value `{v}` into type "{self.__specialised_types__[0]}", ConstParam with description {self.description!r} cannot be set')
                raise
        else:
            self.v = v
    """

@dc.dataclass(slots=True)
class VarParam(Specialisable, Param):
    type : ClassVar[Type]
    description : ClassVar[str]
    unit : ClassVar[str] = 'Unit Not Specified'
    
    v : Any
    
    """
    def __init__(self, v : T):
        if not isinstance(v, self.__expected_types__[0]):
            _lgr.warn(f'VarParam expected type {self.__expected_types__[0]}, but was set with type {type(v)}. Attempting to coerce...', stacklevel=1)
            try:
                self.v = self.__specialised_types__[0](v)
            except:
                _lgr.error(f'Cannot coerce value `{v}` into type "{self.__specialised_types__[0]}", VarParam with description {self.description!r} cannot be set')
                raise
        else:
            self.v = v
    """

@dc.dataclass(slots=True)
class StateParam(Specialisable, Param):
    slice : ClassVar[slice]
    description : ClassVar[str]
    unit : ClassVar[str] = 'Unit Not Specified'
    
    v : np.ndarray
    e : np.ndarray
    log : bool = True # Use logarithm when retrieving this stateparam?
    num_diff : bool = False # Use numerical differentiation for this stateparam?
    
    def __init__(self, pair : tuple[np.ndarray, np.ndarray], log : bool = True, num_diff : bool = False):
        #print( 'StateParam::__init__(...)')
        #print(f'{pair=}')
        #print(f'{log=}')
        
        self.v = pair[0]
        self.e = pair[1]
        self.log = log
        self.num_diff = num_diff






import abc
class ParamMeta(abc.ABCMeta):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        
        if not dc.is_dataclass(cls):
            _lgr.debug(f'Class "{cls.__name__}" is not a dataclass, therefore ParamMeta will not automatically grab details for constparam, varparam, or stateparam attributes.')
            return
            
        if name not in ('ParamTrackerMixin','ParamMixin'):
            cls._constparam_names = cls._get_field_names_of_type(ConstParam)
            cls._constparam_types = cls._get_field_types_of_type(ConstParam)
            cls._n_constparams = len(cls._constparam_names)
            
            cls._varparam_names = cls._get_field_names_of_type(VarParam)
            cls._varparam_types = cls._get_field_types_of_type(VarParam)
            cls._n_varparams = len(cls._varparam_names)
            
            cls._stateparam_names = cls._get_field_names_of_type(StateParam)
            cls._stateparam_types = cls._get_field_types_of_type(StateParam)
            cls._n_stateparams = len(cls._stateparam_names)


    def _get_field_names_of_type(cls, type):
        return tuple(k for k,t in cls.__annotations__.items() if (x:=get_origin(t)) != ClassVar and issubclass((x if x is not None else t), type))
    
    def _get_field_types_of_type(cls, type):
        return tuple(t for k,t in cls.__annotations__.items() if (x:=get_origin(t)) != ClassVar and issubclass((x if x is not None else t), type))

@dc.dataclass(slots=True)
class ParamMixin(metaclass=ParamMeta):
    _n_constparams : ClassVar[int] = 0
    _constparam_names : ClassVar[tuple[str,...]] = tuple()
    _constparam_types : ClassVar[tuple[Type,...]] = tuple()
    
    _n_varparams : ClassVar[int] = 0
    _varparam_names : ClassVar[tuple[str,...]] = tuple()
    _varparam_types : ClassVar[tuple[Type,...]] = tuple()
    
    _n_stateparams : ClassVar[int] = 0
    _stateparam_names : ClassVar[tuple[str,...]] = tuple()
    _stateparam_types : ClassVar[tuple[Type,...]] = tuple()
    
    @classmethod
    def iter_stateparam_names(cls) -> Iterable[type[StateParam]]:
        yield from cls._stateparam_names
    
    @classmethod
    def iter_constparam_names(cls) -> Iterable[type[ConstParam]]:
        yield from cls._constparam_names
    
    @classmethod
    def iter_varparam_names(cls) -> Iterable[type[VarParam]]:
        yield from cls._varparam_names
    
    @classmethod
    def iter_stateparam_types(cls) -> Iterable[tuple[str,type[StateParam]]]:
        yield from zip(cls._stateparam_names, cls._stateparam_types)
    
    @classmethod
    def iter_constparam_types(cls) -> Iterable[tuple[str,type[ConstParam]]]:
        yield from zip(cls._constparam_names, cls._constparam_types)
    
    @classmethod
    def iter_varparam_types(cls) -> Iterable[tuple[str,type[VarParam]]]:
        yield from zip(cls._varparam_names, cls._varparam_types)
    
    def __post_init__(self):
        """
        Change from values to the parameter types
        """
        for k,t in self.iter_stateparam_types():
            setattr(self, k, t(getattr(self,k)))
            
        for k,t in self.iter_constparam_types():
            setattr(self, k, t(getattr(self,k)))
            
        for k,t in self.iter_varparam_types():
            setattr(self, k, t(getattr(self,k)))
    
    def iter_simple_attr_items(self) -> Iterable[tuple[str,Any]]:
        yield from ((k,v) for k,v in get_simple_attributes(self).items())
    
    def iter_stateparam_name_type_value(self) -> Iterable[tuple[str,StateParam]]:
        yield from ((x,t,getattr(self, x)) for x,t in zip(self._stateparam_names, self._stateparam_types))
    
    def iter_constparam_name_type_value(self) -> Iterable[tuple[str,ConstParam]]:
        yield from ((x,t,getattr(self, x)) for x,t in zip(self._constparam_names, self._constparam_types))
    
    def iter_varparam_name_type_value(self) -> Iterable[tuple[str,VarParam]]:
        yield from ((x,t,getattr(self, x)) for x,t in zip(self._varparam_names, self._varparam_types))
    
    def iter_stateparam_items(self) -> Iterable[tuple[str,StateParam]]:
        yield from ((x,getattr(self, x)) for x in self._stateparam_names)
    
    def iter_constparam_items(self) -> Iterable[tuple[str,ConstParam]]:
        yield from ((x,getattr(self, x)) for x in self._constparam_names)
    
    def iter_varparam_items(self) -> Iterable[tuple[str,VarParam]]:
        yield from ((x,getattr(self, x)) for x in self._varparam_names)

    def iter_stateparam_values(self) -> Iterable[np.ndarray]:
        yield from (getattr(self, x).v for x in self._stateparam_names)
    
    def iter_stateparam_errors(self) -> Iterable[np.ndarray]:
        yield from (getattr(self, x).e for x in self._stateparam_names)
    
    def iter_constparam_values(self) -> Iterable[Any]:
        yield from (getattr(self, x).v for x in self._constparam_names)
    
    def iter_varparam_values(self) -> Iterable[Any]:
        yield from (getattr(self, x).v for x in self._varparam_names)
    
    def iter_stateparam_objs(self) -> Iterable[StateParam]:
        yield from (getattr(self, x) for x in self._stateparam_names)
    
    def iter_constparam_objs(self) -> Iterable[ConstParam]:
        yield from (getattr(self, x) for x in self._constparam_names)
    
    def iter_varparam_objs(self) -> Iterable[VarParam]:
        yield from (getattr(self, x) for x in self._varparam_names)
    
    
