import dataclasses as dc
from typing import ClassVar
import textwrap

class TreePrinter:
    """
    Defines a `__repr__` attribute for a dataclass that prints the attributes of a dataclass in a tree-like format.
    
    NOTE: use `repr=False` when defining a dataclass as a subclass of this, otherwise dc.dataclass will overwrite the
    `__repr__` defined here.
    """
    _indent : ClassVar[str] = '    '
    
    def __repr__(self):
        a = f'{self.__class__.__name__}:'
        b = []
        for field in dc.fields(self):
            #print(f'{field.name=}')
            value = getattr(self,field.name)
            value_str = repr(value)

            if isinstance(value, (list,tuple)):
                value_str = '['
                for item in value:
                    #print(f'{isinstance(item, ExomolTreePrinter)=}')
                    #print(f'{dc.is_dataclass(item)=}')
                    #if dc.is_dataclass(item):
                    if isinstance(item, TreePrinter):
                        z = textwrap.indent(repr(item), 2*self._indent)
                        value_str += f'\n{z}'
                    else:
                        value_str += f'\n{2*self._indent}{repr(item)}'
                value_str += f'\n{self._indent}]'
            
            b.append(f'{field.name} : {value_str}')
                
            #if isinstance(value, ExomolTreePrinter):
            #    value_str = '\n' + textwrap.indent(value_str, '\t')
            #b.append(f'{field.name} : {value_str}')

        return a + f'\n{self._indent}' + (f'\n{self._indent}').join(b)