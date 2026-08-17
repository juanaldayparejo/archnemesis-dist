

import textwrap
import dataclasses as dc
from typing import NamedTuple, Any

import numpy as np

from archnemesis.helpers.io_helper import OutWidth


def get_from_decendents(cls, item, default=dc.MISSING):
    for x in cls.mro():
        v = x.__dict__.get(item, dc.MISSING)
        if v is not dc.MISSING:
            return v
    
    return default

class Pair(NamedTuple):
    x : Any
    y : Any

@dc.dataclass
class DescriptionList:
    html_class : None | str = None
    header : None | str = None
    
    _elements : list[Pair] = dc.field(default_factory=list, init=False)
    
    @property
    def n_elements(self):
        """
        Get number of elements in description list
        """
        return len(self._elements)
    
    def add(self, x : Pair | Any, y : Any = dc.MISSING):
        """
        Add item to description list
        """
        if isinstance(x, Pair):
            assert y is dc.MISSING, "When adding a pair to a description list, cannot pass a second argument."
            self._elements.add(x)
        else:
            self._elements.append(Pair(x,y))
        return
    
    def html(self) -> str:
        """
        Generate HTML string for description list
        """
        ds = '\n'.join(f'<dt>\n{p.x}\n</dt>\n<dd>\n{p.y if (z:=getattr(p.y, "html", None)) is None else z()}\n</dd>' for p in self._elements)
        if self.header is not None:
            h = f'<p>{self.header}</p>'
            return f'{h}\n<dl>\n{ds}\n</dl>'
        else:
            return ('<dl>' if self.html_class is None else f'<dl class={self.html_class}>') + f'\n{ds}\n</dl>'
    
    
    
    

class ModelTreePrinter:
    
    _description_wrapper_0 = textwrap.TextWrapper(
        width = OutWidth.get(),
        expand_tabs=True,
        tabsize=2,
        replace_whitespace=True,
        initial_indent='|- description: ',
        subsequent_indent=('|  ' + ' '*(len('|- description: ')-3))
    )
    _description_wrapper_1 = textwrap.TextWrapper(
        width = OutWidth.get(),
        expand_tabs=True,
        tabsize=2,
        replace_whitespace=True,
        initial_indent=('|  ' + ' '*(len('|- description: ')-3)),
        subsequent_indent=('|  ' + ' '*(len('|- description: ')-3))
    )
    _description_wrapper_3 = textwrap.TextWrapper(
        width = OutWidth.get(),
        expand_tabs=True,
        tabsize=2,
        replace_whitespace=True,
        initial_indent='',
        subsequent_indent='',
    )
    
    @classmethod
    def html(cls) -> str:
        docstr = textwrap.dedent(cls.__doc__.strip())
        docstr = docstr.replace('\n\n', '\0')
        docstr = docstr.replace('\n', ' ')
        docstr = docstr.replace('\0', '</p><p>')
        
        input_format_str = textwrap.dedent(cls.apr_input_format)
        input_format_str = input_format_str.replace('<', '&lt;')
        input_format_str = input_format_str.replace('>', '&gt;')
        #input_format_str = input_format_str.replace('\n\n', '\0')
        #input_format_str = input_format_str.replace('\n', ' ')
        #input_format_str = input_format_str.replace('\0', '</p><p>')
        
        x = DescriptionList(html_class='model')
        #y = []
        
        x.add('Description', '<p>'+docstr+'</p>')
        x.add('Format of <runname>.apr', '<pre>'+input_format_str+'</pre>')
        
        
        d = DescriptionList(html_class='model-param')
        for k, t in cls.iter_stateparam_types():
            q = DescriptionList(html_class='model-param-attr')
            for name, v in t.iter_class_attrs():
                q.add(name, v.__name__ if isinstance(v, type) else v)
            d.add(k, q)
        if d.n_elements > 0:
            x.add('StateParams', d)
        
        
        d = DescriptionList(html_class='model-param')
        for k, t in cls.iter_constparam_types():
            q = DescriptionList(html_class='model-param-attr')
            for name, v in t.iter_class_attrs():
                q.add(name, v.__name__ if isinstance(v, type) else v)
            d.add(k, q)
        if d.n_elements > 0:
            x.add('ConstParams', d)
        
        
        d = DescriptionList(html_class='model-param')
        for k, t in cls.iter_varparam_types():
            q = DescriptionList(html_class='model-param-attr')
            for name, v in t.iter_class_attrs():
                q.add(name, v.__name__ if isinstance(v, type) else v)
            d.add(k, q)
        if d.n_elements > 0:
            x.add('VarParam', d)
        
        result = f'<details style="margin-bottom: 0.5em;">\n<summary>\n{cls.__name__}\n</summary>\n{x.html()}\n</details>'
        return result
        
    
    @classmethod
    def summary(
            cls,
            indent : str = '', 
    ) -> str:
        docstr = textwrap.dedent(cls.__doc__.strip())
        docstr = docstr.replace('\n\n', '\0')
        docstr = docstr.replace('\n', ' ')
        docstr = docstr.replace('\0', '\n\0\n')
        #docstr_parts = docstr.split('\n')
        docstr_parts = docstr.splitlines()
        
        docstr_lines = list(cls._description_wrapper_0.wrap(docstr_parts[0]))
        for x in docstr_parts[1:]:
            docstr_lines.extend(cls._description_wrapper_1.wrap(x))
        
        
        
        x = []
        y = []
        for k, t in cls.iter_constparam_types():
            #print(f'{k=} {t=}')
            y.append(f'|  |- {k}')
            for name, v in t.iter_class_attrs():
                if isinstance(v, type):
                    y.append(f'|  |  |- {name}: {v.__name__}')
                else:
                    y.append(f'|  |  |- {name}: {v}')
        if len(y) > 0:
            x.append('|- ConstParams:')
            x.extend(y)
            y = []
        
        for k, t in cls.iter_varparam_types():
            #print(f'{k=} {t=}')
            y.append(f'|  |- {k}')
            for name, v in t.iter_class_attrs():
                if isinstance(v, type):
                    y.append(f'|  |  |- {name}: {v.__name__}')
                else:
                    y.append(f'|  |  |- {name}: {v}')
                
        if len(y) > 0:
            x.append('|- VarParams:')
            x.extend(y)
            y = []
        
        for k, t in cls.iter_stateparam_types():
            #print(f'{k=} {t=}')
            y.append(f'|  |- {k}')
            for name, v in t.iter_class_attrs():
                y.append(f'|  |  |- {name}: {v}')
        if len(y) > 0:
            x.append('|- StateParams:')
            x.extend(y)
            y = []
        
        result = indent + f'\n{indent}'.join(
            [
                cls.__name__
            ]
            +docstr_lines
            +x
        )
        
        return result.replace('\0', '')

    def info(
        self,
        apriori_x0 : None | np.ndarray = None, 
        apriori_sx : None | np.ndarray = None,
        posterior_x0 : None | np.ndarray = None,
        posterior_sx : None | np.ndarray = None,
    ) -> str:
        return self.attrs_to_tree(
            apriori_x0 = apriori_x0,
            apriori_sx = apriori_sx,
            posterior_x0 = posterior_x0,
            posterior_sx = posterior_sx
        )
    
    def attrs_to_tree(
            self, 
            indent : str = '', 
            apriori_x0 : None | np.ndarray = None, 
            apriori_sx : None | np.ndarray = None,
            posterior_x0 : None | np.ndarray = None,
            posterior_sx : None | np.ndarray = None,
    ) -> str:
        
        docstr = textwrap.dedent(self.__doc__.strip())
        docstr = docstr.replace('\n\n', '\0')
        docstr = docstr.replace('\n', ' ')
        docstr = docstr.replace('\0', '\n')
        docstr_parts = docstr.split('\n')
        
        docstr_lines = list(self._description_wrapper_0.wrap(docstr_parts[0]))
        for x in docstr_parts[1:]:
            docstr_lines.extend(self._description_wrapper_1.wrap(x))
        
        
        result = '\n'.join((
            indent + self.__class__.__name__,
            indent + (indent+'\n').join(docstr_lines),
            *(indent + f'|- {k}: {v}' for k,v in self.iter_simple_attr_items()),
            indent + '|- ConstParams:',
            *(indent + '|  |- '+k+'\n'+v.attrs_to_tree(indent+'|  |  ') for k,v in self.iter_constparam_items()),
            indent + '|- VarParams:',
            *(indent + '|  |- '+k+'\n'+v.attrs_to_tree(indent+'|  |  ') for k,v in self.iter_varparam_items()),
        ))
        
        
        show_apriori = apriori_x0 is not None and apriori_sx is not None
        show_posterior = posterior_x0 is not None and posterior_sx is not None
        
        if show_apriori or show_posterior:
            stored_stateparam_data = [(x.v, x.e) for k,x in self.iter_stateparam_items()]
        else:
            result += '\n'+indent + '\n'.join((
                indent + '|- StateParams:',
                *(indent + '|  |- '+k+'\n'+v.attrs_to_tree(indent+'|  |  ') for k,v in self.iter_stateparam_items()),
            ))

        if show_apriori:
            self.pull_from_state_vector(apriori_x0)
            self.pull_from_covariance_matrix(apriori_sx)
            result += '\n'+indent + '\n'.join((
                indent + '|- [APRIORI] StateParams:',
                *(indent + '|  |- '+k+'\n'+v.attrs_to_tree(indent+'|  |  ') for k,v in self.iter_stateparam_items()),
            ))
        
        if show_posterior:
            self.pull_from_state_vector(posterior_x0)
            self.pull_from_covariance_matrix(posterior_sx)
            result += '\n'+indent + '\n'.join((
                indent + '|- [POSTERIOR] StateParams:',
                *(indent + '|  |- '+k+'\n'+v.attrs_to_tree(indent+'|  |  ') for k,v in self.iter_stateparam_items()),
            ))
        
        if show_apriori or show_posterior:
            for (value, error), (k, x) in zip(stored_stateparam_data, self.iter_stateparam_items()):
                x.v = value
                x.e = error
        
        return result

