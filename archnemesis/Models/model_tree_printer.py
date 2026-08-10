

import textwrap

import numpy as np

from archnemesis.helpers.io_helper import OutWidth


class ModelTreePrinter:
    
    _description_wrapper_0 = textwrap.TextWrapper(
        width = OutWidth.get(),
        expand_tabs=True,
        tabsize=2,
        replace_whitespace=False,
        initial_indent='|- description: ',
        subsequent_indent=('|  ' + ' '*(len('|- description: ')-3))
    )
    _description_wrapper_1 = textwrap.TextWrapper(
        width = OutWidth.get(),
        expand_tabs=True,
        tabsize=2,
        replace_whitespace=False,
        initial_indent=('|  ' + ' '*(len('|- description: ')-3)),
        subsequent_indent=('|  ' + ' '*(len('|- description: ')-3))
    )
    

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

