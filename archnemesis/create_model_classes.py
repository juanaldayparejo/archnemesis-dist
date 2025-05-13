import os
import os.path
import re
import inspect

import textwrap


def get_indented_block_after_line_regex(block_start):
	return re.compile(
		r'(?P<start>(?P<indent> *)' + block_start + r'\n)'
			+ r'(?P<block>'
				+r'('
					+r'( *\n)' # empty lines don't stop a block
					+r'|( *#.*\n)' # comment lines don't stop a block
					+r'|((?P=indent) +.*\n)' # all code must be more indented than 'block_start'
				+r')*'
			+r')',
	)


def return_default_with_warning(fmt, adict, key, default, warning):
	lines = []
	if key not in adict:
		lines.append(f'_lgr.warning({warning})')
	lines.append(fmt.format(adict.get(key,default)))
	return '\n'.join(lines)


def main():
	from archnemesis.Variables_0_old import Variables_0
	from archnemesis.ForwardModel_0_old import ForwardModel_0
	import archnemesis.Models_old as Models_old

	models = dict((x[0][len('model'):], x[1]) for x in inspect.getmembers(Models_old, lambda x: inspect.isfunction(x) and x.__name__.startswith('model')))
	print(f'{models=}')
	
	read_apr_fn = Variables_0.read_apr
	#print(f'{read_apr_fn=}')
	
	calc_nxvar_fn = Variables_0.calc_NXVAR
	
	subproftretg_fn = ForwardModel_0.subprofretg
	#print(f'{subproftretg_fn=}')
	
	subspecret_fn = ForwardModel_0.subspecret
	
	
	
	models_code = dict(('-1' if id == 'm1' else id, inspect.getsource(x)) for id, x in models.items())
	read_apr_fn_code = inspect.getsource(read_apr_fn)
	calc_nxvar_fn_code = inspect.getsource(calc_nxvar_fn)
	subproftretg_fn_code = inspect.getsource(subproftretg_fn)
	subspecret_fn_code = inspect.getsource(subspecret_fn)
	
	print(f'{models_code["444"]=}')
	
	
	for id in models_code.keys():
		replace_def_regex = re.compile(f'def\\s*model{id.replace("-","m")}\\(')
		models_code[id] = textwrap.dedent(replace_def_regex.sub('def __call__(cls, ', models_code[id]))
	
	
	model_from_varident_regexs =dict()
	for id in models_code.keys():
		if_statement = r'(el)?if +(?P<condition>varident[[] *i *, *2 *] *== *' + id + r') *: *'
		model_from_varident_regexs[id] = get_indented_block_after_line_regex(if_statement)
	
	
	model_calc_nxvar_regexs = dict()
	for id in models_code.keys():
		if_statement = r'(el)?if +(?P<condition>imod *== *' + id + r') *: *'
		model_calc_nxvar_regexs[id] = get_indented_block_after_line_regex(if_statement)
	
	
	
	model_subprofretg_regexs =dict()
	for id in models_code.keys():
		if_statement = r'(el)?if +(?P<condition>self[.]Variables[.]VARIDENT[[] *ivar *, *(0|2) *] *== *' + id + r') *: *'
		model_subprofretg_regexs[id] = get_indented_block_after_line_regex(if_statement)
	
	#print(f'{read_apr_fn_code=}')
	
	valid_model_ids = set()
	
	indent_re_1 = re.compile(r'^(?P<indent> {4})* {1,3}(?P<text>\S)', re.M)
	sanitise_varident_condition_re = re.compile(r'[[] *i *,')
	sanitise_variables_re_1 = re.compile(r'self[.]VARIDENT[[] *\w* *,')
	sanitise_subprofretg_condition_re = re.compile(r'[[] *ivar *,')
	self_re= re.compile(r'self[.]')
	
	
	models_from_varident = dict()
	models_from_varident_condition = dict()
	for id, regex in model_from_varident_regexs.items():
		valid_model_ids.add(id)
		
		print(f'Geting varident code for {id}')
		match = regex.search(read_apr_fn_code)
		if match is None:
			continue
		condition = match.group('condition')
		condition = sanitise_varident_condition_re.sub('[', condition)
		models_from_varident_condition[id] = condition
		
		code = textwrap.dedent(match.group('block'))
		code = indent_re_1.sub(r'\g<indent>\g<text>', code) # remove any uneven indent
		code = re.sub(r'varident[[] *i *,', 'varident[', code)
		code = re.sub(r'varparam[[] *i *,', 'varparam[', code)
		code = sanitise_variables_re_1.sub('varident[', code)
		code = self_re.sub('variables.', code)
		models_from_varident[id] = code
	
	models_calc_nxvar = dict()
	models_calc_nxvar_condition = dict()
	for id, regex in model_calc_nxvar_regexs.items():
		valid_model_ids.add(id)
		print(f'Getting calc_NXVAR code for {id}')
		
		match = regex.search(calc_nxvar_fn_code)
		if match is None:
			continue
		
		condition = match.group('condition')
		condition = condition.replace('imod', 'varident[2]')
		condition = self_re.sub('variables.', condition)
		models_calc_nxvar_condition[id] = condition
		
		code = textwrap.dedent(match.group('block'))
		code = indent_re_1.sub(r'\g<indent>\g<text>', code) #  remove any uneven indent
		code = re.sub(r'(?P<indent> *)nxvar[[] *\S *] *= *(?P<value>.*)', r'\g<indent>return \g<value>', code)
		code = code.replace('ipar3', 'varident[2]')
		code = code.replace('ipar2', 'varident[1]')
		code = code.replace('ipar', 'varident[0]')
		code = code.replace('imod', 'varident[2]')
		code = sanitise_variables_re_1.sub('varident[', code)
		code = self_re.sub('variables.', code)
		models_calc_nxvar[id] = code
	
	
	print(f'{models_calc_nxvar["444"]=}')
	
	models_subprofretg = dict()
	models_subprofretg_patch = dict()
	models_subprofretg_condition = dict()
	for id, regex in model_subprofretg_regexs.items():
		valid_model_ids.add(id)
		
		print(f'Geting subprofretg code for {id}')
		fn_code = subproftretg_fn_code+'\n'+subspecret_fn_code
		match = regex.search(fn_code)
		if match is None:
			#if id in valid_model_ids:
			#	valid_model_ids.remove(id)
			#	continue
			continue
		
		condition = match.group('condition')
		condition = sanitise_subprofretg_condition_re.sub('[', condition)
		condition = condition.replace('self.Variables.VARIDENT', 'varident')
		models_subprofretg_condition[id] = condition
		
		code = textwrap.dedent(match.group('block'))
		code = indent_re_1.sub(r'\g<indent>\g<text>', code) #  remove any uneven indent
		code = self_re.sub('forward_model.', code)
		code = re.sub(r'model'+id.replace('-','m'), 'cls.__call__', code)
		code = re.sub(r'model(?P<str_id>m?\d+)', r'Model\g<str_id>.__call__', code)
		models_subprofretg[id] = code
		
		# search for another match, this is the patch part
		match = regex.search(fn_code[match.end('block'):])
		if match is None:
			continue
		
		code = textwrap.dedent(match.group('block'))
		code = indent_re_1.sub(r'\g<indent>\g<text>', code) #  remove any uneven indent
		code = self_re.sub('forward_model.', code)
		code = re.sub(r'model'+id.replace('-','m'), 'cls.__call__', code)
		code = re.sub(r'model(?P<str_id>m?\d+)', r'Model\g<str_id>.__call__', code)
		models_subprofretg_patch[id] = code
		
		
	
	
	
	
	
	
	
	
	NotImp_v = 'NotImplemented'
	NotImp_r = 'raise NotImplementedError'

	indent = '    '
	
	model_base_class = """
class ModelBase:
    id : int = None
    
    def __init__(self, variables : "Variables_0", varident : np.ndarray[[3],int], NPRO : int, nlocations : int, i_state_vector_start : int):
        self.state_vector_start = i_state_vector_start
        self.n_state_vector_entries = self.get_nxvar(variables, varident, NPRO, nlocations)
        self.state_vector_end = self.state_vector_start + self.n_state_vector_entries
    
    def get_apriori_state_vector_slice(self, variables : "Variables_0"):
        return (
            variables.XA[self.state_vector_start:self.state_vector_end],
            variables.LX[self.state_vector_start:self.state_vector_end],
            variables.FIX[self.state_vector_start:self.state_vector_end],
            variables.NUM[self.state_vector_start:self.state_vector_end],
            variables.DSTEP[self.state_vector_start:self.state_vector_end],
        )
    
    def get_posterior_state_vector_sice(self, variables : "Variables_0"):
        return (
            variables.XN[self.state_vector_start:self.state_vector_end],
            variables.LX[self.state_vector_start:self.state_vector_end],
            variables.FIX[self.state_vector_start:self.state_vector_end],
            variables.NUM[self.state_vector_start:self.state_vector_end],
            variables.DSTEP[self.state_vector_start:self.state_vector_end],
        )
    """
	init_fn = """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    """
	
	with open(f'{os.path.dirname(os.path.realpath(__file__))}/ModelClass.py', 'w') as f:
		f.write('\n'.join((
			f'import numpy as np',
			f'from typing import IO',
			f'',
			f'from archnemesis.Scatter_0 import kk_new_sub',
			f'',
			f'import logging',
			f'_lgr = logging.getLogger(__name__)',
			f'_lgr.setLevel(logging.WARN)',
			f'',
			f'{model_base_class}',
			f''
		)))
		f.write('\n\n')
		
		for id in sorted(valid_model_ids, key=lambda x: int(x)):
			f.write(f'class Model{'m1' if id == '-1' else id}(ModelBase):')
			f.write('\n')
			f.write(f'{indent}id : int = {id}')
			
			f.write('\n\n')
			f.write(init_fn)
			f.write('\n\n')
			
			f.write('\n'.join((
				f'{indent}@classmethod',
				f'{indent}def is_varident_valid(',
				f'{3*indent}cls,',
				f'{3*indent}varident : np.ndarray[[3],int],',
				f'{2*indent}) -> bool:',
				f'{textwrap.indent(
					return_default_with_warning("return {}", models_subprofretg_condition, id, "False", "\"Model with id "+id+" is not implemented\""),
					2*indent)
				}'
			)))
			
			f.write('\n\n')
			
			f.write(textwrap.indent(
				'\n'.join((
					'@classmethod',
					models_code.get(id, NotImp_r)
				)), 
			indent))
			
			f.write('\n\n')
			
			f.write('\n'.join((
				f'{indent}@classmethod',
				f'{indent}def from_apr_to_state_vector(',
				f'{3*indent}cls,',
				f'{3*indent}variables : "Variables_0",',
				f'{3*indent}f : IO,',
				f'{3*indent}varident : np.ndarray[[3],int],',
				f'{3*indent}varparam : np.ndarray[["mparam"],float],',
				f'{3*indent}ix : int,',
				f'{3*indent}lx : np.ndarray[["mx"],int],',
				f'{3*indent}x0 : np.ndarray[["mx"],float],',
				f'{3*indent}sx : np.ndarray[["mx","mx"],float],',
				f'{3*indent}varfile : list[str],',
				f'{3*indent}npro : int,',
				f'{3*indent}nlocations : int,',
				f'{3*indent}sxminfac : float,',
				f'{2*indent}) -> int:'
			)))
			f.write('\n')
			f.write(textwrap.indent(models_from_varident.get(id,NotImp_r), 2*indent))
			f.write(f'{2*indent}return ix')
			
			f.write('\n\n')
			
			f.write('\n'.join((
				f'{indent}@classmethod',
				f'{indent}def calculate_from_state_vector(',
				f'{3*indent}cls,',
				f'{3*indent}forward_model : "ForwardModel_0",',
				f'{3*indent}ix : int,',
				f'{3*indent}ipar : int,',
				f'{3*indent}ivar : int,',
				f'{3*indent}xmap : np.ndarray,',
				f'{2*indent}) -> int:'
			)))
			f.write('\n')
			f.write(textwrap.indent(models_subprofretg.get(id,NotImp_r) +'\nreturn ix', 2*indent))
			
			f.write('\n\n')
			
			f.write('\n'.join((
				f'{indent}@classmethod',
				f'{indent}def patch_from_state_vector(',
				f'{3*indent}cls,',
				f'{3*indent}forward_model : "ForwardModel_0",',
				f'{3*indent}ix : int,',
				f'{3*indent}ipar : int,',
				f'{3*indent}ivar : int,',
				f'{3*indent}xmap : np.ndarray,',
				f'{2*indent}) -> int:'
			)))
			f.write('\n')
			f.write(textwrap.indent(models_subprofretg_patch.get(id,"ix = ix + forward_model.Variables.NXVAR[ivar]") +'\nreturn ix', 2*indent))
			
			f.write('\n\n')
			
			f.write('\n'.join((
				f'{indent}@classmethod',
				f'{indent}def get_nxvar(',
				f'{3*indent}cls,',
				f'{3*indent}variables : "Variables_0",',
				f'{3*indent}varident : np.ndarray[[3],int],',
				f'{3*indent}NPRO : int,',
				f'{3*indent}nlocations : int,',
				f'{2*indent}) -> int:',
				f'{textwrap.indent(models_calc_nxvar.get(id,NotImp_r), 2*indent)}'
			)))
			
			f.write('\n\n')
			
			
		

if __name__=='__main__':
	main()


