
from typing import Self, ClassVar, get_origin
import inspect

import dataclasses as dc

def get_class_attr_names(cls):
	
	if dc.is_dataclass(cls):
		class_attr_names = tuple(k for k,v in cls.__annotations__.items() if get_origin(v) is ClassVar)
	else:
		class_attr_names_a= (k for k,v in cls.__annotations__.items())
		class_attr_names_d = (k for k,v in cls.__dict__.items() if not k.startswith('__') and not inspect.isroutine(v))
		
		# Get in desired order, as things without default values will be filled first "class_attr_names_a" should take priority
		class_attr_names_no_defaults = []
		class_attr_names_with_defaults = []
		
		i = 0
		for x in class_attr_names_a:
			if x not in class_attr_names_d:
				class_attr_names_no_defaults.append(x)
			else:
				while class_attr_names_d.index(x) > i:
					class_attr_names_with_defaults.append(class_attr_names_d[i])
					i+=1
				class_attr_names_with_defaults.append(x)
		
		class_attr_names = tuple(class_attr_names_no_defaults + class_attr_names_with_defaults)
	
	return class_attr_names


class LabelledValue:
	
	@classmethod
	def __class_getitem__(cls, target_type) -> type[Self]:
		assert not isinstance(target_type, tuple), "Must pass a single type to ValueHolder"
		
		class_attr_names = get_class_attr_names(cls)
		
		keys_to_keep=(
			'__module__',
			'__annotations__',
			'__doc__',
			*class_attr_names,
		)
		
		new_dict = dict((k,cls.__dict__[k]) for k in keys_to_keep if hasattr(cls,k))
		new_dict['__wrapped_type__'] = target_type
		new_dict['__class_attr_names__'] = class_attr_names
		
		print('new_dict:')
		for k,v in new_dict.items():
			print(f'    {k} : {v}')
		
		value_holder_class = type(
			f'{cls.__name__}[{target_type}]',
			(target_type, cls),
			new_dict
		)
		
		return value_holder_class
	
	
	@classmethod
	def using(cls, *args, **kwargs) -> type[Self]:
		n_args = len(args)
		
		subclass_dict = dict(cls.__dict__)
		assert hasattr(cls, '__wrapped_type__'), 'Cannot set labels on LabelledValue until the type has been specialised'
		
		# Set class attribute defaults
		for i, k in enumerate(cls.__class_attr_names__):
			if i < n_args:
				assert k not in kwargs, f"Cannot set {k} via positional and keyword arguments at the same time"
				subclass_dict[k] = args[i]
			else:
				subclass_dict[k] = kwargs.pop(k, subclass_dict.get(k, dc.MISSING))
			
			assert subclass_dict.get(k, dc.MISSING) is not dc.MISSING, f"Class attribute '{k}' not set and does not have a default value"
		
		assert len(kwargs) == 0, f"Unknown keyword arguments: {tuple(kwargs.keys())}"
		
		
		print('subclass_dict:')
		for k,v in subclass_dict.items():
			print(f'    {k} : {v}')
		
		specialisation_str = '{'+", ".join(f'{k}={subclass_dict[k]!r}' for k in cls.__class_attr_names__)+'}'
		specialised_subclass = type(
			f'{cls.__name__}{specialisation_str}',
			(cls,),
			subclass_dict
		)
		
		return specialised_subclass