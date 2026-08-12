"""
Enables specialising of a class (in a similar manner to C++ templates). 

Instead of only types being passed to classes via square brackets,
can also pass values via `.using()` method that will specialise class attributes.

"""

from typing import get_args, get_origin, ClassVar, Self, Any, Iterable
import dataclasses as dc


_SPECIALISED_CLASS_CACHE = dict()

def type_args_replace_type_params(typ, type_param_dict):
	"""
	For a type like `ClassVar[tuple[T,float,list[Y]]]`, with type template arguments `T` and `Y`,
	will replace the type template arguments with their mappings in `type_param_dict`.
	
	# EXAMPLE #
	```
	from typing import ClassVar, TypeVar
	T = TypeVar["T"]
	Y = TypeVar["Y"]
	
	templated_type = ClassVar[tuple[T,float,list[Y]]]
	type_param_dict = {
		T : int,
		Y : bool
	}
	
	specialised_type = type_args_replace_type_params(
		templated_type,
		type_param_dict
	)
	
	print(specialised_type) # ClassVar[tuple[int,float,list[bool]]]
	```
	
	
	"""
	#print(f'{type_param_dict=}')
	#print(f'{typ=}')
	if (o := get_origin(typ)) is not None:
		o = type_param_dict.get(typ, o)
		#print(f'{o=}')
		#print(f'{get_args(typ)=}')
		
		new_args = tuple(type_args_replace_type_params(x, type_param_dict) for x in get_args(typ))
		
		if len(new_args) == 1:
			return o[new_args[0]]
		else:
			return o[*new_args] 
	else:
		return type_param_dict.get(typ, typ)





class Specialisable:

	@classmethod
	def __class_getitem__(cls, *args):
		#print(f'Specialisable::__class_getitem__ {cls=} {args=}')
		
		#print('cls.__dict__:')
		#for k,v in cls.__dict__.items():
		#	print(f'    {k} = {v}')
		
		cache_identity = (cls,args)
		cache_result = _SPECIALISED_CLASS_CACHE.get(cache_identity,None)
		if cache_result is not None:
			return cache_result
		
		specialisable_type_parameters = getattr(cls, '__parameters__', None)
		assert specialisable_type_parameters is not None, "Cannot subscript a class that has no type parameters"
		
		assert len(specialisable_type_parameters) == len(args), "Cannot partially specialise a class with type parameters"
		
		subclass_dict = dict(**cls.__dict__)
		subclass_dict['__annotations__'] = dict((k, type_args_replace_type_params(t,dict(zip(specialisable_type_parameters, args)))) for k,t in subclass_dict['__annotations__'].items())
		slots = subclass_dict.pop('__slots__',tuple())
		subclass_dict.pop('__match_args__',None)
		for k in slots:
			subclass_dict.pop(k)
		
		#print('subclass_dict:')
		#for k,v in subclass_dict.items():
		#	print(f'    {k} = {v}')
		
		specialisation_str = '['+", ".join(f'{x}' for x in args)+']'
		specialised_subclass = type(
			f'{cls.__name__}{specialisation_str}',
			(cls,),
			subclass_dict
		)
		
		_SPECIALISED_CLASS_CACHE[cache_identity] = specialised_subclass
		
		return specialised_subclass

	@classmethod
	def using(cls, *args, **kwargs) -> type[Self]:
		#print(f'Specialisable::using {cls=} {args=} {kwargs=}')
		
		# sort args and kwargs for the class first
		
		class_attr_ordering = cls._get_class_attr_ordering(require_classvar=dc.is_dataclass(cls))
		class_attr_overrides = tuple(cls._get_class_attr_overrides(class_attr_ordering, args, kwargs))
		
		cache_identity = (cls,class_attr_overrides)
		cache_result = _SPECIALISED_CLASS_CACHE.get(cache_identity,None)
		
		#print(f'{cls.__name__}')
		#print(f'  {class_attr_ordering=}')
		#print(f'  {class_attr_overrides=}')
		
		if cache_result is not None:
			return cache_result
		
		if dc.is_dataclass(cls):
			specialised_subclass =  cls._using_dataclass(class_attr_ordering, class_attr_overrides)
		else:
			specialised_subclass =  cls._using(class_attr_ordering, class_attr_overrides)
		
		_SPECIALISED_CLASS_CACHE[cache_identity] = specialised_subclass
		
		return specialised_subclass
	
	@classmethod
	def _get_class_attr_overrides(
			cls, 
			class_attr_ordering : tuple[str,...], 
			args : tuple[Any,...], 
			kwargs : dict[str,Any],
	) -> Iterable[Any]:
		i = 0
		n_args = len(args)
		for k in class_attr_ordering:
			result = dc.MISSING
			
			if (result := kwargs.pop(k, dc.MISSING)) is not dc.MISSING:
				pass
			elif i<n_args:
				result = args[i]
				i += 1
			else:
				if k in cls.__dict__: # continue if there is a default defined for this class attribute
					continue
				else:
					raise AttributeError(f"Class attribute '{k}' not set and does not have a default value")
			
			yield result
			
		
		assert len(kwargs) == 0, f"Unknown keyword arguments: {tuple(kwargs.keys())}"
		
	@classmethod
	def _get_class_attr_ordering(cls, require_classvar : bool = False):
		anno_class_attrs = tuple(k for k, t in cls.__dict__.get('__annotations__', dict()).items() if (not require_classvar) or (require_classvar and (get_origin(t) is ClassVar)))
		defaulted_class_attrs = tuple(k for k in cls.__dict__.keys() if ((not k.startswith('__')) and (k not in anno_class_attrs)))
		
		#print(f'{cls.__name__}')
		#print(f'  {anno_class_attrs=}')
		#print(f'  {defaulted_class_attrs=}')
		
		# non defaulted class attributes should always be set first so prioritise them if there is no overlap
		return (*anno_class_attrs, *defaulted_class_attrs)
		

	@classmethod
	def _using(
			cls, 
			class_attr_ordering : tuple[str], 
			class_attr_overrides : tuple[Any]
	) -> type[Self]:
		subclass_dict = dict(zip(class_attr_ordering, class_attr_overrides))
		
		if '__slots__' in cls.__dict__:
			subclass_dict['__slots__'] = tuple()
		
		specialised_subclass = type(
			f'{cls.__name__}{{{", ".join(f"{k}={v}" for k,v in subclass_dict.items())}}}',
			(cls,),
			subclass_dict
		)
		
		return specialised_subclass

	@classmethod
	def _using_dataclass(
			cls, 
			class_attr_ordering : tuple[str], 
			class_attr_overrides : tuple[Any]
	) -> type[Self]:
		#print(f'{args=}')
		#print(f'{kwargs=}')
		
		def passthrough_init(self, *args, **kwargs):
			cls.__init__(self, *args, **kwargs)
		
		subclass_dict = dict(zip(class_attr_ordering, class_attr_overrides))
		
		subclass_dict['__init__'] = passthrough_init
		
		#print('subclass_dict:')
		#for k,v in subclass_dict.items():
		#	print(f'    {k} : {v}')
		
		specialised_subclass = dc.dataclass(
			type(
				f'{cls.__name__}{{{", ".join(f"{k}={v}" for k,v in subclass_dict.items())}}}',
				(cls,),
				subclass_dict
			),
			slots = '__slots__' in cls.__dict__,
		)
		
		return specialised_subclass
