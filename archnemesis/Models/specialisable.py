"""
Enables specialising of a class (in a similar manner to C++ templates). 

Instead of only types being passed to classes via square brackets,
can also pass values via `.using()` method that will specialise class attributes.

"""

from typing import get_args, get_origin, ClassVar, Self
import dataclasses as dc


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
		
		return specialised_subclass

	@classmethod
	def using(cls, *args, **kwargs) -> type[Self]:
		#print(f'Specialisable::using {cls=} {args=} {kwargs=}')
		if dc.is_dataclass(cls):
			return cls._using_dataclass(*args, **kwargs)
		else:
			return cls._using(*args, **kwargs)

	@classmethod
	def _using(cls, *args, **kwargs) -> type[Self]:
		n_args = len(args)
		
		subclass_dict = dict(**cls.__dict__)
		# Set class attribute defaults
		for i, (k,v) in enumerate(subclass_dict.items()):
			if (not k.startswitn('__')) and (not callable(v)):
				if i < n_args:
					assert k not in kwargs, f"Cannot set {k} via positional and keyword arguments at the same time"
					subclass_dict[k] = args[i]
				else:
					subclass_dict[k] = kwargs.pop(k, subclass_dict.get(k, dc.MISSING))
				
				assert subclass_dict.get(k, dc.MISSING) is not dc.MISSING, f"Class attribute '{k}' not set and does not have a default value"
		
		assert len(kwargs) == 0, f"Unknown keyword arguments: {tuple(kwargs.keys())}"
		
		slots = subclass_dict.pop('__slots__',tuple())
		subclass_dict.pop('__match_args__',None)
		for k in slots:
			subclass_dict.pop(k)
		
		specialisation_str = '{'+", ".join(f'{k}={subclass_dict[k]!r}' for k,t in subclass_dict.items() if (not k.startswitn('__')) and (not callable(v)))+'}'
		specialised_subclass = type(
			f'{cls.__name__}{specialisation_str}',
			(cls,),
			subclass_dict
		)
		
		return specialised_subclass

	@classmethod
	def _using_dataclass(cls, *args, **kwargs) -> type[Self]:
		#print(f'{args=}')
		#print(f'{kwargs=}')
		n_args = len(args)
		
		def passthrough_init(self, *args, **kwargs):
			cls.__init__(self, *args, **kwargs)
		
		#print('cls.__dict__:')
		#for k,v in cls.__dict__.items():
		#	print(f'    {k} : {v}')
		
		subclass_dict = dict()
		
		# Set class attribute defaults
		j = 0
		for i, (k,typ) in enumerate(cls.__dict__['__annotations__'].items()):
			if get_origin(typ) is ClassVar:
				v = kwargs.pop(k, dc.MISSING)
				if v is not dc.MISSING:
					subclass_dict[k] = v
				elif n_args > j:
					subclass_dict[k] = args[j]
					j+=1
				else:
					subclass_dict[k] = cls.__dict__.get(k, dc.MISSING)
				
				assert subclass_dict.get(k, dc.MISSING) is not dc.MISSING, f"Class attribute '{k}' not set and does not have a default value"
		
		assert len(kwargs) == 0, f"Unknown keyword arguments: {tuple(kwargs.keys())}"
		
		
		subclass_dict['__init__'] = passthrough_init
		
		#print('subclass_dict:')
		#for k,v in subclass_dict.items():
		#	print(f'    {k} : {v}')
		
		specialised_subclass = dc.dataclass(
			type(
				f'{cls.__name__}{", ".join(f"{k}={v}" for k,v in subclass_dict.items())}',
				(cls,),
				subclass_dict
			),
			slots = '__slots__' in cls.__dict__,
		)
		
		return specialised_subclass
