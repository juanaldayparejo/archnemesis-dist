
from typing import Annotated, get_origin, get_args, Type
import itertools

from archnemesis.database.data_layouts.structured_qualifiers import StructuredQualifier

class RecordLayoutMeta(type):
	def __new__(meta, name, bases, ctx):
		#print(f'Creating class {name}')
		
		
		annotations = ctx.get('__annotations__',dict())
		for attr, annotation in annotations.items():
			assert isinstance(annotation, Type) or (get_origin(annotation) is Annotated)
			if get_origin(annotation) is Annotated:
				#print(f'{len(get_args(annotation))=}')
				#print(f'{get_args(annotation)[1]=}')
				#print(f'{isinstance(get_args(annotation)[1], StructuredQualifier)=}')
				assert (len(get_args(annotation)) <= 2) and isinstance(get_args(annotation)[1], StructuredQualifier)
		
		default_attrs = tuple(k for k,v in ctx.items() if (not k.startswith('__')) and (not hasattr(v, '__func__')) and (not callable(v)))
		anno_attrs = tuple(k for k in annotations)
		
		for attr in default_attrs:
			if attr not in anno_attrs:
				raise AttributeError(f'Attribte "{attr}" has a default but no type annotation')

		field_names = anno_attrs
		field_types = tuple(get_args(anno)[0] if get_origin(anno) is Annotated else anno for anno in annotations.values())
		metadata = tuple(get_args(anno)[1] if get_origin(anno) is Annotated else StructuredQualifier() for anno in annotations.values())

		ctx['_fields'] = (*itertools.chain.from_iterable(getattr(b, "_fields", tuple()) for b in bases), *field_names)
		ctx['_types'] = (*itertools.chain.from_iterable(getattr(b, "_types", tuple()) for b in bases), *field_types)
		ctx['_metadata'] = (*itertools.chain.from_iterable(getattr(b, "_metadata", tuple()) for b in bases), *metadata)
		
		ctx['__slots__'] = ctx['_fields']

		x = super().__new__(meta, name, bases, ctx)
		return x
	
	def __init__(meta, name, bases, ctx):
		#print(f'Initialising class {name}')
		super().__init__(name, bases, ctx)
	
	def __call__(cls, *args, **kwargs):
		return super().__call__(*args, **kwargs)
	
	def as_string(cls, indent='    '):
		return '\n'.join(
			indent + f'{field} : {typ.__name__}' + (('\n' + meta.as_string(indent+indent)) if meta.has_entries() else '') for field, typ, meta in cls.items()
		)
	
	def __repr__(cls):
		return (
			f'{cls.__name__}:'
			+ '\n' + cls.as_string('    ')
		)
	
	def metadata(cls, attr):
		return cls._metadata[cls._fields.index(attr)]
	
	def type(cls, attr):
		return cls._types[cls._fields.index(attr)]
	
	def attrs(cls):
		return cls._fields
	
	def items(cls):
		return zip(cls._fields, cls._types, cls._metadata)
	


class RecordLayout(metaclass = RecordLayoutMeta):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError("Record classes should not be instantiated")
		for attr, value in zip(self._fields, args):
			assert attr not in kwargs, f"Must not have positional and keyword argument setting the same attribute '{attr}'"
			setattr(self, attr, value)
		
		for k, v in kwargs.items():
			assert k in self._fields, f"All keyword arguments must be expected, '{k}' is not an expected attribute"
			setattr(self, k, v)
		