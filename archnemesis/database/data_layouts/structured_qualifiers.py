

import itertools


class StructuredQualifierMeta(type):

	def __new__(meta, name, bases, ctx):
		#print(f'Creating class {name}')
		
		
		annotations = ctx.get('__annotations__',dict())
		default_attrs = tuple(k for k,v in ctx.items() if (not k.startswith('__')) and (not hasattr(v, '__func__')) and (not callable(v)))
		anno_attrs = tuple(k for k in annotations)

		#for attr in anno_attrs:
		#    if attr not in default_attrs:
		#        raise AttributeError(f'Attribte "{attr}" has a type annotation but no default value')
		
		
		for attr in default_attrs:
			if attr not in anno_attrs:
				raise AttributeError(f'Attribte "{attr}" has a default but no type annotation')

		qual_names = tuple(set(anno_attrs) | set(default_attrs))

		ctx['_metadata'] = (*itertools.chain.from_iterable(getattr(b, "_metadata", tuple()) for b in bases), *qual_names)
		ctx['__slots__'] = ctx['_metadata']
		
		x = super().__new__(meta, name, bases, ctx)
		return x
	
	def __init__(meta, name, bases, ctx):
		#print(f'Initialising class {name}')
		super().__init__(name, bases, ctx)
	
	def __call__(cls, *args, **kwargs):
		#return namedtuple(cls.__name__, cls._metadata)(*args, **kwargs)
		return super().__call__(*args, **kwargs)


class StructuredQualifier(metaclass = StructuredQualifierMeta):
	
	def __init__(self, *args, **kwargs):
		for attr, value in zip(self._metadata, args):
			assert attr not in kwargs, f"Must not have positional and keyword argument setting the same attribute '{attr}'"
			setattr(self, attr, value)
		
		for k, v in kwargs.items():
			assert k in self._metadata, f"All keyword arguments must be expected, '{k}' is not an expected attribute"
			setattr(self, k, v)
	
	def __repr__(self):
		return f'{self.__class__.__name__}(' + ', '.join(f'{attr}={repr(getattr(self,attr))}' for attr in self._metadata) + ')'
	
	def as_string(self, indent='    '):
		return '\n'.join(indent + f'{k} : {v}' for k,v in self.as_dict().items())
	
	def has_entries(self):
		return len(self._metadata) > 0
	
	def as_dict(self):
		return dict((attr, getattr(self,attr)) for attr in self._metadata)
	
	def update_from(self, other):
		for attr in self._metadata:
			if hasattr(attr, other):
				setattr(self, attr, getattr(other, attr))
	

class Description(StructuredQualifier):
	description : str

class CategoryElement(Description):
	pass
	
class Index(Description):
	pass

class IDNumber(Description):
	pass

class Quantity(Description):
	unit : str
