

from typing import Type

from archnemesis.database.data_layouts.record_layout import RecordLayout



class TableLayoutMeta(type):
	def __new__(meta, name, bases, ctx):
		#print(f'Creating class {name}')
		
		
		record_format = ctx.get('record_format',None)
		assert record_format is not None and issubclass(record_format, RecordLayout), "TableLayout class must have a `record_layout` class attribute of class (or subclass) `RecordLayout`"
		
		ctx['__slots__'] = record_format._fields

		x = super().__new__(meta, name, bases, ctx)
		return x
	
	def __repr__(cls):
		return f'{cls.__name__}:' + '\n    '+' | '.join(attr for attr in cls.__slots__)


class TableLayout(metaclass = TableLayoutMeta):
	record_format : Type[RecordLayout] = RecordLayout
	min_col_size = 10
	
	def __init__(self, *args, **kwargs):
		#print(f'TableFormat.__init__(...) {[a.shape for a in args]=} {self.__slots__=}')
		for attr, value in zip(self.__slots__, args):
			assert attr not in kwargs, f"Must not have positional and keyword argument setting the same attribute '{attr}'"
			setattr(self, attr, value)
		
		for k, v in kwargs.items():
			assert k in self.__slots__, f"All keyword arguments must be expected, '{k}' is not an expected attribute"
			setattr(self, k, v)
		
		
	def __repr__(self):
		col_size = tuple(len(k) if len(k) > self.min_col_size else self.min_col_size for k in self.__slots__)
		return f'{self.__class__.__name__}:' + '\n    '+' | '.join(('{: ^'+str(s)+'}').format(attr) for attr, s in zip(self.__slots__, col_size)) + '\n    ' + '\n    '.join(' | '.join(('{: '+str(s)+'.3G}').format(getattr(self,k)[i]) for k, s in zip(self.__slots__, col_size)) for i in range(len(getattr(self,self.__slots__[0]))))
