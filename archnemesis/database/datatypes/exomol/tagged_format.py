

from typing import ClassVar, Iterator, Self, Any, get_origin, get_args, Type, Callable
import dataclasses as dc
import re

import numpy as np

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)

class RepeatedFieldError(Exception):
    pass


class PeekableIterator:
    def __init__(self, next_element, source_iterator, exhausted_value = None):
        self.next_element = next_element
        self.source_iterator = source_iterator
        self.source_exhausted = False
        self.exhausted_value = exhausted_value
    
    @property
    def value(self):
        if self.source_exhausted:
            return self.exhausted_value
            
        while self.next_element is None or (isinstance(self.next_element, str) and (len(self.next_element)==0 or self.next_element.isspace())): # skip empty lines
            try:
                self.next_element = next(self.source_iterator)
            except StopIteration:
                self.source_exhausted = True
                self.next_element = self.exhausted_value
                break
        
        return self.next_element
    
    def next(self):
        self.next_element = None
        return self.value
    
    def as_parts(self):
        return self.next_element, self.source_iterator
    
    @property
    def is_exhausted(self) -> bool:
        return self.source_exhausted



@dc.dataclass
class ExomolTaggedFileFormat:
    """
    A recursive dataclass base for reading ExoMol file formats that use tagged lines. Note that the tags
    are often the COMMENTS of the lines, with the values being before the `#` character.
    
    If a field is a list or tuple, multiple entries of that field can be read from the line iterator.
    If a field is not a list or tuple, only one entry will be read. If the field is itself a subclass of ExomolTaggedFileFormat,
    then the `from_line_itr` method of that class will be used to read the field. Therefore we can recursively read
    complex file formats.
    """
    _complete_when_fields_filled : ClassVar[bool] = True # if True, will return instance when all fields are filled, else will keep reading to end of line_itr
    _repeated_fields_overwrite : ClassVar[bool] = True # if True, fields that are not containers (list, tuple) will hold the final value assigned to them, otherwise will hold the first value.
    _expect_contiguous_fields : ClassVar[bool] = True # if this is true, we expect fields of this class to be contiguous (not neccessarily ordered). So one failure in field lookup will mean the instance is created with whatever fields we have found.
    _self_dispatch_check : ClassVar[str | re.Pattern | Callable[[str], bool]] = lambda next_line: True # compared in the same way as `_field_dispatch_map` to the first line of `line_itr`. Will return `None` instead of an instance of the class if the comparison fails.
    _field_dispatch_map : ClassVar[ dict[str, str | re.Pattern | Callable[[str], bool]] ] = dict() # Maps field names to dispatch checks. A string must match the start of the comment, a regex must match the line, a callable must return true. See `from_line_itr` for details.
    _alternate_field_readers : ClassVar[dict[str, Callable[[Type, PeekableIterator, Any],Any]]] = dict() # If a field has an entry in here, use this instead of the "normal" way of reading the field
    
    @staticmethod
    def should_dispatch(dispatch_check, next_line):
        v, c = next_line.split('#', 1)
        return (
            (isinstance(dispatch_check, str) and c.strip().startswith(dispatch_check))
            or (isinstance(dispatch_check, re.Pattern) and (dispatch_check.match(next_line) is not None))
            or (callable(dispatch_check) and dispatch_check(next_line))
        )
    
    @classmethod
    def _ensure_all_fields_in_dispatch_map(cls):
        missing_fields = []
        for field in dc.fields(cls):
            if field.name not in cls._field_dispatch_map:
                missing_fields.append(field.name)
        
        assert len(missing_fields) == 0, \
            f"All fields of {cls.__name__} must have entries in `cls._field_dispatch_map`. The following fields are missing: {missing_fields}"
        return
    
    
    @classmethod
    def cast_peekable_to_type(cls,
            field_name : str,
            field_type : Type, 
            peek_itr : PeekableIterator,
            current_instance : None | Any,
    ) -> tuple[bool, Any | None]: # return: (was_dispatched : bool, instance : None | Any)
        
        if peek_itr.is_exhausted:
            _lgr.debug('Iterator is exhausted')
            return False, None
        
        if (not cls.should_dispatch(cls._field_dispatch_map[field_name], peek_itr.value)):
            _lgr.debug(f'Dispatch check failed for field "{field_name}"')
            return False, None
        
        alternate_field_reader = cls._alternate_field_readers.get(field_name)
        if alternate_field_reader is not None:
            return True, alternate_field_reader(field_type, peek_itr, current_instance)
        
        value, comment = peek_itr.value.split('#', 1)
        
        t_origin = get_origin(field_type)
        t_args = get_args(field_type)
        
        
        if t_origin in (list, tuple):
            item_list = []
            was_dispatched = False
            if t_args[-1] == ...:
                # if the last element is an ellipsis, assume we repeat the previous `t_args[:-1]` until we run out of acceptable input
                continue_reading = True
                while continue_reading:
                    for t_arg in t_args[:-1]:
                        was_dispatched, instance = cls.cast_peekable_to_type(field_name, t_arg, peek_itr, current_instance)
                        if was_dispatched:
                            continue_reading = True
                            item_list.append(instance)
                        else:
                            continue_reading = False
                            break
            else:
                # If the last element is not an ellipsis, we assume each element has an entry in `t_args`
                for t_arg in t_args[:-1]:
                    was_dispatched, instance = cls.cast_peekable_to_type(field_name, t_arg, peek_itr, current_instance)
                    if was_dispatched:
                        item_list.append(instance)
                    else:
                        break
            
            if current_instance is not None:
                result = list(current_instance) + item_list
            else:
                result = item_list
            
            if t_origin is list:
                return True, result
            else:
                return True, tuple(result)
        
        elif t_origin == None:
            if current_instance is not None:
                if cls._repeated_fields_overwrite:
                    _lgr.warn(f'When reading field "{field_name}" instance of {field_type=} has already been found for this file, and as this field is not a list or tuple, cannot add additional instances. Field WILL be overwritten and will use its LAST assigned value')
                else:
                    _lgr.warn(f'When reading field "{field_name}" instance of {field_type=} has already been found for this file, and as this field is not a list or tuple, cannot add additional instances. Field WILL NOT be overwritten and will use its FIRST assigned value')
                    return True, current_instance
            
            if hasattr(field_type, 'from_line_itr'):
                next_line, instance = field_type.from_line_itr(*peek_itr.as_parts())
                _lgr.debug(f'Got {instance=} from `cls.from_line_itr(...)` of {field_type=}')
                _lgr.debug(f'{next_line=}')
                peek_itr.next_element = next_line
                _lgr.debug(f'{peek_itr.value=} {peek_itr.is_exhausted=}')
                return True, instance
            
            elif issubclass(field_type, int):
                v = value.strip()
                peek_itr.next()
                if v.upper() in ('NAN', 'NA'):
                    return True, np.nan
                else:
                    return True, field_type(float(v))
            
            elif issubclass(field_type, float):
                v = value.strip()
                peek_itr.next()
                if v.upper() in ('NAN', 'NA'):
                    return True, np.nan
                else:
                    return True, field_type(v)
                    
            else:
                v = value.strip()
                peek_itr.next()
                return True, field_type(v)
        
        else:
            raise RuntimeError(f'When reading field "{field_name}", unknown {field_type=} to dispatch. Decomposes into {t_origin=} {t_args=}.')
    
    
    
    @classmethod
    def attempt_dispatch_and_set(cls,
            field,
            peek_itr,
            current_instance : None | Any,
            args : dict,
    ) -> tuple[bool, Any | None]: # (was_dispatched : bool : bool, instance : None | Any)
        _lgr.debug(f'TRIAL: Class={cls.__name__} field="{field.name}" type={field.type}')
        was_dispatched = False
        try:
            was_dispatched, instance = cls.cast_peekable_to_type(field.name, field.type, peek_itr, current_instance)
        except Exception as e:
            _lgr.error(f'When class {cls.__name__} reading field "{field.name}" from next_line "{peek_itr.value}" encountered error: {str(e)}')
            raise
        
        if was_dispatched:
            _lgr.debug(f'RESULT: Dispatch was successful, {instance=}')
            args[field.name] = instance
        else:
            _lgr.debug('RESULT: Dispatch was NOT successful')
        return was_dispatched, instance
    
    @classmethod
    def from_line_itr(cls,
            next_line : None | str,
            line_itr : Iterator[str],
    ) -> tuple[None|str, None|Self]:
        """
        Create an instance of this class from a line iterator, return tuple[next_line, instance].
        `next_line` is the next line (or None) of the `line_itr`. If cannot create an `instance` 
        return None.
        
        Lines are assumed to be in the format: `value # comment`
        
        Dispatch to field type constructors based upon the _field_dispatch_map,
        the cases are:
        
            1) If the entry is a string, it is compared to the same length slice
            of the comment (excluding `#` and any starting whitespace).
        
            2) If the entry is a regular expression pattern, the pattern is matched
            against the entire line.
            
            3) If the entry is a callable it is passed the entire line.
        
        If the cases 1) match; 2) match the regex; 3) return `True`, then we
        dispatch to create an instance of the field type.
        """
        
        cls._ensure_all_fields_in_dispatch_map()
        
        args = dict()
        
        peek_itr = PeekableIterator(next_line, line_itr, exhausted_value = None) # `line_itr` should only return strings, so return `None` on iterator empty
        
        if not cls.should_dispatch(cls._self_dispatch_check, peek_itr.value):
            # Do not continue reading if we fail _self_dispatch_check, return `None` for class instance
            _lgr.error(f'{cls.__name__} failed self-dispatch check, returning `None` for class instance.')
            return peek_itr.value, None

        _lgr.debug(f'Filling instance of {cls.__name__} from line iterator starting with line: "{peek_itr.value}"')
        
        fields = dc.fields(cls)
        extendable_fields = [field.name for field in fields if get_origin(field.type) in (list, tuple)]
        any_fields_are_extendable = len(extendable_fields) > 0
        
        _lgr.debug(f'Extendable fields: {extendable_fields}')
        
        while not peek_itr.is_exhausted:
            was_field_found_for_this_line = False
            
            if (not any_fields_are_extendable) and (len(args) == len(dc.fields(cls))) and cls._complete_when_fields_filled:
                _lgr.debug(f'All fields of {cls.__name__} have been filled, `_complete_when_fields_filled` is True, and there are no extendable fileds, stopping read here.')
                break
            
            _lgr.debug(f'Testing all fields of {cls.__name__} against line: "{peek_itr.value}"')
            
            _lgr.debug('Attempting dispatch to non-set fields first')
            for i, field in enumerate(dc.fields(cls)):
                if (field.name in args):
                    continue

                was_dispatched, instance = cls.attempt_dispatch_and_set(field, peek_itr, None, args)
                if was_dispatched:
                    was_field_found_for_this_line = True
                    break
            
            if (not was_field_found_for_this_line):
                _lgr.debug('Dispatch to non set fields failed, attempting to dispatch to extendable fileds')
                for i, field in enumerate(dc.fields(cls)):
                    if not (field.name in extendable_fields):
                        continue

                    was_dispatched, instance = cls.attempt_dispatch_and_set(field, peek_itr, args.get(field.name, None), args)
                    if was_dispatched:
                        was_field_found_for_this_line = True
                        break
            
            if (not was_field_found_for_this_line) and cls._repeated_fields_overwrite:
                _lgr.debug('Dispatch to non-set and extendable fields failed, attempting dispatch to set fields as `_repeated_fields_overwrite` is True')
                for i, field in enumerate(dc.fields(cls)):
                    if (field.name not in args) or (field.name in extendable_fields):
                        continue

                    was_dispatched, instance = cls.attempt_dispatch_and_set(field, peek_itr, args[field.name], args)
                    if was_dispatched:
                        was_field_found_for_this_line = True
                        break
            
            if cls._expect_contiguous_fields and (not was_field_found_for_this_line):
                _lgr.debug(f'No fields of {cls.__name__} matched line: "{peek_itr.value}", and `_expect_contiguous_fields` is True, stopping read here.')
                break
            
            if not was_field_found_for_this_line:
                # skip the line and move on to the next one, moving to next value of iterator in case of a match is handled inside cast_peekable_to_type
                peek_itr.next()
        
        return peek_itr.value, cls(**args)
            