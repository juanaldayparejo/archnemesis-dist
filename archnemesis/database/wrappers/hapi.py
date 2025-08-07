from __future__ import annotations #  for 3.9 compatability

"""
The HITRAN API "HAPI" has quite a few quirks that mean it does
not behave as expected much of the time. Therefore I've created
this wrapper to handle its import.

Ideally you would only ever use this to import HAPI, and never do it
directly.

USAGE:
    `import archnemesis.database.wrappers.hapi as hapi`
"""
import sys
import builtins
from io import StringIO


import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.WARN)

__HAPI_string_buffer = StringIO()

if "hapi" in sys.modules:
    _lgr.warning('\n'.join(('',
        'Module "hapi" that provides the HITRAN API should only be imported by the ',
        '`archnemesis.database.wrappers.hapi` module.',
        '',
        'Use `import archnemesis.database.wrappers.hapi as hapi` to import HAPI in ',
        'any other module.',
        '',
        'This is because HAPI works quite differently from how python modules are ',
        'expected to work. And quite differently from how database managers work ',
        'in particular. Using the normal import method *should* not break anything ',
        'but may result in lots of extraenous output to stdout.',
    )))

# Need to redefine print when we import so we don't get loads of
# superflous output that we cannot get rid of.
__builtin_print = builtins.print

def __HAPI_print_intercept(*args, **kwargs) -> None:
    if kwargs.get('file', sys.stdout) is sys.stdout:
        if _lgr.level <= logging.DEBUG:
            kwargs['file'] = __HAPI_string_buffer
            __builtin_print(*args, **kwargs)
            __HAPI_string_buffer.truncate()
            s = __HAPI_string_buffer.getvalue()
            _lgr.debug(s[:-1] if s.endswith('\n') else s, stacklevel=2)
            __HAPI_string_buffer.seek(0)
    else:
        __builtin_print(*args, **kwargs)

builtins.print = __HAPI_print_intercept


HAPI_MODULE_INSTALLED = True
try:
    # Must import this way to avoid hapi.__init__ shadowing various bits of the module.
    import hapi.hapi as hapi
except ModuleNotFoundError:
    _lgr.warning(f'HAPI (hitran-api) module was not installed, therefore HITRAN online backend is NOT available. All attempts to use the backend will raise a `ModuleNotFoundError` exception')
    HAPI_MODULE_INSTALLED = False
finally:
    builtins.print = __builtin_print

if not HAPI_MODULE_INSTALLED:
    def __getattr__(name):
        raise ModuleNotFoundError(f'HAPI (hitran-api) module was not installed, therefore HITRAN online backend is NOT available.')
    
    def __setattr__(name, value):
        raise ModuleNotFoundError(f'HAPI (hitran-api) module was not installed, therefore HITRAN online backend is NOT available.')

else:
    hapi.LOCAL_TABLE_CACHE = dict()


    # Use these two functions to pass everything through to the "real" `hapi` module.
    def __getattr__(name):
        wrapped = getattr(hapi, name)
        if not callable(wrapped) or name=='describeTable':
            return wrapped
        
        def wrapper(*args, **kwargs):
            builtins.print = __HAPI_print_intercept
            try:
                result = wrapped(*args, **kwargs)
            finally:
                builtins.print = __builtin_print
            return result
        return wrapper

    def __setattr__(name, value):
        setattr(hapi, name, value)
