import warnings
import pytest
import archnemesis as ans
import importlib
from pathlib import Path
import os

def test_no_syntax_warnings_on_compile(recwarn):
    caught_warnings = []
    
    for path in ans.__path__:
        for dirpath, dirnames, filenames in os.walk(str(path)):
            for filename in filenames:
                if filename.endswith('.py'):
                    file = Path(dirpath) / filename
                    loader = importlib.machinery.SourceFileLoader("<test_import>", file)
                    loader.source_to_code(loader.get_data(file), file)

    if len(recwarn) > 0:
        msg = "\n".join(map(lambda w: f'{w.category.__name__} {w.filename} {w.lineno}: {w.message}', filter(lambda w: issubclass(w.category, SyntaxWarning), recwarn)))
        assert len(msg) == 0, f"Caught an warnings on compile, test failed:\n{msg}"
