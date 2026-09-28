"""
Choose which `voigt` lineshape implementations we want to use
"""

from .voigt_impl.voigt_scipy import voigt_scipy as voigt                 # noqa: F401 :: Ignore unused variable
#from .voigt_impl.voigt_schreier import voigt_schreier as voigt            # noqa: F401 :: Ignore unused variable
