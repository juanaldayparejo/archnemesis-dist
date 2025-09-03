
# flake8: noqa

import sys

if sys.version_info[0] == 3 and sys.version_info[1] == 9:
    from . import py_3_9_compat
elif  sys.version_info[0] == 3 and sys.version_info[1] == 10:
    from . import py_3_10_compat
else:
	pass


#from . import logs