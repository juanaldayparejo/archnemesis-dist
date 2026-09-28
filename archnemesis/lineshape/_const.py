"""
Constants used by line profiles
"""

import numpy as np

NUMBA_CACHE     : bool       = True

SQRT_2          : float      = np.sqrt(2.0)
SQRT_PI         : float      = np.sqrt(np.pi)
SQRT_2PI        : float      = SQRT_2 * SQRT_PI
SQRT_log2       : float      = np.sqrt(np.log(2.0))
SQRT_2log2      : float      = np.sqrt(2.0*np.log(2.0))

INV_SQRT_2      : float      = 1.0/(SQRT_2)
INV_SQRT_PI     : float      = 1.0/(SQRT_PI)
INV_SQRT_2PI    : float      = 1.0/(SQRT_2PI)
INV_SQRT_log2   : float      = 1.0/(SQRT_log2)
INV_SQRT_2log2  : float      = 1.0/(SQRT_2log2)

L24             : float      = np.sqrt(24./np.sqrt(2.0))

A24             : np.ndarray = np.array([ 
                                +2.3241983342526162e+00,  # a0 = L/sqrtPi
                                +2.1978589365315417e+00, +1.8562864992055408e+00, +1.3948196733791203e+00, +9.2570871385886788e-01,
                                +5.3611395357291292e-01, +2.6549639598807689e-01, +1.0838723484566792e-01, +3.3723366855316413e-02,
                                +6.2150063629501763e-03, -4.9364269012806686e-04, -7.8166429956142650e-04, -2.0748431511424456e-04,
                                +2.4331415462641969e-05, +3.0471066083243790e-05, +4.1394617248575527e-06, -3.0388931839840047e-06,
                                -1.0856475790698251e-06, +2.5682641346701115e-07, +1.8738343486619108e-07, -1.9122258522976932e-08,
                                -3.0082822811202271e-08, +1.3310461806370372e-09, +4.9048215867870488e-09, -1.5137461654527820e-10
])