from __future__ import annotations #  for 3.9 compatability

import numpy as np # incase it's needed

k_boltzmann          : float    = 1.380649e-23                               # J/K
c_light              : float    = 2.99792458e8                               # m/s
h_planck             : float    = 6.62607015e-34                             # J s
ref_temp             : float    = 296.0                                      # K
c2                   : float    = 100.0 * h_planck * c_light / k_boltzmann   # cm k

N_avogadro           : float    = 6.02214129E+23                             # mol^{-1}