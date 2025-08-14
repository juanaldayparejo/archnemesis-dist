from __future__ import annotations #  for 3.9 compatability

import numpy as np # incase it's needed

k_boltzmann          : float    = 1.380649e-23                                   # J/K
k_boltzmann_cgs      : float    = 1.380649e-16                                   # erg/K
c_light              : float    = 2.99792458e8                                   # m/s
c_light_cgs          : float    = 2.99792458e10                                  # cm/s
h_planck             : float    = 6.62607015e-34                                 # J s
h_planck_cgs         : float    = 6.62607015e-27                                 # erg s
ref_temp             : float    = 296.0                                          # K
c2                   : float    = c_light * h_planck / k_boltzmann               # m K
c2_cgs               : float    = c_light_cgs * h_planck_cgs / k_boltzmann_cgs   # cm K

N_avogadro           : float    = 6.02214129E+23                                 # mol^{-1}