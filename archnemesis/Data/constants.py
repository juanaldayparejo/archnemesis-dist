from __future__ import annotations #  for 3.9 compatability

k_B                  : float    = 1.380649E-23                                   # J/K           Boltzmann constant
k_boltzmann          : float    = 1.380649E-23                                   # J/K           Boltzmann constant
k_boltzmann_cgs      : float    = 1.380649E-16                                   # erg/K         Boltzman constant (cgs)

sig_B                : float    = 5.67037E-8                                     # W m-2 K-4     Stephan Boltzmann constant

R                    : float    = 8.31446261815324                               # J mol-1 K-1   universal gas constant
R_cgs                : float    = 8.31446261815324E7                             # erg mol-1 K-1 universal gas constant (cgs)

G                    : float    = 6.67199976E-11                                 # m3 kg-1 s-2   universal gravitational constant

eps_LJ               : float    = 59.7*5.67037E-8                                # J             depth of the Lennard-Jones potential well for H2

H2_c_p               : float    = 14300.0                                        # J K-1         hydrogen specific heat

c_light              : float    = 2.99792458E8                                   # m/s           Speed of light
c_light_cgs          : float    = 2.99792458E10                                  # cm/s          Speed of light (cgs)

h_planck             : float    = 6.62607015E-34                                 # J s           Plancks constant
h_planck_cgs         : float    = 6.62607015E-27                                 # erg s         Plancks constant (cgs)
hbar_planck          : float    = 1.05457182E-34                                 # J s           Plancks constant divided by 2*PI

ref_temp             : float    = 296.0                                          # K             A reference temperature

c2                   : float    = c_light * h_planck / k_boltzmann               # m K
c2_cgs               : float    = c_light_cgs * h_planck_cgs / k_boltzmann_cgs   # cm K

N_avogadro           : float    = 6.02214129E+23                                 # mol^{-1}      Avogadro's number, number of items in one mole