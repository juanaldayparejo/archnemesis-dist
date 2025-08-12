#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# gas_data.py - Gas properties and constants.
#
# Copyright (C) 2025 Juan Alday, Joseph Penn, Patrick Irwin,
# Jack Dobinson, Jon Mason, Jingxuan Yang
#
# This file is part of archNEMESIS.
#
# archNEMESIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Data reference files.

Contains definition of units, constants, gas identifiers (RADTRAN ID),
gas isotope identifiers, gas isotope relative abundances, atomic mass,
partitiona function coefficients and planets parameters.

Contains callable functions:
Calc_MMW(VMR, ID, ISO=0):
    calculate mean molecular weight using the radtrans data base.
look_up(gas_name):
    look up information about a molecule using the RADTRAN data base.
"""

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

def Calc_mmw(VMR, ID, ISO=0):
    NGAS = len(ID)
    MMW = 0
    if ISO == 0:
        for i in range(NGAS):
            mass = gas_info['{}'.format(ID[i])]['mmw']
            MMW += VMR[i] * mass
    return MMW

def look_up(gas_name):
    index = gas_id[gas_name]
    index = str(index)
    return gas_info[index]

def count_isotopes(mol_id):
    """
    Count the number of isotopes for a given molecule ID.
    """
    mol_entry = gas_info.get(str(mol_id))
    if mol_entry:
        isotopes = mol_entry.get('isotope', {})
        return len(isotopes)
    else:
        return 0  # Molecule ID not found


def id_to_name(gasid,isoid):
    """
    Return the number of the molecule or isotope
    """
    
    if isoid==0:
        return gas_info[str(gasid)]["name"]
    else:
        return gas_info[str(gasid)]["isotope"][str(isoid)]["name"]


def molecule_to_latex(formula):
    import re
    # Step 1: Replace isotopes in parentheses (e.g., (14N) -> ^{14}N)
    formula = re.sub(r'\((\d+)([A-Z][a-z]*)\)', r'^{\1}\2', formula)

    # Step 2: Add \text{} around elements and subscripts for numbers
    def add_subscripts(match):
        element, number = match.groups()
        if number:
            return f"\\text{{{element}}}_{{{number}}}"
        else:
            return f"\\text{{{element}}}"

    formula = re.sub(r'([A-Z][a-z]*)(\d*)', add_subscripts, formula)
    return formula



unit = {
    'pc': 3.08567e16,        # m parsec
    'ly': 9.460730e15,       # m lightyear
    'AU': 1.49598e11,        # m astronomical unit
    'R_Sun': 6.95700e8,      # m solar radius
    'R_Jup': 7.1492e7,       # m nominal equatorial Jupiter radius (1 bar pressure level)
    'R_E': 6.371e6,          # m nominal Earth radius
    'd_H2': 2.827e-10,       # m molecular diameter of H2
    'M_Sun': 1.989e30,       # kg solar mass
    'M_Jup': 1.8982e27,      # kg Jupiter mass
    'M_E': 5.972e24,         # kg Earth mass
    'amu': 1.66054e-27,      # kg atomic mass unit
    'atm': 101325,           # Pa atmospheric pressure
}

const = {
    'k_B': 1.38065e-23,         # J K-1 Boltzmann constant
    'sig_B': 5.67037e-8,        # W m-2 K-4 Stephan Boltzmann constant
    #'R': 8.31446,               # J mol-1 K-1 universal gas constant
    'R': 8.31,                  # J mol-1 K-1 universal gas constant
    'G': 6.67199976E-11,        # m3 kg-1 s-2 universal gravitational constant
    'eps_LJ': 59.7*5.67037e-8,  # J depth of the Lennard-Jones potential well for H2
    'c_p': 14300,               # J K-1 hydrogen specific heat
    'h': 6.62607,               # Js Planck's constant
    'hbar': 1.05457,            # Js
    'N_A': 6.02214e23,          # Avagadro's number
}

gas_id = {
    'H2O': 1,
    'CO2': 2,
    'O3': 3,
    'N2O': 4,
    'CO': 5,
    'CH4': 6,
    'O2': 7,
    'NO': 8,
    'SO2': 9,
    'NO2': 10,
    'NH3': 11,
    'HNO3': 12,
    'OH': 13,
    'HF': 14,
    'HCl': 15,
    'HBr': 16,
    'HI': 17,
    'ClO': 18,
    'OCS': 19,
    'H2CO': 20,
    'HOCl': 21,
    'N2': 22,
    'HCN': 23,
    'CH3Cl': 24,
    'H2O2': 25,
    'C2H2': 26,
    'C2H6': 27,
    'PH3': 28,
    'C2N2': 29,
    'C4H2': 30,
    'HC3N': 31,
    'C2H4': 32,
    'GeH4': 33,
    'C3H8': 34,
    'HCOOH': 35,
    'H2S': 36,
    'COF2': 37,
    'SF6': 38,
    'H2': 39,
    'He': 40,
    'AsH3': 41,
    'C3H4': 42,
    'ClONO2': 43,
    'HO2': 44,
    'O': 45,
    'NO+': 46,
    'CH3OH': 47,
    'H': 48,
    'C6H6': 49,
    'CH3CN': 50,
    'CH2NH': 51,
    'C2H3CN': 52,
    'HCP': 53,
    'CS': 54,
    'HC5N': 55,
    'HC7N': 56,
    'C2H5CN': 57,
    'CH3NH2': 58,
    'HNC': 59,
    'Na': 60,
    'K': 61,
    'TiO': 62,
    'VO': 63,
    'CH2CCH2': 64,
    'C4N2': 65,
    'C5H5N': 66,
    'C5H4N2': 67,
    'C7H8': 68,
    'C8H6': 69,
    'C5H5CN': 70,
    'HOBr': 71,
    'CH3Br': 72,
    'CF4': 73,
    'SO3': 74,
    'Ne': 75,
    'Ar': 76,
    'COCl2': 77,
    'SO': 78,
    'H2SO4': 79,
    'e–': 80,
    'H3+': 81,
    'FeH': 82,
    'AlO': 83,
    'AlCl': 84,
    'AlF': 85,
    'AlH': 86,
    'BeH': 87,
    'C2': 88,
    'CaF': 89,
    'CaH': 90,
    'H–': 91,
    'CaO': 92,
    'CH': 93,
    'CH3': 94,
    'CH3F': 95,
    'CN': 96,
    'CP': 97,
    'CrH': 98,
    'HD+': 99,
    'HeH+': 100,
    'KCl': 101,
    'KF': 102,
    'LiCl': 103,
    'LiF': 104,
    'LiH': 105,
    'LiH+': 106,
    'MgF': 107,
    'MgH': 108,
    'MgO': 109,
    'NaCl': 110,
    'NaF': 111,
    'NaH': 112,
    'NH': 113,
    'NS': 114,
    'OH+': 115,
    'cis-P2H2': 116,
    'trans-P2H2': 117,
    'PH': 118,
    'PN': 119,
    'PO': 120,
    'PS': 121,
    'ScH': 122,
    'SH': 123,
    'SiH': 124,
    'SiH2': 125,
    'SiH4': 126,
    'SiO': 127,
    'SiS': 129,
    'TiH': 130,
    'Cl2': 131,
    'ClO2': 132,
    'BrO': 133,
    'IO': 134,
    'NO3': 135,
    'BrO2': 136,
    'IO2': 137,
    'CS2': 138,
    'CH3I': 139,
    'NF3': 140,
}

atom_mass = {
    'H': 1.00794,
    'He': 4.002602,
    'Li': 6.941,
    'Be': 9.012182,
    'B': 10.811,
    'C': 12.0107,
    'N': 14.0067,
    'O': 15.9994,
    'F': 18.9984032,
    'Ne': 20.1797,
    'Na': 22.98976928,
    'Mg': 24.305,
    'Al': 26.9815386,
    'Si': 28.0855,
    'P': 30.973762,
    'S': 32.065,
    'Cl': 35.453,
    'Ar': 39.948,
    'K': 39.0983,
    'Ca': 40.078,
    'Sc': 44.955912,
    'Ti': 47.867,
    'V': 50.9415,
    'Cr': 51.9961,
    'Mn': 54.938045,
    'Fe': 55.845,
    'Co': 58.933195,
    'Ni': 58.6934,
    'Cu': 63.546,
    'Zn': 65.409,
    'Ga': 69.723,
    'Ge': 72.64,
    'As': 74.9216,
    'Se': 78.96,
    'Br': 79.904,
    'Kr': 83.798,
    'Rb': 85.4678,
    'Sr': 87.62,
    'Y': 88.90585,
    'Zr': 91.224,
    'Nb': 92.90638,
    'Mo': 95.94,
    'Tc': 98.9063,
    'Ru': 101.07,
    'Rh': 102.9055,
    'Pd': 106.42,
    'Ag': 107.8682,
    'Cd': 112.411,
    'In': 114.818,
    'Sn': 118.71,
    'Sb': 121.76,
    'Te': 127.6,
    'I': 126.90447,
    'Xe': 131.293,
    'Cs': 132.9054519,
    'Ba': 137.327,
    'La': 138.90547,
    'Ce': 140.116,
    'Pr': 140.90465,
    'Nd': 144.242,
    'Pm': 146.9151,
    'Sm': 150.36,
    'Eu': 151.964,
    'Gd': 157.25,
    'Tb': 158.92535,
    'Dy': 162.5,
    'Ho': 164.93032,
    'Er': 167.259,
    'Tm': 168.93421,
    'Yb': 173.04,
    'Lu': 174.967,
    'Hf': 178.49,
    'Ta': 180.9479,
    'W': 183.84,
    'Re': 186.207,
    'Os': 190.23,
    'Ir': 192.217,
    'Pt': 195.084,
    'Au': 196.966569,
    'Hg': 200.59,
    'Tl': 204.3833,
    'Pb': 207.2,
    'Bi': 208.9804,
    'Po': 208.9824,
    'At': 209.9871,
    'Rn': 222.0176,
    'Fr': 223.0197,
    'Ra': 226.0254,
    'Ac': 227.0278,
    'Th': 232.03806,
    'Pa': 231.03588,
    'U': 238.02891,
    'Np': 237.0482,
    'Pu': 244.0642,
    'Am': 243.0614,
    'Cm': 247.0703,
    'Bk': 247.0703,
    'Cf': 251.0796,
    'Es': 252.0829,
    'Fm': 257.0951,
    'Md': 258.0951,
    'No': 259.1009,
    'Lr': 262,
    'Rf': 267,
    'Db': 268,
    'Sg': 271,
    'Bh': 270,
    'Hs': 269,
    'Mt': 278,
    'Ds': 281,
    'Rg': 281,
    'Cn': 285,
    'Nh': 284,
    'Fl': 289,
    'Mc': 289,
    'Lv': 292,
    'Ts': 294,
    'Og': 294,
    'ZERO': 0,
}

# 'mass' is molecular mass in grams / mol
gas_info = {
    "1": {
        "name": "H2O",
        "isotope": {
            "1": {
                "name": "H2(16O)",
                "abun": 0.997317,
                "mass": 18.010560,
                "id": 161,
            },
            "2": {
                "name": "H2(18O)",
                "abun": 0.002000,
                "mass": 20.014810,
                "id": 181,
            },
            "3": {
                "name": "H2(17O)",
                "abun": 0.000372,
                "mass": 19.014780,
                "id": 171,
            },
            "4": {
                "name": "HD(16O)",
                "abun": 0.000311,
                "mass": 19.016740,
                "id": 162,
            },
            "5": {
                "name": "HD(18O)",
                "abun": 6e-07,
                "mass": 21.020980,
                "id": 182,
            },
            "6": {
                "name": "HD(17O)",
                "abun": 0.000,
                "mass": 20.020960,
                "id": 172,
            },
            "7": {
                "name": "D2(16O)",
                "abun": 0.000,
                "mass": 20.022920,
                "id": 622,
            },
        },
        "mmw": 18.01529586265
    },
    "2": {
        "name": "CO2",
        "isotope": {
            "1": {
                "name": "(12C)(16O)2",
                "abun": 0.984204,
                "mass": 43.989830,
                "id": 626,
            },
            "2": {
                "name": "(13C)(16O)2",
                "abun": 0.011057,
                "mass": 44.993180,
                "id": 636,
            },
            "3": {
                "name": "(16O)(12C)(18O)",
                "abun": 0.003947,
                "mass": 45.994080,
                "id": 628,
            },
            "4": {
                "name": "(16O)(12C)(17O)",
                "abun": 0.000734,
                "mass": 44.994040,
                "id": 627,
            },
            "5": {
                "name": "(16O)(13C)(18O)",
                "abun": 0.000044,
                "mass": 46.997430,
                "id": 638,
            },
            "6": {
                "name": "(16O)(13C)(17O)",
                "abun": 0.000008,
                "mass": 45.997400,
                "id": 637,
            },
            "7": {
                "name": "(12C)(18O)2",
                "abun": 4e-06,
                "mass": 47.998320,
                "id": 828,
            },
            "8": {
                "name": "(17O)(12C)(18O)",
                "abun": 1e-07,
                "mass": 46.998290,
                "id": 728,
            },
            "9": {
                "name": "(12C)(17O)2",
                "abun": 1e-20,
                "mass": 45.998260,
                "id": 727,
            },
            "10": {
                "name": "(13C)(18O)2",
                "abun": 1e-20,
                "mass": 49.001670,
                "id": 838,
            },
            "11": {
                "name": "(18O)(13C)(17O)",
                "abun": 1e-20,
                "mass": 48.001650,
                "id": 837,
            },
            "12": {
                "name": "(13C)(17O)2",
                "abun": 1e-20,
                "mass": 47.001620,
                "id": 737,
            },
        },
        "mmw": 44.00967449352
    },
    "3": {
        "name": "O3",
        "isotope": {
            "1": {
                "name": "(16O)3",
                "abun": 0.992901,
                "mass": 47.984740,
                "id": 666,
            },
            "2": {
                "name": "(16O)(16O)(18O)",
                "abun": 0.003982,
                "mass": 49.988990,
                "id": 668,
            },
            "3": {
                "name": "(16O)(18O)(16O)",
                "abun": 0.001991,
                "mass": 49.988990,
                "id": 686,
            },
            "4": {
                "name": "(16O)(16O)(17O)",
                "abun": 0.000740,
                "mass": 48.988960,
                "id": 667,
            },
            "5": {
                "name": "(16O)(17O)(16O)",
                "abun": 0.000370,
                "mass": 48.988960,
                "id": 676,
            },
        },
        "mmw": 47.99704799509999
    },
    "4": {
        "name": "N2O",
        "isotope": {
            "1": {
                "name": "(14N)2(16O)",
                "abun": 0.990333,
                "mass": 44.001060,
                "id": 446,
            },
            "2": {
                "name": "(14N)(15N)(16O)",
                "abun": 0.003641,
                "mass": 44.998100,
                "id": 456,
            },
            "3": {
                "name": "(15N)(14N)(16O)",
                "abun": 0.003641,
                "mass": 44.998100,
                "id": 546,
            },
            "4": {
                "name": "(14N)2(18O)",
                "abun": 0.001986,
                "mass": 46.005310,
                "id": 448,
            },
            "5": {
                "name": "(14N)2(17O)",
                "abun": 0.000369,
                "mass": 45.005280,
                "id": 447,
            },
        },
        "mmw": 44.01138631291001
    },
    "5": {
        "name": "CO",
        "isotope": {
            "1": {
                "name": "(12C)(16O)",
                "abun": 0.986544,
                "mass": 27.994910,
                "id": 26,
            },
            "2": {
                "name": "(13C)(16O)",
                "abun": 0.011084,
                "mass": 28.998270,
                "id": 36,
            },
            "3": {
                "name": "(12C)(18O)",
                "abun": 0.001978,
                "mass": 29.999160,
                "id": 28,
            },
            "4": {
                "name": "(12C)(17O)",
                "abun": 0.000368,
                "mass": 28.999130,
                "id": 27,
            },
            "5": {
                "name": "(13C)(18O)",
                "abun": 2.22e-05,
                "mass": 31.002520,
                "id": 38,
            },
            "6": {
                "name": "(13C)(17O)",
                "abun": 4.1e-06,
                "mass": 30.002490,
                "id": 37,
            }
        },
        "mmw": 28.010430635559995
    },
    "6": {
        "name": "CH4",
        "isotope": {
            "1": {
                "name": "(12C)H4",
                "abun": 0.988274,
                "mass": 16.031300,
                "id": 211,
            },
            "2": {
                "name": "(13C)H4",
                "abun": 0.01110,
                "mass": 17.034660,
                "id": 311,
            },
            "3": {
                "name": "(12C)H3D",
                "abun": 0.000616,
                "mass": 17.037480,
                "id": 212,
            },
            "4": {
                "name": "(13C)H3D",
                "abun": 6.9179e-06,
                "mass": 18.040830,
                "id": 312,
            }
        },
        "mmw": 16.04307145141211
    },
    "7": {
        "name": "O2",
        "isotope": {
            "1": {
                "name": "(16O)2",
                "abun": 0.995262,
                "mass": 31.989830,
                "id": 66,
            },
            "2": {
                "name": "(16O)(18O)",
                "abun": 0.0039914,
                "mass": 33.994080,
                "id": 68,
            },
            "3": {
                "name": "(16O)(17O)",
                "abun": 0.0007422,
                "mass": 32.994040,
                "id": 67,
            }
        },
        "mmw": 31.998404525139996
    },
    "8": {
        "name": "NO",
        "isotope": {
            "1": {
                "name": "(14N)(16O)",
                "abun": 0.993974,
                "mass": 29.997990,
                "id": 46,
            },
            "2": {
                "name": "(15N)(16O)",
                "abun": 0.0036543,
                "mass": 30.995020,
                "id": 56,
            },
            "3": {
                "name": "(14N)(18O)",
                "abun": 0.0019931,
                "mass": 32.00230,
                "id": 48,
            }
        },
        "mmw": 29.99428066532
    },
    "9": {
        "name": "SO2",
        "isotope": {
            "1": {
                "name": "(32S)(16O)2",
                "abun": 0.945678,
                "mass": 63.961900,
                "id": 626,
            },
            "2": {
                "name": "(34S)(16O)2",
                "abun": 0.041950,
                "mass": 65.957700,
                "id": 646,
            },
            "3": {
                "name": "(33S)(16O)2",
                "abun": 0.007464,
                "mass": 64.961290,
                "id": 636,
            },
            "4": {
                "name": "(16O)(32S)(18O)",
                "abun": 0.003793,
                "mass": 65.966150,
                "id": 628,
            },
        },
        "mmw": 63.25430697051
    },
    "10": {
        "name": "NO2",
        "isotope": {
            "1": {
                "name": "(14N)(16O)2",
                "abun": 0.991616,
                "mass": 45.992900,
                "id": 646,
            },
            "2": {
                "name": "(15N)(16O)2",
                "abun": 0.003646,
                "mass": 46.989940,
                "id": 656,
            },
        },
        "mmw": 45.6072955264
    },
    "11": {
        "name": "NH3",
        "isotope": {
            "1": {
                "name": "(14N)H3",
                "abun": 0.995872,
                "mass": 17.026550,
                "id": 411,
            },
            "2": {
                "name": "(15N)H3",
                "abun": 0.003661,
                "mass": 18.023580,
                "id": 511,
            },
        },
        "mmw": 17.02220441468
    },
    "12": {
        "name": "HNO3",
        "isotope": {
            "1": {
                "name": "H(14N)(16O)3",
                "abun": 0.98911,
                "mass": 62.995640,
                "id": 146,
            },
            "2": {
                "name": "H(15N)(16O)3",
                "abun": 0.003636,
                "mass": 63.992680,
                "id": 156,
            },
        },
        "mmw": 62.30957791600001
    },
    "13": {
        "name": "OH",
        "isotope": {
            "1": {
                "name": "(16O)H",
                "abun": 0.997473,
                "mass": 17.002740,
                "id": 61,
            },
            "2": {
                "name": "(18O)H",
                "abun": 0.0020001,
                "mass": 19.006990,
                "id": 81,
            },
            "3": {
                "name": "(16O)D",
                "abun": 0.0001554,
                "mass": 18.008910,
                "id": 62,
            },
        },
        "mmw": 17.00054866086
    },
    "14": {
        "name": "HF",
        "isotope": {
            "1": {
                "name": "H(19F)",
                "abun": 0.999844,
                "mass": 20.006230,
                "id": 19,
            },
            "2": {
                "name": "D(19F)",
                "abun": 0.000156,
                "mass": 21.012400,
                "id": 29,
            }
        },
        "mmw": 20.0030790328
    },
    "15": {
        "name": "HCl",
        "isotope": {
            "1": {
                "name": "H(35Cl)",
                "abun": 0.757587,
                "mass": 35.976680,
                "id": 15,
            },
            "2": {
                "name": "H(37Cl)",
                "abun": 0.242257,
                "mass": 37.973730,
                "id": 17,
            },
            "3": {
                "name": "D(35Cl)",
                "abun": 0.000118,
                "mass": 36.982850,
                "id": 25,
            },
            "4": {
                "name": "D(37Cl)",
                "abun": 0.000038,
                "mass": 38.979900,
                "id": 27,
            }
        },
        "mmw": 36.4548748638
    },
    "16": {
        "name": "HBr",
        "isotope": {
            "1": {
                "name": "H(79Br)",
                "abun": 0.506781,
                "mass": 79.926160,
                "id": 19,
            },
            "2": {
                "name": "H(81Br)",
                "abun": 0.493063,
                "mass": 81.924120,
                "id": 11,
            },
            "3": {
                "name": "D(79Br)",
                "abun": 0.000079,
                "mass": 80.932340,
                "id": 29,
            },
            "4": {
                "name": "D(81Br)",
                "abun": 0.000077,
                "mass": 82.930290,
                "id": 21,
            },
        },
        "mmw": 80.8988220805
    },
    "17": {
        "name": "HI",
        "isotope": {
            "1": {
                "name": "H(127I)",
                "abun": 0.999844,
                "mass": 127.912300,
                "id": 17,
            },
            "2": {
                "name": "D(127I)",
                "abun": 0.000156,
                "mass": 128.918500,
                "id": 27,
            }
        },
        "mmw": 127.8923456812
    },
    "18": {
        "name": "ClO",
        "isotope": {
            "1": {
                "name": "(35Cl)(16O)",
                "abun": 0.755908,
                "mass": 50.963770,
                "id": 56,
            },
            "2": {
                "name": "(37Cl)(16O)",
                "abun": 0.241720,
                "mass": 52.960820,
                "id": 76,
            }
        },
        "mmw": 51.3256287064
    },
    "19": {
        "name": "OCS",
        "isotope": {
            "1": {
                "name": "(16O)(12C)(32S)",
                "abun": 0.937395,
                "mass": 59.966990,
                "id": 622,
            },
            "2": {
                "name": "(16O)(12C)(34S)",
                "abun": 0.0415828,
                "mass": 61.962780,
                "id": 624,
            },
            "3": {
                "name": "(16O)(13C)(32S)",
                "abun": 0.010531,
                "mass": 60.970340,
                "id": 632,
            },
            "4": {
                "name": "(18O)(12C)(32S)",
                "abun": 0.001880,
                "mass": 61.971230,
                "id": 822,
            },
            "5": {
                "name": "(16O)(12C)(33S)",
                "abun": 0.007399,
                "mass": 60.966370,
                "id": 623,
            },
            "6": {
                "name": "(16O)(13C)(34S)",
                "abun": 0.000467,
                "mass": 62.966140,
                "id": 634,
            },
            "7": {
                "name": "(17O)(12C)(32S)",
                "abun": 1e-07,
                "mass": 60.970340,
                "id": 722,
            },
        },
        "mmw": 59.99905145753
    },
    "20": {
        "name": "H2CO",
        "isotope": {
            "1": {
                "name": "H2(12C)(16O)",
                "abun": 0.986237,
                "mass": 30.01056,
                "id": 126,
            },
            "2": {
                "name": "H2(13C)(16O)",
                "abun": 0.0110802,
                "mass": 31.0139,
                "id": 136,
            },
            "3": {
                "name": "H2(12C)(18O)",
                "abun": 0.0019776,
                "mass": 32.0148,
                "id": 128,
            },
        },
        "mmw": 30.00451679546
    },
    "21": {
        "name": "HOCl",
        "isotope": {
            "1": {
                "name": "H(16O)(35Cl)",
                "abun": 0.75579,
                "mass": 51.971590,
                "id": 165,
            },
            "2": {
                "name": "H(16O)(37Cl)",
                "abun": 0.241683,
                "mass": 53.9686,
                "id": 167,
            },
        },
        "mmw": 52.3229087178
    },
    "22": {
        "name": "N2",
        "isotope": {
            "1": {
                "name": "(14N)2",
                "abun": 0.992687,
                "mass": 28.006150,
                "id": 44,
            },
            "2": {
                "name": "(14N)(15N)",
                "abun": 0.007313,
                "mass": 29.003180,
                "id": 45,
            }
        },
        "mmw": 28.0133998366
    },
    "23": {
        "name": "HCN",
        "isotope": {
            "1": {
                "name": "H(12C)(14N)",
                "abun": 0.985114,
                "mass": 27.010900,
                "id": 124,
            },
            "2": {
                "name": "H(13C)(14N)",
                "abun": 0.0110676,
                "mass": 28.014250,
                "id": 134,
            },
            "3": {
                "name": "H(12C)(15N)",
                "abun": 0.0036217,
                "mass": 28.0079,
                "id": 125,
            }
        },
        "mmw": 27.02030302071
    },
    "24": {
        "name": "CH3Cl",
        "isotope": {
            "1": {
                "name": "(12C)H3(35Cl)",
                "abun": 0.748937,
                "mass": 49.9923,
                "id": 215,
            },
            "2": {
                "name": "(12C)H3(37Cl)",
                "abun": 0.239491,
                "mass": 51.9894,
                "id": 217,
            }
        },
        "mmw": 49.8920765805
    },
    "25": {
        "name": "H2O2",
        "isotope": {
            "1": {
                "name": "H2(16O)2",
                "abun": 0.994952,
                "mass": 34.0055,
                "id": 166,
            }
        },
        "mmw": 33.83384023599999
    },
    "26": {
        "name": "C2H2",
        "isotope": {
            "1": {
                "name": "(12C)2H2",
                "abun": 0.977599,
                "mass": 26.015650,
                "id": 221,
            },
            "2": {
                "name": "(12C)(13C)H2",
                "abun": 0.0219663,
                "mass": 27.019,
                "id": 231,
            },
            "3": {
                "name": "(12C)2HD",
                "abun": 0.000305,
                "mass": 27.021820,
                "id": 222,
            }
        },
        "mmw": 26.03482546833
    },
    "27": {
        "name": "C2H6",
        "isotope": {
            "1": {
                "name": "(12C)2H6",
                "abun": 0.9769,
                "mass": 30.047,
                "id": 226,
            },
            "2": {
                "name": "(12C)H3(13C)H3",
                "abun": 0.0222,
                "mass": 31.050310,
                "id": 236,
            }
        },
        "mmw": 30.042668300000003
    },
    "28": {
        "name": "PH3",
        "isotope": {
            "1": {
                "name": "(31P)H3",
                "abun": 0.999533,
                "mass": 33.9972,
                "id": 131,
            }
        },
        "mmw": 33.9813233076
    },
    "29": {
        "name": "C2N2",
        "isotope": {
            "1": {
                "name": "(12C)2(14N)2",
                "abun": 0.9705,
                "mass": 52.006150,
                "id": 224,
            },
            "2": {
                "name": "(12C)(13C)(14N)2",
                "abun": 0.0222,
                "mass": 53.0,
                "id": 324,
            },
            "3": {
                "name": "(12C)2(14N)(15N)",
                "abun": 0.0073,
                "mass": 53.0,
                "id": 225,
            }
        },
        "mmw": 52.00000000000001
    },
    "30": {
        "name": "C4H2",
        "isotope": {
            "1": {
                "name": "(12C)4H2",
                "abun": 0.9777777,
                "mass": 50.015650,
                "id": 211,
            },
            "2": {
                "name": "(12C)3(13C)H2",
                "abun": 0.0222222,
                "mass": 51.0062,
                "id": 311,
            }
        },
        "mmw": 50.07975052863
    },
    "31": {
        "name": "HC3N",
        "isotope": {
            "1": {
                "name": "H(12C)3(14N)",
                "abun": 0.963346,
                "mass": 51.010900,
                "id": 124,
            },
            "2": {
                "name": "H(12C)2(13C)(14N)",
                "abun": 0.0111,
                "mass": 52.0143,
                "id": 134,
            },
            "3": {
                "name": "H(12C)3(15N)",
                "abun": 0.00365,
                "mass": 52.0079,
                "id": 125,
            }
        },
        "mmw": 51.02567679
    },
    "32": {
        "name": "C2H4",
        "isotope": {
            "1": {
                "name": "(12C)2H4",
                "abun": 0.977294,
                "mass": 28.031300,
                "id": 211,
            },
            "2": {
                "name": "(12C)H2(13C)H2",
                "abun": 0.0219595,
                "mass": 29.0347,
                "id": 311,
            }
        },
        "mmw": 28.03240879685
    },
    "33": {
        "name": "GeH4",
        "isotope": {
            "1": {
                "name": "(74Ge)H4",
                "abun": 0.365172,
                "mass": 77.952480,
                "id": 411,
            },
            "2": {
                "name": "(72Ge)H4",
                "abun": 0.274129,
                "mass": 75.953380,
                "id": 211,
            },
            "3": {
                "name": "(70Ge)H4",
                "abun": 0.205072,
                "mass": 73.955550,
                "id": 11,
            },
            "4": {
                "name": "(73Ge)H4",
                "abun": 0.0773,
                "mass": 76.954760,
                "id": 311,
            },
            "5": {
                "name": "(76Ge)H4",
                "abun": 0.0744,
                "mass": 79.952700,
                "id": 611,
            }
        },
        "mmw": 76.6691
    },
    "34": {
        "name": "C3H8",
        "isotope": {
            "1": {
                "name": "(12C)3H8",
                "abun": 1.0,
                "mass": 44.0,
                "id": 211,
            }
        },
        "mmw": 44.0
    },
    "35": {
        "name": "HCOOH",
        "isotope": {
            "1": {
                "name": "H(12C)(16O)(16O)H",
                "abun": 0.983898,
                "mass": 46.00548,
                "id": 261,
            }
        },
        "mmw": 45.264719439
    },
    "36": {
        "name": "H2S",
        "isotope": {
            "1": {
                "name": "H2(32S)",
                "abun": 0.949884,
                "mass": 33.9877,
                "id": 121,
            },
            "2": {
                "name": "H2(33S)",
                "abun": 0.007498,
                "mass": 34.9871,
                "id": 131,
            },
            "3": {
                "name": "H2(34S)",
                "abun": 0.0421369,
                "mass": 35.9835,
                "id": 141,
            }
        },
        "mmw": 34.06292834762
    },
    "37": {
        "name": "COF2",
        "isotope": {
            "1": {
                "name": "(12C)(16O)(19F)2",
                "abun": 0.986544,
                "mass": 65.991720,
                "id": 269,
            },
            "2": {
                "name": "(13C)(16O)(19F)2",
                "abun": 0.011084,
                "mass": 66.995080,
                "id": 369,
        },
        },
        "mmw": 66.1037156848
    },
    "38": {
        "name": "SF6",
        "isotope": {
            "1": {
                "name": "(32S)(19F)6",
                "abun": 0.95018,
                "mass": 145.9625,
                "id": 29,
            },
        },
        "mmw": 138.69064825
    },
    "39": {
        "name": "H2",
        "isotope": {
            "1": {
                "name": "H2",
                "abun": 0.999955,
                "mass": 2.016,
                "id": 11,
            },
            "2": {
                "name": "HD",
                "abun": 0.000311,
                "mass": 3.021825,
                "id": 12,
            }
        },
        "mmw": 2.01604536
    },
    "40": {
        "name": "He",
        "isotope": {
            "1": {
                "name": "He",
                "abun": 1.0,
                "mass": 4.0021,
                "id": 4,
            }
        },
        "mmw": 4.0021
    },
    "41": {
        "name": "AsH3",
        "isotope": {
            "1": {
                "name": "(75As)H3",
                "abun": 1.0,
                "mass": 78.5,
                "id": 181,
            }
        },
        "mmw": 78.5
    },
    "42": {
        "name": "C3H4",
        "isotope": {
            "1": {
                "name": "(12C)3H4",
                "abun": 1.0,
                "mass": 40.0,
                "id": 341,
            }
        },
        "mmw": 40.0
    },
    "43": {
        "name": "ClONO2",
        "isotope": {
            "1": {
                "name": "(35Cl)(16O)(14N)(16O)2",
                "abun": 0.74957,
                "mass": 96.956670,
                "id": 564,
            },
            "2": {
                "name": "(37Cl)(16O)(14N)(16O)2",
                "abun": 0.239694,
                "mass": 98.953720,
                "id": 764,
            }
        },
        "mmw": 97.
    },
    "44": {
        "name": "HO2",
        "isotope": {
            "1": {
                "name": "H(16O)2",
                "abun": 0.995107,
                "mass": 32.997660,
                "id": 166,
            }
        },
        "mmw": 32.836242253900004
    },
    "45": {
        "name": "O",
        "isotope": {
            "1": {
                "name": "(16O)",
                "abun": 0.997628,
                "mass": 15.994920,
                "id": 6,
            }
        },
        "mmw": 15.9569600972
    },
    "46": {
        "name": "NO+",
        "isotope": {
            "1": {
                "name": "(14N)(16O)+",
                "abun": 0.993974,
                "mass": 29.997990,
                "id": 46,
            }
        },
        "mmw": 29.817232052
    },
    "47": {
        "name": "CH3OH",
        "isotope": {
            "1": {
                "name": "(12C)H3(16O)H",
                "abun": 0.98593,
                "mass": 32.0262,
                "id": 211,
            }
        },
        "mmw": 31.575591366
    },
    "48": {
        "name": "H",
        "isotope": {
            "1": {
                "name": "H",
                "abun": 1.0,
                "mass": 1.00794,
                "id": 1,
            }
        },
        "mmw": 1.00794
    },
    "49": {
        "name": "C6H6",
        "isotope": {
            "1": {
                "name": "(12C)6H6",
                "abun": 1.0,
                "mass": 78.11,
                "id": 221,
            }
        },
        "mmw": 78.11
    },
    "50": {
        "name": "CH3CN",
        "isotope": {
            "1": {
                "name": "(12C)H3(12C)(14N)",
                "abun": 0.973866,
                "mass": 41.026550,
                "id": 241,
            }
        },
        "mmw": 41.05
    },
    "51": {
        "name": "CH2NH",
        "isotope": {
            "1": {
                "name": "(12C)H2(14N)H",
                "abun": 1.0,
                "mass": 29.0,
                "id": 241,
            },
            "2": {
                "name": "(13C)H2(14N)H",
                "abun": 0.01053,
                "mass": 30.0,
                "id": 341,
            },
            "3": {
                "name": "(12C)H2(15N)H",
                "abun": 0.01538,
                "mass": 30.0,
                "id": 251,
            },
            "4": {
                "name": "(12C)H2(14N)D",
                "abun": 0.000117,
                "mass": 30.0,
                "id": 242,
            }
        },
        "mmw": 29.78081
    },
    "52": {
        "name": "C2H3CN",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 53.0,
                "id": 241,
                "partition": [
                    -154.793,
                    58.8874,
                    0.93928,
                    0.000643523
                ]
            },
            "2": {
                "abun": 0.01053,
                "mass": 54.0,
                "id": 441,
                "partition": [
                    -170.787,
                    26.8203,
                    0.276754,
                    -0.000234522
                ]
            },
            "3": {
                "abun": 0.01053,
                "mass": 54.0,
                "id": 541,
                "partition": [
                    -172.984,
                    27.1647,
                    0.280301,
                    -0.000237552
                ]
            },
            "4": {
                "abun": 0.01053,
                "mass": 54.0,
                "id": 641,
                "partition": [
                    -175.702,
                    27.5884,
                    0.284693,
                    -0.000241259
                ]
            },
            "5": {
                "abun": 0.000117,
                "mass": 54.0,
                "id": 242,
                "partition": [
                    -190.29,
                    30.0907,
                    0.314149,
                    -0.000271996
                ]
            },
            "6": {
                "abun": 0.01538,
                "mass": 54.0,
                "id": 251,
                "partition": [
                    -175.435,
                    27.546,
                    0.284265,
                    -0.000240851
                ]
            }
        },
        "mmw": 55.54269800000001
    },
    "53": {
        "name": "HCP",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 44.0,
                "id": 123,
                "partition": [
                    0.407958,
                    1.03821,
                    6.67295e-05,
                    -2.16079e-07
                ]
            }
        },
        "mmw": 44.0
    },
    "54": {
        "name": "CS",
        "isotope": {
            "1": {
                "abun": 0.947333,
                "mass": 44.0,
                "id": 22,
                "partition": [
                    0.353804,
                    0.849631,
                    1.39364e-05,
                    -3.6623e-08
                ]
            },
            "2": {
                "abun": 0.010643,
                "mass": 44.0,
                "id": 32,
                "partition": [
                    0.368163,
                    0.899182,
                    2.41945e-05,
                    -6.10961e-08
                ]
            },
            "3": {
                "abun": 0.042024,
                "mass": 44.0,
                "id": 24,
                "partition": [
                    0.358717,
                    0.86316,
                    1.65761e-05,
                    -4.21693e-08
                ]
            }
        },
        "mmw": 43.99999999999999
    },
    "55": {
        "name": "HC5N",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 75.0,
                "id": 241,
                "partition": [
                    0.333717,
                    15.6509,
                    1.10945e-05,
                    -2.04631e-11
                ]
            }
        },
        "mmw": 75.0
    },
    "56": {
        "name": "HC7N",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 99.0,
                "id": 241,
                "partition": [
                    -246.6,
                    130.5,
                    -0.27785,
                    0.000216506
                ]
            }
        },
        "mmw": 99.0
    },
    "57": {
        "name": "C2H5CN",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 55.0,
                "id": 241,
                "partition": [
                    -242.048,
                    37.9879,
                    0.391607,
                    -0.000332892
                ]
            },
            "2": {
                "abun": 0.01053,
                "mass": 56.0,
                "id": 441,
                "partition": [
                    -922.629,
                    110.609,
                    1.14061,
                    -0.000839605
                ]
            },
            "3": {
                "abun": 0.01053,
                "mass": 56.0,
                "id": 541,
                "partition": [
                    -932.681,
                    111.813,
                    1.15301,
                    -0.000848733
                ]
            },
            "4": {
                "abun": 0.01053,
                "mass": 56.0,
                "id": 641,
                "partition": [
                    -946.039,
                    113.398,
                    1.16931,
                    -0.000860716
                ]
            },
            "5": {
                "abun": 0.01538,
                "mass": 56.0,
                "id": 251,
                "partition": [
                    6.18308,
                    18.21,
                    0.568471,
                    -0.000674954
                ]
            }
        },
        "mmw": 57.630320000000005
    },
    "58": {
        "name": "CH3NH2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 31.0,
                "id": 124,
                "partition": [
                    -813.72,
                    126.598,
                    1.30776,
                    -0.00111831
                ]
            }
        },
        "mmw": 31.0
    },
    "59": {
        "name": "HNC",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 27.0,
                "id": 142,
                "partition": [
                    0.349399,
                    0.459311,
                    2.92041e-06,
                    -3.51095e-09
                ]
            },
            "2": {
                "abun": 0.01053,
                "mass": 28.0,
                "id": 143,
                "partition": [
                    0.357422,
                    0.477576,
                    9.6211e-06,
                    -2.27844e-08
                ]
            },
            "3": {
                "abun": 0.01538,
                "mass": 28.0,
                "id": 152,
                "partition": [
                    0.353799,
                    0.46832,
                    6.59346e-06,
                    -1.42053e-08
                ]
            },
            "4": {
                "abun": 0.000117,
                "mass": 28.0,
                "id": 242,
                "partition": [
                    0.34557,
                    0.545889,
                    2.50195e-06,
                    -2.46926e-09
                ]
            }
        },
        "mmw": 27.728756
    },
    "60": {
        "name": "Na",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 22.99,
                "id": 11,
                "partition": [
                    4.39393,
                    -0.00433822,
                    2.39818e-06,
                    -4.12438e-10
                ]
            }
        },
        "mmw": 22.99
    },
    "61": {
        "name": "K",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 39.1,
                "id": 19,
                "partition": [
                    6.01454,
                    -0.00722715,
                    3.97413e-06,
                    -6.80502e-10
                ]
            }
        },
        "mmw": 39.1
    },
    "62": {
        "name": "TiO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 63.8664,
                "id": 486,
                "partition": [
                    -860.00527,
                    5.25794,
                    0.00411768,
                    4.44797e-07
                ]
            }
        },
        "mmw": 63.8664
    },
    "63": {
        "name": "VO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 66.9409,
                "id": 516,
                "partition": [
                    -922.20713,
                    4.96349,
                    0.0024269,
                    1.66643e-07
                ]
            }
        },
        "mmw": 66.9409
    },
    "64": {
        "name": "CH2CCH2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 40.0646,
                "id": 221,
                "partition": [
                    -0.104157,
                    0.00355223,
                    -1.14095e-05,
                    6.36864e-08
                ]
            }
        },
        "mmw": 40.0646
    },
    "65": {
        "name": "C4N2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 76.0566,
                "id": 224,
                "partition": [
                    -1.92827,
                    0.0572307,
                    -0.000488527,
                    1.72324e-06
                ]
            }
        },
        "mmw": 76.0566
    },
    "66": {
        "name": "C5H5N",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 79.1,
                "id": 241,
                "partition": [
                    -559.899,
                    87.6703,
                    0.901869,
                    -0.000772987
                ]
            }
        },
        "mmw": 79.1
    },
    "67": {
        "name": "C5H4N2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 80.088,
                "id": 241,
                "partition": [
                    -526.099,
                    82.396,
                    0.847528,
                    -0.000726356
                ]
            }
        },
        "mmw": 80.088
    },
    "68": {
        "name": "C7H8",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 92.14,
                "id": 221,
                "partition": [
                    -1127.29,
                    177.06,
                    1.8376,
                    -0.00159469
                ]
            }
        },
        "mmw": 92.14
    },
    "69": {
        "name": "C8H6",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 102.133,
                "id": 221,
                "partition": [
                    -1655.61,
                    268.028,
                    2.94334,
                    -0.00278395
                ]
            }
        },
        "mmw": 102.133
    },
    "70": {
        "name": "C5H5CN",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 103.12,
                "id": 241,
                "partition": [
                    -7009.17,
                    1095.06,
                    11.2784,
                    -0.0096696
                ]
            }
        },
        "mmw": 103.12
    },
    "71": {
        "name": "HOBr",
        "isotope": {
            "1": {
                "abun": 0.505579,
                "mass": 95.921076,
                "id": 169,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            },
            "2": {
                "abun": 0.491894,
                "mass": 97.919027,
                "id": 161,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 96.661463550142
    },
    "72": {
        "name": "CH3Br",
        "isotope": {
            "1": {
                "name": "(12C)H3(79Br)",
                "abun": 0.500995,
                "mass": 93.941810,
                "id": 219,
            },
            "2": {
                "name": "(12C)H3(81Br)",
                "abun": 0.487433,
                "mass": 95.939760,
                "id": 211,
            }
        },
        "mmw": 93.828584587757
    },
    "73": {
        "name": "CF4",
        "isotope": {
            "1": {
                "name": "(12C)(19F)4",
                "abun": 0.98889,
                "mass": 87.993616,
                "id": 29,
            },
        },
        "mmw": 88.01600692624001
    },
    "74": {
        "name": "SO3",
        "isotope": {
            "1": {
                "name": "(32S)(16O)3",
                "abun": 0.943434,
                "mass": 79.956820,
                "id": 26,
            },
        },
        "mmw": 80.431263988
    },
    "75": {
        "name": "Ne",
        "isotope": {
            "1": {
                "name": "(20Ne)",
                "abun": 0.9048,
                "mass": 19.99244,
                "id": 20,
            },
            "2": {
                "name": "(22Ne)",
                "abun": 0.0925,
                "mass": 21.991385,
                "id": 22,
            },
            "3": {
                "name": "(21Ne)",
                "abun": 0.0027,
                "mass": 20.993847,
                "id": 21,
            }
        },
        "mmw": 20.1800462114
    },
    "76": {
        "name": "Ar",
        "isotope": {
            "1": {
                "abun": 0.996035,
                "mass": 39.962383,
                "id": 40,
                "partition": [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            "2": {
                "abun": 0.003336,
                "mass": 35.967545,
                "id": 36,
                "partition": [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ]
            },
            "3": {
                "abun": 0.000629,
                "mass": 37.962732,
                "id": 38,
                "partition": [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 39.947798439953004
    },
    "77": {
        "name": "COCl2",
        "isotope": {
            "1": {
                "name": "(12C)(16O)(35Cl)2",
                "abun": 0.566392,
                "mass": 97.93262,
                "id": 2655,
            },
            "2": {
                "name": "(12C)(16O)(35Cl)(37Cl)",
                "abun": 0.362235,
                "mass": 99.929670,
                "id": 2657,
            }
        },
        "mmw": 98.66627651949
    },
    "78": {
        "name": "SO",
        "isotope": {
            "1": {
                "name": "(32S)(16O)",
                "abun": 0.947926,
                "mass": 47.966990,
                "id": 26,
            },
            "2": {
                "name": "(34S)(16O)",
                "abun": 0.042050,
                "mass": 49.962780,
                "id": 46,
            },
            "3": {
                "name": "(32S)(18O)",
                "abun": 0.001901,
                "mass": 49.971230,
                "id": 48,
            },
        },
        "mmw": 48.064
    },
    "79": {
        "name": "H2SO4",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 98.079,
                "id": 13616,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 98.079
    },
    "80": {
        "name": "e-",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 0.000545,
                "id": 111,
                "partition": [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 0.000545
    },
    "81": {
        "name": "H3+",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 3.02352,
                "id": 111,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 3.02352
    },
    "82": {
        "name": "FeH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 56.85284,
                "id": 61,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 56.85284
    },
    "83": {
        "name": "AlO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 42.980539,
                "id": 76,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 42.980539
    },
    "84": {
        "name": "AlCl",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 62.434539,
                "id": 75,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 62.434539
    },
    "85": {
        "name": "AlF",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 45.979942,
                "id": 79,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 45.979942
    },
    "86": {
        "name": "AlH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 27.989379,
                "id": 71,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 27.989379
    },
    "87": {
        "name": "BeH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 10.020022,
                "id": 91,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 10.020022
    },
    "88": {
        "name": "C2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 24.0214,
                "id": 22,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 24.0214
    },
    "89": {
        "name": "CaF",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 59.076403,
                "id": 409,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 59.076403
    },
    "90": {
        "name": "CaH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 41.08584,
                "id": 401,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 41.08584
    },
    "91": {
        "name": "H-",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 1.008,
                "id": 111,
                "partition": [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 1.008
    },
    "92": {
        "name": "CaO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 56.077,
                "id": 406,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 56.077
    },
    "93": {
        "name": "CH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 13.01854,
                "id": 21,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 13.01854
    },
    "94": {
        "name": "CH3",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 15.03422,
                "id": 211,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 15.03422
    },
    "95": {
        "name": "CH3F",
        "isotope": {
            "1": {
                "name": "(12C)H3(19F)",
                "abun": 0.988428,
                "mass": 34.021880,
                "id": 219,
            }
        },
        "mmw": 34.032623
    },
    "96": {
        "name": "CN",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 26.0174,
                "id": 24,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 26.0174
    },
    "97": {
        "name": "CP",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 42.984462,
                "id": 231,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 42.984462
    },
    "98": {
        "name": "CrH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 53.00394,
                "id": 521,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 53.00394
    },
    "99": {
        "name": "HD+",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 3.02194178,
                "id": 11,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 3.02194178
    },
    "100": {
        "name": "HeH+",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 5.010442,
                "id": 41,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 5.010442
    },
    "101": {
        "name": "KCl",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 74.5513,
                "id": 395,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 74.5513
    },
    "102": {
        "name": "KF",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 58.096703,
                "id": 399,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 58.096703
    },
    "103": {
        "name": "LiCl",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 42.394,
                "id": 75,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 42.394
    },
    "104": {
        "name": "LiF",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 25.939403,
                "id": 79,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 25.939403
    },
    "105": {
        "name": "LiH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 7.94884,
                "id": 71,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 7.94884
    },
    "106": {
        "name": "LiH+",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 7.94884,
                "id": 71,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 7.94884
    },
    "107": {
        "name": "MgF",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 43.303403,
                "id": 249,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 43.303403
    },
    "108": {
        "name": "MgH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 25.31284,
                "id": 241,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 25.31284
    },
    "109": {
        "name": "MgO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 40.304,
                "id": 246,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 40.304
    },
    "110": {
        "name": "NaCl",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 58.4427693,
                "id": 235,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 58.4427693
    },
    "111": {
        "name": "NaF",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 41.9881723,
                "id": 239,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 41.9881723
    },
    "112": {
        "name": "NaH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 23.9976093,
                "id": 231,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 23.9976093
    },
    "113": {
        "name": "NH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 15.01454,
                "id": 41,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 15.01454
    },
    "114": {
        "name": "NS",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 46.0717,
                "id": 42,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 46.0717
    },
    "115": {
        "name": "OH+",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 17.00684,
                "id": 61,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 17.00684
    },
    "116": {
        "name": "cis-P2H2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 63.963204,
                "id": 311,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 63.963204
    },
    "117": {
        "name": "trans-P2H2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 63.963204,
                "id": 311,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 63.963204
    },
    "118": {
        "name": "PH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 31.981602,
                "id": 311,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 31.981602
    },
    "119": {
        "name": "PN",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 44.980462,
                "id": 314,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 44.980462
    },
    "120": {
        "name": "PO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 46.972762,
                "id": 316,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 46.972762
    },
    "121": {
        "name": "PS",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 63.038762,
                "id": 312,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 63.038762
    },
    "122": {
        "name": "ScH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 45.963752,
                "id": 451,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 45.963752
    },
    "123": {
        "name": "SH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 33.07284,
                "id": 21,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 33.07284
    },
    "124": {
        "name": "SiH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 29.09334,
                "id": 281,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 29.09334
    },
    "125": {
        "name": "SiH2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 30.10118,
                "id": 2811,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 30.10118
    },
    "126": {
        "name": "SiH4",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 32.11686,
                "id": 2811,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 32.11686
    },
    "127": {
        "name": "SiO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 44.0845,
                "id": 286,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 44.0845
    },
    "128": {
        "name": "SiO2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 60.0835,
                "id": 2866,
                "partition": [
                    -44.219,
                    5.8047,
                    0.050145,
                    -3.7316e-05
                ]
            }
        },
        "mmw": 60.0835
    },
    "129": {
        "name": "SiS",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 60.1505,
                "id": 282,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 60.1505
    },
    "130": {
        "name": "TiH",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 48.87484,
                "id": 481,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 48.87484
    },
    "131": {
        "name": "Cl2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 69.9377,
                "id": 10000,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 69.9377
    },
    "132": {
        "name": "ClO2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 67.45,
                "id": 10001,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 67.45
    },
    "133": {
        "name": "BrO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 95.903,
                "id": 10001,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 95.903
    },
    "134": {
        "name": "IO",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 142.903,
                "id": 10001,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw": 142.903
    },
    "135": {
        "name": "NO3",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 62.005,
                "id": 10001,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw":62.005
    },
    "136": {
        "name": "BrO2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 111.903,
                "id": 10001,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw":111.903
    },
    "137": {
        "name": "IO2",
        "isotope": {
            "1": {
                "abun": 1.0,
                "mass": 158.903,
                "id": 10001,
                "partition": [
                    0.0,
                    10.0,
                    0.0,
                    0.0
                ]
            }
        },
        "mmw":158.903
    },
    "138": {
        "name": "CS2",
        "isotope": {
            "1": {
                "name": "(12C)(32S)2",
                "abun": 0.892811,
                "mass": 75.944140,
                "id": 222,
            },
            "2": {
                "name": "(32S)(12C)(34S)",
                "abun": 0.079210,
                "mass": 77.939940,
                "id": 224,
            },
            "3": {
                "name": "(32S)(12C)(33S)",
                "abun": 0.014094,
                "mass": 76.943530,
                "id": 223,
            },
            "4": {
                "name": "(13C)(32S)2",
                "abun": 0.010031,
                "mass": 76.947500,
                "id": 322,
            },
        },
        "mmw": 76.
    },
    "139": {
        "name": "CH3I",
        "isotope": {
            "1": {
                "name": "(12C)H3(127I)",
                "abun": 0.988428,
                "mass": 141.927900,
                "id": 217,
            },
        },
        "mmw": 142.
    },
    "140": {
        "name": "NF3",
        "isotope": {
            "1": {
                "name": "(14N)(19F)3",
                "abun": 0.996337,
                "mass": 70.998290,
                "id": 49,
            },
        },
        "mmw": 71.
    },
}

svp_coefficients = {
    1: (15.278, -5980.3, 8.8294e-3, -1.2169e-5),
    2: (25.24826, -3882.969, -2.722391e-2, 0.0),
    6: (10.6944, -1163.83, 0.0, 0.0),
    11: (23.224, -4245.8, -2.2775e-2, 0.0),
    28: (11.44684, -1974.438, -4.358464e-3, 0.0),
    33: (15.85, -2108.6, -4.0205e-2, 8.8775e-5),
    41: (38.7, -3961.2, -0.14707, 2.4937e-4),
    26: (21.41126, -3263.087, -2.262256e-2, 0.0),
    32: (11.80221, -1858.863, -4.911005e-3, 0.0),
    27: (12.96829, -2148.377, -7.132374e-3, 0.0),
    34: (13.47137, -2746.288, -6.834743e-3, 0.0),
    23: (19.62791, -5337.395, 0.0, 0.0),
    31: (13.46153, -4222.267, 0.0, 0.0),
    42: (13.1954, -3049.57, -0.0040089, 0.0),
    30: (37.9230, -6646.29, -0.0510657, 0.0),
    49: (13.60930, -4740.271, 0.0, 0.0),
    5: (9.847951, -816.7098, 0.0, 0.0),
    57: (11.8004, -4352.66, 0.0, 0.0),
    36: (12.8713, -2702.37, 0.0, 0.0)
}
