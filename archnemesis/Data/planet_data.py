#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# planet_data.py - Data reference files for planetary parameters.
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

Contains the parameters describing different planets.
"""

import logging
_lgr = logging.getLogger(__name__)


planet_info = {
    "1": {
        "name": "Mercury",
        "mass": 0.33022e24,  #kg 
        "radius": 2439.7,    #km
        "flatten": 0.0,      
        "rotation": 58.6462, #days
        "isurf": 1,
        "Jcoeff": [0.0,0.0,0.0]
        },
    "2": {
        "name": "Venus",
        "mass": 4.8690e24,  #kg
        "radius": 6051.8,    #km
        "flatten": 0.0,      
        "rotation": -243.0187, #days
        "isurf": 1,
        "Jcoeff": [0.027,0.0,0.0]
        },
    "3": {
        "name": "Earth",
        "mass": 5.9742e24,  #kg
        "radius": 6378.14,    #km
        "flatten": 0.00335364,      
        "rotation": 0.99726968, #days
        "isurf": 1,
        "Jcoeff": [1.08263,-2.54,-1.61]
        },
    "4": {
        "name": "Mars",
        "mass": 0.64191e24,  #kg
        "radius": 3397.0,    #km
        "flatten": 0.006476,      
        "rotation": 1.02595675, #days
        "isurf": 1,
        "Jcoeff": [1.964,36.0,0.0]
        }, 
    "5": {
        "name": "Jupiter",
        "mass": 1898.2e24,  #kg
        "radius": 71492.0,    #km
        "flatten": 0.064874,      
        "rotation": 0.41354, #days
        "isurf": 0,
        "Jcoeff": [14.75,0.0,-580.0]
        },
    "6": {
        "name": "Saturn",
        "mass": 568.5e24,  #kg
        "radius": 60268.0,    #km
        "flatten": 0.097962,      
        "rotation": 0.44401, #days
        "isurf": 0,
        "Jcoeff": [16.45,0.0,-1000.0]
        },
    "7": {
        "name": "Uranus",
        "mass": 86.625e24,  #kg
        "radius": 25559.0,    #km
        "flatten": 0.022000,      
        "rotation": -0.71833, #days
        "isurf": 0,
        "Jcoeff": [12.0,0.0,0.0]
        },
    "8": {
        "name": "Neptune",
        "mass": 102.78e24,  #kg
        "radius": 24764.0,    #km
        "flatten": 0.017081,      
        "rotation": 0.67125, #days
        "isurf": 0,
        "Jcoeff": [4.0,0.0,0.0]
        },
    "9": {
        "name": "Pluto",
        "mass": 0.015e24,  #kg
        "radius": 1151.0,    #km
        "flatten": 0.0,      
        "rotation": -6.3872, #days
        "isurf": 1,
        "Jcoeff": [0.0,0.0,0.0]
        },
    "10": {
        "name": "Sun",
        "mass": 1989000.0e24,  #kg
        "radius": 695000.0,    #km
        "flatten": 0.0,      
        "rotation": 25.38, #days
        "isurf": 0,
        "Jcoeff": [0.0,0.0,0.0]
        },
    "11": {
        "name": "Titan",
        "mass": 0.1353e24,  #kg
        "radius": 2575.0,    #km
        "flatten": 0.0,      
        "rotation": 15.945, #days
        "isurf": 1,
        "Jcoeff": [0.0,0.0,0.0]
        },
    "85": {
        "name": "NGTS-10b",
        "mass": 4103.757e24,  #kg
        "radius": 86147.86,    #km
        "flatten": 0.0,      
        "rotation": 100000.0, #days
        "isurf": 0,
        "Jcoeff": [0.0,0.0,0.0]
        },
    "87": {
        "name": "WASP-43b",
        "mass": 3895.110e24,  #kg
        "radius": 74065.70,    #km
        "flatten": 0.0,      
        "rotation": 100000.0, #days
        "isurf": 0,
        "Jcoeff": [0.0,0.0,0.0]
        }
}