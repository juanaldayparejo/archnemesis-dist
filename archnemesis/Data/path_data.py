#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# path_data.py - Functions to handle file paths within the archNEMESIS project.
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

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

ARCHNEMESIS_PATH_PLACEHOLDER='ARCHNEMESIS_PATH/'

def archnemesis_path():
    import os
    nemesis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../')
    return nemesis_path

def archnemesis_resolve_path(path : str):
    if path.startswith(ARCHNEMESIS_PATH_PLACEHOLDER):
        return archnemesis_path() + path[len(ARCHNEMESIS_PATH_PLACEHOLDER):]
    else:
        return path

def archnemesis_indirect_path(path : str):
    if path.startswith(archnemesis_path()):
        return ARCHNEMESIS_PATH_PLACEHOLDER + path[len(archnemesis_path()):]
    else:
        return path