from __future__ import annotations #  for 3.9 compatability
"""
Perform compatability setup for python version 3.9
"""

import abc
import typing, typing_extensions

typing.Self = typing_extensions.Self


def ABCMeta_or(self, *args):
    return typing.Union[self, *args]

abc.ABCMeta.__or__ = ABCMeta_or


