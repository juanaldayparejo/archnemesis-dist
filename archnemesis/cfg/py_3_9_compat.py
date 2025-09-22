from __future__ import annotations #  for 3.9 compatability
"""
Perform compatability setup for python version 3.9
"""

import abc
import typing, typing_extensions
from typing import NoneType

typing.Self = typing_extensions.Self


def UNION_or(self, other):
    return typing.Union[self, other]

abc.ABCMeta.__or__ = UNION_or

NoneType.__or__ = UNION_or

