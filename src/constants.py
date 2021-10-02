#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:07:11 2021

@author: johnlau
"""

from enum import Enum
from typing import List


class Period(Enum):
    d = 1
    w = 5
    m = 21
    q = 63


FEATURE_SET_ALL: List[str] = [
    "RV^d",
    "RV^w",
    "RV^m",
    "RV^q",
    "MIDAS^d",
    "MIDAS^w",
    "MIDAS^m",
    "MIDAS^q",
    "RVP^d",
    "RVN^d",
    "HARQ^d",
    "HARQ^w",
    "HARQ^m",
    "HARQ^q",
    "ExpRV^1",
    "ExpRV^5",
    "ExpRV^25",
    "ExpRV^125",
    "ExpGlRV",
]

T_START: int = 2004
"""The starting year of testing set (or validation set for ML methods) (inclusive)."""

T_END: int = 2022
"""The ending year of testing set (noninclusive)."""
