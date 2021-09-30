#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:07:11 2021

@author: johnlau
"""

from enum import Enum


class Period(Enum):
    d = 1
    w = 5
    m = 21
    q = 63


FEATURE_SET = [
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

RESPONSE_SET = ["RV_res^d", "RV_res^w", "RV_res^m", "RV_res^q"]
