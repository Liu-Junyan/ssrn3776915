# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 11:42:02 2021

@author: John
"""

import os
from util import *
import pickle
import pandas as pd
import numpy as np



def main():
    # for filename in os.scandir('./data/'):
    #     filter_by_nrows(filename)
    # for filename in os.scandir('./pkl/'):
    #     mat: pd.core.frame.DataFrame = pickle.load(open(filename, 'rb'))
    #     res: pd.core.frame.DataFrame = mat.groupby(0)[[2]].agg(realized_variance_d_vec)
    #     res.to_pickle('./res/' + filename.name)
    pass


if __name__ == '__main__':
    main()
    pass

