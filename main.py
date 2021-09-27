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
from sys import exit
import random



def main():
    # for filename in os.scandir('./data/'):
    #     filter_by_nrows(filename)
    # for filename in os.scandir('./pkl/'):
    #     mat: pd.core.frame.DataFrame = pickle.load(open(filename, 'rb'))
    #     res: pd.core.frame.DataFrame = mat.groupby(0)[[2]].agg(realized_variance_d_vec)
    #     res.to_pickle('./res/' + filename.name)
    # filter_by_ndates()
    # for filename in os.scandir('./res/'):
    #     res: pd.core.frame.DataFrame = pd.read_pickle(filename.path)
    #     res.rename(columns={2: 'RV^d'}, inplace=True)
    #     res.to_pickle(filename.path)
    # for filename in os.scandir('./res/'):
    #     res: pd.core.frame.DataFrame = pd.read_pickle(filename.path)
    #     res['RV^w'] = res['RV^d'].rolling(5).mean()
    #     res['RV^m'] = res['RV^d'].rolling(21).mean()
    #     res['RV^q'] = res['RV^d'].rolling(63).mean()
    #     res.to_pickle(filename.path)
    # sample = random.sample(list(os.scandir('./res/')), 100)
    # for filename in sample:
    #     os.popen(f'cp {filename.path} ./sp/{filename.name}')
    MIDAS_grid_search()
    # for filename in os.scandir('./sp/'):
    #     res: pd.core.frame.DataFrame = pd.read_pickle(filename.path)
    #     temp = res['RV^d'].rolling(50).apply(MIDAS)
    #     exit(1)
    #     pass
    
    pass

if __name__ == '__main__':
    main()
    pass

