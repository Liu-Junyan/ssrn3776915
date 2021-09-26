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
from timeit import default_timer as timer


def main():
    # for filename in os.scandir('./data/'):
    #     filter_by_nrows(filename)
    i = 0
    for filename in os.scandir('./pkl/'):
        i += 1
        if i == 10:
            break
        mat: pd.core.frame.DataFrame = pickle.load(open(filename, 'rb'))
        res: pd.core.frame.DataFrame = mat.groupby(0)[[2]].agg(realized_variance_d)
        res.to_pickle('./res/' + filename.name)


if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print(end - start)
    pass

# mat: pd.core.frame.DataFrame = pickle.load(open('./pkl/SH600000.pkl', 'rb'))
# grouped = mat.groupby(0)[[2]]
# temp = grouped.agg(realized_variance_d)
# temp1 = grouped.agg(realized_variance_d_1)
