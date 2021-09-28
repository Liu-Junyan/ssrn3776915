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
    #     mat: pd.DataFrame = pickle.load(open(filename, 'rb'))
    #     res: pd.DataFrame = mat.groupby(0)[[2]].agg(realized_variance_d_vec)
    #     res.to_pickle('./res/' + filename.name)
    # filter_by_ndates()
    # for filename in os.scandir('./res/'):
    #     res: pd.DataFrame = pd.read_pickle(filename.path)
    #     res.rename(columns={2: 'RV^d'}, inplace=True)
    #     res.to_pickle(filename.path)
    # for filename in os.scandir('./res/'):
    #     res: pd.DataFrame = pd.read_pickle(filename.path)
    #     res['RV^w'] = res['RV^d'].rolling(5).mean()
    #     res['RV^m'] = res['RV^d'].rolling(21).mean()
    #     res['RV^q'] = res['RV^d'].rolling(63).mean()
    #     res.to_pickle(filename.path)
    # sample = random.sample(list(os.scandir('./res/')), 100)
    # for filename in sample:
    #     os.popen(f'cp {filename.path} ./sp/{filename.name}')
    # (theta2_d, theta2_w, theta2_m, theta2_q) = MIDAS_grid_search()
    # (theta2_d, theta2_w, theta2_m, theta2_q) = (20, 10, 5, 4)
    
    # for filename in os.scandir('./sp/'):
    #     res: pd.DataFrame = pd.read_pickle(filename.path)
    #     res['MIDAS^d'] = res['RV^d'].rolling(50).apply(MIDAS, args=(20,))
    #     res['MIDAS^w'] = res['RV^d'].rolling(50).apply(MIDAS, args=(10,))
    #     res['MIDAS^m'] = res['RV^d'].rolling(50).apply(MIDAS, args=(5,))
    #     res['MIDAS^q'] = res['RV^d'].rolling(50).apply(MIDAS, args=(4,))
    #     res.to_pickle(filename.path)
    
    # for filename in os.scandir('./sp/'):
    #     mat: pd.DataFrame = pd.read_pickle('./pkl/' + filename.name)
    #     df: pd.Series = mat.groupby(0)[2].agg(realized_semivariance_d_vec)
    #     df = df.apply(pd.Series)
    #     res = pd.read_pickle(filename)
    #     res[['RVP^d', 'RVN^d']] = df
    #     res.to_pickle(filename)
        
    # for filename in os.scandir('./sp/'):
    #     mat: pd.DataFrame = pd.read_pickle('./pkl/' + filename.name)
    #     df = mat.groupby(0)[[2]].agg(realized_quarticity_d)
    #     df.rename(columns={2: 'RQ^d'}, inplace=True)
    #     df.to_pickle('./temp/' + filename.name)
        
    # for filename in os.scandir('./temp/'):
    #     df = pd.read_pickle(filename)
    #     for period in Period:
    #         df[f'RQ^{period.name}'] = df['RQ^d'].rolling(period.value).mean()
    #     df = np.sqrt(df)
    #     res = pd.read_pickle('./res/' + filename.name)
    #     for period in Period:
    #         res[f'HARQ^{period.name}'] = df[f'RQ^{period.name}'] * res[f'RV^{period.name}']
    #     res.to_pickle('./res/' + filename.name)
    
    # for filename in os.scandir('./sp/'):
    #     res = pd.read_pickle(filename)
    #     for CoM in (1, 5, 25, 125):    
    #         res[f'ExpRV^{CoM}'] = res['RV^d'].rolling(500).apply(Exp_realized_variance, args=(CoM,))
    #     res.to_pickle(filename)
    
    # all_tradedays = []
    # for filename in os.scandir('./sp/'):
    #     res = pd.read_pickle(filename)
    #     all_tradedays = np.union1d(all_tradedays, res.index)
    # RV_all = pd.DataFrame(index=all_tradedays)
    # for filename in os.scandir('./sp/'):
    #     res = pd.read_pickle(filename)
    #     RV_all[filename.name[:-4]] = res['RV^d']
    # RV_all.to_pickle('RV_all.pkl')

    # RV_all = pd.read_pickle('RV_all.pkl')
    # LM_all = RV_all
    # for i in range(RV_all.shape[0]):
    #     print(i)
    #     LM_all.iloc[i] = np.mean(RV_all.iloc[:i+1])
    # LM_all.to_pickle('LM_all.pkl')

    # RV_all = pd.read_pickle('RV_all.pkl')
    # RV_all = RV_all.interpolate()
    # LM_all = pd.read_pickle('LM_all.pkl')
    # for filename in os.scandir('./sp/'):
    #     res = pd.read_pickle(filename)
    #     RV_sub = RV_all.loc[res.index]
    #     LM_sub = LM_all.loc[res.index]
    #     res['GlRV'] = 1 / RV_all.shape[1] * np.sum(RV_sub / LM_sub, axis=1) * LM_sub[filename.name[:-4]]
    #     res.to_pickle(filename)

    for filename in os.scandir('./sp/'):
        print(filename.name)
        res = pd.read_pickle(filename)
        CoM = 5
        lmbda = np.log(1 + 1 / CoM)
        exp_list = [np.exp(-i * lmbda) for i in range(1, 501)]
        res['ExpGlRV'] = res['GlRV'].rolling(500).apply(Exp_realized_variance_expl, args=(exp_list,))
        res.to_pickle(filename)

    pass

if __name__ == '__main__':
    main()
    pass

