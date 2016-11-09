# -*- coding:utf-8 -*- 
from hmmlearn.hmm import GaussianHMM
import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import tushare as ts
import datetime

from scipy import stats # To perfrom box-cox transformation
from sklearn import preprocessing # To center and standardize the data.

# Source Code from previous HMM modeling

# Note that numbers of hidden states are modified to be 3, instead of 6.

beginDate = '2005-01-01'
endDate = '2015-12-31'
n = 3 # Hidden states are set to be 3 instead of 6
_index=ts.get_hist_data('sh')
volume = _index['volume'][:600]
close = _index['close'][:600]

#volume = data['TotalVolumeTraded']
#close = data['ClosingPx']

logDel = np.log(np.array(_index['high'][:600])) - np.log(np.array(_index['low'][:600]
))
#day change
logDel

logRet_1 = np.array(np.diff(np.log(close)))#这个作为后面计算收益使用
#one day change
logRet_5 = np.log(np.array(close[5:])) - np.log(np.array(close[:-5]))
#five day change
logRet_5

logVol_5 = np.log(np.array(volume[5:])) - np.log(np.array(volume[:-5]))
#five day volumn change
logVol_5

logDel = logDel[5:]
logRet_1 = logRet_1[4:]
close = close[5:]
#omit first several days
Date = pd.to_datetime(_index['high'])

# Box-Cox Transformation of the observation sequences

boxcox_logDel, _ = stats.boxcox(logDel)

# Standardize the observation sequence distribution

rescaled_boxcox_logDel = preprocessing.scale(boxcox_logDel, axis=0, with_mean=True, with_std=True, copy=False)

rescaled_logRet_5 = preprocessing.scale(logRet_5, axis=0, with_mean=True, with_std=True, copy=False)

rescaled_logVol_5 = preprocessing.scale(logVol_5, axis=0, with_mean=True, with_std=True, copy=False)

# Box-Cox Transformation of the observation sequences

boxcox_logDel, _ = stats.boxcox(logDel)

# Standardize the observation sequence distribution

rescaled_boxcox_logDel = preprocessing.scale(boxcox_logDel, axis=0, with_mean=True, with_std=True, copy=False)

rescaled_logRet_5 = preprocessing.scale(logRet_5, axis=0, with_mean=True, with_std=True, copy=False)

rescaled_logVol_5 = preprocessing.scale(logVol_5, axis=0, with_mean=True, with_std=True, copy=False)

# Box-Cox Transformation of the observation sequences

boxcox_logDel, _ = stats.boxcox(logDel)

# Standardize the observation sequence distribution

rescaled_boxcox_logDel = preprocessing.scale(boxcox_logDel, axis=0, with_mean=True, with_std=True, copy=False)

rescaled_logRet_5 = preprocessing.scale(logRet_5, axis=0, with_mean=True, with_std=True, copy=False)

rescaled_logVol_5 = preprocessing.scale(logVol_5, axis=0, with_mean=True, with_std=True, copy=False)

# Observation sequences matrix 
A = np.column_stack([logDel,logRet_5,logVol_5]) 

# Rescaled observation sequences matrix 
rescaled_A = np.column_stack([rescaled_boxcox_logDel, rescaled_logRet_5, rescaled_logVol_5]) 

# HMM modeling based on raw observation sequences

model = GaussianHMM(n_components= 3, covariance_type="full", n_iter=2000).fit([A])
hidden_states = model.predict(A)
hidden_states

plt.figure(figsize=(25, 18)) 
for i in range(model.n_components):
    pos = (hidden_states==i)
    plt.plot_date(Date[pos],close[pos],'o',label='hidden state %d'%i,lw=2)
    plt.legend(loc="left")