import pandas as pd
import numpy as np
from datetime import datetime
import psutil 
import gc #memory garbage collection

import caffeinated_pandas_utils as cp
from scipy import stats as scipy_stats

import time


def add_stats(df, bars1, bars2, bars3):

    #----- Groupby symbol and store into an object which can be used for all calculations
    groups = df.groupby('symbol', observed=True)['close'] #observed=True needed for categorical data, otherwise results may be unpredictable

    #----- Rolling Simple Moving Averages
    df['sma_1'] = groups.rolling(bars1).mean().reset_index(drop=True)
    df['sma_2'] = groups.rolling(bars2).mean().reset_index(drop=True)
    df['sma_3'] = groups.rolling(bars3).mean().reset_index(drop=True)

    #----- Rolling Standard Deviation
    df['std_1'] = groups.rolling(bars1).std(ddof=0).reset_index(drop=True)
    df['std_2'] = groups.rolling(bars2).std(ddof=0).reset_index(drop=True)
    df['std_3'] = groups.rolling(bars3).std(ddof=0).reset_index(drop=True)

    #----- Rollling Higher Highs and Lower Lows - Donchian Channels - https://www.investopedia.com/terms/d/donchianchannels.asp
    df['hh_1']  = groups.rolling(bars1).max().reset_index(drop=True)
    df['hh_2']  = groups.rolling(bars2).max().reset_index(drop=True)
    df['hh_3']  = groups.rolling(bars3).max().reset_index(drop=True)

    df['ll_1']  = groups.rolling(bars1).min().reset_index(drop=True)
    df['ll_2']  = groups.rolling(bars2).min().reset_index(drop=True)
    df['ll_3']  = groups.rolling(bars3).min().reset_index(drop=True)

    #----- Maximum Drawdown - https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp
    df['maxdd_1'] = groups.rolling(bars1).apply(lambda x: np.min(x / np.maximum.accumulate(x)) - 1).reset_index(drop=True)
    df['maxdd_2'] = groups.rolling(bars2).apply(lambda x: np.min(x / np.maximum.accumulate(x)) - 1).reset_index(drop=True)
    df['maxdd_3'] = groups.rolling(bars3).apply(lambda x: np.min(x / np.maximum.accumulate(x)) - 1).reset_index(drop=True)

    #----- R-Value not used as it is demonstrated here for any type of technical analysis
    #----- Included here because it's a CPU intensive function and good for this demo!
    def calc_rvalue(ts):
        x = np.arange(len(ts))
        log_ts = np.log(ts)
        slope, intercept, rvalue, pvalue, std_err = scipy_stats.linregress(x, log_ts)
        return rvalue

    df['rvalue_1'] = groups.rolling(bars1).apply(calc_rvalue).reset_index(drop=True)
    df['rvalue_2'] = groups.rolling(bars2).apply(calc_rvalue).reset_index(drop=True)
    df['rvalue_3'] = groups.rolling(bars3).apply(calc_rvalue).reset_index(drop=True)

    return df


if __name__ == '__main__':  #--- confirms that the code is under main function

    #----- Core check
    print('\nPhysical Cores: '+str(psutil.cpu_count(logical=False)), ' / Logical Cores: '+ str(psutil.cpu_count(logical=True)), '\n')

    #----- Create a test Dataframe
    fn = 'stock_test.feather'
    #num_symbols = 10_000
    #num_symbols = 1_000
    num_symbols = 100 #try this first and build up bigger
    df = cp.create_test_dataframe(start_date='2000-01-01', end_date='2019-12-31', num_symbols=num_symbols, squeeze=True) #comment out after first run to re-use file

    #----- Dataframe needs to be sorted by the splitby parameter in the multiproc_dataframe as well as by intended groupby ordering (e.g., groupby symbol, ordered by date) 
    df = df.sort_values(by=['symbol','date'], ascending=True).reset_index(drop=True) #comment out after first run to re-use file

    #----- Write to file for next time
    cp.write_file(df, fn) #comment out after first run to re-use file

    #----- Read the file
    df = cp.read_file(fn)
    print('\n------- Test Dataframe -------\n', df)

    #----- Run Single Processor test
    start = datetime.utcnow()
    df1 = add_stats(df=df, bars1=21, bars2=63, bars3=252)
    end = datetime.utcnow()
    print('add_stats Single-processor: 1 Time: '+str(end-start)[0:11])

    print()


#    for procs in [2,4,8,12,16,20,24,28,32,36]: # test on 32-CPU server
    for procs in [2,4,6,8,10,12]: # test for quad-core PC
#    for procs in [2,4,6,8]: # test for dual-core PC
        start = datetime.utcnow()
        df1 = cp.multiproc_dataframe(function=add_stats, df=df, bars1=21, bars2=63, bars3=252, procs=procs, splitby=['symbol'])
        end = datetime.utcnow()
        print('add_stats Multi-processor: '+str(procs)+' Time: '+str(end-start)[0:11])

        print()


