import pandas as pd
import numpy as np

import psutil
import os
import glob
import sqlite3
import hashlib
from datetime import datetime
import multiprocessing


def squeeze_dataframe(df):

    #----- Get columns in dataframe
    cols = dict(df.dtypes)

    #----- Check each column's type downcast or categorize as appropriate
    for col, type in cols.items():
        if type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif type == 'object':
            df[col] = df[col].astype('category')

    return df


def panda_mem_usage(df, detail='full'):

    dtypes = df.dtypes.reset_index(drop=False)
    memory = df.memory_usage(deep=True).reset_index(drop=False)

    df1 = pd.merge(dtypes, memory, on='index')
    df1 = df1.rename(columns = {'index': 'col', '0_x': 'type', '0_y': 'bytes'})
    total = df1['bytes'].sum()

    objects = df.select_dtypes(include=['object', 'category'])
    df_objs = objects.select_dtypes(include=['object', 'category']).describe().T.reset_index()

    if detail == 'full':
        print('')
        print('{:<15} {:<15} {:>15} {:>8} {:>8}'.format('Column', 'Data Type', 'Bytes', 'MBs', 'GBs'))
        print('{} {} {} {} {}'.format('-'*15, '-'*15, '-'*15, '-'*8, '-'*8))

        for index, row in df1.iterrows():
            print('{:<15} {:<15} {:>15,.0f} {:>8,.1f} {:>8,.2f}'.format(row['col'], str(row['type']), row['bytes'], row['bytes']/1024**2, row['bytes']/1024**3))

        print('\nTotal: {:,.0f} Rows, {:,.0f} Bytes, {:,.1f} MBs, {:,.2f} GBs\n'.format(len(df), total, total/1024**2, total/1024**3))

        print('{:<15} {:>13} {:>13}'.format('Column', 'Count', 'Unique'))
        print('{} {} {}'.format('-'*15, '-'*13, '-'*13))
        for index, row in df_objs.iterrows():
            print('{:<15} {:>13,.0f} {:>13,.0f}'.format(row['index'], row['count'], row['unique']))

    elif detail == 'return_short':
        return len(df), total


def write_file(df, fn, compression=''):

    fn_ext = os.path.splitext(fn)[1]

    if fn_ext == '.csv':
        df.to_csv(fn, index=False)

    elif fn_ext == '.zip':
        df.to_csv(fn, compression=dict(method='zip', archive_name='data'), index=False)

    elif fn_ext == '.parquet':
        compression = 'brotli' if compression == '' else compression
        df.to_parquet(fn, engine='pyarrow', compression=compression)

    elif fn_ext == '.feather':
        compression = 'zstd' if compression == '' else compression
        df.to_feather(fn, compression=compression)

    elif fn_ext == '.h5':
        compression = 'blosc:lz4' if compression == '' else compression
        df.to_hdf(fn, key='data', mode='w', format='table', index=False, complevel=9, complib=compression)

    elif fn_ext == '.pkl':
        compression = 'zip' if compression == '' else compression
        df.to_pickle(fn, compression=compression)

    elif fn_ext == '.sqlite':
        con = sqlite3.connect(fn)
        df.to_sql('data', con=con, if_exists='replace', index=False)
        con.close()

    elif fn_ext == '.json':
        df.to_json(fn, orient='records')

    elif fn_ext == '.xlsx':
        writer = pd.ExcelWriter(fn, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='quotes', index=False)
        #add more sheets by repeating df.to_excel() and change sheet_name
        writer.save()

    else:
        print('oopsy in write_file()! File extension unknown:', fn_ext)
        quit(0)

    return


def read_file(fn, compression='', sql=''):

    fn_ext = os.path.splitext(fn)[1]

    if fn_ext == '.csv' or fn_ext == '.zip':
        df = pd.read_csv(fn, keep_default_na=False)

    elif fn_ext == '.parquet':
        df = pd.read_parquet(fn)

    elif fn_ext == '.feather':
        df = pd.read_feather(fn)

    elif fn_ext == '.h5':
        df = pd.read_hdf(fn, key='data')

    elif fn_ext == '.pkl':
        df = pd.read_pickle(fn, compression=compression).copy() #copy added because of some trouble with categories not fully read by mem util on first pass

    elif fn_ext == '.sqlite':
        if sql == '':
            sql = 'SELECT * FROM data'
        con = sqlite3.connect(fn)
        df = pd.read_sql(sql, con)
        con.close()

    elif fn_ext == '.json':
        df = pd.read_json(fn, convert_dates=False)

    elif fn_ext == '.xlsx':
        df = pd.read_excel(fn, sheet_name='quotes', keep_default_na=False)

    else:
        print('oopsy in read_file()! File extension unknown:', fn_ext)
        quit(0)

    return df


def sample_id(string, samples=100):
    #----- given a string, return a repeatable integer between 0 and q-1
    #      don't use expecting a perfect sample
    #      this is really just a quick way to split up a dataset into roughly equal parts in a way that's repeatable
    #      from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    sample = int(hashlib.sha256(string.encode('utf8')).hexdigest(),16) % samples

    return sample



def create_test_dataframe(start_date, end_date, num_symbols, squeeze=True, out=''):

    #----- Create skeleton dataframe for one symbol, 20-years, business days only
    print('cp.create_test_dataframe --> create skeleton')
    np.random.seed(0) # seed so there's consistency between testing runs
    dfs = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='B').strftime('%Y-%m-%d'),
                        'vendor':'StockDataCo', 'interval':'1day', 'symbol':'BEAN'})


    #----- Duplicate skeleton and populate with psudo-random values
    print('cp.create_test_dataframe --> duplicate symbols by', num_symbols)
    df = dfs.loc[np.repeat(dfs.index.values, num_symbols)]
    print('cp.create_test_dataframe --> created {:,.0f} rows'.format(len(df)))


    #----- for each duplicate added, create a unique symbol name
    print('cp.create_test_dataframe --> make symbol names')
    df['dupe_num'] = df.groupby(['date']).cumcount()+1 #asssigns a sequence
    df['dupe_num'] = df['dupe_num'].astype(str).str.zfill(len(str(num_symbols))) #pad with 0's based on num_symbols length
    df['symbol'] = dfs['symbol'].str.cat('.'+df['dupe_num'])
    df = df.drop('dupe_num', axis=1).reset_index(drop='true')


    #----- For each column, populate values based on a random open to demonstrate compression.
    #      Note that this is not a true depiction of random prices or indicators!
    print('cp.create_test_dataframe --> populate prices, indicators and signals')
    df['open'] = [round(np.random.uniform(1,200),2) for k in df.index]
    df['high'] = round(df['open'] * 1.11, 2)
    df['low'] = round(df['open'] * 0.91, 2)
    df['close'] = round(df['open'] * 1.06, 2)
    df['volume'] = (df['open'] * 1123211).astype(int)
    df['dividend'] = round(df['open'] * 0.021, 2)
    df['ind1'] = round(df['open'] * 0.5, 2)
    df['ind2'] = round(df['open'] * 1.2, 2)
    df['ind3'] = round(df['open'] * 0.9, 2)
    df['trend1'] = (df['open'] % 2).astype(int)
    df['trend2'] = (df['close'] % 2).astype(int)
    df['signal'] = df['open'].apply(lambda x: 'buy' if (int(x) % 2) == 0 else 'sell')
    df['sample'] = df['symbol'].apply(lambda x: sample_id(x, samples=100))


    #----- Squeeze if specified
    print('cp.create_test_dataframe --> squeezing')
    if squeeze == True:
        df = squeeze_dataframe(df)


    #----- Write to file if specified
    if out != '':
        print('cp.create_test_dataframe --> writing to fn=', out)
        write_file(df=df, fn=out)

    print('cp.create_test_dataframe --> done')

    return df


def multiproc_run_target(*args):
    #----- a companion to multiproc_dataframe - sends to data enhancement function and appends results
    #      done this way so the data enhancement function can remain unchanged from one that does a straight apply()

    function = args[0]
    return_list = args[1]

    args1 = []
    for arg in args[2:]:
        args1.append(arg)

    df1 = function(*args1)

    return_list.append(df1)


def multiproc_dataframe(**kwargs):

    #----- load required parameter to pass onto the function, and load optionally added parameters to a list 
    added_args = []

    for key, val in kwargs.items():

        if key == 'function':
            function = val

        elif key == 'df':
            df = val

        elif key == 'procs':
            procs = val

        elif key == 'splitby':
            splitby = val

        else:
            added_args.append(val)

    if 'procs' not in kwargs:
        procs = multiprocessing.cpu_count()

    if 'splitby' not in kwargs or splitby == None:
        df_splits = np.array_split(df, procs)

    else:
        groups = df[splitby].unique()
        splits_list = np.array_split(groups, procs)

        #----- Create splits
        df_splits = []

        for i in range(procs):
            dfs = df[df[splitby].isin(splits_list[i])].copy()
            df_splits.append(dfs)

    del df


    #----- Initialize
    manager = multiprocessing.Manager()
    return_list = manager.list()
    processes = []


    #----- Process splits and concatinate all element in the return_list
    for i in range(procs):
        process = multiprocessing.Process(target=multiproc_run_target, args=[function, return_list, df_splits[i]]+added_args)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    df = pd.concat(return_list).reset_index(drop=True)

    return df


def del_files(path):

    files = glob.glob(path)

    if len(files) == 0:
        print('no files to delete matching "'+path+'"')
        return

    for file in files:
        print('Deleting', file)
        os.remove(file)

