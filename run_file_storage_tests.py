import pandas as pd
import numpy as np

from datetime import datetime
import os
import gc #memory garbage collection
import sys

import caffeinated_pandas_utils as cp


def write_read_test(df, fn, compression='', iterations=3):

    gc.collect()

    print('\nFile:', fn)

    timer = []
    for i in range(iterations):
        print('writing...', end='', flush=True)
        start = datetime.utcnow()
        cp.write_file(df=df, fn=fn, compression=compression)
        end = datetime.utcnow()
        timer.append(end - start)
        file_size = os.path.getsize(fn)/(1024**2)
        df_size_written = cp.panda_mem_usage(df, detail='return_short')
    write_time = np.mean(timer)
    print()

    timer = []
    for i in range(iterations):
        print('reading...', end='', flush=True)
        start = datetime.utcnow()
        df1 = cp.read_file(fn=fn, compression=compression)
        end = datetime.utcnow()
        timer.append(end - start)
        df_size_read = cp.panda_mem_usage(df1, detail='return_short')
    read_time = np.mean(timer)
    print()

    timer = []
    for i in range(iterations):
        print('selecting...', end='', flush=True)
        start = datetime.utcnow()
        df2 = df1[(df1['symbol'].str.contains('666') | df1['symbol'].str.contains('777')) & (df1['date'].str[8:10] <= '10')].reset_index(drop=True)
        df2 = df1[df1['dividend'] > (df1['dividend'].max() * .99)].reset_index(drop=True) #highest dividends paid
        df2 = df1[(df1['ind1'] > df1['ind2'].shift()) & (df1['ind2'] > df1['ind3'].shift(3))].reset_index(drop=True) #highest dividends paid
        df2 = df1.pivot(index='date', columns='symbol', values='close') 
        end = datetime.utcnow()
        timer.append(end - start)
    select_time = np.mean(timer)
    print()

    timer = []
    for i in range(iterations):
        print('sorting...', end='', flush=True)
        start = datetime.utcnow()
        df2 = df1.sort_values(by=['symbol','date'], ascending=[True, False]).reset_index(drop=True)        
        end = datetime.utcnow()
        timer.append(end - start)
    sort_time = np.mean(timer)
    print()


    #----- Clean up memory
    print('deleting dataframes...', end='', flush=True)
    df = pd.DataFrame()
    del df

    df1 = pd.DataFrame()
    del df1

    df2 = pd.DataFrame()
    del df2

    gc.collect()

    print('done', end='', flush=True)


    #----- Print stats
    print('')
    print('Time Write to Disk - '+str(write_time)[0:11])
    print('Disk File Size - {:,.2f} MB'.format(file_size))

    print('Read Time - '+str(read_time)[0:11])
    print('Dataframe After Read - {:,.1f} MB'.format(df_size_read[1]/1024**2))

    print('Time Selects - '+str(select_time)[0:11])
    print('Time Sorts - '+str(sort_time)[0:11])



#===== Main program

#----- Create a test Dataframe
fn = 'stock_test.csv'
#num_symbols = 10_000
#num_symbols = 1_000
num_symbols = 100 #try this first and build up bigger
cp.create_test_dataframe(start_date='2000-01-01', end_date='2019-12-31', num_symbols=num_symbols, squeeze=False, out=fn) #comment out after first run to re-use file
df = cp.read_file(fn)
print('\n------- Test Dataframe -------\n', df)

#----- Squeeze the dataframe to ideal memory size (see "compressing" Medium article and run_dataframe_squeeze.py for background)
df_size_read = cp.panda_mem_usage(df, detail='return_short')

print('\nSqueezing Dataframe')
df = cp.squeeze_dataframe(df)
df_size_squeezed = cp.panda_mem_usage(df, detail='return_short')

print('Dataframe Rows - {:,.0f}'.format(df_size_read[0]))
print('Dataframe Size - {:,.1f} MB'.format(df_size_read[1]/1024**2))
print('Dataframe Size Squeezed - {:,.1f} MB'.format(df_size_squeezed[1]/1024**2))


#---- Delete previous test files
cp.del_files('test-file*') 


#----- Test various file format and compression tests
iterations = 3

write_read_test(df=df, fn='test-file.csv', iterations=iterations)
write_read_test(df=df, fn='test-file.zip', iterations=iterations)

write_read_test(df=df, fn='test-file-snappy.parquet', compression='snappy', iterations=iterations)
write_read_test(df=df, fn='test-file-gzip.parquet', compression='gzip', iterations=iterations)
write_read_test(df=df, fn='test-file-brotli.parquet', compression='brotli', iterations=iterations)

write_read_test(df=df, fn='test-file-zstd.feather', compression='zstd', iterations=iterations)
write_read_test(df=df, fn='test-file-lz4.feather', compression='lz4', iterations=iterations)

write_read_test(df=df, fn='test-file-bzip2.h5', compression='bzip2', iterations=iterations)
write_read_test(df=df, fn='test-file-lzo.h5', compression='lzo', iterations=iterations)
write_read_test(df=df, fn='test-file-zlib.h5', compression='zlib', iterations=iterations)
write_read_test(df=df, fn='test-file-blosc-blosclz.h5', compression='blosc:blosclz', iterations=iterations)
write_read_test(df=df, fn='test-file-blosc-lz4.h5', compression='blosc:lz4', iterations=iterations)
write_read_test(df=df, fn='test-file-blosc-lz4hc.h5', compression='blosc:lz4hc', iterations=iterations)
write_read_test(df=df, fn='test-file-blosc-zlib.h5', compression='blosc:zlib', iterations=iterations)
write_read_test(df=df, fn='test-file-blosc-zstd.h5', compression='blosc:zstd', iterations=iterations)

write_read_test(df=df, fn='test-file-zip.pkl', compression='zip', iterations=iterations)
write_read_test(df=df, fn='test-file-gzip.pkl', compression='gzip', iterations=iterations)
write_read_test(df=df, fn='test-file-xz.pkl', compression='xz', iterations=iterations)
write_read_test(df=df, fn='test-file-bz2.pkl', compression='bz2', iterations=iterations)

write_read_test(df=df, fn='test-file.sqlite', iterations=iterations)

