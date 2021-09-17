import pandas as pd
import numpy as np

from datetime import datetime

import caffeinated_pandas_utils as cp


#----- Create a test Dataframe
fn = 'stock_test.feather'
#num_symbols = 10_000
num_symbols = 1_000 #try this first and build up bigger
cp.create_test_dataframe(start_date='2000-01-01', end_date='2019-12-31', num_symbols=num_symbols, squeeze=True, out=fn) #comment out after first run to re-use file
df = cp.read_file(fn)
print('\n------- Test Dataframe -------\n', df)

cp.panda_mem_usage(df)


#----- Create sample in 1% chunks (range 0 to 99)
print('\n\n======== Applying 1% samples ========')

print('Number of input rows: {:>13,.0f}'.format(len(df)))
start = datetime.utcnow()
df['sample'] = df['symbol'].apply(lambda x: cp.sample_id(string=x, samples=100))
end = datetime.utcnow()
print('Time to apply 1% sample: '+str(end-start)[0:12])


#----- Show 1% distributions
start = datetime.utcnow()
total_rows = len(df)
sample_rows = df.groupby(['sample']).size().values
end = datetime.utcnow()
print('Time to summarize: '+str(end-start)[0:12])
for i, rows in enumerate(sample_rows):
	print('{:>3d} {:>10,d} {:>6,.2f}%'.format(i, rows, 100*(rows/total_rows)))


#---- Select a "rough" 1% sample of all sample keys equaling 64
start = datetime.utcnow()
df1 = df[df['sample'] == 99].reset_index(drop=True)
end = datetime.utcnow()
print('\n------- 1% sample -------\n', df1)
print('Time to select: '+str(end-start)[0:12])


#---- Select a "rough" 3% sample of all sample keys equaling 0, 1, or 2
start = datetime.utcnow()
df1 = df[df['sample'].isin([0,1,2])].reset_index(drop=True)
end = datetime.utcnow()
print('\n------- 3% sample -------\n', df1)
print('Time to select: '+str(end-start)[0:12])


#---- Select a "rough" 10% sample of all sample keys between and including 10 through 19
start = datetime.utcnow()
df1 = df[(df['sample'] >= 10) & (df['sample'] <= 19)].reset_index(drop=True)
end = datetime.utcnow()
print('\n------- 10% sample -------\n', df1)
print('Time to select: '+str(end-start)[0:12])

