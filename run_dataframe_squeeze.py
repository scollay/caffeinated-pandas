import pandas as pd
import numpy as np
from datetime import datetime

import caffeinated_pandas_utils as cp


#----- Create a test Dataframe
fn = 'stock_test.csv'
#num_symbols = 10_000
#num_symbols = 1_000
num_symbols = 100 #try this first and build up bigger
cp.create_test_dataframe(start_date='2000-01-01', end_date='2019-12-31',
			 num_symbols=num_symbols,
			 squeeze=False,
			 out=fn)
df = cp.read_file(fn)
print('\n------- Test Dataframe -------\n', df)


#----- Print memory usage of dataframe
cp.panda_mem_usage(df)


#----- Squeeze test Dataframe
print('Squeezing Dataframe')
start = datetime.utcnow()
df = cp.squeeze_dataframe(df)
end = datetime.utcnow()
print('\nTime to compress: '+str(end-start)[0:11])

#----- Print usage of compressed datafrme
cp.panda_mem_usage(df)
