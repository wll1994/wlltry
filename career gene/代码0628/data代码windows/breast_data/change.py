import numpy as np
import pandas as pd
first=pd.read_csv('breast1.txt',header=None,sep='\t')

data2=first.T
print data2.shape
print data2.head()

data2.to_csv('breast2.txt',sep='\t',index=False,header=False)