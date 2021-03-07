import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
UTIL_PATH = os.path.join(PATH, os.pardir, 'utilities')
sys.path.append(UTIL_PATH)

import pandas as pd
from data_utility import loaddata

df = loaddata(r'C:\Users\Michiel Jacobs\Research\Master Thesis\Experimental-Reactivity-Prediction\data\CNOS_sub1_10to20_10k\Full')
print(df.head())

result = df[['cid','homo','lumo','coulomb_matrix']]

print(result.head())