import os
import pandas as pd
import numpy as np

def loaddata(path):
    dataframes = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            dataframes.append(pd.read_json(os.path.join(path, file)))

    transp = []
    for df in dataframes:
        transp.append(df.transpose())
    
    return pd.concat(transp)

def savedataJson(path, df, cnt, name):
    sections = np.array_split(df, cnt)
    
    for i,s in enumerate(sections):
        s.to_json(r'{}\{}_{}.json'.format(path, name, i), orient='index', indent=2)