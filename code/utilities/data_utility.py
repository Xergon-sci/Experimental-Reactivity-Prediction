import os
import pandas as pd

def loaddata(path):
    dataframes = []
    for file in os.listdir(path):
        if file.endswith(".json"):
            dataframes.append(pd.read_json(os.path.join(path, file)))

    transp = []
    for df in dataframes:
        transp.append(df.transpose())
    
    return pd.concat(transp)