# ===== Imports =====
import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
UTIL_PATH = os.path.join(PATH, os.pardir, 'utilities')
sys.path.append(UTIL_PATH)

import pandas as pd
from data_utility import loaddata

baseData = pd.read_csv(r'C:\Users\Michiel Jacobs\Research\Master Thesis\Experimental-Reactivity-Prediction\data\CNOS_sub1_10to20_10k\backup\CNOS_sub1_10to20_10k.csv')
resultData = loaddata(r'C:\Users\Michiel Jacobs\Research\Master Thesis\Experimental-Reactivity-Prediction\data\CNOS_sub1_10to20_10k\backup')

# Base analytical data
baseData = baseData.rename(str.lower, axis='columns')
baseData = baseData.sort_values(by=['cid'])
baseData = baseData.reset_index()
baseData = baseData.drop(['index'], axis = 1)
print(baseData.head())

# Results from caluclation
resultData['cid'] = resultData['cid'].astype('int64')
resultData = resultData.sort_values(by=['cid'])
resultData = resultData.reset_index()
resultData = resultData.drop(['index'], axis = 1)
print(resultData.head())

# resulting data
result = pd.merge(baseData, resultData, on="cid")
result['jid'] = result['jid_x']
result = result.drop(['jid_x','jid_y'], axis = 1)
result = result[['jid','cid', 'smiles', 'carbon', 'fluorine', 'chlorine', 'bromine', 'iodine', 'sulfur', 'phosphorous', 'acyclic nitrogen', 'cyclic nitrogen', 'acyclic oxygen', 'cyclic oxygen', 
'heavy atoms', 'acyclic single bonds', 'acyclic double bonds', 'acyclic triple bonds', 'cyclic single bonds', 'cyclic double bonds', 'cyclic triple bonds', 'rotable bonds', 'h-bond acceptor sites', 'h-bond acceptor atoms', 'h-bond donor sites', 'h-bond donor atoms', 'negative charges', 'positive charges', 'acyclic single valent nodes', 'acyclic divalent nodes', 'acyclic trivalent nodes', 'acyclic tetravalent nodes', 'cyclic divalent nodes', 'cyclic trivalent nodes', 'cyclic tetravalent nodes', '3-membered rings', '4-membered rings', '5-membered rings', '6-membered rings', '7-membered rings', '8-membered rings', '9-membered rings', '>= 10-membered rings', 'nodes shared by >= 2 rings', 'edges shared by >= 2 rings', 'functional groups', 'optimized_geometry', 'self_consistent_field', 'zero_point_energy', 'thermal_correction_to_energy', 'thermal_correction_to_enthalpy', 'thermal_correction_to_gibbs_free_energy', 'homo', 'lumo', 'natural_population_analysis', 'coulomb_matrix']]

result.to_json('test.json', orient='index', indent=2)