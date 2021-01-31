# ===== Config =====
AUTHOR = 'Michiel Jacobs'
VERSION = '0.1.0'
MODELTYPE = '3D CNN'
FEATURE = 'Coulomb Matrix'
LABEL = 'Zero point energy'

# ===== Imports =====
import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(PATH, os.pardir, 'utilities')
sys.path.append(PATH)

from datetime import datetime
import logging as log
import pandas as pd
from data_utility import loaddata

# ===== Initialization phase =====
now = datetime.now()
MODELNAME = '{} {} on {}'.format(MODELTYPE, VERSION, now.strftime("%m-%d-%Y %H.%M.%S"))
MODELNAME = MODELNAME.replace(' ', '_')

log.basicConfig(filename='{}.log'.format(MODELNAME),
                level=log.DEBUG,
                format='%(asctime)s : %(message)s')

log.info('Starting model {}'.format(MODELNAME))
log.info('=================== Model info ===================')
log.info('Author: {}'.format(AUTHOR))
log.info('Version: {}'.format(VERSION))
log.info('Modeltype: {}'.format(MODELTYPE))
log.info('Feature: {}'.format(FEATURE))
log.info('Labels: {}'.format(LABEL))

# ===== Step 1: Load data =====
log.info('============== Step 1: Loading Data ==============')



# ===== Save and Cleanup =====
