# ===== Config =====
AUTHOR = 'Michiel Jacobs'
VERSION = '0.1.0'
MODELTITLE = 'Report System Test'
MODELTYPE = '3D CNN'
MAXHEAVYATOMS = 20
FEATURE = 'Coulomb Matrix'
LABEL = 'Zero point energy'

DEVMODE = True

# ===== Imports =====
import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
UTIL_PATH = os.path.join(PATH, os.pardir, 'utilities')
sys.path.append(UTIL_PATH)

from datetime import datetime
import logging as log
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from qubit.preprocessing.descriptors import tensorise_coulomb_matrix
from qubit.preprocessing.matrix_operations import pad_matrix
from data_utility import loaddata
from report_template import Template
from plot_utility import plot_predictions
from plot_utility import plot_errorhist
from plot_utility import plot_loss

# ===== Initialization =====
now = datetime.now()
MODELNAME = '{} {} on {}'.format(MODELTYPE, VERSION, now.strftime("%m-%d-%Y %H.%M.%S"))
MODELNAME = MODELNAME.replace(' ', '_')

LOGPATH = os.path.join(PATH, os.pardir, os.pardir, 'logs', '{}.log'.format(MODELNAME))
log.basicConfig(filename=LOGPATH,
                level=log.INFO,
                format='%(asctime)s:%(levelname)s:%(message)s')

log.info('Starting model {}'.format(MODELNAME))
log.info('=================== Model info ===================')
log.info('Author: {}'.format(AUTHOR))
log.info('Version: {}'.format(VERSION))
log.info('Modeltype: {}'.format(MODELTYPE))
log.info('Maximum heavy atoms: {}'.format(MAXHEAVYATOMS))
log.info('Feature: {}'.format(FEATURE))
log.info('Labels: {}'.format(LABEL))
log.info('DEVELOPMENT: {}'.format(DEVMODE))

# ===== Step 1: Load data =====
log.info('============== Step 1: loading data ==============')
data = loaddata(os.path.join(PATH, os.pardir, os.pardir, 'data'))
log.info('Data loaded')

# ===== Step 2: data preprocessing =====
log.info('============== Step 2: data preprocessing ==============')

if DEVMODE:
    data = data.iloc[:100,:]

log.info('Trimming dataset...')
# Drop unused data
data = data[['coulomb_matrix', 'zero_point_energy']]

log.info('Loading arrays...')
# pandas loads data as list, convert it to numpy arrays
data['coulomb_matrix'] = data['coulomb_matrix'].apply(lambda x: np.array(x))

log.info('Shuffeling data...')
# shuffle the data
data = data.sample(frac=1)

log.info('Calculating maximum size of molecules...')
# The data set contains molecules with maximum 20 heavy atoms
# So XnH2n+2 can be used to calculate the maximum atoms that can be present and thus
# the maximum shape of our molecules
n = MAXHEAVYATOMS
maxsize = n + ((2*n)+2)
log.info('The maximumsize of molecules is {}'.format(maxsize))

log.info('Normalizing data...')
# normalize the data to a constant size
data['coulomb_matrix'] = data['coulomb_matrix'].apply(pad_matrix, size=maxsize)

log.info('Tensorisation of the coulomb matrices...')
# tensorize the data
data['tensors'] = data['coulomb_matrix'].apply(tensorise_coulomb_matrix, negative_dimensions=5)

log.info('Building channels...')
# wrap the matrix to provide channels
data['features'] = data['tensors'].apply(np.expand_dims, axis=3)

log.info('Calculating train test split...')
DATASETLENGHT = data.shape[0]
log.info('There are {} entries in this dataset.'.format(DATASETLENGHT))
split_ratio = 0.80
log.info('Split ratio set to {}.'.format(split_ratio))
train_size = int(split_ratio * DATASETLENGHT)
log.info('Trainingset contains {} molecules.'.format(train_size))
# test size is whatever is left after the 0.75 split

log.info('Building train and test sets...')
# split into train,val and test sets.
train = data.iloc[:train_size,:]
test = data.iloc[train_size+1:,:]

# Convert data to tf tensors
log.info('Converting train features to tf.tensors...')
train_tensor_list = []
for t in train['features'].values:
    t = tf.convert_to_tensor(t)
    train_tensor_list.append(t)
train_features = np.array(train_tensor_list)

log.info('Converting test features to tf.tensors...')
test_tensor_list = []
for t in test['features'].values:
    t = tf.convert_to_tensor(t)
    test_tensor_list.append(t)
test_features = np.array(test_tensor_list)

log.info('Converting train labels to array...')
zpe = train['zero_point_energy'].values.astype(np.float).tolist()
train_labels = np.array(zpe)

log.info('Converting test labels to array...')
test_zpe = test['zero_point_energy'].values.astype(np.float).tolist()
test_labels = np.array(test_zpe)

# ===== Step 3: Model compilation =====
log.info('============== Step 3: Model compilation ==============')

# Settings of the NN
settings = pd.DataFrame(
    np.array([
            [(6,62,62,1), '', (1,3,3), '', 64, ''],
            ['', '', '', '', '', 0.2],
            ['', '', (1,3,3), '', '64', ''],
            ['', '', '', '', '', 0.2],
            ['', '', (1,3,3), '', '64', ''],
            ['', '', '', '', '', ''],
            ['', '', '', '', '', '0.2']
            ], dtype=object),
    index=['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'Layer 7'],
    columns=['Input shape', 'Batch size', 'kernel size', 'pool size', 'filters', 'dropout'])

input_shape = (6,62,62,1)
batch_size = 128
kernel_size = (1,3,3)
pool_size = (2,2,2)
filters = 64
dropout = 0.2

log.info('Building model...')
# Define a sequential model for testing purposes.
model = keras.Sequential()

model.add(layers.Conv3D(
    filters=filters,
    kernel_size=kernel_size,
    activation = 'relu',
    input_shape = input_shape))

model.add(layers.MaxPool3D(pool_size))

model.add(layers.Conv3D(
    filters=filters,
    kernel_size=kernel_size,
    activation = 'relu'))

model.add(layers.MaxPool3D(pool_size))

model.add(layers.Conv3D(
    filters=filters,
    kernel_size=kernel_size,
    activation = 'relu'))

model.add(layers.Flatten())

model.add(layers.Dropout(dropout))

model.add(layers.Dense(1))

# Print the model summary to the log
model.summary(print_fn=log.info)

# Compile the model
log.info('Compiling the model...')
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
    )

# ===== Step 4: Model training =====
log.info('============== Step 4: Model training ==============')

history = model.fit(
    train_features,
    train_labels,
    batch_size=512,
    epochs=2,
    validation_split=0.2)

plot_loss('model_loss.jpg', history.history['loss'], history.history['val_loss'], 'mse')

# ===== Step 5: Model evaluation =====
log.info('============== Step 5: Model evaluation ==============')

test_scores = model.evaluate(
    test_features,
    test_labels,
    verbose=1)

log.info('Test scores: {}'.format(test_scores))

test_predictions = model.predict(test_features).flatten()
plot_predictions('ToP.jpg', test_labels, test_predictions, 'ev')
plot_errorhist('Error_hist.jpg', test_labels, test_predictions, 'ev')

# ===== Step 6: Saving, reporting and cleanup =====
log.info('============== Step 6: Saving, reporting and cleanup ==============')

# Save the model
log.info('Saving model...')
model.save(os.path.join(PATH, os.pardir, os.pardir, 'models', '{}.tf'.format(MODELNAME)))
log.info('Model saved.')

# Generate the report
log.info('Generating report...')
report = Template(MODELTITLE, MODELTYPE, AUTHOR, MODELNAME)

# ===== Second page =====
report.add_page()

report.head1('Run information')
report.head2('Model:')
report.label('Author: ', AUTHOR)
report.label('Version: ', VERSION)
report.label('Type: ', MODELTYPE)
report.label('Feature: ', FEATURE)
report.label('Label: ', LABEL)

report.head2('Data:')
report.label('Maximum heavy atoms: ', MAXHEAVYATOMS)
report.label('Maximum molecule size: ', maxsize)
report.label('Split ratio: ', split_ratio)
report.label('Molecules for training: ', train_features.shape[0])
report.label('Molecules for testing: ', test_features.shape[0])

# ===== Third page =====
report.add_page()
report.head1('Neural Network')
report.head2('Network settings')
settings = settings.transpose()
report.text(settings.to_string())
report.head2('NN summary')
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
report.text(short_model_summary)

# ===== Results section =====
report.add_page()
report.head1('Results')
report.head2('Training evaluation')
report.image('model_loss.jpg', w=170)
report.add_page()
report.head2('Model evaluation')
report.image('ToP.jpg', x=30, w=150)
report.add_page()
report.image('Error_hist.jpg', w=170)

# ===== Last pages =====
report.add_page()
report.head1('Log')
with open(LOGPATH, 'r') as f:
    report.text(f.read())

# Save the report
report.output(os.path.join(PATH, os.pardir, os.pardir, 'reports', '{}.pdf'.format(MODELNAME)))

# cleanup the images
os.remove('model_loss.jpg')
os.remove('ToP.jpg')
os.remove('Error_hist.jpg')
log.info('Removed images')

log.info('Shutting down...')

# Stop logging
log.shutdown()
