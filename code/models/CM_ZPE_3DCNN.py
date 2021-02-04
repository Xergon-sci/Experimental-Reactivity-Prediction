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
import pprint
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import tstd
from scipy.stats import bayes_mvs
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from qubit.preprocessing.descriptors import tensorise_coulomb_matrix
from qubit.preprocessing.matrix_operations import pad_matrix
from data_utility import loaddata
from report_template import Template
from plot_utility import plot_predictions
from plot_utility import plot_errorhist
from plot_utility import plot_loss
from plot_utility import plot_metric
from plot_utility import plot_errorbox

# ===== Config =====
# Model Info
AUTHOR = 'Michiel Jacobs'
VERSION = '0.1.1'
MODELTITLE = 'Predicting\nZero Point energies'
MODELTYPE = '3D CNN'
MAXHEAVYATOMS = 20
FEATURE = 'Coulomb Matrix'
LABEL = 'Zero point energy'

# Development
DEVMODE = True

# Data preprocessing
POSITIVE_DIMENSIONS = 0
NEGATIVE_DIMENSIONS = 5
SPLIT_RATIO = 0.80

# Settings of the NN
# Compilation
LEARNINGRATE = 0.001
LOSS = 'mean_squared_error'
OPTIMIZER = optimizers.Adam(LEARNINGRATE)
METRICS = ['mean_absolute_error', 'mean_squared_error']

# Fit
BATCH_SIZE = 32
EPOCHS = 2
VALIDATION_SPLIT = 0.2
SHUFFLE = True

# Early stopping
MIN_DELTA = 0.0001
PATIENCE = 5
RESTORE_BEST_WEIGHTS = True

# ===== Initialization =====
FOLDER = os.path.basename(__file__).replace('.py','')

now = datetime.now()
MODELNAME = '{} {} on {}'.format(MODELTYPE, VERSION, now.strftime("%m-%d-%Y %H.%M.%S"))
MODELNAME = MODELNAME.replace(' ', '_')

LOGFOLDER = os.path.join(PATH, os.pardir, os.pardir, 'logs', FOLDER)
LOGPATH = os.path.join(LOGFOLDER, '{}.log'.format(MODELNAME))

MODELFOLDER = os.path.join(PATH, os.pardir, os.pardir, 'models', FOLDER)
MODELPATH = os.path.join(MODELFOLDER, '{}.tf'.format(MODELNAME))

REPORTFOLDER = os.path.join(PATH, os.pardir, os.pardir, 'reports', FOLDER)
REPORTPATH = os.path.join(REPORTFOLDER, '{}.pdf'.format(MODELNAME))

if not os.path.exists(LOGFOLDER):
    os.mkdir(LOGFOLDER)

if not os.path.exists(MODELFOLDER):
    os.makedirs(MODELFOLDER)

if not os.path.exists(REPORTFOLDER):
    os.makedirs(REPORTFOLDER)

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
    data = data.iloc[:500,:]

log.info('Trimming dataset...')
# Drop unused data
data = data[['coulomb_matrix', 'zero_point_energy']]

log.info('Loading arrays...')
# pandas loads json data as list, convert it to numpy arrays
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
data['tensors'] = data['coulomb_matrix'].apply(
    tensorise_coulomb_matrix,
    positive_dimension=POSITIVE_DIMENSIONS,
    negative_dimensions=NEGATIVE_DIMENSIONS)

log.info('Building channels...')
# wrap the matrix to provide channels
data['features'] = data['tensors'].apply(np.expand_dims, axis=3)

log.info('Calculating train test split...')
DATASETLENGHT = data.shape[0]
log.info('There are {} entries in this dataset.'.format(DATASETLENGHT))
log.info('Split ratio set to {}.'.format(SPLIT_RATIO))
train_size = int(SPLIT_RATIO * DATASETLENGHT)
log.info('Trainingset contains {} molecules.'.format(train_size))
# test size is whatever is left after the split

log.info('Building train and test sets...')
# split into train,val and test sets.
train = data.iloc[:train_size,:]
test = data.iloc[train_size:,:]

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

log.info('Building model...')
# Define a sequential model for testing purposes.
model = keras.Sequential()

# Settings of the Network layers
MODEL_SETTINGS = {
    'input_shape': (
        POSITIVE_DIMENSIONS + 1 + NEGATIVE_DIMENSIONS,
        maxsize,
        maxsize,
        1),
    'kernel_size': (1,3,3),
    'activation': 'relu',
    'pool_size': (2,2,2),
    'filters': 64,
    'dropout': 0.2,
    'dense_units': 32,
    'output_shape': 1
}

model.add(layers.Conv3D(
    filters=MODEL_SETTINGS['filters'],
    kernel_size=MODEL_SETTINGS['kernel_size'],
    activation = MODEL_SETTINGS['activation'],
    input_shape = MODEL_SETTINGS['input_shape']))

model.add(layers.MaxPool3D(MODEL_SETTINGS['pool_size']))

model.add(layers.Conv3D(
    filters=MODEL_SETTINGS['filters'],
    kernel_size=MODEL_SETTINGS['kernel_size'],
    activation = MODEL_SETTINGS['activation']))

model.add(layers.MaxPool3D(MODEL_SETTINGS['pool_size']))

model.add(layers.Conv3D(
    filters=MODEL_SETTINGS['filters'],
    kernel_size=MODEL_SETTINGS['kernel_size'],
    activation = MODEL_SETTINGS['activation']))

model.add(layers.Flatten())

model.add(layers.Dropout(MODEL_SETTINGS['dropout']))

model.add(layers.Dense(MODEL_SETTINGS['dense_units']))
model.add(layers.Dense(MODEL_SETTINGS['dense_units']))
model.add(layers.Dense(MODEL_SETTINGS['output_shape']))

# Print the model summary to the log
model.summary(print_fn=log.info)

# Compile the model
log.info('Compiling the model...')
model.compile(
    loss=LOSS,
    optimizer=OPTIMIZER,
    metrics=METRICS
    )

# ===== Step 4: Model training =====
log.info('============== Step 4: Model training ==============')

log.info('Enabeling early stopping...')
earlystopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=MIN_DELTA,
    patience=PATIENCE,
    verbose=2,
    restore_best_weights=RESTORE_BEST_WEIGHTS)

log.info('Start training...')
history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[earlystopping],
    validation_split=VALIDATION_SPLIT,
    shuffle=SHUFFLE)

log.info('Plotting loss...')
plot_loss('model_loss.jpg', history.history['loss'], history.history['val_loss'], 'mse')

for m in METRICS:
    log.info('Plotting metric {}'.format(m))
    plot_metric('{}.jpg'.format(m), history.history[m], m)

# ===== Step 5: Model evaluation =====
log.info('============== Step 5: Model evaluation ==============')

log.info('Evaluating model...')
test_scores = model.evaluate(
    test_features,
    test_labels,
    batch_size=BATCH_SIZE,
    verbose=1,
    return_dict=True)

log.info('Test scores:')
log.info(pprint.pformat(test_scores))

log.info('Making test predictions...')
test_predictions = model.predict(test_features).flatten()
log.info('Plotting ToP plot...')
plot_predictions('ToP.jpg', test_labels, test_predictions, 'ev')
log.info('Plotting Error histogram plot...')
plot_errorhist('Error_hist.jpg', test_labels, test_predictions, 'ev')
log.info('Plotting boxplot...')
plot_errorbox('boxplot.jpg', test_labels, test_predictions, 'ev')

error = test_predictions - test_labels
test_error_mean = np.mean(error)
test_error_min = np.min(error)
test_error_max = np.max(error)
test_error_median = np.median(error)
test_error_skewness = skew(error)
test_error_kurtosis = kurtosis(error)
test_error_sd = tstd(error)
test_error_CI = bayes_mvs(error)

# ===== Step 6: Saving, reporting and cleanup =====
log.info('============== Step 6: Saving, reporting and cleanup ==============')

# Save the model
log.info('Saving model...')
model.save(MODELPATH)
log.info('Model saved.')

# Generate the report
log.info('Generating report...')
report = Template(MODELTITLE, MODELTYPE + ' v' + VERSION, AUTHOR, MODELNAME)

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
report.label('Total molecules: ', DATASETLENGHT)

report.head2('Tensorisation:')
report.label('Positive dimensions: ', POSITIVE_DIMENSIONS)
report.label('Negative dimensions: ', NEGATIVE_DIMENSIONS)

report.head2('Test and train sets:')
report.label('Split ratio: ', SPLIT_RATIO)
report.label('Molecules for training: ', train_features.shape[0])
report.label('Molecules for testing: ', test_features.shape[0])

# ===== Third page =====
report.add_page()
report.head1('Neural Network')
report.head2('Network compile parameters:')
report.label('Learningrate: ', LEARNINGRATE)
report.label('Loss: ', LOSS)
report.label('Optimizer: ', OPTIMIZER._name)
report.label('Metrics: ', ', '.join(METRICS))

report.head2('Network fit parameters:')
report.label('Batch size: ', BATCH_SIZE)
report.label('Epochs: ', EPOCHS)
report.label('Validation split: ', VALIDATION_SPLIT)
report.label('Shuffle data each epoch: ', SHUFFLE)

report.head2('Early stopping parameters:')
report.label('Minimum change required: ', MIN_DELTA)
report.label('Epochs no change is allowed before stopping: ', PATIENCE)
report.label('Restore best weights: ', RESTORE_BEST_WEIGHTS)

report.head2('Neural network Layer settings:')
for k in MODEL_SETTINGS:
    report.dict_label(k.replace('_', ' '), MODEL_SETTINGS[k])

report.add_page()
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
for m in METRICS:
    report.image('{}.jpg'.format(m), w=170)

report.add_page()
report.head2('Model evaluation')
report.image('ToP.jpg', x=30, w=150)
for m in METRICS:
    report.dict_label(m.replace('_', ' '), round(test_scores[m],4))

report.add_page()
report.head2('Error evaluation')
report.image('Error_hist.jpg', w=170)
report.image('boxplot.jpg', w=170)

report.add_page()
report.label('Mean: ', round(test_error_mean, 4))
report.label('Median: ', round(test_error_median, 4))
report.label('Minimum error: ', round(test_error_min, 4))
report.label('Maximum error: ', round(test_error_max, 4))
report.label('Skewness: ', round(test_error_skewness, 4))
report.label('Kurtosis: ', round(test_error_kurtosis, 4))
report.label('Standard deviation: ', round(test_error_sd, 4))
report.label('90% Confidence interval: ', '[{};{}]'.format(round(test_error_CI[0][1][0], 4), round(test_error_CI[0][1][1], 4)))

# ===== Last pages =====
report.add_page()
report.head1('Log')
with open(LOGPATH, 'r') as f:
    report.text(f.read())

# Save the report
report.output(REPORTPATH)

# cleanup the images
os.remove('model_loss.jpg')
for m in METRICS:
    os.remove('{}.jpg'.format(m))
os.remove('ToP.jpg')
os.remove('Error_hist.jpg')
os.remove('boxplot.jpg')

log.info('Removed images')

log.info('Shutting down...')

# Stop logging
log.shutdown()
