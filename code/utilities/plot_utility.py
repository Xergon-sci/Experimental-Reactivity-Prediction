import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

def plot_predictions(name, labels, predictions, unit):
    plt.style.use('seaborn')
    plt.figure(figsize=(4.5, 4.1), dpi=300)
    a = plt.axes(aspect='equal')
    plt.tight_layout(w_pad=300)
    minimum = np.amin(labels)
    maximum = np.amax(labels)
    diff = (maximum - minimum) /2
    lim = (minimum-diff, maximum+diff)
    plt.plot(lim, lim, color='grey', linestyle='--', linewidth=1, zorder=1)
    plt.scatter(labels, predictions, marker='.', s=30, color='black', zorder=2)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel('True values / {}'.format(unit))
    plt.ylabel('Predictions / {}'.format(unit))
    plt.title("Truth of predictions")
    plt.savefig(name)

def plot_errorhist(name, labels, predictions, unit):
    errors = predictions - labels
    plt.style.use('seaborn')
    plt.figure(dpi=300)
    plt.hist(errors, bins=int(sqrt(len(errors))), color='black')
    plt.axvline(np.mean(errors), color='orange', label='mean')
    plt.xlabel('Prediction error / {}'.format(unit))
    plt.ylabel('Error count')
    plt.title("Error histogram")
    plt.legend()
    plt.savefig(name)

def plot_errorbox(name, labels, predictions, unit):
    errors = predictions - labels
    plt.style.use('seaborn')
    plt.figure(dpi=300)
    plt.boxplot(errors)
    plt.title("Boxplot of errors")
    plt.savefig(name)

def plot_loss(name, loss, val_loss, loss_unit):
    plt.style.use('seaborn')
    plt.figure(dpi=300)
    plt.plot(loss, label='loss', color='black')
    plt.plot(val_loss, label='validation loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel(loss_unit)
    plt.title("Loss during training")
    plt.legend()
    plt.savefig(name)

def plot_metric(name, metric, unit):
    unit = unit.replace('_', ' ')
    plt.style.use('seaborn')
    plt.figure(dpi=300)
    plt.plot(metric, label=unit, color='black')
    plt.xlabel('Epoch')
    plt.ylabel(unit)
    plt.title("{} during training".format(unit))
    plt.legend()
    plt.savefig(name)