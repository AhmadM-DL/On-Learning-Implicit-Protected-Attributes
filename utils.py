import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(data, labels, title):
    fig, ax = plt.subplots()
    ax.matshow(data, cmap='seismic')
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    ax.set_ylabel("Ground Truth")
    ax.set_yticklabels([""]+labels)
    ax.set_xlabel("Predictions")
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels([""]+labels)
    ax.set_title(title, position=(0.5, 1.2))
    return ax


def plot_aur_roc_curves():
    return