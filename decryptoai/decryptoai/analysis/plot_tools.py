import seaborn as sns
import matplotlib.pyplot as plt
import decryptoai.players.unsupervised.numpy_guesser as npg
import numpy as np

def plot_random_var(random_var: npg.NumpyRandomVariable, ax: plt.Axes):
    data = dict(zip(random_var.keyword_indices, np.exp(random_var.log_probabilities)))
    sns.barplot(data, ax=ax).set(ylim=(0, 1))

def plot_random_vars(random_vars, axs, round=0):
    for vi, random_var in enumerate(random_vars):
        axs[vi, round].get_xaxis().set_visible(False)
        plot_random_var(random_var, axs[vi, round])