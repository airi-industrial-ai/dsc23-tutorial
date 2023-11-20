from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import numpy as np

def hist2samp(target0, target1, label0, label1, nbins=20):
    bins = np.linspace(-3, 3, nbins)
    width = 6/nbins/3
    (target0 - target0.shift(1)).hist(bins=bins+width, width=width/1.2, density=True, label=label0, ax=plt.gca())
    (target1 - target1.shift(1)).hist(bins=bins+3*width, width=width/1.2, density=True, label=label1, ax=plt.gca())
    plt.legend()
    plt.grid(False)
    plt.show()

def pacf2samp(target0, target1, label0, label1, nlags=40):
    plt.figure(figsize=(6, 7))
    ax = plt.subplot(2, 1, 1)
    plot_pacf(target0, lags=nlags, title=f'PACF {label0}', ax=ax)
    ax = plt.subplot(2, 1, 2)
    plot_pacf(target1, lags=nlags, title=f'PACF {label1}', ax=ax)
    plt.show()
