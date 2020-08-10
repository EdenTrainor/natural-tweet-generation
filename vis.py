import numpy as np
import random

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense


class PyplotHistoryCallback(Callback):

    def __init__(self, save_path='./logs'):
        self.save_path = save_path

    # def plot_history(self, history):
    #     if hasattr(self, "fig"):
    #         return self.update_plot(history)
    #     t_metrics, v_metrics = self.split_metrics(history)
    #     nmetrics = len(t_metrics)
    #     self.fig, self.ax = plt.subplots(nrows=)

    #     self.fig.canvas.draw()

    # def split_metrics(self, history):
    #     def skim_key(key): return "_".join(key.split("_")[1:])
    #     t_metrics = {
    #         skim_key(k): history[k] for k in history.keys() if k.startswith("train")}
    #     v_metrics = {
    #         skim_key(k): history[k] for k in history.keys() if k.startswith("val")}
    #     return t_metrics, v_metrics

    # def update_plot(self, history):
    #     pass


def get_data(nsamples):
    data = np.random.rand(nsamples, 10)
    targets = np.zeros((nsamples, 5))
    for row in range(nsamples):
        targets[row, random.randint(0, 4)] = 1
    return data, targets


def main():
    nepochs = 100
    model = Sequential()
    model.add(Dense(10, input_shape=(10,)))
    model.add(Dense(5), activation='softmax')
    model.compile(loss="cross_entropy", optimizer='adam')
    tp = PyplotHistoryCallback()
    X, y = get_data(100)
    for _ in range(nepochs):
        history = model.fit(X, y, call)
        tp.plot_history(history.history)


if __name__ == '__main__':
    main()
