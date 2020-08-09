import numpy as np
import matplotlib.pyplot as pyplot


class ModelMock:

    def __init__(self, nvals):
        np.random.seed(42)
        self.nvals = nvals
        self.train_loss = np.random.rand(nvals).cumsum()
        self.val_loss = np.random.rand(nvals).cumsum()
        self.train_acc = np.random.rand(nvals).cumsum()
        self.val_acc = np.random.rand(nvals).cumsum()

    @property
    def history(self):
        return next(self.history_gen())

    def history_gen(self):
        for i in range(1, self.nvals+1):
            yield {
                "train_loss": self.train_loss[:i],
                "train_acc": self.train_acc[:i],
                "val_loss": self.val_loss[:i],
                "val_acc": self.val_acc[:i]
            }


class TrainingPlotter:

    def __init__(self, save_path='./logs'):
        pass

    def plot_history(self, history):
        if hasattr(self, "fig"):
            return self._update_plot(history)

    def _update_plot(self, history):
        pass


def main():
    nepochs = 100
    model = ModelMock(nepochs)
    tp = TrainingPlotter()
    for i in range(nepochs):
        tp.plot_history(model.history)


if __name__ == '__main__':
    main()
