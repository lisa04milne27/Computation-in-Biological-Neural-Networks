"""Small JAX-runner logging and plotting helpers."""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def log(data, filename):
    with open(filename, "a") as f:
        np.savetxt(f, np.asarray(data), newline=" ", fmt="%0.6f")
        f.write("\n")


def accuracy(logits, labels, use_jax=True):
    return jnp.mean(jnp.argmax(logits, axis=1) == labels)


class Plot:
    def __init__(self, directory, window_size=10):
        self.directory, self.window = directory, window_size

    def __call__(self):
        for filename, title, ylim in (("acc_meta.txt", "Meta Accuracy", (0, 1)),
                                      ("loss_meta.txt", "Meta Loss", None)):
            values = np.atleast_1d(np.loadtxt(os.path.join(self.directory, filename)))
            window = min(self.window, len(values))
            plotted = np.convolve(values, np.ones(window) / window, mode="valid")
            plt.plot(plotted)
            plt.title(title)
            if ylim:
                plt.ylim(ylim)
            plt.savefig(os.path.join(self.directory, title.lower().replace(" ", "_")), bbox_inches="tight")
            plt.close()
