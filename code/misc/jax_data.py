"""EMNIST task loading and batching for the JAX inner loop."""

import os
import random
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image


class EmnistDataset:
    def __init__(self, minTrainingDataPerClass, maxTrainingDataPerClass, queryDataPerClass,
                 dimensionOfImage=28, use_jax=True):
        self.root = str(Path(__file__).resolve().parents[2] / "data" / "emnist")
        self.classes = sorted(folder for folder in os.listdir(self.root)
                              if os.path.isdir(os.path.join(self.root, folder)))
        self.k = maxTrainingDataPerClass
        self.q = queryDataPerClass
        self.image_size = dimensionOfImage

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        files = sorted(os.listdir(os.path.join(self.root, self.classes[index])))
        class_dir = os.path.join(self.root, self.classes[index])
        images = [np.asarray(Image.open(os.path.join(class_dir, f), mode="r").convert("L")
                             .resize((self.image_size, self.image_size)), dtype=np.float32)[None] / 255
                  for f in files[:self.k + self.q]]
        images = jnp.asarray(np.stack(images))
        labels = jnp.full((self.k + self.q,), index)
        return images[:self.k], labels[:self.k], images[self.k:], labels[self.k:]


def emnist_batches(dataset, epochs, classes, batch_size):
    for _ in range(epochs):
        indices = random.sample(range(len(dataset)), classes)
        tasks = [dataset[i] for i in indices]
        yield tuple(jnp.asarray(np.stack([task[col] for task in tasks])) for col in range(4))


class DataProcess:
    def __init__(self, minTrainingDataPerClass, maxTrainingDataPerClass,
                 queryDataPerClass, dimensionOfImage, iid=True, use_jax=True,
                 permutation=False):
        self.min_k = minTrainingDataPerClass
        self.max_k = maxTrainingDataPerClass
        self.q = queryDataPerClass
        self.dimension = dimensionOfImage
        self.iid = iid
        self.permutation = permutation

    def __call__(self, data, classes):
        x_train, y_train, x_query, y_query = data
        k = random.randint(self.min_k, self.max_k)
        x_train, y_train = x_train[:, :k], y_train[:, :k]
        x_train = x_train.reshape(1, classes * k, self.dimension**2)
        y_train = y_train.reshape(1, classes * k, 1)
        x_query = x_query.reshape(1, classes * self.q, self.dimension**2)
        y_query = y_query.reshape(1, classes * self.q, 1)
        if self.permutation:
            perm = np.random.permutation(self.dimension**2)
            x_train, x_query = x_train[:, :, perm], x_query[:, :, perm]
        if self.iid:
            perm = np.random.permutation(classes * k)
            x_train, y_train = x_train[:, perm], y_train[:, perm]
        return x_train, y_train, x_query, y_query, k, 1
