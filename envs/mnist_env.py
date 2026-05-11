import os
import gzip
import struct
import numpy as np


def _load_images(filepath):
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rb") as f:
        f.read(4)
        n = struct.unpack(">I", f.read(4))[0]
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 784).astype(np.float64)


def _load_labels(filepath):
    opener = gzip.open if filepath.endswith(".gz") else open
    with opener(filepath, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist(path="datasets/MNIST/raw"):
    images, labels = None, None
    for f in os.listdir(path):
        if "train-images" in f and (images is None or not f.endswith(".gz")):
            images = _load_images(os.path.join(path, f))
        elif "train-labels" in f and (labels is None or not f.endswith(".gz")):
            labels = _load_labels(os.path.join(path, f))
    if images is None or labels is None:
        raise FileNotFoundError(f"MNIST train files not found under {path}")
    return images, labels


def group_by_class(features, labels, n_classes=10):
    return {k: features[labels == k] for k in range(n_classes)}


class MNISTBanditEnv:
    def __init__(self, clusters, target_class=0):
        self.clusters = clusters
        self.target_class = target_class
        self.K = 10
        self.d = 784
        self.reset()

    def reset(self):
        self.t = 0
        self.mistakes = 0
        self.cumulative_mistakes = []

    def get_contexts(self):
        contexts = np.zeros((self.K, self.d))
        for k in range(self.K):
            idx = np.random.randint(len(self.clusters[k]))
            contexts[k] = self.clusters[k][idx]
        return contexts

    def step(self, action):
        self.t += 1
        if action == self.target_class:
            reward = 1
        else:
            reward = 0
            self.mistakes += 1
        self.cumulative_mistakes.append(self.mistakes)
        return reward
