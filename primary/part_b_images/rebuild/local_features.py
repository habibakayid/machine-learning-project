"""Describe small patches and pool their differences from learned visual words."""

import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.preprocessing import normalize

from dataset_preparation import SEED
from feature_extraction import CENTER, histogram, quantize

PATCHES = 98
DESCRIPTOR_SIZE = 55


def describe_patches(rgb, mask):
    gray = rgb2gray(rgb)
    h, s, v = rgb2hsv(rgb).transpose(2, 0, 1)
    hue = np.where(s < .15, 0, 1 + quantize((h + 1 / 24) % 1, 12))
    sat, light = quantize(s, 3), quantize(v, 3)
    patches = []
    for size, step in ((8, 2), (16, 1)):
        grid = hog(gray, orientations=9, pixels_per_cell=(size, size),
                   cells_per_block=(2, 2), feature_vector=False)
        # The 8-pixel patches use the same 7 by 7 sampling grid as the larger patches.
        offset = 1 if step == 2 else 0
        for i in range(7):
            for j in range(7):
                a, b = i * step + offset, j * step + offset
                region = np.s_[a * size:(a + 2) * size, b * size:(b + 2) * size]
                visible = mask[region]
                weight = visible.mean() * (.5 + CENTER[region].mean())
                descriptor = [grid[a, b].ravel(), histogram(hue[region], 13, visible) * .7,
                              histogram(sat[region], 3, visible) * .35,
                              histogram(light[region], 3, visible) * .35]
                patches.append(np.concatenate([*descriptor, [weight]]))
    return np.asarray(patches, dtype=np.float32).ravel()


class LocalVisualWords(TransformerMixin, BaseEstimator):
    """Fit patch centers on training images and pool signed residuals per photo."""

    def __init__(self, n_words=24, sample_count=20000):
        self.n_words = n_words
        self.sample_count = sample_count

    def fit(self, x, y=None):
        patches = x.reshape(-1, DESCRIPTOR_SIZE + 1)
        valid = np.flatnonzero(patches[:, -1] >= .65)
        rng = np.random.default_rng(SEED)
        sample = rng.choice(valid, size=min(self.sample_count, len(valid)), replace=False)
        codebook = MiniBatchKMeans(n_clusters=self.n_words, random_state=SEED,
                                   n_init=3, batch_size=1024)
        self.centers_ = codebook.fit(patches[sample, :-1]).cluster_centers_
        return self

    def transform(self, x):
        output = np.empty((len(x), self.n_words * DESCRIPTOR_SIZE), dtype=np.float32)
        for start in range(0, len(x), 128):
            batch = x[start:start + 128].reshape(-1, PATCHES, DESCRIPTOR_SIZE + 1)
            patches = batch[:, :, :-1].reshape(-1, DESCRIPTOR_SIZE)
            weights = batch[:, :, -1].ravel()
            words = pairwise_distances_argmin(patches, self.centers_)
            residuals = (patches - self.centers_[words]) * weights[:, None]
            pooled = np.zeros((len(batch), self.n_words, DESCRIPTOR_SIZE), dtype=np.float32)
            np.add.at(pooled, (np.repeat(np.arange(len(batch)), PATCHES), words), residuals)
            pooled = np.sign(pooled) * np.sqrt(np.abs(pooled))
            pooled /= np.maximum(np.linalg.norm(pooled, axis=2, keepdims=True), 1e-8)
            output[start:start + len(batch)] = normalize(pooled.reshape(len(batch), -1))
        return output
