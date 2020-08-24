import tensorflow as tf
import numpy as np
import os

from bot_utils import (
    TextCorpus,
    load_corpus,
)

DATA_FILE = "./data/archive.json"


def generate_dataset(corpus, sample_size, step_size, batch_size, shuffle):
    dataset = tf.data.Dataset.from_tensor_slices(corpus)
    dataset = make_window_dataset(dataset, sample_size, step_size)
    if shuffle:
        dataset.shuffle(, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    return dataset


def main():
    corpus = TextCorpus(load_corpus(DATA_FILE, False))
    data = corpus.encode_numerical(corpus.corpus)
    dataset = tf.
