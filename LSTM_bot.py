from bot_utils import (
    DataGenerator,
    TextCorpus,
    one_hot_features,
    generate_tweet,
    pred_indicies,
    create_class_weight,
    load_corpus,
)
from tensorflow.keras.layers import Dropout, Input, Dense, LSTM
from tensorflow.keras import Sequential
import tensorflow as tf
from datetime import datetime
import random
import sys
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

DATA_FILE = "./data/archive.json"
MODEL_NAME = 'full_run1'
NUM_EPOCHS = 15
SAMPLE_LEN = 32
STEP_SIZE = 3
BATCH_SIZE = 32
MU = 0.15


def get_model(in_shape):
    model = Sequential([
        Input(shape=in_shape, dtype=np.float32),
        LSTM(256, return_sequences=True),
        Dropout(0.4),
        LSTM(128),
        Dropout(0.2),
        Dense(in_shape[-1], activation="softmax"),
    ],
        name=MODEL_NAME,
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print(model.summary())
    return model


def main():
    full_corpus = TextCorpus(load_corpus(DATA_FILE))

    train_corpus = full_corpus[: int(len(full_corpus) * 0.9)]
    test_corpus = full_corpus[int(len(full_corpus) * 0.9):]

    class_weights = create_class_weight(train_corpus.corpus, MU)

    # make_window_dataset
    train_gen = DataGenerator(train_corpus, SAMPLE_LEN,
                              STEP_SIZE, batch_size=BATCH_SIZE)
    test_gen = DataGenerator(test_corpus, SAMPLE_LEN,
                             STEP_SIZE, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model(in_shape=(SAMPLE_LEN, full_corpus.get_num_chars()))

    tweet_ends = np.where(full_corpus.corpus == "@")[0]

    for epoch in range(1, NUM_EPOCHS):
        print('-' * 40)
        print('Epoch', epoch)

        model.fit(
            train_gen,
            steps_per_epoch=train_gen.epoch_size,
            epochs=1,
            validation_data=test_gen,
            validation_steps=test_gen.epoch_size,
            class_weight=class_weights,
            workers=4,
            use_multiprocessing=True,
        )

        seed_index = 1 + np.random.choice(tweet_ends, 1)[0]
        for diversity in [0.2, 0.7, 1.2]:
            generate_tweet(model, seed_index, diversity, full_corpus)


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    main()
