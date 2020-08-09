from datetime import datetime

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Input, Dense, LSTM
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from eda_func import tweet_cleaner, rmv_uncommon
from bot_func import (
    DataGenerator,
    TextCorpus,
    one_hot_features,
    generate_tweet,
    create_class_weight,
)


def main():
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(
    #     physical_devices[0], True)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    DATA_FILE = "./data/archive.json"

    df = pd.read_json(DATA_FILE)

    # Remove hyerlinks etc
    df["text"] = df["text"].apply(tweet_cleaner)
    df["text"] = df.text.apply(rmv_uncommon)
    df = df[(df["text"] != "") | (df["text"] != " ")]
    df = df[df.is_retweet == False]

    # Use "special" symbol @ to indicate end of tweet, since we removed them all before
    corpus = "@".join(df["text"].values)

    full_corp = TextCorpus(corpus)

    train_corp = full_corp[: int(len(full_corp) * 0.9)]
    test_corp = full_corp[int(len(full_corp) * 0.9):]

    SAMPLE_LEN = 32
    STEP_SIZE = 3
    BATCH_SIZE = 32
    MU = 0.15

    class_weights = create_class_weight(train_corp.corpus, MU)
    train_gen = DataGenerator(train_corp, SAMPLE_LEN,
                              STEP_SIZE, batch_size=BATCH_SIZE)
    test_gen = DataGenerator(test_corp, SAMPLE_LEN,
                             STEP_SIZE, batch_size=BATCH_SIZE, shuffle=False)

    nchars = full_corp.get_num_chars()

    model = Sequential(name="test_bot-dropout")
    model.add(Input(shape=(SAMPLE_LEN, nchars), dtype=np.float32))
    model.add(LSTM(256, return_sequences=False))
    # model.add(Dropout(0.4))
    # model.add(LSTM(128))
    # model.add(Dropout(0.2))
    model.add(Dense(nchars, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print(model.summary())

    # logdir = "logs/" + model.name + "/scalars/" + \
    #     datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    tweet_ends = np.where(np.asarray(list(corpus)) == "@")[0]

    for epoch in range(1, 100):
        print('-' * 40)
        print('Epoch', epoch)
        model.fit(
            train_gen,
            steps_per_epoch=train_gen.epoch_size,
            epochs=1,
            validation_data=test_gen,
            validation_steps=test_gen.epoch_size,
            class_weight=class_weights,
            # callbacks=[tensorboard_callback],
        )

        seed_index = 1 + np.random.choice(tweet_ends, 1)[0]
        for diversity in [0.2, 0.7, 1.2]:
            generate_tweet(model, seed_index, diversity, full_corp)
        print('-' * 40)


if __name__ == '__main__':
    main()
