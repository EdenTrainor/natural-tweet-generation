import os
import sys
import json
import numpy as np
from numba import njit
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import model_from_json


@njit
def one_hot_features(start_inds, data, sample_len, nchars):
    """
    Generates the one hot encoded feature and target arrays. 
    As it's a nested for-loop invoving just numpy it's massivly more 
    performant to compile it with numba.

    Args
    ----
    start_inds, np.ndarray, (batch_size,)
        Array of starting indices for the samples in this batch

    data, np.ndarray, (corpus_size, )
        The full corpus of text in ordinal form

    sample_len, int
        Number of characters before target character

    nchars, int
        Number of unique characters in corpus and the size of each one hot array
    """
    out = np.zeros((len(start_inds), sample_len + 1, nchars), np.float32)
    for j, start_ind in enumerate(start_inds):
        for k, val in enumerate(np.arange(start_ind, start_ind + sample_len + 1)):
            out[j, k, data[val]] = 1
    return out[:, :-1, :], out[:, -1, :]


class DataGenerator(Sequence):
    """
    Generates data as the model trains.
    This means we can prepare the next batch on the
    cpu as we train the model on the GPU, speeding up the whole process.        
    """

    def __init__(self, data, sample_len, step_size, batch_size=32, shuffle=True):
        """
        Args
        ----
        data, TextCorpus
            Training text.

        sample_len, int
            Number of characters before target character.

        step_size, int
            Spacing in text between the start of each sample. This will control 
            how much overlap there is in each sample.

        batch_size, int
            The number of samples per batch.

        shuffle, bool
            Shuffle data after each epoch.
        """
        assert isinstance(data, TextCorpus)
        self.corpus = data.encode_numerical(data.corpus)
        self.nchars = data.get_num_chars()
        self.sample_len = sample_len
        self.batch_size = batch_size
        self.epoch_size = len(data) // batch_size
        self.shuffle = shuffle
        # Steps of sample_len +1 for target to be captured in same array
        self.train_index = np.arange(
            0, len(data) - sample_len - 1, step_size)

    def __len__(self):
        """
        Must return the number of batches in an epoch
        """
        return self.epoch_size

    def __getitem__(self, pos):
        """
        Must return the batch given by the position in the epoch.

        Args
        ----
        pos, int
            Position (index) in the epoch.
        """
        sample_starts = self.train_index[pos: pos + self.batch_size]
        features, targets = one_hot_features(
            sample_starts, self.corpus, self.sample_len, self.nchars,
        )
        return features, targets

    def on_epoch_end(self):
        """
        Shuffle the data after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.train_index)

    def shape(self):
        return (self.batch_size, self.sample_len, self.nchars)


class TextCorpus:

    """
    Stores a text corpus and has utility methods.
    Persists text to char conversion. 
    """

    def __init__(self, corpus, save_dir="./model/", embed=True):
        """
        Args
        ----
        corpus, str
            A string containing the data set

        save_dir, str
            Directory to save the string-intiger embedding files

        embed, bool
            Embed if this is the full text corpus. Don't embed if you're taking 
            a subset but want to retain the previous embedding. e.g. see 
            __getitem__ method.
        """
        self.save_dir = save_dir
        if not isinstance(corpus, np.ndarray):
            # Allows for memory/embedding sharing when using train/test split
            # with __getitem__
            self.corpus = np.fromiter(corpus, np.dtype('U1'))
        else:
            self.corpus = corpus
        if embed:
            self.embed(self.corpus)

    def embed(self, corpus):
        """
        Checks if embedding exists, if so, it loads it, if not, creates it and 
        saves it as a json in self.save_dir

        Args
        ----
        corpus, str
            Concatenated string of entire text corpus

        Returns
        -------
        char_to_indices, dict
            Dictionary mapping from characters to indices

        indicies_to_char, dict
            Dictionary mapping from indices to characters
        """
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        try:
            self.char_to_indicies = json.load(
                self.save_dir + "char_to_indicies.json")
            self.indicies_to_char = json.load(
                self.save_dir + "indicies_to_char.json")
        except:
            characters = sorted(list(set(corpus)))
            self.char_to_indicies = dict((c, i)
                                         for i, c, in enumerate(characters))
            self.indicies_to_char = dict((i, c)
                                         for i, c in enumerate(characters))
            with open(self.save_dir + "char_to_indicies.json", "w") as out_file:
                json.dump(self.char_to_indicies, out_file)
            with open(self.save_dir + "indicies_to_char.json", "w") as out_file:
                json.dump(self.indicies_to_char, out_file)

    def encode_numerical(self, text):
        """
        Encodes text into ordinal representation.

        Returns
        -------
        data, np.ndarray(np.int32)
            Numerical corpus, ordinally encoded
        """
        data = np.zeros((len(text),), dtype=np.int32)
        for i, t in enumerate(text):
            data[i] = self.char_to_indicies[t]
        return data

    def decode_numerical(self, nums):
        """
        Decodes ordinal representation into string.

        Returns
        -------
        data, np.ndarray(np.int32)
            Numerical corpus, ordinally encoded
        """
        ret = []
        for n in nums:
            ret.append(self.indicies_to_char[n])
        return ''.join(ret)

    def __len__(self):
        """
        Utility wrapper around corpus string

        Returns
        -------
        _, int
            Lenght of the text corpus
        """
        return len(self.corpus)

    def __getitem__(self, key):
        """
        Creates a second TextCorpus object of the data with the same embedding 
        but a view of the data in order to conserve memory.
        """
        out = TextCorpus(self.corpus[key], save_dir=self.save_dir, embed=False)
        out.char_to_indicies = self.char_to_indicies
        out.indicies_to_char = self.indicies_to_char
        return out

    def get_num_chars(self):
        """
        Returns the number of unique characters in the full text corpus
        """
        return len(self.char_to_indicies)


def pred_indicies(preds, metric=1.0):
    """Function to convert predictions into index
    (metric decides how to flatten the probability distribution -> makes bot more 'imaginative')
    """

    preds = np.asarray(preds).astype("float64")

    # Scale Preds by metric value
    preds = np.log(preds) / metric
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # Resample
    probs = np.random.multinomial(1, preds, 1)

    return np.argmax(probs)


def save_model(model):
    # Save model config
    with open("./logs/" + model.name + "_config.json", "w+") as json_file:
        json_file.write(model.to_json())

    # Save weights to acompanying file
    model.save_weights("./logs/" + model.name + "_weights.h5")


def load_model(model_name):

    # Load model configuration
    with open("./logs/" + model_name + "_config.json", "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    # Load weights into new model
    model.load_weights("./logs/" + model_name + "_weights.h5")
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    print("Model loaded from file...")
    return model


def generate_tweet(model, seed_index, diversity, corp):
    with open('./logs/' + model.name + '_logs.txt', "a+") as log_file:
        logwrite(log_file, '----- diversity: {}\n'.format(diversity))

        sentence = ''.join(corp.corpus[seed_index: seed_index +
                                       model.input_shape[1] + 1])
        generated = sentence

        logwrite(log_file, '----- Generating with seed: "' + sentence + '"\n')
        logwrite(log_file, generated, True)

        pred_char = ""
        def end_of_sentence(char): return char == "@"

        while not end_of_sentence(pred_char):
            x = corp.encode_numerical(sentence)
            xOH, _ = one_hot_features(
                np.array([0]), x, model.input_shape[1], corp.get_num_chars()

            )
            preds = model.predict(xOH, verbose=0)[0]

            next_index = pred_indicies(preds, diversity)
            pred_char = corp.indicies_to_char[next_index]
            generated += pred_char
            sentence = sentence[1:] + pred_char
            logwrite(log_file, pred_char, True)

        logwrite(log_file, '\n', True)
        sys.stdout.flush()


def logwrite(file, string, to_std=False):
    file.write(string)
    if to_std:
        sys.stdout.write(string)
    else:
        print(string)
