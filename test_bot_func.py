import unittest as ut
import bot_func as bf
from numpy.testing import assert_array_equal
import numpy as np


class TestOneHotFeatures(ut.TestCase):
    def test_one_hot_features(self):

        nchars = 5
        sample_len = 5
        data = np.array([1, 2, 2, 4, 2, 0, 2, 3, 3, 2, 1])
        start_inds = np.array([0, 3, 4])

        want_features = np.array([
            [[0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0]],
            [[0, 0, 0, 0, 1],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0]],
            [[0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0]],
        ])
        want_targets = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ])

        features, targets = bf.one_hot_features(
            start_inds, data, sample_len, nchars)

        self.assertIsNone(
            assert_array_equal(want_targets, targets),
            "Resulting targets doesn't match:\nWanted: {}\nGot: {}".format(
                want_targets, targets),
        )
        self.assertIsNone(
            assert_array_equal(want_features, features),
            "Resulting features doesn't match:\nWanted: {}\nGot: {}".format(
                want_features, features),
        )


if __name__ == "__main__":
    ut.main()
