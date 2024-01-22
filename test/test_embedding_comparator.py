import unittest

import numpy as np
import numpy.testing as np_test

from embedding_comparator import get_cosine_similarity


class EmbeddingComparatorTest(unittest.TestCase):
    def test_cosine_similarity(self):
        array = np.array([[1, 2, 3]])
        self.assertEqual(
            get_cosine_similarity(array, np.array([1, 2, 3])),
            1
        )
        np_test.assert_array_almost_equal(
            get_cosine_similarity(np.array([[1, 1]]), np.array([0, 1])),
            [0.707],
            decimal=3
        )

    def test_nd_cosine_similarity(self):
        np_test.assert_array_almost_equal(
            np.array([1, 1]),
            get_cosine_similarity(
                np.array([[1, 1], [1, 1]]),
                np.array([1, 1])
            )
        )
        np_test.assert_array_almost_equal(
            np.array([1, 0.707]),
            get_cosine_similarity(
                np.array([[1, 1], [0, 1]]),
                np.array([1, 1])
            ),
            decimal=3
        )

    def test_error_messages(self):
        array_1d = np.array([1, 2, 3])
        array_2d = np.array([[1, 2, 3]])
        with self.assertRaises(AssertionError) as exception_context:
            get_cosine_similarity(array_1d, array_1d)
        self.assertEqual(exception_context.exception.args[0], "matrix_2d must be a 2D matrix")
        with self.assertRaises(AssertionError) as exception_context:
            array_1d = np.array([1, 2, 3])
            get_cosine_similarity(array_2d, array_2d)
        self.assertEqual(exception_context.exception.args[0], "reference_1d_array must be a 1D array")
