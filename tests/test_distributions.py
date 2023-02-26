from unittest import TestCase

import numpy as np

from bwb.distributions import DiscreteDistribution


class TestDiscreteDistribution(TestCase):

    def setUp(self) -> None:
        self.d1 = DiscreteDistribution(weights=[0.1, 0.2, 0.5, 0.2],
                                       support=[1, 1, 2, 3])

    def test_pdf(self):
        self.assertAlmostEqual(
            0.3,
            self.d1.pdf(1), msg="'pdf' method does not works."
        )
        self.assertAlmostEqual(
            0.5,
            self.d1.pdf(2), msg="'pdf' method does not works."
        )
        self.assertAlmostEqual(
            0.0,
            self.d1.pdf(0), msg="'pdf' method does not works."
        )

        self.assertTrue(
            np.allclose(
                np.array([0.3, 0.5]),
                self.d1.pdf([1, 2])
            ), "'pdf' method does not works for list of values."
        )

    def test_rvs(self):
        arr1 = self.d1.rvs(size=10, random_state=42)
        arr2 = self.d1.rvs(size=10, random_state=42)
        self.assertEqual(
            10,
            len(arr1),
            "The length does not match with the expected."
        )
        self.assertTrue(
            np.all(arr1 == arr2),
            "The seed does not works."
        )

    def test_draw(self):
        s1 = self.d1.draw(42)
        s2 = self.d1.draw(42)
        self.assertEqual(s1, s2, "Seed does not works well.")
