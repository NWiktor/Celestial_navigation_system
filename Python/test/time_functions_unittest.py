# pylint: disable = missing-module-docstring
import unittest
from utils import time_functions as tf


# pylint: disable = missing-class-docstring
class TestTimeFunctions(unittest.TestCase):

    def test_julian_date(self):
        """ Check ulian_date function. """
        self.assertEqual(tf.julian_date(2024, 1, 1, 13, 26, 34),
                         2460311.0601157406, "Check date of 2024-01-01 13:26:34")

    def test_j2000_date(self):
        """ Check J2000_date function. """
        self.assertEqual(tf.j2000_date(2000, 1, 1, 12, 00), 2000.0,
                         "Check epoch date with itself (2000-01-01 12:00:00)")


if __name__ == '__main__':
    unittest.main()
