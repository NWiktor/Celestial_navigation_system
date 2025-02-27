# -*- coding: utf-8 -*-
# !/usr/bin/python3
""" Unittesting time_functions.py """
import unittest
from datetime import datetime
from utils import time_functions as tf


# pylint: disable = missing-class-docstring
class TestTimeFunctions(unittest.TestCase):

    def test_julian_date(self):
        self.assertEqual(
            tf.julian_date(datetime(2024, 1, 1, 13, 26, 34)),
            2460311.0601157406, "Check date of 2024-01-01 13:26:34")

    def test_j2000_date(self):
        self.assertEqual(
            tf.j2000_date(datetime(2000, 1, 1, 12, 00)),
            0.0,
            "Check epoch date with itself (2000-01-01 12:00:00), should be 0.0")


if __name__ == '__main__':
    unittest.main()
