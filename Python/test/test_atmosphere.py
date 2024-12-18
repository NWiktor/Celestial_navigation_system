# -*- coding: utf-8 -*-
# !/usr/bin/python3
""" Unittesting atmosphere.py """
import unittest
from cls import EarthAtmosphereUS1976


class TestEarthAtmosphereUS1976(unittest.TestCase):
    """ Testing EarthAtmosphereUS1976 class. """
    @classmethod
    def setUpClass(cls):
        cls.atm = EarthAtmosphereUS1976()

    def test_tempretatures(self):
        """ Test temperatures. """
        self.assertAlmostEqual(self.atm.get_temperature(0), 288.15, 3,
                               "Should be 288.15 K° at ground!")
        self.assertAlmostEqual(self.atm.get_temperature(11000), 216.65, 3,
                               "Should be 216.65 K° for 11000 (m) altitude!")
        self.assertAlmostEqual(self.atm.get_temperature(20000), 216.65, 3,
                               "Should be 216.65 K° for 20000 (m) altitude!")
        self.assertAlmostEqual(self.atm.get_temperature(100000), 0, 3,
                               "Should be 0 K° for 100000 (m) altitude!")

    def test_pressure(self):
        """ Test pressures. """
        self.assertAlmostEqual(self.atm.get_pressure(0), 101.325, 3,
                               "Should be 101.325 kPa at ground!")
        self.assertAlmostEqual(self.atm.get_pressure(11000), 22.632, 3,
                               "Should be 22.632 kPa for 11000 (m) altitude!")
        self.assertAlmostEqual(self.atm.get_pressure(100_000), 0, 3,
                               "Should be 0 kPa for 100000 (m) altitude!")


if __name__ == '__main__':
    unittest.main()
