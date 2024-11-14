# pylint: disable = missing-module-docstring
import unittest
from cls import EarthAtmosphereUS1976


class TestEarthAtmosphere(unittest.TestCase):
    """ Unittest functions for EarthAtmosphere class. """

    @classmethod
    def setUpClass(self):
        self.atm = EarthAtmosphereUS1976()

    def test_apply_limits(self):
        """ Test for put of limit values. """
        self.assertEqual(self.atm._apply_limits(-2), 0,
                         "Should be zero for negativ values!")
        self.assertEqual(self.atm._apply_limits(0), 0,
                         "Should be zero for 0!")
        self.assertEqual(self.atm._apply_limits(100000), 100000,
                         "Should be 120000 for 120000!")
        self.assertEqual(self.atm._apply_limits(100001), 100000,
                         "Should be 120000 for values higher than 120000!")

    def test_tempretatures(self):
        """ Test temperatures. """
        self.assertAlmostEqual(self.atm.get_temperature(0), 288.15, 3,
                               "Should be 288.15 at ground!")
        self.assertAlmostEqual(self.atm.get_temperature(11000), 216.65, 3,
                               "Should be 216.65 for 11000 (m) altitude!")
        self.assertAlmostEqual(self.atm.get_temperature(20000), 216.65, 3,
                               "Should be 216.65 for 20000 (m) altitude!")


if __name__ == '__main__':
    unittest.main()
