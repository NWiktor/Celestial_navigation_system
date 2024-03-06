# pylint: disable = missing-module-docstring
import unittest
from cls import EarthAtmosphere


class TestEarthAtmosphere(unittest.TestCase):
    """ Unittest functions for EarthAtmosphere class. """

    @classmethod
    def setUpClass(self):
        self.atm = EarthAtmosphere()

    def test_apply_limits(self):
        """ Test for put of limit values. """

        self.assertEqual(self.atm.apply_limits(-2), 0, "Should be zero for negativ values!")
        self.assertEqual(self.atm.apply_limits(0), 0, "Should be zero for 0!")
        self.assertEqual(self.atm.apply_limits(120000), 120000, "Should be 120000 for 120000!")
        self.assertEqual(self.atm.apply_limits(120001), 120000, "Should be 120000 for values higher than 120000!")


if __name__ == '__main__':
    unittest.main()
