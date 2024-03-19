# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module describes the abstract Atmosphere class, and different atmospheric models (e.g. Earth and Mars).

Libs
----
* Matplotlib - for data visualization and test

Help
----
* https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
* https://www.digitaldutch.com/atmoscalc/US_Standard_Atmosphere_1976.pdf
* https://en.wikipedia.org/wiki/Barometric_formula
* http://www.luizmonteiro.com/StdAtm.aspx
* https://www.grc.nasa.gov/www/k-12/airplane/atmosmrm.html

Contents
--------
"""

# Standard library imports
import logging
import math as m
import matplotlib.pyplot as plt

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)

# Add here rocket hardware specs
# Falcon
# Vulcan
# Starship


class Attitude:
    """ Baseclass for a generic atmospheric model. """

    def __init__(self, model_name, atm_lower_limit_m: int, atm_upper_limit_m: int):
        self.model_name = model_name
        self.atm_lower_limit_m = atm_lower_limit_m  # Lower limit
        self.atm_upper_limit_m = atm_upper_limit_m  # Upper limit

    def _apply_limits(self, altitude) -> int:
        """ If the given value is out of range of the valid atmospheric model, returns the applicable value.
        This function is needed to in case an iterative calculation accidentally runs through the limits.
        When the altitude smaller than 0, there is solid ground, no atmosphere; this case should be handled
        anyway elsewhere. When altitude is above range, there is space (no pressure, no atmosphere).
        """
        return min(max(self.atm_lower_limit_m, altitude), self.atm_upper_limit_m)

    # pylint: disable = unused-argument
    # @override
    def _atmospheric_model(self, altitude) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values. """
        return 0.0, 0.0, 0.0

    def get_atm_params(self, altitude) -> tuple[float, float, float]:
        """ Enforces the atmospheric model limits, and returns the temperature, pressure, density values. """
        return self._atmospheric_model(self._apply_limits(altitude))

    def get_temperature(self, altitude) -> float:
        """ Returns the temperature at a given altitude. """
        return self._atmospheric_model(altitude)[0]

    def get_pressure(self, altitude) -> float:
        """ Returns the pressure at a given altitude. """
        return self._atmospheric_model(altitude)[1]

    def get_density(self, altitude) -> float:
        """ Returns the density at a given altitude. """
        return self._atmospheric_model(altitude)[2]


def module_test():
    pass


# Include guard
if __name__ == '__main__':
    pass
