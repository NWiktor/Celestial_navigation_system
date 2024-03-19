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


def module_test():
    pass


# Include guard
if __name__ == '__main__':
    pass
