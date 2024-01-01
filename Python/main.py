# -*- coding: utf-8 -*-
#!/usr/bin/python3

""" Summary of this code file goes here. The purpose of this module can be
expanded in multiple sentences. Below a short free-text summary of the included
classes and functions to give an overview. More detailed summary of the
functions can be provided inside the function's body.

Libs
----
* some_module - This is used for imported, non-standard modules, to help track
    dependencies. Summary is not needed.

Help
----
* https://en.wikipedia.org/wiki/Truncated_icosahedron

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import sys
import datetime
import time
import math as m

# Third party imports
import numpy as np

# Local application imports
from logger import MAIN_LOGGER as l
from modules.time_functions import julian_date, j2000_date

# Class initializations and global variables
gravitational_constant = 6.67430 * pow(10,-11) # m^3 kg-1 s-2

# Class and function definitions


class Orbit():

    def __init__(self, eccentricity, semimajor_axis, inclination,
        longitude_of_ascending_node, argument_of_periapsis, mean_anomaly_at_epoch):
        # Keplerian elements needed for calculating orbit
        # https://en.wikipedia.org/wiki/Orbital_elements
        # https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion
        # https://en.wikipedia.org/wiki/Ellipse
        """  """
        self.eccentricity = eccentricity # Eccentricity (e),
        self.semimajor_axis = semimajor_axis # Semimajor axis (a), km
        self.inclination = inclination # (i), deg
        self.longitude_of_ascending_node = longitude_of_ascending_node
        # Longitude of the ascending node (Ω), deg
        self.argument_of_periapsis = argument_of_periapsis # Argument of periapsis (ω), deg
        self.mean_anomaly_at_epoch = mean_anomaly_at_epoch # Mean anonaly, deg


    def orbital_parameters(self, orbital_time, mean_motion, orbital_period):
        """  """
        self.orbital_time = orbital_time # s
        self.mean_motion = mean_motion # (n) 1/s
        self.orbital_period = orbital_period # s


    def get_position(self, time):
        # calculate time difference from epoch
        # calculate true anomaly from time diff.
        # convert position coordinates to x,y,z coordinates for animation
        pass


    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)



class CelestialObject():

    def __init__(self, name, uuid, mass, radius, parent_object=None):
        self._name = name
        self._uuid = uuid # Unique identifier
        self._mass = mass # kg
        self._radius = radius # km ???
        self._parent_object = parent_object
        self._orbit = None


    def set_orbit(self, orbit):
        self._orbit = orbit


    def get_position(self, time):
        return self._orbit.get_position(time)


    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# Main function for module testing
def main():
    """  """

    nap = CelestialObject("Nap", "0001", 2500, 3400)
    print(nap)

    fold = CelestialObject("Fold", "0002", 2500, 3400)
    fold_orbit = Orbit(0.0167086, 149598023, 0.00005, -11.26064, 114.20783, 358.617)

    fold.set_orbit(fold_orbit)
    print(fold)
    print(fold_orbit)


    julian_date(2000,1,1,12,00)
    j2000_date(2000,1,1,12,00)


# Include guard
if __name__ == '__main__':
    main()
