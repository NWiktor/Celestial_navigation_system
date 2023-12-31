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

# Class initializations and global variables
gravitational_constant = 6.67430 * pow(10,-11) # m^3 kg-1 s-2

# Class and function definitions

class CelestialObject():

    def __init__(self, name, uuid, mass, radius, parent_object=None):
        self._name = name
        self._uuid = uuid # Unique identifier
        self._mass = mass # kg
        self._radius = radius # km ???
        self._parent_object = parent_object


    def keplerian_elements(self, eccentricity, semimajor_axis, inclination,
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


    def get_orbital_coords_at_time(self):
        pass


    def orbital_parameters(self, orbital_time, mean_motion, orbital_period):
        """  """
        self.orbital_time = orbital_time # s
        self.mean_motion = mean_motion # (n) 1/s
        self.orbital_period = orbital_period # s


    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}


    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)



def julian_date(year, month, day, hour, minute, second=0):
    # https://stackoverflow.com/questions/13943062/extract-day-of-year-and-julian-day-from-a-string-date
    # https://en.wikipedia.org/wiki/Julian_day
    """ Calculates full Julian date of a given date according to:
    JD = JDN + hour/24 + minute/1440
    """
    jdn = datetime.date(year,month,day).toordinal() + 1721424.5
    julian_date = jdn + hour/24 + minute/1440

    # print(f"Date is {year}-{month}-{day} {hour}:{minute}:{second}")
    # print(f"Julian date number of given Gregorian date is: {julian_date}")

    return julian_date


def j2000_date(year, month, day, hour, minute, second=0):
    # https://en.wikipedia.org/wiki/Epoch_(astronomy)
    """ Calculates J200 date of a given date from Julian date. """
    julian_d = julian_date(year, month, day, hour, minute, second)
    j2000 = 2000 + (julian_d - 2451545.0) / 365.25

    # print(f"Date is {year}-{month}-{day} {hour}:{minute}:{second}")
    # print(f"J2000 date of the given Gregorian date is: {j2000}")

    return j2000


# Main function for module testing
def main():
    """  """

    nap = CelestialObject("Nap", "0001", 2500, 3400)
    print(nap)

    fold = CelestialObject("Fold", "0002", 2500, 3400)
    fold.keplerian_elements(0.0167086, 149598023, 0.00005, -11.26064, 114.20783, 358.617)
    print(fold)


    julian_date(2000,1,1,12,00)
    j2000_date(2000,1,1,12,00)


# Include guard
if __name__ == '__main__':
    main()
