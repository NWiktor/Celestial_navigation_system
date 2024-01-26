# -*- coding: utf-8 -*-
# !/usr/bin/python3

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

# Third party imports
import numpy as np

# Local application imports
from logger import MAIN_LOGGER as L
from kepler_orbit import KeplerOrbit

# Class initializations and global variables
gravitational_constant = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2


# Class and function definitions

# TODO: break it to CelestialObject() and Planet() child class ??
class CelestialBody:

    def __init__(self, name, uuid, mass, radius, parent_object=None):
        self._name = name
        self._uuid = uuid  # Unique identifier
        self._mass = mass  # kg
        self._radius = radius  # km ???
        self._parent_object = parent_object
        self._orbit = None
        self._rotation = None

    def set_orbit(self, parent_object, orbit: KeplerOrbit):
        # Relate a Kepler orbit, defined in the parent star inertial reference frame
        self._parent_object = parent_object
        self._orbit = orbit

    def set_rotation(self, rotation: Rotation):
        # Define object rotation, in the parent star inertial reference frame
        self._rotation = rotation

    def get_position(self, j2000_time):
        """  """

        # If parent object is not defined
        if self._parent_object is None or self._orbit is None:
            return np.array([0, 0, 0])

        # Else we add object position + parent object position
        return self._orbit.get_position(j2000_time) + self._parent_object.get_position(j2000_time)

    def clear(self):
        """ Erases all loaded and existing training data to allow refresh. """
        self.__dict__ = {}

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# TODO: rethink what this does
class Rotation:
    """ Creates connection between the inertial and the non-inertial reference frame between the same object. """

    def __init__(self, obliquity_vector, rotation_vector):
        self.obliquity_vector = obliquity_vector
        self.rotation_vector = rotation_vector


# Include guard
if __name__ == '__main__':
    pass
