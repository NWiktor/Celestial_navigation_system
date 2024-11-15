# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" CelestialBody baseclass for describing different celestial bodies. Specified
classes can be inherited from this.

Libs
----
* Numpy

Help
----
* https://en.wikipedia.org/wiki/Truncated_icosahedron

Contents
--------
"""

# Standard library imports
import logging
import numpy as np

# Local application imports
from cls.kepler_orbit import KeplerOrbit
from cls.celestial_body_utils import Composition, AsteriodType
from utils import unit_vector, angle_of_vectors

# Class initializations and global variables
logger = logging.getLogger(__name__)
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2


# Class and function definitions
# TODO: Create children objects for the inner planets ??
# TODO: merge with PlanetLocation class, or make it CelestialBody's children
class CelestialBody:
    """ Class for celestial bodies (planet, moon, asteroid, sun, etc.). """

    def __init__(self, uuid: str, name: str, mass_kg: float,
                 other_names: list[str] = None,
                 composition: Composition = None):
        self.uuid = uuid  # Unique identifier
        self.name = name  # Name of CB
        self.other_names = other_names
        self.composition = composition

        # Phisycal properties
        self.mass_kg = mass_kg
        self.std_gravitational_parameter = 0.0  # m^3/s^2

        # Properties to be set by function
        # NOTE: Orbit and rotational vector is defined in the same
        #  inertial reference frame
        self.parent_object = None
        self.orbit = None
        self.rotation_vector = None
        self.axial_tilt = None
        self.angular_velocity_rad_per_s = None

        # Calculate params
        self.set_std_gravitational_param()

    def set_orbit(self, parent_object, orbit: KeplerOrbit):
        """ Set a Kepler orbit to the object, defined in the parent object
        inertial reference frame. """
        self.parent_object = parent_object
        self.orbit = orbit

    def set_rotation_params(self, rotation_vector: np.array):
        """ Set a rotation vector, which describes the celestial body rotation
        in the parent object inertial reference frame.

        Also calculates axial tilt and angular velocity.
        """
        self.rotation_vector = rotation_vector
        self.axial_tilt: float = angle_of_vectors(
                np.array([0.0, 0.0, 1.0]),
                rotation_vector)  # rad
        self.angular_velocity_rad_per_s: float = unit_vector(rotation_vector)

    # TODO: This is valid for a given object-pair, this value is only universal
    #  for objects with zero mass.
    def set_std_gravitational_param(self, mass2_kg: float = 0.0):
        """ Sets standard gravitational parameter using the CB's own mass, and
        the given M2 value. By default
        """
        self.std_gravitational_parameter = (
                gravitational_constant * (self.mass_kg + mass2_kg))  # m^3/s^2

    # TODO: rework this !!!
    # def get_relative_velocity(self, state: np.array) -> float:
    #     """ Returns the speed of the rocket relative to the atmosphere.
    #
    #     The atmosphere of the planet is modelled as static (no winds). The function calculates the atmospheric
    #     velocity (in inertial ref. frame), and substracts it from the rocket's speed in inertial frame, then takes
    #     the norm of the resulting vector.
    #     """
    #     return np.linalg.norm(state[3:6] - np.cross(np.array([0, 0, self.angular_velocity_rad_per_s]), state[0:3]))

    # def get_position(self, j2000_time: float):
    #     """ Returns a 3D position vector of the object at a given time, since epoch. """
    #
    #     # If parent object is not defined
    #     if self.parent_object is None or self.orbit is None:
    #         return np.array([0, 0, 0])
    #
    #     # Else we add object position + parent object position
    #     return self.orbit.get_position(j2000_time) + self.parent_object.get_position(j2000_time)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# TODO: rework when the time comes
class CelestialBodyVisual:
    """ Abstract class for visual / graphical representation of the
    CelestialBody.
    """

    def __init__(self, celestial_body: CelestialBody, radius: int,
                 color: tuple[int, int, int]):
        self.celestial_body = celestial_body
        self.radius = radius  # For 'visualization' only
        self.color = color


class Asteroid(CelestialBody):
    def __init__(self, *args, asteroid_type: AsteriodType):
        super().__init__(*args)
        self.asteroid_type = asteroid_type


class Comet(CelestialBody):
    def __init__(self, *args):
        super().__init__(*args)


# Include guard
if __name__ == '__main__':
    pass
