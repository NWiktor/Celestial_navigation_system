# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" CelestialBody baseclass for describing different celestial bodies.
Specified classes (planet, star) can be inherited from this class.

Libs
----
* Numpy

Help
----
* https://en.wikipedia.org/wiki/Stellar_classification
* https://en.wikipedia.org/wiki/Earth%27s_rotation
* https://en.wikipedia.org/wiki/Density_of_air
* https://en.wikipedia.org/wiki/Earth

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
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg^-1 s^-2


# Class and function definitions
class CelestialBody:
    """ Class for generic celestial bodies (stars, planets, moons, asteroids,
    etc.).
    """
    def __init__(self, uuid: str, name: str, mass_kg: float,
                 other_names: list[str] = None,
                 composition: Composition = None):
        self.uuid = uuid  # Unique identifier
        self.name = name  # Name of CB
        self.other_names = other_names

        # Phisycal properties
        self.mass_kg = mass_kg
        self.composition = composition

        # Properties to be set by function
        # NOTE: Orbit is defined in the parent object's inertial reference frame
        #  and describes the current object inertial reference frame
        self.parent_object = None
        self.orbit: KeplerOrbit | None = None

        # Set rotational params
        # TODO: transform between IRF and it's parent IRF
        self.transformation_matrix = None
        # self.rotation_vector: np.array = None - derived from transform matrix
        # self.axial_tilt: float = 0.0 - derived from transform matrix

        # NOTE: value in planet IRF
        self.angular_velocity_rad_per_s: float = 0.0
        self._tidal_lock: bool = False  # TODO: add setter

    def set_orbit(self, parent_object, orbit: KeplerOrbit):
        """ Set a Kepler orbit to the object, defined in the parent object
        inertial reference frame. """
        self.parent_object = parent_object
        self.orbit = orbit

    # TODO: think about reversing implementation or keep both methods?
    def set_rotation_params(self, rotation_vector: np.array):
        """ Set a rotation vector, which describes the celestial body rotation
        in the parent object inertial reference frame.

        Also calculates axial tilt and angular velocity.
        """
        self.rotation_vector = rotation_vector
        self.axial_tilt: float = angle_of_vectors(
                np.array([0.0, 0.0, 1.0]),
                unit_vector(rotation_vector))  # deg
        self.angular_velocity_rad_per_s: float = np.linalg.norm(rotation_vector)

    # NOTE: This is valid for a given object-pair!
    def get_std_gravitational_param(self, mass2_kg: float = 0.0):
        """ Returns the standard gravitational parameter using the CB's own
        mass and the given mass2 value. If mass2 is omitted, zero value is used,
        as it is considered negligible.

        .. math::
          \\mu = G \cdot (M + m) \\approx GM \\qquad (m^3 / s^2)

        :param mass2_kg: Mass of second object (kg)
        :return: standard gravitational parameter (m^3/s^2)
        """
        return gravitational_constant * (self.mass_kg + mass2_kg)  # m^3/s^2

    # TODO: check time format
    # TODO: is this needed ?
    # TODO: test this later, bc. this is needed for complex orbiting
    def get_position(self, j2000_time: float) -> np.array:
        """ Returns a 3D position vector of the object at a given time,
        since epoch.
        """

        # If parent object or orbit is not defined, return default
        if self.parent_object is None or self.orbit is None:
            return np.array([0, 0, 0])

        # Else we add object position + parent object position
        return (self.orbit.get_position(j2000_time) +
                self.parent_object.get_position(j2000_time))

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Asteroid(CelestialBody):
    """ Class describing asteroids. """
    def __init__(self, *args, asteroid_type: AsteriodType):
        super().__init__(*args)
        self.asteroid_type = asteroid_type


class Comet(CelestialBody):
    """ Class describing comets. """
    def __init__(self, *args):
        super().__init__(*args)


# Include guard
if __name__ == '__main__':
    pass
