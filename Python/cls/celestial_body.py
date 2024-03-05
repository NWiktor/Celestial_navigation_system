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
import math as m
import numpy as np

# Local application imports
from utils.kepler_orbit import KeplerOrbit

# Class initializations and global variables
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2


# Class and function definitions
# TODO: rethink what this does
# class Rotation:
#     """ Creates connection between the inertial and the non-inertial reference frame between the same object. """
#
#     def __init__(self, obliquity_vector, rotation_vector):
#         self.obliquity_vector = obliquity_vector
#         self.rotation_vector = rotation_vector


# TODO: break it to Planet(), Asteriod() child class ??
# TODO: merge with PlanetLocation class, or make it CelestialBody's children
class Atmosphere:
    """ Baseclass for a generic atmospheric model. """

    def __init__(self, model_name, atmosphere_limit_m: int):
        self.model_name = model_name
        self.atmosphere_limit_m = atmosphere_limit_m  # Upper limit, lower is always zero

    # TODO: remove this function
    def apply_limits(self, altitude) -> int:
        """ Checks if given value is within range of the valid atmospheric model. Returns the applicable value. """
        return min(max(0, altitude), self.atmosphere_limit_m)

    # pylint: disable = unused-argument
    # @override
    def atmospheric_model(self, altitude) -> tuple[float, float, float]:
        """ Returns the pressure, temperature, density values. """
        return 0.0, 0.0, 0.0

    def get_pressure(self, altitude) -> float:
        """ Returns the pressure at a given altitude. """
        return self.atmospheric_model(altitude)[0]

    def get_temperature(self, altitude) -> float:
        """ Returns the temperature at a given altitude. """
        return self.atmospheric_model(altitude)[1]

    def get_density(self, altitude) -> float:
        """ Returns the density at a given altitude. """
        return self.atmospheric_model(altitude)[2]


class EarthAtmosphere(Atmosphere):
    """  Standard model of Earth Atmosphere according to NASA:
    https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    """

    def __init__(self):
        super().__init__("standard atmosphere", 120000)

    def atmospheric_model(self, altitude: float) -> tuple[float, float, float]:
        """ Returns the pressure, temperature, density values of the atmosphere depending on the altitude measured
        from sea level. Calculation is based on the standard atmosphere model, which has three separate zones.
        """
        alt = self.apply_limits(altitude)
        temperature = 0.0
        pressure = 0.0

        if 0 <= altitude < 11000:  # Troposhere
            temperature = 15.04 - 0.00649 * alt
            pressure = 101.29 * pow((temperature + 273.1) / 288.08, 5.256)

        elif 11000 <= altitude < 25000:  # Lower stratosphere
            temperature = -56.46
            pressure = 22.65 * m.exp(1.73 - 0.000157 * alt)

        elif 25000 <= altitude:  # Upper stratosphere
            temperature = -131.21 + 0.00299 * alt
            pressure = 2.488 * pow((temperature + 273.1) / 216.6, -11.388)

        # Calculate air density and return values
        air_density = pressure / (0.2869 * (temperature + 273.1))
        return pressure, temperature, air_density


class CelestialBody:
    """ Class for celestial bodies (planet, moon, asteroid, etc.). """

    def __init__(self, name, uuid, mass, radius, angular_velocity, std_gravity, std_gravitational_parameter,
                 parent_object=None):
        self.name = name
        self.uuid = uuid  # Unique identifier

        # Phisycal properties
        self.mass = mass  # kg
        self.surface_radius_m = radius  # m
        self.angular_velocity_rad_per_s = angular_velocity  # rad/s
        self.std_gravity = std_gravity  # m/s^2
        self.std_gravitational_parameter = std_gravitational_parameter  # m^3/s^2
        self.atmosphere = None

        # Orbital properties
        self.parent_object = parent_object
        self.orbit = None
        self.rotation = None

    def get_relative_velocity(self, state: np.array) -> float:
        """ Returns the speed of the rocket relative to the atmosphere.

        The atmosphere of the planet is modelled as static (no winds). The function calculates the atmospheric
        velocity (in inertial ref. frame), and substracts it from the rocket's speed in inertial frame, then takes
        the norm of the resulting vector.
        """
        return np.linalg.norm(state[3:6] - np.cross(np.array([0, 0, self.angular_velocity_rad_per_s]), state[0:3]))

    def set_atmosphere(self, atmosphere: Atmosphere):
        """ Set an Atmosphere object to the Celestial body. """
        self.atmosphere = atmosphere

    def set_orbit(self, parent_object, orbit: KeplerOrbit):
        """ Set a Kepler orbit to the object, defined in the parent object inertial reference frame. """
        self.parent_object = parent_object
        self.orbit = orbit

    # def set_rotation(self, rotation: Rotation):
    #     """ Define object rotation, in the parent object inertial reference frame. """
    #     self.rotation = rotation

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


# Include guard
if __name__ == '__main__':
    pass
