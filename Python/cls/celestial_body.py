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
import logging
from enum import StrEnum
import numpy as np

# Local application imports
from cls.kepler_orbit import KeplerOrbit
from cls.atmosphere import Atmosphere
from utils import unit_vector, angle_of_vectors

# Class initializations and global variables
logger = logging.getLogger(__name__)
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2


# Class and function definitions
class AsteriodType(StrEnum):
    """ Describes the asteroid by composition. """
    CARBON = "Carbon"
    METAL = "Metal"
    SILICONE = "Silicone"


class TemperatureClass(StrEnum):
    """ Stellar classification by its surface temperature.

    https://en.wikipedia.org/wiki/Stellar_classification
    """
    O_class = "O"
    B = "B"
    A = "A"
    F = "F"
    G = "G"
    K = "K"
    M = "M"
    W = "W"  # Wolf–Rayet star
    S = "S"  # S-type star
    C = "C"  # Carbon star
    D = "D"  # White dwarf
    L = "L"  # L-dwarf
    T = "T"  # T-dwarf
    Y = "Y"  # Y-dwarf


class LuminosityClass(StrEnum):
    """ Stellar classification by its luminosity.

    https://en.wikipedia.org/wiki/Stellar_classification
    """
    Iap = "Ia+"  # Hypergiant
    Ia = "Ia"  # Supergiant
    Iab = "Iab"  # Intermediate supergiant
    Ib = "Ib"  # Less-luminous Supergiant
    II = "II"  # Bright giant
    III = "III"  # Giant
    IV = "IV"  # Sub-giant
    V = "V"  # Main-sequence star (dwarf)
    VI = "VI"  # Subdwarf
    VII = "VII"  # Subdwarf


class SpectralClass:
    """ Spectral class according to the Morgan–Keenan-Kellman (MKK) stellar
    classification system.
    """

    def __init__(self, temp_class: TemperatureClass, rel_temp: float,
                 lum_class: LuminosityClass):
        self.temp_class = temp_class
        self.rel_temp = rel_temp
        self.lum_class = lum_class
        self.stellar_class = f"{self.temp_class.value}{self.rel_temp}{self.lum_class.value}"

    @property
    def rel_temp(self):
        return self._rel_temp

    @rel_temp.setter
    def rel_temp(self, value):
        if 0 <= value < 10:
            self._rel_temp = f"{value:.4g}"
        else:
            raise ValueError("Value must be within 0-10!")

    def __str__(self) -> str:
        return self.stellar_class


class CelestialBodyRotationVector:
    """ Defines a rotation vector (pseudovector) of a celestial body in
    inertial frame.

    Precession is omitted.
    """

    def __init__(self, rotation_vector: np.array):
        self.rotation_vector: np.array = rotation_vector
        self.axial_tilt: float = angle_of_vectors(
                np.array([0.0, 0.0, 1.0]),
                rotation_vector)  # rad
        self.angular_velocity_rad_per_s: float = unit_vector(rotation_vector)


# TODO: Create children objects for the inner planets
# TODO: merge with PlanetLocation class, or make it CelestialBody's children
class CelestialBody:
    """ Class for celestial bodies (planet / moon, asteroid, sun). """

    def __init__(self, uuid: str, name: str, mass_kg: float,
                 outer_radius_m: float):
        self.uuid = uuid  # Unique identifier
        self.name = name  # Name of CB

        # Phisycal properties
        self.mass_kg = mass_kg
        self.outer_radius_m = outer_radius_m  # For 'visualization' only
        self.std_gravitational_parameter = 0.0  # m^3/s^2

        # Properties to be set by function
        self.parent_object = None
        self.orbit = None
        self.rotation = None

        # Calculate params
        self.set_std_gravitational_param()

    def set_orbit(self, parent_object, orbit: KeplerOrbit):
        """ Set a Kepler orbit to the object, defined in the parent object
        inertial reference frame. """
        self.parent_object = parent_object
        self.orbit = orbit

    def set_rotation(self, rotation: CelestialBodyRotationVector):
        """ Set a rotation vector, which describes the celestial body rotation
        in the parent object inertial reference frame.
        """
        self.rotation = rotation

    def set_std_gravitational_param(self, mass2_kg: float = 0.0):
        """ Sets standard gravitational parameter using the CB's own mass, and
        the given M2 value.
        """
        self.std_gravitational_parameter = gravitational_constant * (self.mass_kg + mass2_kg)  # m^3/s^2

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


class Planet(CelestialBody):
    """  """
    def __init__(self, *args, std_gravity, surface_radius_m):
        super().__init__(*args)
        self.surface_radius_m = surface_radius_m  # m
        self.std_gravity = std_gravity  # m/s^2
        self.atmosphere = None

    def set_atmosphere(self, atmosphere: Atmosphere):
        """ Set an Atmosphere object to the Celestial body. """
        self.atmosphere = atmosphere


class Asteroid(CelestialBody):
    def __init__(self, *args, asteroid_type: AsteriodType):
        super().__init__(*args)
        self.asteroid_type = asteroid_type


class Star(CelestialBody):
    def __init__(self, *args, std_gravity, surface_radius_m,
                 stellar_class: SpectralClass):
        super().__init__(*args)
        self.surface_radius_m = surface_radius_m  # m
        self.std_gravity = std_gravity  # m/s^2
        self.stellar_class = stellar_class


class CelestialBodyLocation:
    # ezmiez???
    pass


# Include guard
if __name__ == '__main__':
    sun = SpectralClass(TemperatureClass.G, 2, LuminosityClass.V)
    print(sun)
    pollux = SpectralClass(TemperatureClass.K, 0, LuminosityClass.III)
    print(pollux)
    print(pollux.temp_class)
    print(pollux.rel_temp)
    randomstar = SpectralClass(TemperatureClass.W, 9.123456789,
                               LuminosityClass.Iab)
    print(randomstar)
