# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Planet class inherited from CelestialBody class for representing planets.

Help
----
* https://en.wikipedia.org/wiki/Earth%27s_rotation
* https://en.wikipedia.org/wiki/Density_of_air
* https://en.wikipedia.org/wiki/Earth

Contents
--------
"""

# Standard library imports
import logging
from enum import StrEnum
import numpy as np

# Local application imports
from cls.atmosphere import Atmosphere, EarthAtmosphereUS1976
from cls.celestial_body import CelestialBody
from cls.celestial_body_utils import Component, Composition

# Class initializations and global variables
logger = logging.getLogger(__name__)

EarthCoreComposition = Composition([
    Component("Iron", 32.1, "Fe"),
    Component("Oxygen", 30.1, "O"),
    Component("Silicon", 15.1, "Si"),
    Component("Magnesium", 13.9, "Mg"),
    Component("Sulfur", 2.9, "S"),
    Component("Nickel", 1.8, "Ni"),
    Component("Calcium", 1.5, "Ca"),
    Component("Aluminium", 1.4, "Al")],
    source="https://en.wikipedia.org/wiki/Earth"
)


class PlanetType(StrEnum):
    """ Describes the asteroid types by composition. """
    ICE_GIANT = "Ice giant"
    GAS_GIANT = "Gas giant"
    ROCKY = "Rocky"
    TERRESTIAL = "Terrestial"
    EXO = "Exo"


# Class and function definitions
class Planet(CelestialBody):

    def __init__(self, uuid, name, mass_kg, other_names, composition,
                 planettype: PlanetType | None, std_gravity: float,
                 surface_radius_m: float):
        super().__init__(uuid, name, mass_kg, other_names, composition)
        self.planettype = planettype
        self.surface_radius_m = surface_radius_m  # m
        self.std_gravity = std_gravity  # m/s^2
        self.atmosphere = None

        # Setters
        self.outer_radius_m = None
        self.set_outer_radius_m()

    def set_atmosphere(self, atmosphere: Atmosphere):
        """ Set an Atmosphere object to the Celestial body. """
        self.atmosphere = atmosphere
        self.set_outer_radius_m()

    def set_outer_radius_m(self):
        """ Sets outer radius as the sum of surface radius and athmospheric
        thickness (if defined).
        """
        if self.atmosphere is not None:
            self.outer_radius_m = (
                self.surface_radius_m + self.atmosphere.atm_upper_limit_m
            )
        else:
            self.outer_radius_m = self.surface_radius_m


# TODO: refactor Planetlocation to Launchlocation,
#  only to collect launch relevant data
class PlanetLocation:
    """ General class for representing a location on a planet. """

    def __init__(self, planet: Planet, location_name: str,
                 latitude: float, longitude: float):
        self.planet = planet
        self.name = f"{location_name} ({self.planet.name})"
        self.latitude = latitude
        self.longitude = longitude


class LaunchSite(PlanetLocation):
    """ Class for representing a launch-site on a planet. """

    def __init__(self, planet, location_name, latitude, longitude):
        super().__init__(planet, location_name, latitude, longitude)

        self.surface_radius = self.planet.surface_radius_m
        self.angular_velocity = self.planet.angular_velocity_rad_per_s
        self.std_gravity = self.planet.std_gravity  # m/s^2

        # TODO: work on implementation
        # For zero-mass spacecraft
        self.std_gravitational_parameter = (
            self.planet.std_gravitational_parameter)  # m^3/s^2

    def get_density(self, altitude) -> float:
        return self.planet.atmosphere.get_density(altitude)

    def get_pressure(self, altitude) -> float:
        return self.planet.atmosphere.get_pressure(altitude)

    def get_relative_velocity(self, state: np.array) -> float:
        """ Returns the speed of the rocket relative to the atmosphere.

        The atmosphere of the planet is modelled as static (no winds). The
        function calculates the atmospheric velocity (in inertial ref. frame),
        and substracts it from the rocket's speed in inertial frame, then takes
        the norm of the resulting vector.
        """
        return float(np.linalg.norm(
                state[3:6] - np.cross(
                        np.array([0, 0, self.angular_velocity]),
                        state[0:3]
                )))

    def get_surface_velocity(self):
        """ Placeholder func. """


class Earth(Planet):
    def __init__(self):
        super().__init__("0001", "Earth", 5.972e24,
                         None, EarthCoreComposition,
                         PlanetType.TERRESTIAL,
                         9.80665, 6_371_000)
        self.set_atmosphere(EarthAtmosphereUS1976())
        self.set_outer_radius_m()
        self.set_std_gravitational_param()
        # self.set_orbit()
        # self.set_rotation_params()
        # TODO: replace direct access with setter function above
        self.angular_velocity_rad_per_s = 7.292115e-5


# Include guard
if __name__ == '__main__':
    pass
