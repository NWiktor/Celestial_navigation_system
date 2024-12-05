# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Planet class inherited from CelestialBody class for representing planets
and planetezoids (e.g. moons).

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
from cls.atmosphere import Atmosphere
from cls.celestial_body import CelestialBody

# Class initializations and global variables
logger = logging.getLogger(__name__)


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

        # Properties to be set by function
        self.atmosphere = None
        self.outer_radius_m = None

        # Setter function call
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


class PlanetLocation:
    """ General class for representing a location (point) on a planet
    (spherical body).
    """
    def __init__(self, planet: Planet, location_name: str,
                 latitude: float, longitude: float, radius: float):
        self.planet = planet
        self.name = f"{location_name} ({self.planet.name})"
        self.latitude = latitude
        self.longitude = longitude
        self.radius = radius


class LaunchSite(PlanetLocation):
    """ Class for representing a launch-site on a planet. """
    def __init__(self, planet: Planet, location_name: str,
                 latitude: float, longitude: float,
                 launch_azimuth_range: tuple[float, float] | None = None):
        # self.surface_radius = self.planet.surface_radius_m
        super().__init__(planet, location_name, latitude, longitude,
                         planet.surface_radius_m)
        self.launch_azimuth_range = launch_azimuth_range
        self.angular_velocity = self.planet.angular_velocity_rad_per_s
        # TODO: ez kell ide?
        self.std_gravity = self.planet.std_gravity  # m/s^2

        # TODO: work on implementation ??
        # For zero-mass spacecraft
        self.std_gravitational_parameter = (
            self.planet.std_gravitational_parameter)  # m^3/s^2

    def get_density(self, altitude) -> float:
        """ Get atmospheric density at altitude. """
        return self.planet.atmosphere.get_density(altitude)

    def get_pressure(self, altitude) -> float:
        """ Get atmospheric pressure at altitude. """
        return self.planet.atmosphere.get_pressure(altitude)

    def get_relative_velocity(self, state: np.array) -> float:
        """ Returns the speed of the rocket relative to the atmosphere.

        The atmosphere of the planet is modelled as static (no winds). The
        function calculates the atmospheric velocity (in inertial ref. frame),
        and substracts it from the rocket's speed in inertial frame, then takes
        the norm of the resulting vector.
        """
        pos = state[0:3]  # Rocket position
        vel = state[3:6]  # Rocket velocity
        # NOTE: Cross product of the angular velocity and the position, which is
        #  the speed of the atmosphere at this given location
        atm_velocity = np.cross(np.array([0, 0, self.angular_velocity]), pos)
        return float(np.linalg.norm(vel - atm_velocity))


# Include guard
if __name__ == '__main__':
    pass
