# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Planet class inherited from CelestialBody class for representing planets
and planetary-mass moons (i.e. Moon).

Help
----
* https://en.wikipedia.org/wiki/Planetary-mass_moon
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
from cls.celestial_body_utils import Composition
# Class initializations and global variables
logger = logging.getLogger(__name__)


# TODO: expand this, and split to more categories
class PlanetType(StrEnum):
    """ Describes the planet types by main categories. """
    ICE_GIANT = "Ice giant"
    GAS_GIANT = "Gas giant"
    ROCKY = "Rocky"
    TERRESTIAL = "Terrestial"
    EXO = "Exo"
    DWARF = "Dwarf"
    MINOR = "Minor"


# Class and function definitions
class Planet(CelestialBody):
    """ Describes a planet. """
    def __init__(self, uuid: str, name: str, mass_kg: float,
                 surface_radius_m: float, other_names: list[str] = None,
                 composition: Composition = None,
                 planettype: PlanetType | None = None,
                 ):
        super().__init__(uuid, name, mass_kg, other_names, composition)
        self.planettype = planettype
        self.surface_radius_m = surface_radius_m  # m

        # Properties to be set by function
        self.atmosphere = None
        self.outer_radius_m: float = surface_radius_m  # default: no atmosphere
        self.surface_gravity_m_per_s2: float = (
            self._set_surface_gravity_m_per_s2())  # Set automatically

    def set_atmosphere(self, atmosphere: Atmosphere):
        """ Set an Atmosphere object to the Celestial body. """
        self.atmosphere = atmosphere
        self._set_outer_radius_m()

    def _set_outer_radius_m(self):
        """ Sets outer radius (m) as the sum of the surface radius (m) and
        athmospheric thickness (m), if defined.
        """
        if self.atmosphere is not None:
            self.outer_radius_m = (
                self.surface_radius_m + self.atmosphere.atm_upper_limit_m
            )
        else:
            logger.debug("Atmosphere not defined.")
            self.outer_radius_m = self.surface_radius_m

        logger.info("Planet %s outer radius is set to %s.3f (m).",
                    self.name, self.outer_radius_m)

    def _set_surface_gravity_m_per_s2(self) -> float:
        """ Set surface gravity for planet using the surface radius and the
        standard gravitational parameter.

        .. math::
          g = \\mu / r^2 \\qquad (m/s^2)

        :return: Surface gravity (m/s^2)
        """
        return (self.get_std_gravitational_param()
                / pow(self.surface_radius_m, 2))


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
        super().__init__(planet, location_name, latitude, longitude,
                         planet.surface_radius_m)
        # self.surface_radius = self.planet.surface_radius_m
        self.launch_azimuth_range = launch_azimuth_range
        self.angular_velocity = self.planet.angular_velocity_rad_per_s

        # For zero-mass spacecraft
        self.std_gravitational_parameter = (
            self.planet.get_std_gravitational_param())  # m^3/s^2

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
