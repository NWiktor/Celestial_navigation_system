# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Planet and LaunchSite data using the Planet and LaunchSite classes.

Help
----
* https://en.wikipedia.org/wiki/Earth%27s_rotation
* https://en.wikipedia.org/wiki/Density_of_air
* https://en.wikipedia.org/wiki/Earth
* https://ofrohn.github.io/seh-doc/list-lc.html

Contents
--------
"""

# Standard library imports
import logging

# Local application imports
from cls.atmosphere import EarthAtmosphereUS1976
from cls.star import Star
from cls.kepler_orbit import CircularOrbit
from cls.planet import Planet, PlanetType, LaunchSite
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


class Sun(Star):
    pass


class Earth(Planet):
    def __init__(self):
        super().__init__("0001", "Earth", 5.972e24,
                         6_371_000, None,
                         EarthCoreComposition, PlanetType.TERRESTIAL)
        self.set_atmosphere(EarthAtmosphereUS1976())
        # self.set_orbit()
        # self.set_rotation_params()
        # TODO: replace direct setting of value with setter function ?
        self.angular_velocity_rad_per_s = 7.292115e-5


class Moon(Planet):
    def __init__(self):
        super().__init__("0002", "Moon", 7.34767309e22,
                         1_737_000, None, None,
                         PlanetType.TERRESTIAL)
        self.set_orbit(Earth(),
                       CircularOrbit(384_748, 28.58,
                                     45, 90,
                                     0))
        # self.set_rotation_params()
        # TODO: replace direct access with setter function above
        self.angular_velocity_rad_per_s = 7.292115e-5


CAPE_CANEVERAL = LaunchSite(
    Earth(), "Cape Canaveral, SLC 40",
    28.562106, -80.57718, (35, 120))
KSC_LC39A = LaunchSite(
    Earth(), "Kennedy Space Center, LC 39 A",
    28.608389, -80.604333, (35, 120))
KSC_LC39B = LaunchSite(
    Earth(), "Kennedy Space Center, LC 39 B",
    28.627222, -80.620833, (35, 120))
VANDENBERG_AFB = LaunchSite(
    Earth(), "Vanderberg AFB",
    34.75133, -120.52023, (147, 235))
SPACEX_STARBASE = LaunchSite(
    Earth(), "SpaceX Starbase, Boca Chica",
    25.997, -97.157, (93, 113))
BAIKONUR_COSMODROME = LaunchSite(
    Earth(), "Baikonur Cosmodrome",
    45.965, 63.305, (347, 65))
KOUROU_GSC = LaunchSite(
    Earth(), "Kourou, Guyana Space Center",
    5.169, -52.6903, (349, 90))

CAPE_TEST = LaunchSite(
    Earth(), "Cape Canaveral",
    28.5, 0, (35, 120))

# Include guard
if __name__ == '__main__':
    pass
