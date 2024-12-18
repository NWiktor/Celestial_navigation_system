# -*- coding: utf-8 -*-
# !/usr/bin/python3
""" Star class inherited from CelestialBody class for representing stars.

Help
----
* https://en.wikipedia.org/wiki/Stellar_classification

Contents
--------
"""

# Standard library imports
import logging
from enum import StrEnum

# Local application imports
from cls.celestial_body import CelestialBody

# Class initializations and global variables
logger = logging.getLogger(__name__)
GRAVITATIONAL_CONSTANT: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2


# Class and function definitions
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
        self.stellar_class =\
            f"{self.temp_class.value}{self.rel_temp}{self.lum_class.value}"

    @property
    def rel_temp(self):
        """ Returns relative temperature. """
        return self._rel_temp

    @rel_temp.setter
    def rel_temp(self, value):
        if 0 <= value < 10:
            self._rel_temp = f"{value:.4g}"
        else:
            raise ValueError("Value must be within 0-10!")

    def __str__(self) -> str:
        return self.stellar_class


class Star(CelestialBody):
    """ Star class. """
    def __init__(self, *args, std_gravity: float, surface_radius_m: float,
                 spectral_class: list[SpectralClass | tuple[SpectralClass]]):
        super().__init__(*args)
        self.surface_radius_m = surface_radius_m  # m
        self.std_gravity = std_gravity  # m/s^2
        # NOTE: spectral class is a list, because it could be uncertain
        #  (multiple-values) and values could also be inbetween (tuple)
        self.spectral_class = spectral_class
        self.surface_temperature_K = None  # Kelvin
        self.bv_color_index = None
        self.ub_color_index = None


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
