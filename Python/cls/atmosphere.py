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
import math as m
import numpy as np

# Local application imports
from cls.kepler_orbit import KeplerOrbit

# Class initializations and global variables
logger = logging.getLogger(__name__)
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2

class Atmosphere:
    """ Baseclass for a generic atmospheric model. """

    def __init__(self, model_name, atmosphere_limit_m: int):
        self.model_name = model_name
        self.atmosphere_limit_m = atmosphere_limit_m  # Upper limit, lower is always zero

    def apply_limits(self, altitude) -> int:
        """ If the given value is out of range of the valid atmospheric model, returns the applicable value.
        This function is needed to in case an iterative calculation accidentally runs through the limits.
        """
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
        super().__init__("US. standard atmosphere, 1976", 120000)

    def atmospheric_model(self, altitude: float) -> tuple[float, float, float]:
        """ Returns the pressure, temperature, density values of the atmosphere depending on the altitude measured
        from sea level. Calculation is based on the US. standard atmosphere model (1976), which has three separate zones.
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

def test_plot_atmosphere():
    """ aaa """

    import matplotlib.pyplot as plt

    alt = []
    tmp = []
    pres = []
    rho = []
    for i in range(0, 100_000):
        data = EarthAtmosphere().atmospheric_model(i)
        alt.append(i)
        pres.append(data[0])
        tmp.append(data[1])
        rho.append(data[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle("Atmospheric test")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Temperature")
    ax1.set_xlabel('altitude (m)')
    ax1.set_ylabel('temperature (°C)', color="m")
    ax1.plot(alt, tmp, color="m")

    # Flight velocity, acceleration
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Pressure")
    ax2.set_xlabel('altitude (m)')
    ax2.set_ylabel('pressure (kPa)', color="b")
    # ax2.set_xlim(0, len(time_data))
    # ax2.set_ylim(0, 10)
    ax2.scatter(alt, pres, s=0.5, color="b")
    # ax2.tick_params(axis='y', labelcolor="b")

    # Flight velocity, acceleration
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Air density")
    ax3.set_xlabel('altitude (m)')
    ax3.set_ylabel('air density (kg/m3)', color="g")
    # ax3.set_xlim(0, len(time_data))
    # ax3.set_ylim(0, 10)
    ax3.scatter(alt, rho, s=0.5, color="g")
    ax3.tick_params(axis='y', labelcolor="g")

    plt.show()


# Include guard
if __name__ == '__main__':
    test_plot_atmosphere()