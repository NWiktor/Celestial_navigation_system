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
import matplotlib.pyplot as plt

# Local application imports

# Class initializations and global variables
logger = logging.getLogger(__name__)
gravitational_constant: float = 6.67430 * pow(10, -11)  # m^3 kg-1 s-2
standard_gravity = 9.80665  # m/s2
air_molar_mass = 0.028964425278793993  # kg/mol
universal_gas_constant = 8.3144598  # N·m/(mol·K)


class Atmosphere:
    """ Baseclass for a generic atmospheric model. """

    def __init__(self, model_name, atm_lower_limit_m: int, atm_upper_limit_m: int):
        self.model_name = model_name
        self.atm_lower_limit_m = atm_lower_limit_m  # Lower limit
        self.atm_upper_limit_m = atm_upper_limit_m  # Upper limit

    def _apply_limits(self, altitude) -> int:
        """ If the given value is out of range of the valid atmospheric model, returns the applicable value.
        This function is needed to in case an iterative calculation accidentally runs through the limits.
        When the altitude smaller than 0, there is solid ground, no atmosphere; this case should be handled
        anyway elsewhere. When altitude is above range, there is space (no pressure, no atmosphere).
        """
        return min(max(self.atm_lower_limit_m, altitude), self.atm_upper_limit_m)

    # pylint: disable = unused-argument
    # @override
    def _atmospheric_model(self, altitude) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values. """
        return 0.0, 0.0, 0.0

    def get_atm_params(self, altitude) -> tuple[float, float, float]:
        """ Enforces the atmospheric model limits, and returns the temperature, pressure, density values. """
        return self._atmospheric_model(self._apply_limits(altitude))

    def get_temperature(self, altitude) -> float:
        """ Returns the temperature at a given altitude. """
        return self._atmospheric_model(altitude)[0]

    def get_pressure(self, altitude) -> float:
        """ Returns the pressure at a given altitude. """
        return self._atmospheric_model(altitude)[1]

    def get_density(self, altitude) -> float:
        """ Returns the density at a given altitude. """
        return self._atmospheric_model(altitude)[2]


class EarthAtmosphere(Atmosphere):
    """  Standard model of Earth Atmosphere according to NASA:
    https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    """

    def __init__(self):
        super().__init__("Standard atmosphere, simplified, NASA", 0, 30000)

    def _atmospheric_model(self, altitude: float) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values of the atmosphere depending on the altitude measured
        from sea level. Calculation is loosely based on the US. standard atmosphere model (1976), and uses three
        separate zones (layers).
        """
        temperature = 0.0
        pressure = 0.0

        if 0 <= altitude < 11000:  # Troposhere
            temperature = 15.04 - 0.00649 * altitude
            pressure = 101.29 * pow((temperature + 273.1) / 288.08, 5.256)

        elif 11000 <= altitude < 25000:  # Lower stratosphere
            temperature = -56.46
            pressure = 22.65 * m.exp(1.73 - 0.000157 * altitude)

        elif 25000 <= altitude:  # Upper stratosphere
            temperature = -131.21 + 0.00299 * altitude
            pressure = 2.488 * pow((temperature + 273.1) / 216.6, -11.388)

        # Calculate air density and return values
        air_density = pressure / (0.2869 * (temperature + 273.1))
        logging.debug("Atmospheric temp.: %.6f (K), pres.: %.6f (kPa) and air density: %.6f (kg/m3) @  %s (m)",
                      temperature, pressure, air_density, altitude)
        return temperature, pressure, air_density


class EarthAtmosphereUS1976(Atmosphere):
    """  Standard model of Earth Atmosphere according (1976):
    https://www.digitaldutch.com/atmoscalc/US_Standard_Atmosphere_1976.pdf
    """

    def __init__(self):
        super().__init__("US. standard atmosphere, 1976", 0, 100000)

    def _atmospheric_model(self, altitude: float) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values of the atmosphere depending on the altitude measured
        from sea level. Calculation is based on the US. standard atmosphere model (1976),
        which has seven separate zones.

        https://en.wikipedia.org/wiki/Barometric_formula
        http://www.luizmonteiro.com/StdAtm.aspx
        """
        temperature: float = 288.15  # Kelvin
        pressure: float = 101.325  # kPa
        # air_density: float = 1.225  # kg/m3

        if 0 < altitude <= 11000:  # Troposphere
            temperature = 288.15 - 0.0065 * altitude
            pressure = 101.325 * pow(288.15 / temperature, -5.228)

        elif 11000 < altitude <= 20000:  # Tropopause
            temperature = 216.65
            pressure = 22.632 * m.exp(-0.03416265012915032 * (altitude - 11000) / 216.65)

        elif 20000 < altitude <= 32000:  # Lower stratosphere
            temperature = 216.65 + 0.001 * (altitude - 20000)
            pressure = 5.47489 * pow(216.65 / temperature, 34.1626)

        elif 32000 < altitude <= 47000:  # Upper stratosphere
            temperature = 228.65 + 0.0028 * (altitude - 32000)
            pressure = 0.86802 * pow(228.65 / temperature, 12.2009)

        elif 47000 < altitude <= 51000:  # Stratopause
            temperature = 270.65
            pressure = 0.11091 * m.exp(-0.03416265012915032 * (altitude - 47000) / 270.65)

        elif 51000 < altitude <= 71000:  # Lower mesosphere
            temperature = 270.65 - 0.0028 * (altitude - 51000)
            pressure = 0.06694 * pow(270.65 / temperature, -12.2009)

        elif 71000 < altitude <= 86000:  # Upper mesosphere
            temperature = 214.65 - 0.002 * (altitude - 71000)
            pressure = 0.00396 * pow(214.65 / temperature, -17.0813)

        elif 86000 < altitude <= 91000:  # Thermosphere
            temperature = 184.65
            pressure = 0.0003

        elif 91000 < altitude:
            temperature = 184.65
            pressure = 0.0003

        # Calculate air density and return values
        air_density = pressure / (0.2869 * temperature)
        logging.debug("Atmospheric temp.: %.6f (K), pres.: %.6f (kPa) and air density: %.6f (kg/m3) @  %s (m)",
                      temperature, pressure, air_density, altitude)
        return temperature, pressure, air_density


def plot_atmosphere(model):
    """ Plot pressure, temperature and density data of the given model, to check its validity. """

    alt = []
    tmp = []
    pres = []
    rho = []
    for i in range(0, 10000):
        data = model.get_atm_params(i * 10)
        alt.append(i*10)
        tmp.append(data[0])
        pres.append(data[1])
        rho.append(data[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle("Atmospheric parameters")
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Temperature")
    ax1.set_xlabel('altitude (m)')
    ax1.set_ylabel('temperature (K)', color="m")
    ax1.plot(alt, tmp, color="m")
    ax1.tick_params(axis='y', labelcolor="m")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("Pressure")
    ax2.set_xlabel('altitude (m)')
    ax2.set_ylabel('pressure (kPa)', color="b")
    ax2.plot(alt, pres, color="b")
    ax2.tick_params(axis='y', labelcolor="b")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Air density")
    ax3.set_xlabel('altitude (m)')
    ax3.set_ylabel('air density (kg/m3)', color="g")
    ax3.plot(alt, rho, color="g")
    ax3.tick_params(axis='y', labelcolor="g")

    plt.show()


def module_test():
    plot_atmosphere(EarthAtmosphereUS1976())


# Include guard
if __name__ == '__main__':
    pass
