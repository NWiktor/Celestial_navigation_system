# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module describes the abstract Atmosphere class, and different
atmospheric models (e.g. Earth and Mars).

Libs
----
* Matplotlib - for data visualization and test

Help
----
* https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
* https://www.digitaldutch.com/atmoscalc/US_Standard_Atmosphere_1976.pdf
* https://en.wikipedia.org/wiki/Barometric_formula
* http://www.luizmonteiro.com/StdAtm.aspx
* https://www.grc.nasa.gov/www/k-12/airplane/atmosmrm.html

Contents
--------
"""

# Standard library imports
import logging
import math as m
import matplotlib.pyplot as plt

# Local application imports
from cls.celestial_body_utils import Composition, Component

# Class initializations and global variables
logger = logging.getLogger(__name__)

EarthAthmosphericComposition = Composition([
    Component("Nitrogen", 78.084, "N2"),
    Component("Oxygen", 20.946, "O2"),
    Component("Argon", 0.9340, "Ar")
    # Component("Carbon-dioxid", 0.0417, "CO2"),
    # Component("Neon", 0.001818, "Ne"),
    # Component("Helium", 0.000524, "He"),
    # Component("Methane", 0.000191, "CH4"),
    # Component("Krypton", 0.000114, "Kr")
    ],
    source="https://en.wikipedia.org/wiki/Atmosphere_of_Earth")

MarsAthmosphericComposition = Composition([
    Component("Carbon-dioxid", 94.9, "CO2"),
    Component("Nitrogen", 2.8, "N2"),
    Component("Argon", 2.0, "Ar"),
    Component("Oxygen", 0.174, "O2"),
    Component("Carbon-monoxid", 0.0747, "CO"),
    Component("Water vapour", 0.03, "H2O")
    ],
    source=["https://en.wikipedia.org/wiki/Atmosphere_of_Mars",
            "https://www.sciencedirect.com/topics/earth-and-planetary-sciences/martian-atmosphere"])


class Atmosphere:
    """ Baseclass for a generic atmospheric model. """

    def __init__(self, model_name: str, atm_upper_limit_m: int):
        self.model_name = model_name
        self.atm_upper_limit_m = atm_upper_limit_m  # Upper limit
        self.composition = None

    # pylint: disable = unused-argument
    # @override
    def _atmospheric_model(self, altitude_m) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values.

        When the altitude is lass than 0, there is solid ground, no atmosphere;
        when altitude is above the upper limit, there is space (no pressure, no
        atmosphere).
        """
        return 0.0, 0.0, 0.0

    def get_atm_params(self, altitude_m) -> tuple[float, float, float]:
        """ Enforces the atmospheric model limits, and returns the temperature,
        pressure, density values.
        """
        return self._atmospheric_model(altitude_m)

    def get_temperature(self, altitude_m) -> float:
        """ Returns the temperature (K) at a given altitude (m). """
        return self._atmospheric_model(altitude_m)[0]

    def get_pressure(self, altitude_m) -> float:
        """ Returns the pressure (kPa) at a given altitude (m). """
        return self._atmospheric_model(altitude_m)[1]

    def get_density(self, altitude_m) -> float:
        """ Returns the density (kg/m3) at a given altitude (m). """
        return self._atmospheric_model(altitude_m)[2]

    def set_composition(self, composition: Composition) -> None:
        """ Sets the chemical composition of the atmosphere. """
        self.composition = composition

    def get_composition(self) -> Composition:
        """ Returns the chemical composition of the atmosphere. """
        return self.composition


class EarthAtmosphere(Atmosphere):
    """  Standard model of Earth Atmosphere according to NASA:
    https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    """

    def __init__(self):
        super().__init__("Standard atmosphere, simplified, NASA",
                         30000)
        self.set_composition(EarthAthmosphericComposition)

    def _atmospheric_model(self, alt_m: float) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values of the atmosphere
        depending on the altitude measured from sea level. Calculation is
        loosely based on the US. standard atmosphere model (1976), and uses
        three separate zones (layers).
        """
        alt_m = max(0.0, alt_m)  # Handle negative values

        if 0 <= alt_m < 11000:  # Troposhere
            temp_celsius = 15.04 - 0.00649 * alt_m
            pressure_kpa = 101.29 * pow((temp_celsius + 273.1) / 288.08, 5.256)

        elif 11000 <= alt_m < 25000:  # Lower stratosphere
            temp_celsius = -56.46
            pressure_kpa = 22.65 * m.exp(1.73 - 0.000157 * alt_m)

        elif 25000 <= alt_m < self.atm_upper_limit_m:  # Upper stratosphere
            temp_celsius = -131.21 + 0.00299 * alt_m
            pressure_kpa = 2.488 * pow((temp_celsius + 273.1) / 216.6, -11.388)

        else:
            return 0.0, 0.0, 0.0  # in space zero kelvin, zero athmosphere

        # Calculate air density and return values
        temp_kelvin = temp_celsius + 273.15
        air_density = pressure_kpa / (0.2869 * temp_kelvin)
        logger.debug("Atmospheric temp.: %.6f (K°), pres.: %.6f (kPa) and"
                     " air density: %.6f (kg/m3) @ %s (m)",
                     temp_kelvin, pressure_kpa, air_density, alt_m)
        return temp_kelvin, pressure_kpa, air_density


class EarthAtmosphereUS1976(Atmosphere):
    """  Standard model of Earth Atmosphere according (1976):
    https://www.digitaldutch.com/atmoscalc/US_Standard_Atmosphere_1976.pdf
    """

    def __init__(self):
        super().__init__("US. standard atmosphere, 1976",
                         100000)
        self.set_composition(EarthAthmosphericComposition)

    def _atmospheric_model(self, alt_m: float) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values of the atmosphere
        depending on the altitude measured from sea level. Calculation is based
        on the US. standard atmosphere model (1976), which has seven separate
        zones.

        https://en.wikipedia.org/wiki/Barometric_formula
        http://www.luizmonteiro.com/StdAtm.aspx
        """
        alt_m = max(0.0, alt_m)  # Handle negative values

        if 0 <= alt_m < 11000:  # Troposphere
            temp_kelvin = 288.15 - 0.0065 * alt_m  # == 15 C°
            pressure_kpa = 101.325 * pow(288.15 / temp_kelvin, -5.228)

        elif 11000 <= alt_m < 20000:  # Tropopause
            temp_kelvin = 216.65
            pressure_kpa = 22.632 * m.exp(-0.03416265012915032
                                          * (alt_m - 11000) / 216.65)

        elif 20000 <= alt_m < 32000:  # Lower stratosphere
            temp_kelvin = 216.65 + 0.001 * (alt_m - 20000)
            pressure_kpa = 5.47489 * pow(216.65 / temp_kelvin, 34.1626)

        elif 32000 <= alt_m < 47000:  # Upper stratosphere
            temp_kelvin = 228.65 + 0.0028 * (alt_m - 32000)
            pressure_kpa = 0.86802 * pow(228.65 / temp_kelvin, 12.2009)

        elif 47000 <= alt_m < 51000:  # Stratopause
            temp_kelvin = 270.65
            pressure_kpa = 0.11091 * m.exp(-0.03416265012915032
                                           * (alt_m - 47000) / 270.65)

        elif 51000 <= alt_m < 71000:  # Lower mesosphere
            temp_kelvin = 270.65 - 0.0028 * (alt_m - 51000)
            pressure_kpa = 0.06694 * pow(270.65 / temp_kelvin, -12.2009)

        elif 71000 <= alt_m < 86000:  # Upper mesosphere
            temp_kelvin = 214.65 - 0.002 * (alt_m - 71000)
            pressure_kpa = 0.00396 * pow(214.65 / temp_kelvin, -17.0813)

        elif 86000 <= alt_m < 91000:  # Thermosphere
            temp_kelvin = 184.65
            pressure_kpa = 0.0003

        elif 91000 <= alt_m < self.atm_upper_limit_m:
            temp_kelvin = 184.65
            pressure_kpa = 0.0003

        else:
            return 0.0, 0.0, 0.0  # in space zero kelvin, zero athmosphere

        # Calculate air density and return values
        air_density = pressure_kpa / (0.2869 * temp_kelvin)
        logger.debug("Atmospheric temp.: %.6f (K), pres.: %.6f (kPa) and"
                     " air density: %.6f (kg/m3) @ %s (m)",
                     temp_kelvin, pressure_kpa, air_density, alt_m)
        return temp_kelvin, pressure_kpa, air_density


# TODO: fix celsius to kelvin !!!
class MarsAtmosphere(Atmosphere):
    """  Standard model of Mars Atmosphere according to NASA:
    https://www.grc.nasa.gov/www/k-12/airplane/atmosmrm.html
    """

    def __init__(self):
        super().__init__("Standard Martian atmosphere, NASA",
                         60000)
        self.set_composition(MarsAthmosphericComposition)

    def _atmospheric_model(self, alt_m: float) -> tuple[float, float, float]:
        """ Returns the temperature, pressure, density values of the atmosphere
        depending on the altitude measured from surface level. Calculation uses
        two separate zones (layers) and based on measurements made by the
        Mars Global Surveyor in April 1996.
        """
        alt_m = max(0.0, alt_m)  # Handle negative values

        if 0 <= alt_m < 7000:  # Troposhere
            temp_celsius = -31 - 0.000998 * alt_m
            pressure_kpa = 0.699 * m.exp(-0.00009 * alt_m)

        elif 7000 <= alt_m < self.atm_upper_limit_m:  # Upper stratosphere
            temp_celsius = -23.4 - 0.00222 * alt_m
            pressure_kpa = 0.699 * m.exp(-0.00009 * alt_m)

        else:
            return 0.0, 0.0, 0.0  # in space zero kelvin, zero athmosphere

        # Calculate air density and return values
        temp_kelvin = temp_celsius + 273.15
        air_density = pressure_kpa / (0.1921 * temp_kelvin)
        logger.debug("Atmospheric temp.: %.6f (K°), pres.: %.6f (kPa) and"
                     " athmospheric density: %.6f (kg/m3) @ %s (m)",
                     temp_celsius, pressure_kpa, air_density, alt_m)
        return temp_kelvin, pressure_kpa, air_density


def plot_atmosphere(model: Atmosphere):
    """ Plot pressure, temperature and density data of the given model,
    to check its validity.
    """

    alt = []
    tmp = []
    pres = []
    rho = []
    for i in range(1, int(model.atm_upper_limit_m/10 * 1.1)):
        data = model.get_atm_params(i * 10)
        alt.append(i*10)
        tmp.append(data[0])
        pres.append(data[1])
        rho.append(data[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle(f"Atmospheric parameters ({model.model_name})")
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
    ax3.set_title("Atmospheric density")
    ax3.set_xlabel('altitude (m)')
    ax3.set_ylabel('atmospheric density (kg/m3)', color="g")
    ax3.plot(alt, rho, color="g")
    ax3.tick_params(axis='y', labelcolor="g")

    plt.show()


# Include guard
if __name__ == '__main__':
    plot_atmosphere(EarthAtmosphere())
    # plot_atmosphere(EarthAtmosphereUS1976())
    # plot_atmosphere(MarsAtmosphere())
    # for comp in MarsAthmosphericComposition.get_composition():
    #    print(comp)
    # plot_atmosphere(MarsAtmosphere())
