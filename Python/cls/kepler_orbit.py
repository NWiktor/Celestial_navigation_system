# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" Class for describing a Keplerian orbit.

Libs
----
* Numpy

Help
----
* https://en.wikipedia.org/wiki/Orbital_elements
* https://space.stackexchange.com/questions/55356/how-to-find-eccentric-anomaly\
-by-mean-anomaly
* https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion
* https://en.wikipedia.org/wiki/Ellipse

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import math as m
import logging

# Third party imports
import numpy as np

# Local application imports

# Class initializations and global variables
gravitational_constant = 6.67430e-11  # m^3 kg^-1 s^-2


# Class and function definitions
class KeplerOrbit:
    """ This class defines a Keplerian orbit in an inertial frame of reference
    (IFR). In order to initialize, the six orbital element must be defined.
    """

    def __init__(self, eccentricity, semimajor_axis_km, inclination_deg,
                 longitude_of_ascending_node_deg, argument_of_periapsis_deg,
                 mean_anomaly_at_epoch_deg):

        # Keplerian elements
        self.eccentricity = eccentricity  # (e), -
        self.semimajor_axis_km = semimajor_axis_km  # (a)
        self.inclination_deg = inclination_deg  # (i)
        self.longitude_of_ascending_node_deg = longitude_of_ascending_node_deg  # (Ω)
        self.argument_of_periapsis_deg = argument_of_periapsis_deg  # (ω)
        self.mean_anomaly_at_epoch_deg = mean_anomaly_at_epoch_deg  # (M0)

        # Orbital parameters
        self.orbital_period = None  # (T) day
        self.mean_angular_motion = None  # (n) deg/day
        self.rotational_matrix = None

        # Calculate known attributes
        self._calculate_rotational_matrix()

    def _calculate_rotational_matrix(self):
        """ Calculates the rotational matrix (Euler rotation 3-1-3 (Z-X-Z))
        between the orbital plane and the inertial reference frame.
        """
        loan = self.longitude_of_ascending_node_deg * m.pi / 180
        aop = self.argument_of_periapsis_deg * m.pi / 180
        incl = self.inclination_deg * m.pi / 180

        self.rotational_matrix = np.array([
            [m.cos(loan) * m.cos(aop) - m.sin(loan) * m.cos(incl) * m.sin(aop),
             -m.cos(loan) * m.sin(aop) - m.sin(loan) * m.cos(incl) * m.cos(aop),
             m.sin(loan) * m.sin(incl)],
            [m.sin(loan) * m.cos(aop) + m.cos(loan) * m.cos(incl) * m.sin(aop),
             -m.sin(loan) * m.sin(aop) + m.cos(loan) * m.cos(incl) * m.cos(aop),
             -m.cos(loan) * m.sin(incl)],
            [m.sin(incl) * m.sin(aop),
             m.sin(incl) * m.cos(aop),
             m.cos(incl)]])

    def _calculate_mean_angular_motion(self) -> None:
        """ Calculate mean angular motion from orbital period. """
        self.mean_angular_motion = 360 / self.orbital_period  # deg/day
        logging.debug("Mean angular motion is %s °/solar day",
                      self.mean_angular_motion)

    def set_orbital_period(self, orbital_period: float) -> None:
        """ Set the value of orbital period attribute directly (from external
        source), and calculates mean angular motion.
        """
        self.orbital_period = orbital_period  # 24*60*60 seconds aka 1 solar day
        self._calculate_mean_angular_motion()

    def calculate_orbital_period(self, mass1_kg: float, mass2_kg: float = 0.0):
        """ Calculate orbital period from keplerian elements. """
        # Converting km to m in semimajor axis, and convert seconds to days
        self.orbital_period = 2 * m.pi * m.sqrt(
            pow(self.semimajor_axis_km * 1000, 3)
            / ((mass1_kg + mass2_kg) * gravitational_constant)) / 86400
        logging.debug("Orbital period is %s", self.orbital_period)
        self._calculate_mean_angular_motion()

    def get_current_mean_anomaly(self, j2000_years: float) -> float:
        """ Calculate mean anomaly at current time, in deg.

        This method can be used to calculate angular position at an
        initial value.
        """
        mean_anomaly = (self.mean_anomaly_at_epoch_deg +
                        (self.mean_angular_motion * j2000_years * 365.25)) % 360
        logging.debug("Mean anomaly is %s°", mean_anomaly)
        return mean_anomaly

    def get_position(self, j2000_time: float):
        """ Calculate position and velocity vectors of an object at a given
        orbit, at a given time since J2000 epoch in the inertial reference
        frame (IRF).
        """
        # Calculate mean anomaly at J2000 in deg
        mean_anomaly = self.get_current_mean_anomaly(j2000_time)

        # Solving for E = M + e*sin(E) using iteration
        eca0 = mean_anomaly * m.pi / 180

        i = 0
        while True:
            eca1 = (mean_anomaly * m.pi / 180) + self.eccentricity * m.sin(eca0)
            i += 1
            if abs(eca1 - eca0) > 0.0000001:
                eca0 = eca1
            else:
                logging.debug("Eccentric anomaly is %s rad, found at "
                              "%s. iteration.", eca1, i)
                break

        # Calculate state variables (position and velocity) in orbital plane
        rx = self.semimajor_axis_km * (m.cos(eca1) - self.eccentricity)  # km
        ry = (self.semimajor_axis_km *
              m.sqrt(1 - pow(self.eccentricity, 2)) * m.sin(eca1))  # km
        # TODO: add velocity calculation
        # vx =
        # vy =

        # Convert position vector coordinates to IRF coordinates
        return np.dot(self.rotational_matrix, (np.array([rx, ry, 0])))

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class CircularOrbit(KeplerOrbit):
    """ Simplified child of the KeplerOrbit class, where eccentricity is zero,
    thus the orbit is circular.

    The function takes only 5 parameters.
    """

    def __init__(self, radius_km, inclination_deg,
                 longitude_of_ascending_node_deg,
                 argument_of_periapsis_deg, mean_anomaly_at_epoch_deg):
        self.radius_km = radius_km  # equals to semimajor axis
        super().__init__(0, radius_km, inclination_deg,
                         longitude_of_ascending_node_deg,
                         argument_of_periapsis_deg, mean_anomaly_at_epoch_deg)

    def get_position(self, j2000_time: float):
        """ Calculate position and velocity vectors of an object at a given
        orbit, at a given time since J2000 epoch in the inertial reference frame
        (IRF).
        """
        # Calculate mean anomaly at J2000 in deg
        mean_anomaly = self.get_current_mean_anomaly(j2000_time)
        eca0 = mean_anomaly * m.pi / 180

        # Calculate state variables (position and velocity) in orbital plane
        rx = self.radius_km * m.cos(eca0)  # km
        ry = self.radius_km * m.sin(eca0)  # km
        # TODO: add velocity calculation
        # vx =
        # vy =

        # Convert position vector coordinates to IRF coordinates
        return np.dot(self.rotational_matrix, (np.array([rx, ry, 0])))


# Include guard
if __name__ == '__main__':
    pass
