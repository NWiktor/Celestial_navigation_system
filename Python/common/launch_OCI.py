# -*- coding: utf-8 -*-
# !/usr/bin/python3

""" This module contains all relevant class and function for orbit propagation
around a celestial body. The module calculates the trajectory of a two-stage
rocket launched from surface in Object-Centered Inertial reference frame (OCI).

Libs
----
* Numpy
* Matplotlib

Help
----
* https://en.wikipedia.org/wiki/Orbital_speed

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import logging
import math as m
from datetime import datetime

# Third party imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from utils import (secs_to_mins, convert_spherical_to_cartesian_coords,
                   runge_kutta_4, unit_vector, rodrigues_rotation,
                   angle_of_vectors, cross)
from cls import (LaunchSite, CircularOrbit, Rocket, RocketAttitudeStatus,
                 RocketEngineStatus, FALCON9)
from database import CAPE_TEST, CAPE_CANEVERAL

logger = logging.getLogger(__name__)

# NOTE: for visualizing only
launch_plane_normal: np.array = np.array([0, 0, 0])


class RocketFlightProgram:
    """ Describes the rocket launch program - staging, engine throttling, roll
    and pitch maneuvers in function of time (seconds) as well as fairing
    jettison. The class main function is to collect constants and return the
    status Enums depending on time.

    Example values for LEO: meco: 145 s, stage separation: meco + 3 s,
    Second engine start-1: meco + 11s,
    fairing jettison: 195 s (LEO) - 222 s (GTO).

    Example for throttle_program - tuple(list[t], list[y]):
    t is the list of time-points since launch, and y is the list of throttling
    factor at the corresponding t values. Outside the given timerange, 1.0
    (100%) is the default value. Burn duration, staging is not considered!
    Example: 80% throttling between 34 and 45 seconds after burn. Before and
    after no throttling (100%) => throttle_map = ([34, 45], [0.8, 0.8])

    :param float meco: Main engine cut-off time in T+seconds.
    :param float ses_1: Second engine start-1 in T+seconds.
    :param float seco_1: Second engine cut-off in T+seconds.
    :param throttle_program: Throttle program: time (s) - throttle (%)
        value-pairs.
    :param float fairing_jettison: Time of jettison in T+seconds.
    :param float pitch_maneuver_start: Time of pitch manuever start in T+seconds.
    :param float pitch_maneuver_end: Time of pitch manuever end in T+seconds.
    :param float ss_1: Time of stage separation in T+seconds
        (default: meco+3 s).
    :param bool manned: The flight includes humans or not (limits acceleration)
    """
    # pylint: disable = too-many-arguments
    def __init__(self,
                 meco: float,  # Main (most) engine cut-off
                 ses_1: float,  # Second engine start-1
                 seco_1: float,  # Second engine cut-off-1
                 throttle_program: list[list[float]] | None,
                 fairing_jettison: float,
                 pitch_maneuver_start: float = 5,
                 pitch_maneuver_end: float = 15,
                 ss_1: float = None,  # Stage separation-1
                 manned: bool = False
                 ):
        # Staging parameters
        self.meco = meco  # s
        self.ses_1 = ses_1  # s
        self.seco_1 = seco_1  # s
        self.throttle_program = throttle_program  # second - % mapping
        self.fairing_jettison = fairing_jettison  # s
        self.manned = manned

        # Stage separation
        if ss_1 is None:
            self.ss_1 = meco + 3  # s
        else:
            self.ss_1 = ss_1  # s

        # Attitude control
        self.pitch_maneuver_start = pitch_maneuver_start  # s
        self.pitch_maneuver_end = pitch_maneuver_end  # s

        # Log data
        self._log_rocketflightprogram()

    def get_engine_status(self, t: float) -> RocketEngineStatus:
        """ Return RocketEngineStatus at a given t time since launch. """
        if t < self.meco:
            return RocketEngineStatus.STAGE_1_BURN
        if self.meco <= t < self.ses_1:
            return RocketEngineStatus.STAGE_1_COAST
        if self.ses_1 <= t < self.seco_1:
            return RocketEngineStatus.STAGE_2_BURN

        return RocketEngineStatus.STAGE_2_COAST

    def get_throttle(self, time: float) -> float:
        """ Return engine throttling factor (0 - 1) at a given t (s) time
        since launch.
        """
        if self.throttle_program is None:
            return 1  # If no throttle_program - always use full throttle

        throttle = np.interp(
            time, self.throttle_program[0], self.throttle_program[1],
            left=1, right=1
            )
        return float(throttle)

    def get_attitude_status(self, t: float) -> RocketAttitudeStatus:
        """ Return RocketAttitudeStatus at a given t time since launch. """
        if t < self.pitch_maneuver_start:
            return RocketAttitudeStatus.VERTICAL_FLIGHT
        if self.pitch_maneuver_start <= t < self.pitch_maneuver_end:
            return RocketAttitudeStatus.PITCH_PROGRAM

        return RocketAttitudeStatus.GRAVITY_ASSIST

    def _log_rocketflightprogram(self):
        """ Print flight program. """
        logger.info("--- FLIGHT PROFILE DATA ---")
        logger.info("MAIN ENGINE CUT OFF at T+%s (%s s)",
                    secs_to_mins(self.meco), self.meco)
        logger.info("STAGE SEPARATION 1 at T+%s (%s s)",
                    secs_to_mins(self.ss_1), self.ss_1)
        logger.info("SECOND ENGINE START 1 at T+%s (%s s)",
                    secs_to_mins(self.ses_1), self.ses_1)
        logger.info("PAYLOAD FAIRING JETTISON at T+%s (%s s)",
                    secs_to_mins(self.fairing_jettison),
                    self.fairing_jettison)
        logger.info("SECOND ENGINE CUT OFF 1 at T+%s (%s s)",
                    secs_to_mins(self.seco_1), self.seco_1)


class RocketLaunch:
    """ RocketLaunch class, defined by name, payload mass, drag coefficient and
    diameter; and stages.
    """

    def __init__(self, name: str, rocket: Rocket, payload_mass: float,
                 flightprogram: RocketFlightProgram,
                 target_orbit: CircularOrbit, launchsite: LaunchSite,
                 earliest_launch_date: float = None,
                 flight_angle_corr: float = 0.87):
        self.name = name
        self.rocket = rocket
        self.rocket.set_payload_mass_kg(payload_mass)  # Call to set payload!
        self.flightprogram = flightprogram
        self.target_orbit = target_orbit
        self.launchsite = launchsite
        self.launch_azimuth: list[float | None] = [None, None]
        self.flight_angle_corr = flight_angle_corr
        self._density_at_surface: float = 0.0  # Automatically set
        self.earliest_launch_date = earliest_launch_date

        if earliest_launch_date is None:
            pass
            # FIXME: self.earliest_launch_date = time.now()
        else:
            self.earliest_launch_date = earliest_launch_date

        # Check if orbit is reachable
        self._check_radius()
        self.target_velocity = self._get_target_velocity(
            self.target_orbit.radius_km * 1000)
        self._check_inclination()
        self._get_launch_azimuth()  # Calculate lauch azimuth
        self._get_launch_date()  # Time of launch to get desired LoAN

        # Launchsite vectors
        self.r_launch = None
        self.local_north = None
        self.local_east = None
        self.launch_plane_unit = None

        # State variables / dynamic vectors and mass (m, m/s and kg)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _check_radius(self):
        """ Check if specified target orbit radius is valid: greater than the
        planet surface.
        """
        if (self.target_orbit.radius_km * 1000 <=
                self.launchsite.planet.surface_radius_m):
            logger.error(f"ERROR: orbit radius"
                         f"({self.target_orbit.radius_km:.3f} km) is smaller "
                         f"than surface radius!")
            raise ValueError

        logger.info(f"Orbit radius: {self.target_orbit.radius_km:.3f} km")

    def _check_inclination(self):
        """ Check if specified target orbit inclination is valid: greater than
        the launch-site latitude.
        """
        if self.target_orbit.inclination_deg < self.launchsite.latitude:
            logger.error("ERROR: Cannot launch directly into orbit with"
                         f"inclination ({self.target_orbit.inclination_deg:.3f}°)"
                         "smaller than launchsite latitude!")
            raise ValueError

        logger.info(f"Inclination: {self.target_orbit.inclination_deg:.3f}°")

    # TODO: test baikonour and korou azimuth limit (349 to 90)
    def _get_launch_azimuth(self):
        """ Check if target orbit is feasible.

        A handy formula to remember is: cos(i) = cos(φ) * sin(β), where i is
        the inclination, β is the launch azimuth, and φ is the launch
        latitude.
        https: // www.orbiterwiki.org / wiki / Launch_Azimuth
        """
        launch_azimuth = m.asin(
                m.cos(self.target_orbit.inclination_deg * m.pi / 180)
                / m.cos(self.launchsite.latitude * m.pi / 180)
                )  # rad

        v_eqrot = (self.launchsite.planet.surface_radius_m
                   * self.launchsite.planet.angular_velocity_rad_per_s)

        launch_azimuth_corr = m.atan2(
                self.target_velocity * m.sin(launch_azimuth)
                - v_eqrot * m.cos(self.launchsite.latitude * m.pi / 180),
                self.target_velocity * m.cos(launch_azimuth)
        ) / m.pi * 180  # (for deg)

        launch_azimuth1 = launch_azimuth_corr
        launch_azimuth2 = (180 - launch_azimuth_corr)

        if self.launchsite.launch_azimuth_range is not None:
            if not (self.launchsite.launch_azimuth_range[0]
                    <= launch_azimuth1
                    <= self.launchsite.launch_azimuth_range[1]):
                logger.warning(f"WARNING: Launch azimuth for ascending node "
                               f"({launch_azimuth1:.3f}°) "
                               "is out of permitted range!")
            self.launch_azimuth[0] = launch_azimuth1
            if not (self.launchsite.launch_azimuth_range[0]
                    <= launch_azimuth2
                    <= self.launchsite.launch_azimuth_range[1]):
                logger.warning(f"WARNING: Launch azimuth for descending node "
                               f"({launch_azimuth2:.3f}°) "
                               "is out of permitted range!")
            self.launch_azimuth[1] = launch_azimuth2
            if not self.launch_azimuth:
                logger.error("ERROR: Launch is not possible from this location"
                             "because of launch azimuth limitations!")
                raise ValueError

        else:
            logger.warning("No checks for launch azimuth!")
            self.launch_azimuth[0] = launch_azimuth1
            self.launch_azimuth[1] = launch_azimuth2

    def _get_target_velocity(self, radius_m):
        """ Calculates orbital velocity for the given radius (m). """
        target_velocity = m.sqrt(
                self.launchsite.std_gravitational_parameter / radius_m
        )
        logger.info(f"Target velocity for orbit: {target_velocity:.3f} m/s")
        return target_velocity

    def _get_launch_date(self):
        """ xxx """
        # TODO: implement
        pass

    # TODO: implement
    def _check_end_condition(self):
        pass

    def _set_inital_params(self):
        """  """
        global launch_plane_normal
        # Update state vector with initial conditions
        self.r_launch = convert_spherical_to_cartesian_coords(
            self.launchsite.radius,
            self.launchsite.latitude * m.pi / 180,
            self.launchsite.longitude * m.pi / 180
        )
        self.v_init = cross(
            np.array([0, 0, self.launchsite.angular_velocity]), self.r_launch
        )

        # TODO: check this in alfonso's program, how to account for j2000 frame
        omega_planet = np.array(
            [0, 0, self.launchsite.angular_velocity])  # rad/s
        self.state = np.concatenate(
            (self.r_launch,
             cross(omega_planet, self.r_launch),
             [self.rocket.total_mass])
        )

        self._density_at_surface = self.launchsite.get_density(0.0)

        # Local orientations at lauchsite
        east_launch = convert_spherical_to_cartesian_coords(
            self.launchsite.radius,
            0,
            ((self.launchsite.longitude + 90) % 360) * m.pi / 180
        )
        self.local_east = unit_vector(east_launch)
        self.local_north = unit_vector(cross(self.r_launch, east_launch))
        local_zenith = unit_vector(self.r_launch)

        # NOTE: launch plane vector is static, it represents the initial plane,
        #  in which the rocket launched.
        launch_plane_normal = rodrigues_rotation(
            self.local_north,
            local_zenith,
            (90 - self.launch_azimuth[0]) * m.pi / 180
        )
        self.launch_plane_unit = unit_vector(launch_plane_normal)

    # pylint: disable = too-many-locals
    def _launch_ode(self, time: float, state, dt: float):
        """ 2nd order ODE of the state-vectors, during launch.

        The function returns the second derivative of position vector at a given
        time (acceleration vector), using the position (r) and velocity (v)
        vectors. All values represent the same time.

        Trick: While passing through the velocity vector unchanged, we can
        numerically integrate both functions in the RK4-solver in one step (this
        is outside of this functions scope).

        State-vector: rx, ry, rz, vx, vy, vz, m
        State-vector_dot: vx, vy, vz, ax, ay, az, m_dot
        """
        global launch_plane_normal
        r = state[:3]  # Position vector
        v = state[3:6]  # Velocity vector
        mass = state[6]  # Mass

        # Calculate flight characteristics at the actual step
        v_relative = self.launchsite.get_relative_velocity(state)
        air_density = self.launchsite.get_density(
            np.linalg.norm(r) - self.launchsite.radius)
        pressure_ratio = air_density / self._density_at_surface
        drag_const = self.rocket.drag_constant * air_density / 2

        # Calculate acceleration: drag and gravity
        a_drag = - unit_vector(v) * drag_const * v_relative ** 2 / mass
        a_gravity = (-r * self.launchsite.std_gravitational_parameter
                     / np.linalg.norm(r) ** 3)

        # Calculate acceleration: thrust
        thrust_force = (self.rocket.get_thrust()
                        * self.flightprogram.get_throttle(time))
        thrust = thrust_force / mass

        # Orbital parameters
        # NOTE: angular velocity bc. of the planet rotation is a one-time
        #  addition, only the initial value (v_init) must be substracted, and
        #  not the running value! However v_rel is dynamic value, because of
        #  'v' is derived from the state vector.
        v_rel = v - self.v_init
        # NOTE: flight angle represents the angle between the local zenith, and
        #  the rocket relative velocity vector. It should start from 0°
        #  (vertical flight) to 90° as the rocket reaches orbit.
        flight_angle = angle_of_vectors(unit_vector(r), unit_vector(v_rel))

        # Vertical flight until tower is cleared in non-inertial frame
        if time < self.flightprogram.pitch_maneuver_start:
            # NOTE: should use dynamic 'r_tower' vector, because launch site is
            #  rotating in the inertial frame with the central body; however
            #  this is very hard to compensate later with hand-made
            #  thrust-vectoring, so just use 'r_launch' to eliminate offset
            #  from the launch-plane
            a_thrust = thrust * unit_vector(self.r_launch)

        # Initial pitch-over maneuver -> Slight offset of thrust from velocity
        elif (self.flightprogram.pitch_maneuver_start <= time
              < self.flightprogram.pitch_maneuver_end):
            # NOTE: incrementally rotating the relative velocity vector in the
            #  orbital plane, to imitate pitch manuever and start gravity assist
            # TODO: find universally applicable parameters, or implement checks
            #  to set it automatically
            v_pitch = rodrigues_rotation(
                    unit_vector(r),
                    self.launch_plane_unit,
                    (flight_angle + self.flight_angle_corr) * m.pi / 180)
            a_thrust = thrust * unit_vector(v_pitch)

        else:  # Gravity assist -> Thrust is parallel with velocity
            # NOTE: keep velocity vector in launch plane
            v_pitch = rodrigues_rotation(
                unit_vector(r),
                self.launch_plane_unit,
                flight_angle * m.pi / 180)
            a_thrust = thrust * unit_vector(v_pitch)

        # Calculate acceleration (v_dot) and m_dot
        a = a_gravity + a_thrust + a_drag  # 2nd order ODE function (acc.)
        m_dot = (- thrust_force
                 / (self.rocket.get_isp(pressure_ratio)
                    * self.launchsite.planet.surface_gravity_m_per_s2 * dt
                    * self.flightprogram.get_throttle(time)  # throttle corr.
                    )
                 )
        return np.concatenate((v, a, [m_dot]))  # vx, vy, vz, ax, ay, az, m_dot

    def launch(self, total_steps: int = 16000, increment: int = 1):
        """ Yield rocket's status variables during launch, every second. """

        # Update state vector with initial conditions, and calculate
        # orientation vectors at launchsite
        self._set_inital_params()

        # Yield initial values
        yield 0, self.state, np.array([0.0, 0.0, 0.0])  # time, state, acc.

        logger.info("--- FLIGHT CALCULATION START ---")
        time_step = 0  # Current step
        while time_step <= total_steps:
            # Calculate stage status according to time
            self.rocket.set_stage_status(
                self.flightprogram.get_engine_status(time_step))

            # Calculate state-vector, acceleration and delta_m
            # NOTE: The ODE is solved for the acceleration vector and m_dot,
            #  which is used as an initial condition for the RK4 numerical
            #  integrator function, which then solves for the velocity function.
            #  Passing not only the acceleration vector, but the velocity vector
            #  to the RK4, we can numerically integrate twice with one
            #  function-call, thus we get back the full state-vector.
            self.state, state_dot = runge_kutta_4(
                self._launch_ode, time_step, self.state, increment, increment
            )
            acceleration = state_dot[3:6]

            # Post-processing
            # NOTE: angular velocity bc. of the planet rotation is a one-time
            #  addition, only the initial value (v_init) must be substracted,
            #  and not the running value! However v_rel is dynamic value,
            #  because of 'v' is derived from the state vector.
            v_rel = self.state[3:6] - self.v_init
            # NOTE: flight angle represents the angle between the local zenith,
            #  and the rocket relative velocity vector. It should start from 0°
            #  (vertical flight) to 90° as the rocket reaches orbit.
            flight_angle = angle_of_vectors(
                unit_vector(self.state[0:3]), unit_vector(v_rel))
            # NOTE: the current orbital plane is determined by the current pos.
            #  and velocity vectors. It is changing bc. of the planet rotation.
            orbital_plane_current = cross(
                unit_vector(self.state[0:3]), unit_vector(self.state[3:6])
            )
            # NOTE: current inclination is changing as the current orbital plane
            #  changes. It should reach the target value at the same time as
            #  orbital radius and velocity.
            inclination_current = angle_of_vectors(
                unit_vector(orbital_plane_current),
                np.array([0, 0, 1])
            )
            acc_deviation = angle_of_vectors(
                state_dot[3:6], launch_plane_normal
            )

            # Logging variables
            logger.debug(f"--- TIMESTEP: T+{time_step} s ---")
            logger.debug(f"Current inclination: {inclination_current:.3f}°")
            logger.debug(f"Flight angle: {flight_angle:.3f}°")
            logger.debug(
                f"Angle between thrust and launch plane: {acc_deviation:.3f}°")

            # Burn mass from stage
            # TODO: remove ifs
            stage_status = self.rocket.get_stage_status()
            if stage_status == RocketEngineStatus.STAGE_1_BURN:
                self.rocket.stages[0].burn_mass(state_dot[6], time_step)
            if stage_status == RocketEngineStatus.STAGE_2_BURN:
                self.rocket.stages[1].burn_mass(state_dot[6], time_step)

            # Evaluate staging events, and update statevector with new mass
            if time_step == self.flightprogram.fairing_jettison:
                self.state[6] = self.rocket.fairing_jettison_event()
            if time_step == self.flightprogram.ss_1:
                self.state[6] = self.rocket.staging_event_1()

            # Log new data and end-conditions
            # TODO: implement checks for target height, arget velocity, and
            #  flight angle. Implement report for deviations to refine params
            r_current = np.linalg.norm(self.state[0:3])
            v_current = np.linalg.norm(self.state[3:6])
            altitude_above_surface = (r_current - self.launchsite.radius)

            if altitude_above_surface <= 0:
                logger.warning("WARNING! LITHOBRAKING!")
                break

            delta_r = self.target_orbit.radius_km - r_current
            delta_v = self.target_velocity - v_current
            limit_r = self.target_orbit.radius_km * 0.01
            limit_v = self.target_velocity * 0.01

            if abs(delta_r) <= limit_r and abs(delta_v) <= limit_v:
                logger.info(f"Target orbit reached at {time_step} s:"
                            f"{altitude_above_surface:.3f} m, "
                            f"{v_current:.3f} m/s")
                break

            delta_v2 = self._get_target_velocity(r_current) - v_current
            if delta_r <= 0 and abs(delta_v2) <= limit_v:
                logger.info(f"Stable orbit reached at {time_step} s: "
                            f"{altitude_above_surface:.3f} m, "
                            f"{v_current:.3f} m/s")
                break

            # Yield values
            time_step += increment
            yield time_step, self.state, acceleration


class RocketLanding:

    def __init__(self):
        # TODO: use the same with launch but in reverse order
        pass


# TODO: implement this
class LaunchTrajectory3D:
    """ xxx """

    def __init__(self, name: str, rocketlaunch: RocketLaunch,
                 no_earlier_date: datetime):
        self.name = name
        self.rocketlaunch = rocketlaunch
        self.no_earlier_date = no_earlier_date

    def get_position(self, time):
        """  """
        # return postion at given time, just like the orbit functions

        # if no stable orbit: return launch func

        # if stable orbit, create orbit and return values from there - to skip iteration

        return


# Main function for module testing
# pylint: disable = too-many-statements
# TODO: keep plotting but delete sphere, add new plot for flight angle variation
#  add option for continous plotting?
# TODO: use 3d visualization with ursina
def plot(rocketlaunch: RocketLaunch):
    """ Plots the given RocketLaunch parameters. """
    global launch_plane_normal
    # Launch
    time_data = []
    alt_data = []
    rx = []
    ry = []
    rz = []
    vx = []
    vy = []
    vz = []
    vel_data = []
    acc_data = []
    mass_data = []
    cbsr = rocketlaunch.launchsite.radius
    plot_title = (f"{rocketlaunch.name} launch from"
                  f" {rocketlaunch.launchsite.name}")

    for time, state, acc, in rocketlaunch.launch(500, 1):
        time_data.append(time)
        rx.append(state[0])
        ry.append(state[1])
        rz.append(state[2])
        vx.append(state[3])
        vy.append(state[4])
        vz.append(state[5])
        alt_data.append((np.linalg.norm(state[0:3]) - cbsr) / 1000)  # Alt. - km
        vel_data.append(np.linalg.norm(state[3:6]))  # Velocity - m/s
        acc_data.append(np.linalg.norm(acc) / 9.82)  # Acceleration - g
        mass_data.append(state[6] / 1000)  # Mass - 1000 kg

    pts_x = []
    pts_y = []
    pts_z = []
    vector = np.array([rx[0], ry[0], rz[0]])
    for i in range(0, 90):
        rot_v = rodrigues_rotation(vector, launch_plane_normal, i * m.pi / 180)
        pts_x.append(rot_v[0])
        pts_y.append(rot_v[1])
        pts_z.append(rot_v[2])

    # Plotting
    plt.style.use('_mpl-gallery')

    fig = plt.figure(layout='tight', figsize=(19, 9.5))
    fig.suptitle(plot_title)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title("Flight profile")
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('flight altitude (km)', color="m")
    ax1.plot(time_data, alt_data, color="m")

    # Flight velocity, acceleration
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.set_title("Flight velocity and acceleration")
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('acceleration (g)', color="b")
    ax3.set_xlim(0, len(time_data))
    ax3.set_ylim(0, 10)
    ax3.scatter(time_data, acc_data, s=0.5, color="b")
    ax3.tick_params(axis='y', labelcolor="b")

    ax4 = ax3.twinx()
    ax4.set_ylabel('velocity (m/s)', color="g")
    ax4.set_ylim(0, 10000)
    ax4.plot(time_data, vel_data, color="g")
    ax4.tick_params(axis='y', labelcolor="g")

    # Mass
    ax6 = fig.add_subplot(2, 2, 3)
    ax6.set_title("Mass")
    ax6.set_xlabel('time (s)')
    ax6.set_ylabel('mass (kg)')
    ax6.set_xlim(0, len(time_data))
    ax6.set_ylim(0, 600)
    ax6.scatter(time_data, mass_data, s=0.5, color="b")

    # Plot trajectory in 3D
    ax5 = fig.add_subplot(2, 2, 4, projection='3d')
    ax5.set_title("Flight trajectory")
    ax5.plot(rx, ry, rz, label="Trajectory", color="m")

    ax5.plot(pts_x, pts_y, pts_z, label="Orbit_plane", color="y")

    # Plot CB surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = cbsr * np.outer(np.cos(u), np.sin(v))
    y = cbsr * np.outer(np.sin(u), np.sin(v))
    z = cbsr * np.outer(np.ones(np.size(u)), np.cos(v))
    ax5.plot_surface(x, y, z)
    ax5.set_aspect('equal')

    # Reference vectors
    ax5.plot([0, launch_plane_normal[0] * cbsr * 1.1],
             [0, launch_plane_normal[1] * cbsr * 1.1],
             [0, launch_plane_normal[2] * cbsr * 1.1],
             label="launch_plane_normal", color="y")
    ax5.plot([0, cbsr * 1.1], [0, 0], [0, 0], label="x axis", color="r")
    ax5.plot([0, 0], [0, cbsr * 1.1], [0, 0], label="y axis", color="g")
    ax5.plot([0, 0], [0, 0], [0, cbsr * 1.1], label="z axis", color="b")
    # Launch site:
    ax5.plot([0, rx[0]], [0, ry[0]], [0, rz[0]], label="launch", color="w")

    plt.show()


# Include guard
def main():
    """ Example SpaceX Falcon 9 launch demonstrating the use of the
    LaunchSite, Stage and RocketLaunch classes.
    
    The function calculates and plots flight parameters during and after
    liftoff.
    """
    # TODO: Modelling throttle to 80% properly, and test it
    throttle_map = [[70, 80, 81, 150, 550],
                    [0.8, 0.8, 1.0, 0.88, 0.88]]
    flight_program = RocketFlightProgram(145, 156, 514,
                                         throttle_map, 195)
    targetorbit = CircularOrbit(300 + 6_378, 51.6,
                                -45,
                                90,
                                0)
    mission_414_falcon9 = RocketLaunch("Mission 414",
                                       FALCON9, 15000,
                                       flight_program,
                                       targetorbit,
                                       CAPE_TEST)

    # Plot launch
    plot(mission_414_falcon9)


# Include guard
if __name__ == '__main__':
    main()
