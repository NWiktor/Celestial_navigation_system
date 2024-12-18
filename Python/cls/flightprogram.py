# -*- coding: utf-8 -*-
# !/usr/bin/python3
""" This module defines the RocketFlightProgram class.

Libs
----
* Numpy
* Matplotlib

Help
----

Contents
--------
"""

# Standard library imports
# First import should be the logging module if any!
import logging
from enum import Enum
import numpy as np
from cls import EngineStatus
from utils import secs_to_mins

logger = logging.getLogger(__name__)


class AttitudeStatus(Enum):
    """ Describes the status of the rocket attitude control programs during
    liftoff.
    """
    VERTICAL_FLIGHT = 0
    ROLL_PROGRAM = 1
    PITCH_PROGRAM = 2
    GRAVITY_ASSIST = 3


class FlightProgram:
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
    :param float pitch_maneuver_start: Time of pitch manuever start in T+secs.
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
                 pitch_maneuver_start: float = 15,
                 pitch_maneuver_end: float = 25,
                 pitch_angle: float = 6.4,
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
        self.pitch_angle = pitch_angle  # deg

        # Log data
        self._log_rocketflightprogram()

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

    def get_engine_status(self, time: float) -> EngineStatus:
        """ Return EngineStatus at a given 't' time since launch. """
        if time < self.meco:
            return EngineStatus.STAGE_1_BURN
        if self.meco <= time < self.ses_1:
            return EngineStatus.STAGE_1_COAST
        if self.ses_1 <= time < self.seco_1:
            return EngineStatus.STAGE_2_BURN

        return EngineStatus.STAGE_2_COAST

    def get_attitude_status(self, time: float) -> AttitudeStatus:
        """ Return RocketAttitudeStatus at a given t time since launch. """
        if time < self.pitch_maneuver_start:
            return AttitudeStatus.VERTICAL_FLIGHT
        if self.pitch_maneuver_start <= time < self.pitch_maneuver_end:
            return AttitudeStatus.PITCH_PROGRAM

        return AttitudeStatus.GRAVITY_ASSIST

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
