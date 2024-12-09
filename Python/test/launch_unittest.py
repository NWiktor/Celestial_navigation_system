# pylint: disable = missing-module-docstring
import unittest
from common.launch_OCI import RocketFlightProgram, RocketLaunch
from cls.hardware import RocketEngineStatus, RocketAttitudeStatus


# pylint: missing-class-docstring
class TestRocketFlightProgram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Setting up instance for testing. """
        throttle_map = [[70, 80, 81, 150, 550], [0.8, 0.8, 1.0, 0.88, 0.88]]
        cls.test_class = RocketFlightProgram(
                145, 156, 514, throttle_map, 195
        )

    def test_get_engine_status(self):
        self.assertEqual(
            self.test_class.get_engine_status(1),
            RocketEngineStatus.STAGE_1_BURN,
            "Shuld be STAGE_1_BURN at the first (1) second!")
        self.assertEqual(
            self.test_class.get_engine_status(self.test_class.meco),
            RocketEngineStatus.STAGE_1_COAST,
            f"Should be STAGE_1_COAST at {self.test_class.meco} s.")
        self.assertEqual(
            self.test_class.get_engine_status(self.test_class.ses_1 - 1),
            RocketEngineStatus.STAGE_1_COAST,
            f"Should be STAGE_1_COAST at {self.test_class.ses_1-1} s.")
        self.assertEqual(
            self.test_class.get_engine_status(self.test_class.ses_1),
            RocketEngineStatus.STAGE_2_BURN,
            f"Should be STAGE_2_BURN at {self.test_class.ses_1} s.")
        self.assertEqual(
            self.test_class.get_engine_status(self.test_class.seco_1),
            RocketEngineStatus.STAGE_2_COAST,
            f"Should be STAGE_2_COAST at {self.test_class.seco_1} s.")

    def test_get_throttle(self):
        self.assertAlmostEqual(self.test_class.get_throttle(1),
                               1.0, 3,
                               "Should be 1.0 at start!")
        self.assertAlmostEqual(self.test_class.get_throttle(70),
                               0.8, 3,
                               "Should be 0.8 at 70 s!")
        self.assertAlmostEqual(self.test_class.get_throttle(81),
                               1.0, 3,
                               "Should be 1.0 at 81 s!")
        self.assertAlmostEqual(self.test_class.get_throttle(150),
                               0.88, 3,
                               "Should be 0.88 at 150 s!")

    def test_get_attitude_status(self):
        self.assertEqual(
            self.test_class.get_attitude_status(1),
            RocketAttitudeStatus.VERTICAL_FLIGHT,
            "Shuld be VERTICAL_FLIGHT at the first (1) second!")
        self.assertEqual(
            self.test_class.get_attitude_status(
                self.test_class.pitch_maneuver_start),
            RocketAttitudeStatus.PITCH_PROGRAM,
            f"Should be PITCH_PROGRAM at "
            f"{self.test_class.pitch_maneuver_start} s.")
        self.assertEqual(
            self.test_class.get_attitude_status(
                self.test_class.pitch_maneuver_end - 1),
            RocketAttitudeStatus.PITCH_PROGRAM,
            f"Should be PITCH_PROGRAM at "
            f"{self.test_class.pitch_maneuver_end - 1} s.")
        self.assertEqual(
            self.test_class.get_attitude_status(
                self.test_class.pitch_maneuver_end),
            RocketAttitudeStatus.GRAVITY_ASSIST,
            f"Should be GRAVITY_ASSIST at "
            f"{self.test_class.pitch_maneuver_end} s.")
    

if __name__ == '__main__':
    unittest.main()
