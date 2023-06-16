import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


class DeoxysResetJoints:
    def __init__(self, 
                 interface_cfg="charmander.yml", controller_cfg="joint-position-controller.yml"):
        self.interface_cfg = interface_cfg
        self.controller_cfg = controller_cfg
        self.controller_type = "JOINT_POSITION"
        self.reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

    def run(self):
        robot_interface = FrankaInterface(
            config_root + f"/{self.interface_cfg}", use_visualizer=False
        )
        controller_cfg = YamlConfig(config_root + f"/{self.controller_cfg}").as_easydict()

        self.reset_joint_positions = [
            e + np.clip(np.random.randn() * 0.005, -0.005, 0.005)
            for e in self.reset_joint_positions
        ]
        action = self.reset_joint_positions + [-1.0]

        while True:
            if len(robot_interface._state_buffer) > 0:
                logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
                logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

                if (
                    np.max(
                        np.abs(
                            np.array(robot_interface._state_buffer[-1].q)
                            - np.array(self.reset_joint_positions)
                        )
                    )
                    < 1e-3
                ):
                    break
            robot_interface.control(
                controller_type=self.controller_type,
                action=action,
                controller_cfg=controller_cfg,
            )
        robot_interface.close()


if __name__ == "__main__":
    reset_joint = DeoxysResetJoints()

    reset_joint.run()
