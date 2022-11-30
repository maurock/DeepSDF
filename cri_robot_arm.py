import numpy as np

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.rl_envs.example_envs.example_arm_env.rest_poses import rest_poses_dict


class CRIRobotArm(Robot):

    def __init__(
            self,
            pb,
            workframe_pos,
            workframe_rpy,
            image_size=[128, 128],
            turn_off_border=True,
            arm_type="ur5",
            t_s_name="tactip",
            t_s_type="standard",
            t_s_core="no_core",
            t_s_dynamics={},
            show_gui=True,
            show_tactile=True,
    ):

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[arm_type][t_s_name][t_s_type]

        # limits
        TCP_lims = np.zeros(shape=(6, 2))
        TCP_lims[0, 0], TCP_lims[0, 1] = -np.inf, +np.inf  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -np.inf, +np.inf  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -np.inf, +np.inf  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = -np.inf, +np.inf  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = -np.inf, +np.inf  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = -np.inf, +np.inf  # yaw lims

        super(CRIRobotArm, self).__init__(
            pb,
            rest_poses,
            workframe_pos,
            workframe_rpy,
            TCP_lims,
            image_size=image_size,
            turn_off_border=turn_off_border,
            arm_type=arm_type,
            t_s_type=t_s_type,
            t_s_core=t_s_core,
            t_s_dynamics=t_s_dynamics,
            show_gui=show_gui,
            show_tactile=show_tactile,
        )

    def move_linear(self, targ_pos, targ_rpy):
        self.arm.tcp_direct_workframe_move(targ_pos, targ_rpy)

        # slow but more realistic moves
        #self.blocking_move(max_steps=5000, constant_vel=0.00025)

        # fast but unrealistic moves (bigger_moves = worse)
        #self.blocking_move(max_steps=1000, constant_vel=None)

        # medium speed
        self.blocking_move(max_steps=5000, constant_vel=0.00075)

    def process_sensor(self):
        """
        Rename to be more like real CRI envs
        """
        return self.get_tactile_observation()
