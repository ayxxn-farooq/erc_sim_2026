from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class ManipulationController:

    ARM_JOINTS = [
        'arm_right_1_joint',
        'arm_right_2_joint',
        'arm_right_3_joint',
        'arm_right_4_joint',
        'arm_right_5_joint',
        'arm_right_6_joint',
        'arm_right_7_joint',
    ]

    ARM_LEFT_JOINTS = [
        'arm_left_1_joint',
        'arm_left_2_joint',
        'arm_left_3_joint',
        'arm_left_4_joint',
        'arm_left_5_joint',
        'arm_left_6_joint',
        'arm_left_7_joint',
    ]

    GRIPPER_JOINTS = [
        'gripper_right_finger_joint'
    ]

    TORSO_JOINTS = [
        'torso_lift_joint'
    ]

    HEAD_JOINTS = [
        'head_1_joint',
        'head_2_joint'
    ]

    def __init__(
        self,
        arm_pub,
        arm_left_pub,
        gripper_pub,
        torso_pub,
        head_pub
    ):
        self.arm_pub = arm_pub
        self.arm_left_pub = arm_left_pub
        self.gripper_pub = gripper_pub
        self.torso_pub = torso_pub
        self.head_pub = head_pub

    def send_trajectory(self, publisher, joints, positions, duration=2):
        msg = JointTrajectory()
        msg.joint_names = joints

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = duration

        msg.points = [point]
        publisher.publish(msg)

    def move_head(self, pan, tilt):
        self.send_trajectory(
            self.head_pub,
            self.HEAD_JOINTS,
            [pan, tilt],
            2
        )

    def move_torso_for_row(self, row):
        # Initial calibrated torso heights for the four
        # book-containing shelf rows.
        heights = {
            1: 0.35,
            2: 0.27,
            3: 0.16,
            4: 0.06,
        }

        if row not in heights:
            raise ValueError(
                f'Invalid shelf row: {row}'
            )

        self.move_torso(
            heights[row]
        )

        return heights[row]


    def move_torso(self, height):
        self.send_trajectory(
            self.torso_pub,
            self.TORSO_JOINTS,
            [height],
            2
        )

    def set_gripper(self, position):
        self.send_trajectory(
            self.gripper_pub,
            self.GRIPPER_JOINTS,
            [position],
            1
        )

    def open_gripper(self):
        # Official PAL TIAGo Pro right-gripper open position
        self.set_gripper(0.07)

    def close_gripper(self):
        # Official PAL TIAGo Pro right-gripper closed position
        self.set_gripper(0.0)

    def move_home_both(self):
        # Official PAL TIAGo Pro spherical-wrist home motions.
        # Both arms are moved to their safe/home positions before
        # the base approaches the shelf.

        times = [1, 5, 10]

        left_positions = [
            [
                1.8557,
                -1.5919,
                0.35538,
                -2.0502,
                0.10524,
                -1.5976,
                0.0,
            ],
            [
                0.26,
                -1.6008,
                0.3489,
                -1.9818,
                0.0,
                -1.5829,
                0.0,
            ],
            [
                0.36,
                -1.83,
                0.47,
                -2.35,
                0.0,
                -1.2,
                0.0,
            ],
        ]

        right_positions = [
            [
                -1.8614,
                -1.6008,
                -0.34892,
                -1.9818,
                0.10153,
                -1.5829,
                0.0,
            ],
            [
                -0.26,
                -1.6008,
                -0.3489,
                -1.9818,
                0.0,
                -1.5829,
                0.0,
            ],
            [
                -0.36,
                -1.83,
                -0.47,
                -2.35,
                0.0,
                -1.2,
                0.0,
            ],
        ]

        left_msg = JointTrajectory()
        left_msg.joint_names = self.ARM_LEFT_JOINTS

        right_msg = JointTrajectory()
        right_msg.joint_names = self.ARM_JOINTS

        torso_msg = JointTrajectory()
        torso_msg.joint_names = self.TORSO_JOINTS

        torso_positions = [
            0.17202,
            0.1761,
            0.1761,
        ]

        for positions, seconds in zip(left_positions, times):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = seconds
            left_msg.points.append(point)

        for positions, seconds in zip(right_positions, times):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = seconds
            right_msg.points.append(point)

        for position, seconds in zip(torso_positions, times):
            point = JointTrajectoryPoint()
            point.positions = [position]
            point.time_from_start.sec = seconds
            torso_msg.points.append(point)

        self.arm_left_pub.publish(left_msg)
        self.arm_pub.publish(right_msg)
        self.torso_pub.publish(torso_msg)

    def move_home_right(self):
        """
        Official TIAGo Pro spherical-wrist home_right motion.

        Source:
        tiago_pro_motions_general_spherical-wrist.yaml

        Moves torso and right arm into the PAL-provided safe/home pose
        before approaching the shelf.
        """

        # PAL official home_right motion:
        # torso + seven right-arm joints, three waypoints.

        torso_positions = [
            0.17202,
            0.1761,
            0.1761,
        ]

        arm_positions = [
            [
                -1.8614,
                -1.6008,
                -0.34892,
                -1.9818,
                0.10153,
                -1.5829,
                0.0,
            ],
            [
                -0.26,
                -1.6008,
                -0.3489,
                -1.9818,
                0.0,
                -1.5829,
                0.0,
            ],
            [
                -0.36,
                -1.83,
                -0.47,
                -2.35,
                0.0,
                -1.2,
                0.0,
            ],
        ]

        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.ARM_JOINTS

        torso_msg = JointTrajectory()
        torso_msg.joint_names = self.TORSO_JOINTS

        # Slightly offset from zero so every point has a valid
        # positive time_from_start.
        times = [1, 5, 10]

        for positions, seconds in zip(arm_positions, times):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.sec = seconds
            arm_msg.points.append(point)

        for position, seconds in zip(torso_positions, times):
            point = JointTrajectoryPoint()
            point.positions = [position]
            point.time_from_start.sec = seconds
            torso_msg.points.append(point)

        self.arm_pub.publish(arm_msg)
        self.torso_pub.publish(torso_msg)

    def move_arm_target(self, positions, duration=3):
        if len(positions) != 7:
            raise ValueError(
                'Right arm requires exactly 7 joint positions'
            )

        self.send_trajectory(
            self.arm_pub,
            self.ARM_JOINTS,
            positions,
            duration
        )

    def move_arm(self, positions, duration=3):
        if len(positions) != 7:
            raise ValueError('Right arm requires exactly 7 joint positions')

        self.send_trajectory(
            self.arm_pub,
            self.ARM_JOINTS,
            positions,
            duration
        )
