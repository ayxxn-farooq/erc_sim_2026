import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from urdf_parser_py.urdf import URDF


class RightArmIK:

    ARM_JOINTS = [
        'arm_right_1_joint',
        'arm_right_2_joint',
        'arm_right_3_joint',
        'arm_right_4_joint',
        'arm_right_5_joint',
        'arm_right_6_joint',
        'arm_right_7_joint',
    ]

    def __init__(
        self,
        robot_description_xml,
        root_link='torso_lift_link',
        tip_link='gripper_right_grasping_link'
    ):
        self.robot = URDF.from_xml_string(
            robot_description_xml
        )

        self.root_link = root_link
        self.tip_link = tip_link

        self.chain = self._find_chain(
            root_link,
            tip_link
        )

        self.movable = [
            j for j in self.chain
            if j.type in (
                'revolute',
                'continuous',
                'prismatic'
            )
        ]

        self.joint_names = [
            j.name for j in self.movable
        ]

        print('URDF movable joints:')

        for name in self.joint_names:
            print('  ', name)

        if self.joint_names != self.ARM_JOINTS:
            raise RuntimeError(
                '\nUnexpected arm chain.\n'
                f'Expected: {self.ARM_JOINTS}\n'
                f'Got:      {self.joint_names}'
            )

        self.lower = []
        self.upper = []

        for joint in self.movable:

            if joint.type == 'continuous':
                self.lower.append(-math.pi)
                self.upper.append(math.pi)

            else:
                self.lower.append(
                    float(joint.limit.lower)
                )

                self.upper.append(
                    float(joint.limit.upper)
                )

        self.lower = np.array(
            self.lower,
            dtype=float
        )

        self.upper = np.array(
            self.upper,
            dtype=float
        )

    # ============================================================
    # URDF CHAIN
    # ============================================================

    def _find_chain(
        self,
        root_link,
        tip_link
    ):

        child_to_joint = {
            joint.child: joint
            for joint in self.robot.joints
        }

        chain = []

        current = tip_link

        while current != root_link:

            if current not in child_to_joint:
                raise RuntimeError(
                    f'Cannot trace "{current}" '
                    f'back to "{root_link}"'
                )

            joint = child_to_joint[current]

            chain.append(joint)

            current = joint.parent

        chain.reverse()

        return chain

    # ============================================================
    # TRANSFORMS
    # ============================================================

    @staticmethod
    def _origin_matrix(joint):

        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]

        if joint.origin is not None:

            if joint.origin.xyz is not None:
                xyz = joint.origin.xyz

            if joint.origin.rpy is not None:
                rpy = joint.origin.rpy

        T = np.eye(4)

        T[:3, :3] = Rotation.from_euler(
            'xyz',
            rpy
        ).as_matrix()

        T[:3, 3] = np.asarray(
            xyz,
            dtype=float
        )

        return T

    @staticmethod
    def _joint_motion_matrix(
        joint,
        q
    ):

        T = np.eye(4)

        if joint.type == 'fixed':
            return T

        axis = np.asarray(
            joint.axis,
            dtype=float
        )

        norm = np.linalg.norm(axis)

        if norm == 0:
            raise RuntimeError(
                f'Joint {joint.name} has zero axis'
            )

        axis = axis / norm

        if joint.type in (
            'revolute',
            'continuous'
        ):

            T[:3, :3] = (
                Rotation.from_rotvec(
                    axis * float(q)
                ).as_matrix()
            )

        elif joint.type == 'prismatic':

            T[:3, 3] = (
                axis * float(q)
            )

        else:

            raise RuntimeError(
                f'Unsupported joint type: '
                f'{joint.type}'
            )

        return T

    # ============================================================
    # FORWARD KINEMATICS
    # ============================================================

    def forward_matrix(
        self,
        joints
    ):

        if len(joints) != 7:
            raise ValueError(
                'Expected seven right-arm joints'
            )

        joint_values = dict(
            zip(
                self.ARM_JOINTS,
                joints
            )
        )

        T = np.eye(4)

        for joint in self.chain:

            # URDF semantics:
            #
            # parent
            #   -> joint.origin
            #   -> joint motion
            #   -> child
            #
            T = T @ self._origin_matrix(
                joint
            )

            if joint.type != 'fixed':

                q = joint_values[
                    joint.name
                ]

                T = (
                    T
                    @ self._joint_motion_matrix(
                        joint,
                        q
                    )
                )

        return T

    def forward(
        self,
        joints
    ):

        T = self.forward_matrix(
            joints
        )

        quat = Rotation.from_matrix(
            T[:3, :3]
        ).as_quat()

        return {
            'x': float(T[0, 3]),
            'y': float(T[1, 3]),
            'z': float(T[2, 3]),

            'qx': float(quat[0]),
            'qy': float(quat[1]),
            'qz': float(quat[2]),
            'qw': float(quat[3]),
        }

    # ============================================================
    # INVERSE KINEMATICS
    # ============================================================

    def solve(
        self,
        target_x,
        target_y,
        target_z,
        seed,
        orientation=None,
        orientation_weight=0.35
    ):

        seed = np.asarray(
            seed,
            dtype=float
        )

        if seed.shape != (7,):
            raise ValueError(
                'Seed must contain seven joints'
            )

        # Keep seed safely inside optimizer bounds.
        seed = np.clip(
            seed,
            self.lower + 1e-6,
            self.upper - 1e-6
        )

        target_position = np.array([
            target_x,
            target_y,
            target_z
        ], dtype=float)

        if orientation is None:

            seed_T = self.forward_matrix(
                seed
            )

            target_rotation = (
                Rotation.from_matrix(
                    seed_T[:3, :3]
                )
            )

        else:

            target_rotation = (
                Rotation.from_quat(
                    orientation
                )
            )

        def residual(q):

            T = self.forward_matrix(q)

            position = T[:3, 3]

            current_rotation = (
                Rotation.from_matrix(
                    T[:3, :3]
                )
            )

            position_error = (
                position
                - target_position
            )

            rotation_error = (
                target_rotation
                * current_rotation.inv()
            ).as_rotvec()

            return np.concatenate([
                position_error,
                orientation_weight
                * rotation_error
            ])

        result = least_squares(
            residual,
            seed,
            bounds=(
                self.lower,
                self.upper
            ),
            max_nfev=500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10
        )

        if not result.success:
            return None

        final_error = residual(
            result.x
        )

        position_error = np.linalg.norm(
            final_error[:3]
        )

        # Reject bad solutions.
        if position_error > 0.015:
            return None

        return [
            float(x)
            for x in result.x
        ]
