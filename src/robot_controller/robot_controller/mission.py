import math
from pathlib import Path

import rclpy
from rclpy.node import Node

from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan, JointState
from std_msgs.msg import Int32
from trajectory_msgs.msg import JointTrajectory

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from robot_controller.navigation import NavigationController
from robot_controller.perception import PerceptionController
from robot_controller.manipulation import ManipulationController
from robot_controller.ik_solver import RightArmIK


class ERCMission(Node):

    def __init__(self):
        super().__init__('erc_mission')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ---------------------------------------------------------
        # Competition arguments
        # ---------------------------------------------------------

        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('book_colour', 'red')

        self.target_column = int(
            self.get_parameter('shelf_column_number').value
        )

        self.target_colour = str(
            self.get_parameter('book_colour').value
        ).lower()

        self.get_logger().info(
            f'Target: shelf {self.target_column}, '
            f'book {self.target_colour}'
        )

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.shelf_column_pub = self.create_publisher(
            Int32,
            '/erc/shelf_column_identification',
            10
        )

        self.shelf_row_pub = self.create_publisher(
            Int32,
            '/erc/shelf_row_identification',
            10
        )

        self.arm_pub = self.create_publisher(
            JointTrajectory,
            '/arm_right_controller/joint_trajectory',
            10
        )

        self.arm_left_pub = self.create_publisher(
            JointTrajectory,
            '/arm_left_controller/joint_trajectory',
            10
        )

        self.gripper_pub = self.create_publisher(
            JointTrajectory,
            '/gripper_right_controller_raw/joint_trajectory',
            10
        )

        self.torso_pub = self.create_publisher(
            JointTrajectory,
            '/torso_controller/joint_trajectory',
            10
        )

        self.head_pub = self.create_publisher(
            JointTrajectory,
            '/head_controller/joint_trajectory',
            10
        )

        # ---------------------------------------------------------
        # Helpers
        # ---------------------------------------------------------

        self.navigation = NavigationController(
            self.cmd_vel_pub
        )

        self.perception = PerceptionController(
            self.get_logger()
        )

        self.manipulation = ManipulationController(
            self.arm_pub,
            self.arm_left_pub,
            self.gripper_pub,
            self.torso_pub,
            self.head_pub
        )

        robot_xml = Path(
            '/opt/erc_ws/robot_description_clean.urdf'
        ).read_text()

        self.right_arm_ik = RightArmIK(
            robot_xml
        )

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self.rgb_callback,
            sensor_qos
        )

        self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/depth/image_rect_raw',
            self.depth_callback,
            sensor_qos
        )

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.create_subscription(
            LaserScan,
            '/scan_front_raw',
            self.scan_callback,
            sensor_qos
        )

        # ---------------------------------------------------------
        # Pose
        # ---------------------------------------------------------

        self.x = None
        self.y = None
        self.yaw = None

        self.front_distance = None

        self.start_x = None
        self.start_y = None

        # ---------------------------------------------------------
        # State machine
        # ---------------------------------------------------------

        self.state = 'WAIT_FOR_SENSORS'

        self.column_published = False
        self.row_published = False

        self.row_torso_positioned = False
        self.row_torso_time = None

        self.book_scan_index = 0
        self.book_scan_time = None
        self.book_scan_started = False

        # Look from high shelf rows down toward low rows.
        self.book_scan_tilts = [
            0.55,
            0.35,
            0.15,
            0.00,
            -0.20,
            -0.40,
        ]

        self.shelf_detection = None
        self.book_detection = None

        self.current_right_arm = None
        self.current_right_gripper = None

        self.pregrasp_solution = None
        self.grasp_solution = None

        self.grasp_state_time = None

        self.book_align_attempts = 0
        self.book_align_time = None
        self.book_align_direction = 0
        self.book_base_x = None
        self.book_base_y = None
        self.book_base_z = None
        self.grasp_prepare_started = False
        self.grasp_prepare_time = None

        self.approach_prepared = False
        self.approach_prepare_time = None

        self.book_search_prepared = False
        self.book_search_prepare_time = None

        self.shelf_approach_start_x = None
        self.shelf_approach_start_y = None

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

    # =============================================================
    # CALLBACKS
    # =============================================================

    def rgb_callback(self, msg):
        self.perception.update_rgb(msg)

    def depth_callback(self, msg):
        self.perception.update_depth(msg)

    def joint_state_callback(self, msg):

        wanted = [
            'arm_right_1_joint',
            'arm_right_2_joint',
            'arm_right_3_joint',
            'arm_right_4_joint',
            'arm_right_5_joint',
            'arm_right_6_joint',
            'arm_right_7_joint',
        ]

        values = dict(
            zip(msg.name, msg.position)
        )

        if not all(
            name in values
            for name in wanted
        ):
            return

        self.current_right_arm = [
            values[name]
            for name in wanted
        ]

        if 'gripper_right_finger_joint' in values:
            self.current_right_gripper = values[
                'gripper_right_finger_joint'
            ]

        if 'gripper_right_finger_joint' in values:
            self.current_right_gripper = values[
                'gripper_right_finger_joint'
            ]


    def scan_callback(self, msg):

        count = len(msg.ranges)

        if count == 0:
            return

        # Scan around straight ahead.
        center = int(
            (0.0 - msg.angle_min)
            / msg.angle_increment
        )

        # Wide window helps detect shelf structure instead of
        # looking through the open shelf gaps.
        window_angle = math.radians(20.0)

        half_window = max(
            1,
            int(
                window_angle
                / abs(msg.angle_increment)
            )
        )

        start_index = max(
            0,
            center - half_window
        )

        end_index = min(
            count,
            center + half_window + 1
        )

        values = []

        for r in msg.ranges[
            start_index:end_index
        ]:

            if (
                math.isfinite(r)
                and r > msg.range_min
                and r < msg.range_max
            ):
                values.append(r)

        if not values:
            self.front_distance = None
            return

        values.sort()

        # Low percentile detects nearby shelf structure even if
        # many laser rays pass through shelf openings.
        index = max(
            0,
            int(len(values) * 0.15)
        )

        self.front_distance = values[index]


    def odom_callback(self, msg):

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        self.x = position.x
        self.y = position.y

        self.yaw = self.quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )

        if self.start_x is None:
            self.start_x = self.x
            self.start_y = self.y

            self.get_logger().info(
                f'Start position saved: '
                f'x={self.start_x:.2f}, '
                f'y={self.start_y:.2f}'
            )

    # =============================================================
    # MAIN LOOP
    # =============================================================

    def control_loop(self):

        # ---------------------------------------------------------
        # Wait until camera + odometry are alive
        # ---------------------------------------------------------

        if self.state == 'WAIT_FOR_SENSORS':

            self.navigation.stop()

            if (
                self.x is not None
                and self.perception.latest_bgr is not None
            ):
                self.get_logger().info(
                    'Sensors ready.'
                )

                self.state = 'SEARCH_SHELF'

            return

        # ---------------------------------------------------------
        # Search for shelf marker
        # ---------------------------------------------------------

        if self.state == 'SEARCH_SHELF':

            found, result = (
                self.perception.detect_target_shelf(
                    self.target_column
                )
            )

            if not found:
                self.navigation.rotate_right(0.08)
                return

            self.shelf_detection = result

            error = self.perception.horizontal_error(
                result
            )

            self.get_logger().info(
                f'Target shelf {self.target_column} detected. '
                f'Horizontal error={error:.2f}, '
                f'match={result["score"]:.2f}'
            )

            # Once the shelf is visible, use the mecanum base to
            # line up laterally with the requested column.
            #
            # Positive image error = target is on right side.
            # Negative image error = target is on left side.

            if error > 0.08:
                self.navigation.strafe_right(0.08)
                return

            if error < -0.08:
                self.navigation.strafe_left(0.08)
                return

            self.navigation.stop()

            # Publish official ERC scoring topic once
            if not self.column_published:
                msg = Int32()
                msg.data = self.target_column
                self.shelf_column_pub.publish(msg)
                self.column_published = True

                self.perception.save_annotated(
                    result['annotated'],
                    f'shelf_column_{self.target_column}'
                )

                self.get_logger().info(
                    f'Published shelf column {self.target_column}'
                )

            # Save odometry reference for approach
            self.shelf_approach_start_x = self.x
            self.shelf_approach_start_y = self.y

            self.state = 'PREPARE_APPROACH'
            return

        # ---------------------------------------------------------
        # Approach the selected shelf column
        # ---------------------------------------------------------

        if self.state == 'PREPARE_APPROACH':

            self.navigation.stop()

            if not self.approach_prepared:

                self.get_logger().info(
                    'Putting both arms into safe home positions before approach.'
                )

                # Official PAL right-arm home motion.
                self.manipulation.move_home_both()

                # Keep the camera generally facing the shelf.
                self.manipulation.move_head(
                    0.0,
                    -0.15
                )

                # Open the gripper before manipulation begins.
                self.manipulation.open_gripper()

                self.approach_prepare_time = (
                    self.get_clock().now()
                )

                self.approach_prepared = True
                return

            elapsed = (
                self.get_clock().now()
                - self.approach_prepare_time
            ).nanoseconds / 1e9

            # PAL's official home motion lasts 10 seconds.
            # Never move the base until the arm motion has completed.
            if elapsed < 10.5:
                return

            self.get_logger().info(
                'Both arms safe. Beginning shelf approach.'
            )

            self.state = 'APPROACH_SHELF'
            return

        if self.state == 'APPROACH_SHELF':

            if self.front_distance is None:
                self.navigation.stop()
                self.get_logger().warn(
                    'Waiting for front LiDAR distance.'
                )
                return

            distance = self.front_distance

            self.get_logger().info(
                f'Front shelf distance: {distance:.2f} m'
            )

            # Stop with enough clearance for perception and arm motion.
            target_distance = 1.20

            if distance > 1.50:
                self.navigation.forward(0.45)
                return

            if distance > 1.20:
                self.navigation.forward(0.25)
                return

            if distance > target_distance:
                self.navigation.forward(0.10)
                return

            self.navigation.stop()

            self.get_logger().info(
                f'Shelf approach complete at {distance:.2f} m'
            )

            self.state = 'PREPARE_BOOK_SEARCH'
            return

        # ---------------------------------------------------------
        # Book search
        # ---------------------------------------------------------

        if self.state == 'PREPARE_BOOK_SEARCH':

            self.navigation.stop()

            if not self.book_search_prepared:

                self.get_logger().info(
                    'Preparing torso and head for book detection.'
                )

                # Raise torso so camera/arm can work across shelf rows.
                self.manipulation.move_torso(0.20)

                # Keep head centered and tilt toward the book rows.
                self.manipulation.move_head(
                    0.0,
                    -0.30
                )

                self.book_search_prepare_time = (
                    self.get_clock().now()
                )

                self.book_search_prepared = True
                return

            elapsed = (
                self.get_clock().now()
                - self.book_search_prepare_time
            ).nanoseconds / 1e9

            # Allow trajectory controllers time to settle.
            if elapsed < 3.0:
                return

            self.get_logger().info(
                'Torso/head ready. Searching for target book.'
            )

            self.state = 'SEARCH_BOOK'
            return

        if self.state == 'SEARCH_BOOK':

            found, result = (
                self.perception.detect_book_colour(
                    self.target_colour
                )
            )

            if not found:

                self.navigation.stop()

                now = self.get_clock().now()

                if not self.book_scan_started:

                    tilt = self.book_scan_tilts[
                        self.book_scan_index
                    ]

                    self.get_logger().info(
                        f'Book not visible. '
                        f'Scanning head tilt={tilt:.2f}'
                    )

                    self.manipulation.move_head(
                        0.0,
                        tilt
                    )

                    self.book_scan_time = now
                    self.book_scan_started = True
                    return

                elapsed = (
                    now - self.book_scan_time
                ).nanoseconds / 1e9

                # Give head enough time to move and camera to update.
                if elapsed < 0.45:
                    return

                self.book_scan_index += 1

                if self.book_scan_index >= len(
                    self.book_scan_tilts
                ):
                    self.book_scan_index = 0

                tilt = self.book_scan_tilts[
                    self.book_scan_index
                ]

                self.get_logger().info(
                    f'Scanning next shelf height: '
                    f'head tilt={tilt:.2f}'
                )

                self.manipulation.move_head(
                    0.0,
                    tilt
                )

                self.book_scan_time = now
                return

            self.book_detection = result

            # The detector is already restricted to the selected
            # shelf-column ROI. Fine-align the mecanum base very
            # slowly so the target book is almost exactly centered
            # before any arm motion.

            error = self.perception.horizontal_error(
                result
            )

            self.get_logger().info(
                f'Target book image error={error:.3f}'
            )

            # Shelf column was already centered.
            # Never chase the book sideways: each column contains
            # the same colours and doing so can move us to a neighbour.
            self.navigation.stop()

            row = result['row']

            if not self.row_published:
                msg = Int32()
                msg.data = row

                self.shelf_row_pub.publish(msg)

                self.row_published = True

                self.perception.save_annotated(
                    result['annotated'],
                    f'book_{self.target_colour}_row_{row}'
                )

                self.get_logger().info(
                    f'Published row {row}'
                )

            if not self.row_torso_positioned:

                height = (
                    self.manipulation.move_torso_for_row(
                        row
                    )
                )

                self.get_logger().info(
                    f'Moving torso for row {row}: '
                    f'{height:.2f} m'
                )

                self.row_torso_time = (
                    self.get_clock().now()
                )

                self.row_torso_positioned = True
                self.state = 'WAIT_ROW_TORSO'
                return

            self.state = 'BOOK_CENTERED'
            return

        # ---------------------------------------------------------
        # Stop before manipulation
        # ---------------------------------------------------------

        if self.state == 'WAIT_ROW_TORSO':

            self.navigation.stop()

            elapsed = (
                self.get_clock().now()
                - self.row_torso_time
            ).nanoseconds / 1e9

            if elapsed < 3.0:
                return

            self.get_logger().info(
                'Torso reached target row height. '
                'Re-detecting book.'
            )

            # Previous image/depth is stale because camera moved.
            self.book_detection = None

            self.state = 'SEARCH_BOOK'
            return


        if self.state == 'BOOK_CENTERED':

            self.navigation.stop()

            if self.book_detection is None:
                return

            depth = self.book_detection['depth']

            if depth is None:
                self.get_logger().warn(
                    'No valid target-book depth.'
                )
                self.state = 'SEARCH_BOOK'
                return

            u = self.book_detection['center_x']
            v = self.book_detection['center_y']

            p_cam = self.perception.pixel_to_camera_xyz(
                u,
                v,
                depth
            )

            if p_cam is None:
                self.get_logger().warn(
                    'Could not calculate book camera position.'
                )
                return

            self.get_logger().info(
                'Book camera measurement: '
                f'u={u}, v={v}, depth={depth:.3f} m, '
                f'xyz=({p_cam["x"]:.3f}, '
                f'{p_cam["y"]:.3f}, '
                f'{p_cam["z"]:.3f})'
            )

            p_torso = self.camera_point_to_torso(
                p_cam
            )

            if p_torso is None:
                return

            self.book_torso_x = p_torso['x']
            self.book_torso_y = p_torso['y']
            self.book_torso_z = p_torso['z']

            self.get_logger().info(
                'LIVE book position in torso_lift_link: '
                f'x={self.book_torso_x:.3f}, '
                f'y={self.book_torso_y:.3f}, '
                f'z={self.book_torso_z:.3f}'
            )

            # Reject detections that are obviously outside the
            # centered target shelf column.
            if abs(self.book_torso_y) > 0.32:
                self.get_logger().warn(
                    f'Rejecting book outside target column: '
                    f'y={self.book_torso_y:.3f}'
                )
                self.book_detection = None
                self.state = 'SEARCH_BOOK'
                return

            # Align the book with the right shoulder/arm rather
            # than forcing the arm to reach diagonally.
            desired_book_y = -0.16
            lateral_error = (
                self.book_torso_y - desired_book_y
            )

            if (
                abs(lateral_error) > 0.045
                and self.book_align_attempts < 20
            ):
                self.get_logger().info(
                    f'Fine-aligning book to right arm: '
                    f'book_y={self.book_torso_y:.3f}, '
                    f'target_y={desired_book_y:.3f}'
                )

                # Positive relative Y = book is left of desired arm line.
                if lateral_error > 0:
                    self.book_align_direction = 1
                else:
                    self.book_align_direction = -1

                self.book_align_time = (
                    self.get_clock().now()
                )

                self.book_align_attempts += 1
                self.state = 'ALIGN_BOOK_TO_ARM'
                return

            if self.current_right_arm is None:
                self.get_logger().warn(
                    'Waiting for current right-arm joints.'
                )
                return

            # Test whether the arm can reach a safe pre-grasp
            # point 18 cm in front of the detected book.
            test_pre_x = self.book_torso_x - 0.18

            test_solution = self.right_arm_ik.solve(
                test_pre_x,
                self.book_torso_y,
                self.book_torso_z,
                self.current_right_arm,
                orientation_weight=0.0
            )

            if test_solution is not None:

                self.get_logger().info(
                    'Book is inside reachable pre-grasp workspace.'
                )

                self.state = 'GRASP_READY'

            else:

                self.get_logger().info(
                    f'Pre-grasp IK unreachable at '
                    f'x={test_pre_x:.3f}. '
                    'Moving base closer.'
                )

                self.state = 'MOVE_TO_GRASP_RANGE'

            return

        if self.state == 'PREPARE_GRASP':

            self.navigation.stop()

            if not self.grasp_prepare_started:

                self.get_logger().info(
                    'Preparing robot for grasp.'
                )

                # Raise torso further for high shelf rows.
                #
                # Clamp inside TIAGo torso travel.
                target_torso = min(
                    0.35,
                    max(
                        0.10,
                        self.book_base_z - 1.20
                    )
                )

                self.get_logger().info(
                    f'Setting torso to {target_torso:.3f} m'
                )

                self.manipulation.move_torso(
                    target_torso
                )

                # Keep right gripper open.
                self.manipulation.open_gripper()

                self.grasp_prepare_time = (
                    self.get_clock().now()
                )

                self.grasp_prepare_started = True
                return

            elapsed = (
                self.get_clock().now()
                - self.grasp_prepare_time
            ).nanoseconds / 1e9

            if elapsed < 3.0:
                return

            self.get_logger().info(
                'Torso ready for grasp planning.'
            )

            self.state = 'GRASP_READY'
            return

        if self.state == 'MOVE_TO_GRASP_RANGE':

            # Keep both arms tucked while creeping closer.
            # Never continue if LiDAR data disappears.
            if self.front_distance is None:
                self.navigation.stop()
                return

            self.get_logger().info(
                f'Grasp approach distance: '
                f'{self.front_distance:.2f} m'
            )

            # Stop before getting dangerously close to shelf.
            grasp_base_distance = 0.75

            if self.front_distance > grasp_base_distance:

                # Slow controlled approach only.
                self.navigation.forward(0.05)
                return

            self.navigation.stop()

            self.get_logger().info(
                f'Grasp staging distance reached: '
                f'{self.front_distance:.2f} m'
            )

            # IMPORTANT:
            # Previous book coordinates are now stale because
            # the base moved. Detect the target again.
            self.book_detection = None

            self.state = 'SEARCH_BOOK'
            return

        if self.state == 'ALIGN_BOOK_TO_ARM':

            elapsed = (
                self.get_clock().now()
                - self.book_align_time
            ).nanoseconds / 1e9

            # Short 0.18-second mecanum correction.
            if elapsed < 0.18:

                if self.book_align_direction > 0:
                    self.navigation.strafe_left(0.05)
                else:
                    self.navigation.strafe_right(0.05)

                return

            self.navigation.stop()

            # Base moved, so all old book XYZ data is stale.
            self.book_detection = None
            self.state = 'SEARCH_BOOK'
            return


        if self.state == 'GRASP_READY':

            self.navigation.stop()

            if self.current_right_arm is None:
                return

            bx = self.book_torso_x
            by = self.book_torso_y
            bz = self.book_torso_z

            self.get_logger().info(
                'Planning book pickup at '
                f'x={bx:.3f}, '
                f'y={by:.3f}, '
                f'z={bz:.3f}'
            )

            pre_x = bx - 0.12

            self.pregrasp_solution = (
                self.right_arm_ik.solve(
                    pre_x,
                    by,
                    bz,
                    self.current_right_arm,
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    orientation_weight=1.0
                )
            )

            if self.pregrasp_solution is None:
                self.get_logger().error(
                    'No IK solution for pre-grasp pose.'
                )
                self.state = 'GRASP_FAILED'
                return

            # FIRST open the hand.
            self.get_logger().info(
                'Opening right gripper.'
            )

            self.manipulation.open_gripper()

            self.grasp_state_time = (
                self.get_clock().now()
            )

            self.state = 'WAIT_GRIPPER_OPEN'
            return


        if self.state == 'WAIT_GRIPPER_OPEN':

            self.navigation.stop()

            elapsed = (
                self.get_clock().now()
                - self.grasp_state_time
            ).nanoseconds / 1e9

            if self.current_right_gripper is None:
                return

            if self.current_right_gripper >= 0.065:

                self.get_logger().info(
                    f'Right gripper confirmed open: '
                    f'{self.current_right_gripper:.3f}'
                )

                self.manipulation.move_arm_target(
                    self.pregrasp_solution,
                    duration=4
                )

                self.grasp_state_time = (
                    self.get_clock().now()
                )

                self.state = 'WAIT_PREGRASP'
                return

            if elapsed > 3.0:

                self.get_logger().error(
                    f'Gripper failed to open. '
                    f'Current={self.current_right_gripper:.3f}'
                )

                self.state = 'GRASP_FAILED'

            return


        if self.state == 'WAIT_PREGRASP':

            self.navigation.stop()

            elapsed = (
                self.get_clock().now()
                - self.grasp_state_time
            ).nanoseconds / 1e9

            if elapsed < 2.2:
                return

            self.get_logger().info(
                'Pre-grasp position reached.'
            )

            self.state = 'FINAL_GRASP_APPROACH'
            return


        if self.state == 'FINAL_GRASP_APPROACH':

            self.navigation.stop()

            if self.current_right_arm is None:
                return

            bx = self.book_torso_x
            by = self.book_torso_y
            bz = self.book_torso_z

            # Stop just short of the book face.
            grasp_x = bx + 0.01

            self.grasp_solution = (
                self.right_arm_ik.solve(
                    grasp_x,
                    by,
                    bz,
                    self.current_right_arm,
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    orientation_weight=1.0
                )
            )

            if self.grasp_solution is None:
                self.get_logger().error(
                    'No IK solution for final grasp.'
                )
                self.state = 'GRASP_FAILED'
                return

            self.get_logger().info(
                f'Extending gripper to book: '
                f'x={grasp_x:.3f}'
            )

            self.manipulation.move_arm_target(
                self.grasp_solution,
                duration=1
            )

            self.grasp_state_time = (
                self.get_clock().now()
            )

            self.state = 'WAIT_FINAL_APPROACH'
            return


        if self.state == 'WAIT_FINAL_APPROACH':

            self.navigation.stop()

            elapsed = (
                self.get_clock().now()
                - self.grasp_state_time
            ).nanoseconds / 1e9

            if elapsed < 1.3:
                return

            self.get_logger().info(
                'Gripper at book. Closing gripper.'
            )

            self.manipulation.close_gripper()

            self.grasp_state_time = (
                self.get_clock().now()
            )

            self.state = 'WAIT_GRIPPER_CLOSE'
            return


        if self.state == 'WAIT_GRIPPER_CLOSE':

            self.navigation.stop()

            elapsed = (
                self.get_clock().now()
                - self.grasp_state_time
            ).nanoseconds / 1e9

            if self.current_right_gripper is None:
                return

            # With a book between the fingers, the joint may not
            # reach exactly zero. We mainly wait for closure motion
            # to complete before retracting.
            if (
                self.current_right_gripper <= 0.04
                or elapsed > 2.5
            ):

                self.get_logger().info(
                    f'Gripper closure complete: '
                    f'{self.current_right_gripper:.3f}'
                )

                if self.pregrasp_solution is None:
                    self.state = 'GRASP_FAILED'
                    return

                self.manipulation.move_arm_target(
                    self.pregrasp_solution,
                    duration=3
                )

                self.grasp_state_time = (
                    self.get_clock().now()
                )

                self.state = 'RETRACT_BOOK'
                return

            return


        if self.state == 'RETRACT_BOOK':

            self.navigation.stop()

            elapsed = (
                self.get_clock().now()
                - self.grasp_state_time
            ).nanoseconds / 1e9

            if elapsed < 2.2:
                return

            if self.current_right_gripper is None:
                self.get_logger().warn(
                    'Cannot verify grasp: no gripper joint state.'
                )
                self.state = 'GRASP_FAILED'
                return

            self.get_logger().info(
                f'Post-retract gripper position: '
                f'{self.current_right_gripper:.3f}'
            )

            # Fully closed is 0.0. If the gripper remains clearly
            # open after retracting, something is likely between
            # the fingers.
            if self.current_right_gripper > 0.015:

                self.get_logger().info(
                    'Grasp appears successful: object held.'
                )

                self.state = 'BOOK_GRASPED'

            else:

                self.get_logger().error(
                    'Gripper closed almost fully; book likely missed.'
                )

                self.state = 'GRASP_FAILED'

            return


        if self.state == 'BOOK_GRASPED':

            self.navigation.stop()
            return


        if self.state == 'GRASP_FAILED':

            self.navigation.stop()

            self.get_logger().error(
                'Book grasp failed. Robot stopped safely.'
            )

            self.state = 'STOPPED'
            return


        if self.state == 'STOPPED':

            self.navigation.stop()
            return

    # =============================================================
    # HELPERS
    # =============================================================

    def camera_point_to_torso(self, point):
        """
        Transform XYZ from
        head_front_camera_color_optical_frame
        directly into torso_lift_link using live TF.
        """

        try:
            transform = self.tf_buffer.lookup_transform(
                'torso_lift_link',
                'head_front_camera_color_optical_frame',
                rclpy.time.Time()
            )

        except Exception as e:
            self.get_logger().warn(
                f'Camera->torso TF unavailable: {e}'
            )
            return None

        tr = transform.transform.translation
        q = transform.transform.rotation

        rotation = Rotation.from_quat([
            q.x,
            q.y,
            q.z,
            q.w
        ])

        camera_xyz = [
            point['x'],
            point['y'],
            point['z']
        ]

        rotated = rotation.apply(
            camera_xyz
        )

        return {
            'x': float(rotated[0] + tr.x),
            'y': float(rotated[1] + tr.y),
            'z': float(rotated[2] + tr.z),
        }


    @staticmethod
    def quaternion_to_yaw(x, y, z, w):

        siny_cosp = 2.0 * (
            w * z +
            x * y
        )

        cosy_cosp = 1.0 - 2.0 * (
            y * y +
            z * z
        )

        return math.atan2(
            siny_cosp,
            cosy_cosp
        )


def main(args=None):

    rclpy.init(args=args)

    node = ERCMission()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.navigation.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
