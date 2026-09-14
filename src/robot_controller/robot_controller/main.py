import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Int32

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class ERCController(Node):

    def __init__(self):
        super().__init__('erc_controller')

        # ---------------------------------------------------------
        # Competition parameters
        # ---------------------------------------------------------

        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter('book_colour', 'red')

        self.target_column = (
            self.get_parameter('shelf_column_number')
            .get_parameter_value()
            .integer_value
        )

        self.target_colour = (
            self.get_parameter('book_colour')
            .get_parameter_value()
            .string_value
        ).lower()

        self.get_logger().info(
            f'Target shelf column: {self.target_column}'
        )

        self.get_logger().info(
            f'Target book colour: {self.target_colour}'
        )

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------

        # TIAGo base
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Required competition scoring topics
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

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.rgb_sub = self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/color/image_raw',
            self.rgb_callback,
            sensor_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/head_front_camera/head_front_camera/depth/image_rect_raw',
            self.depth_callback,
            sensor_qos
        )

        # ---------------------------------------------------------
        # Robot pose
        # ---------------------------------------------------------

        self.x = None
        self.y = None
        self.yaw = None

        self.start_x = None
        self.start_y = None
        self.start_yaw = None

        # ---------------------------------------------------------
        # Camera data
        # ---------------------------------------------------------

        self.latest_rgb = None
        self.latest_depth = None

        # ---------------------------------------------------------
        # Competition state
        # ---------------------------------------------------------

        self.state = 'INITIALISE'

        self.target_shelf_detected = False
        self.target_book_detected = False

        self.detected_column = None
        self.detected_row = None

        self.book_grasped = False
        self.bin_detected = False

        # Used by movement functions
        self.motion_start_x = None
        self.motion_start_y = None
        self.motion_start_yaw = None

        # Prevent repeated score publication
        self.column_published = False
        self.row_published = False

        # Main control loop: 10 Hz
        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

    # =============================================================
    # SENSOR CALLBACKS
    # =============================================================

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

        # Save original start position
        if self.start_x is None:
            self.start_x = self.x
            self.start_y = self.y
            self.start_yaw = self.yaw

            self.get_logger().info(
                f'Start pose recorded: '
                f'x={self.start_x:.2f}, '
                f'y={self.start_y:.2f}, '
                f'yaw={math.degrees(self.start_yaw):.1f} deg'
            )

    def rgb_callback(self, msg):
        self.latest_rgb = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    # =============================================================
    # MAIN STATE MACHINE
    # =============================================================

    def control_loop(self):

        if self.x is None:
            return

        # ---------------------------------------------------------
        # 1. Initialise
        # ---------------------------------------------------------

        if self.state == 'INITIALISE':

            self.stop_robot()

            self.get_logger().info(
                'Robot ready. Beginning Phase 1 solution.'
            )

            self.state = 'SEARCH_SHELF'

        # ---------------------------------------------------------
        # 2. Find requested shelf column using vision
        # ---------------------------------------------------------

        elif self.state == 'SEARCH_SHELF':

            found = self.detect_target_shelf()

            if found:

                self.get_logger().info(
                    f'Target shelf {self.target_column} detected.'
                )

                if not self.column_published:

                    msg = Int32()
                    msg.data = self.target_column

                    self.shelf_column_pub.publish(msg)

                    self.column_published = True

                    self.get_logger().info(
                        'Published shelf column identification.'
                    )

                self.reset_motion_reference()

                self.state = 'NAVIGATE_TO_SHELF'

            else:

                # Slowly rotate while looking for shelf marker
                self.rotate(-0.15)

        # ---------------------------------------------------------
        # 3. Navigate to target shelf
        # ---------------------------------------------------------

        elif self.state == 'NAVIGATE_TO_SHELF':

            #
            # TEMPORARY navigation:
            # move approximately 2.0 m toward shelves.
            #
            # This should eventually be replaced by Nav2 or
            # vision/depth-assisted navigation.
            #

            reached = self.move_forward_distance(
                distance=2.0,
                speed=0.20
            )

            if reached:

                self.stop_robot()

                self.get_logger().info(
                    'Reached shelf approach area.'
                )

                self.state = 'SEARCH_BOOK'

        # ---------------------------------------------------------
        # 4. Find requested book
        # ---------------------------------------------------------

        elif self.state == 'SEARCH_BOOK':

            found, row = self.detect_target_book()

            if found:

                self.detected_row = row

                self.get_logger().info(
                    f'{self.target_colour} book found '
                    f'on row {row}.'
                )

                if not self.row_published:

                    msg = Int32()
                    msg.data = row

                    self.shelf_row_pub.publish(msg)

                    self.row_published = True

                    self.get_logger().info(
                        'Published shelf row identification.'
                    )

                self.state = 'ALIGN_WITH_BOOK'

        # ---------------------------------------------------------
        # 5. Align robot with book
        # ---------------------------------------------------------

        elif self.state == 'ALIGN_WITH_BOOK':

            aligned = self.align_with_book()

            if aligned:

                self.stop_robot()

                self.get_logger().info(
                    'Robot aligned with target book.'
                )

                self.state = 'GRASP_BOOK'

        # ---------------------------------------------------------
        # 6. Pick up book
        # ---------------------------------------------------------

        elif self.state == 'GRASP_BOOK':

            success = self.grasp_book()

            if success:

                self.book_grasped = True

                self.get_logger().info(
                    'Book successfully grasped.'
                )

                self.state = 'RETURN_HOME'

        # ---------------------------------------------------------
        # 7. Return to starting zone
        # ---------------------------------------------------------

        elif self.state == 'RETURN_HOME':

            reached = self.return_to_start()

            if reached:

                self.stop_robot()

                self.get_logger().info(
                    'Returned to Start/End Zone.'
                )

                self.state = 'SEARCH_BIN'

        # ---------------------------------------------------------
        # 8. Detect collection bin
        # ---------------------------------------------------------

        elif self.state == 'SEARCH_BIN':

            found = self.detect_collection_bin()

            if found:

                self.get_logger().info(
                    'Collection bin detected.'
                )

                self.state = 'PLACE_BOOK'

            else:

                self.rotate(0.10)

        # ---------------------------------------------------------
        # 9. Place book
        # ---------------------------------------------------------

        elif self.state == 'PLACE_BOOK':

            success = self.place_book()

            if success:

                self.get_logger().info(
                    'Book placed into collection bin.'
                )

                self.state = 'FINISHED'

        # ---------------------------------------------------------
        # 10. Finished
        # ---------------------------------------------------------

        elif self.state == 'FINISHED':

            self.stop_robot()

    # =============================================================
    # BASE MOVEMENT
    # =============================================================

    def move_forward_distance(self, distance, speed=0.2):

        if self.motion_start_x is None:

            self.motion_start_x = self.x
            self.motion_start_y = self.y

        travelled = math.sqrt(
            (self.x - self.motion_start_x) ** 2 +
            (self.y - self.motion_start_y) ** 2
        )

        if travelled >= distance:

            self.stop_robot()

            self.reset_motion_reference()

            return True

        msg = Twist()

        msg.linear.x = speed
        msg.linear.y = 0.0
        msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)

        return False

    def move_sideways(self, speed):

        msg = Twist()

        msg.linear.x = 0.0
        msg.linear.y = speed
        msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)

    def rotate(self, speed):

        msg = Twist()

        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = speed

        self.cmd_vel_pub.publish(msg)

    def stop_robot(self):

        msg = Twist()

        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.angular.z = 0.0

        self.cmd_vel_pub.publish(msg)

    # =============================================================
    # ODOMETRY MOVEMENT HELPERS
    # =============================================================

    def reset_motion_reference(self):

        self.motion_start_x = None
        self.motion_start_y = None
        self.motion_start_yaw = None

    def return_to_start(self):

        if self.start_x is None:
            return False

        dx = self.start_x - self.x
        dy = self.start_y - self.y

        distance = math.sqrt(
            dx ** 2 +
            dy ** 2
        )

        if distance < 0.15:

            self.stop_robot()

            return True

        desired_yaw = math.atan2(
            dy,
            dx
        )

        yaw_error = self.normalize_angle(
            desired_yaw - self.yaw
        )

        # Turn toward start
        if abs(yaw_error) > 0.15:

            msg = Twist()

            if yaw_error > 0:
                msg.angular.z = 0.20
            else:
                msg.angular.z = -0.20

            self.cmd_vel_pub.publish(msg)

            return False

        # Drive toward start
        msg = Twist()

        msg.linear.x = 0.20

        self.cmd_vel_pub.publish(msg)

        return False

    # =============================================================
    # PERCEPTION
    # =============================================================

    def detect_target_shelf(self):

        """
        TODO:
        Use the RGB camera to detect the numerical marker
        corresponding to self.target_column.

        Competition requirement:
        Shelf numbers are randomized each simulation run,
        therefore their locations must be identified using vision.

        When detected:
            return True

        Otherwise:
            return False
        """

        if self.latest_rgb is None:
            return False

        return False

    def detect_target_book(self):

        """
        TODO:
        Detect the requested book colour using the live camera.

        Books may be:
            red
            blue
            green
            yellow

        Determine which shelf row (1-4) contains the requested
        colour.

        Return:
            (True, row_number)

        Example:
            return True, 3

        If nothing detected:
            return False, None
        """

        if self.latest_rgb is None:
            return False, None

        return False, None

    def align_with_book(self):

        """
        TODO:
        Use RGB + depth data to position the base so the
        chosen arm can reach the target book.

        This function will later use the target book's image
        position and depth.

        Return True once aligned.
        """

        return True

    # =============================================================
    # MANIPULATION
    # =============================================================

    def grasp_book(self):

        """
        TODO:
        Implement MoveIt 2 / trajectory control for ONE arm.

        Required sequence will approximately be:

            1. Set torso height
            2. Move arm to pre-grasp pose
            3. Open gripper
            4. Approach book
            5. Close gripper
            6. Retract arm

        Return True after confirming successful grasp.
        """

        self.get_logger().warn(
            'grasp_book() has not been implemented yet.'
        )

        return False

    # =============================================================
    # COLLECTION BIN
    # =============================================================

    def detect_collection_bin(self):

        """
        TODO:
        Detect the red collection bin from the RGB camera.

        Depth information can then be used to estimate its
        distance.

        Return True once detected.
        """

        if self.latest_rgb is None:
            return False

        return False

    def place_book(self):

        """
        TODO:
        Position the arm above/inside the collection bin,
        open the gripper and release the target book.

        A controlled placement should be preferred over simply
        dropping the book.

        Return True when complete.
        """

        self.get_logger().warn(
            'place_book() has not been implemented yet.'
        )

        return False

    # =============================================================
    # MATH HELPERS
    # =============================================================

    def quaternion_to_yaw(self, x, y, z, w):

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

    def normalize_angle(self, angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle


def main(args=None):

    rclpy.init(args=args)

    node = ERCController()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
