import math
from geometry_msgs.msg import Twist


class NavigationController:

    def __init__(self, publisher):
        self.publisher = publisher

    def stop(self):
        msg = Twist()
        self.publisher.publish(msg)

    def forward(self, speed=0.2):
        msg = Twist()
        msg.linear.x = speed
        self.publisher.publish(msg)

    def backward(self, speed=0.2):
        msg = Twist()
        msg.linear.x = -abs(speed)
        self.publisher.publish(msg)

    def strafe_left(self, speed=0.15):
        msg = Twist()
        msg.linear.y = abs(speed)
        self.publisher.publish(msg)

    def strafe_right(self, speed=0.15):
        msg = Twist()
        msg.linear.y = -abs(speed)
        self.publisher.publish(msg)

    def rotate_left(self, speed=0.15):
        msg = Twist()
        msg.angular.z = abs(speed)
        self.publisher.publish(msg)

    def rotate_right(self, speed=0.15):
        msg = Twist()
        msg.angular.z = -abs(speed)
        self.publisher.publish(msg)

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle
