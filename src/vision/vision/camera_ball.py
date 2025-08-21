from .vision_constants import (
    CAMERA_TOPIC
)
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        # ROS utils
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        # Variables de OpenCV
        path = "lut_orange2_generated.npy"
        self.lut = np.load(path)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        self.last_center = None
        self.image = None

    def image_callback(self, msg):
        """Callback cuando llega una imagen de la cámara"""
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.process_image()

    def process_image(self):
        frame = self.image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # bajar brillo
        v = np.clip(v - 40, 0, 255)
        hsv_darker = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv_darker, cv2.COLOR_HSV2BGR)

        # segmentación
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        U = yuv[:, :, 1]
        V = yuv[:, :, 2]
        mask = self.lut[V, U]

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        possible_ellipses = []
        for cnt in contours:
            if len(cnt) >= 4:
                area = cv2.contourArea(cnt)
                if 400 < area < 3000:
                    ellipse = cv2.fitEllipse(cnt)
                    possible_ellipses.append(ellipse)

        chosen_ellipse = None
        if possible_ellipses:
            if self.last_center is None:
                chosen_ellipse = max(possible_ellipses,
                                     key=lambda e: np.pi * (e[1][0]/2) * (e[1][1]/2))
            else:
                def distance(e):
                    center = e[0]
                    return np.linalg.norm(np.array(center) - np.array(self.last_center))
                chosen_ellipse = min(possible_ellipses, key=distance)

        if chosen_ellipse is not None:
            self.last_center = chosen_ellipse[0]
            (x, y) = self.last_center
            self.get_logger().info(f"Ball center: {x}, {y}")

            # Publicar la transformación en tf
            self.publish_tf(x, y)
        else:
            self.get_logger().info("Ball not detected")
            self.last_center = None

    def publish_tf(self, x, y):
        """Publica la posición de la pelota como un frame en tf"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "ball"

        # aquí x,y son pixeles; si tuvieras calibración podrías pasarlo a metros
        t.transform.translation.x = float(x)
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
