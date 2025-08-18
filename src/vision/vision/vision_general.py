#!/usr/bin/env python3

import cv2
from .vision_constants import (
    CAMERA_TOPIC,
    YOLO_LOCATION,
    CONF_THRESH
)

import rclpy 
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class CameraDetections(Node):
    def __init__(self):
        super().__init__('camera_detections')
        self.bridge = CvBridge()
        # self.yolo_model = YOLO(YOLO_LOCATION)  # Uncomment when YOLO/model is ready
        self.camera_view = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10
        )
        self.image = None
        
    def image_callback(self, data):
        """Callback to receive image from camera"""
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    


def main(args=None):
    rclpy.init(args=args)
    node = CameraDetections()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()