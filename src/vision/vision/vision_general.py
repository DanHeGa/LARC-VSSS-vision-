#!/usr/bin/env python3

import cv2
from .vision_constants import (
    CAMERA_TOPIC,
    YOLO_LOCATION,
    CONF_THRESH,
    MODEL_VIEW_TOPIC
)

import rclpy 
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations
import math

class CameraDetections(Node):
    def __init__(self):
        super().__init__('camera_detections')
        self.bridge = CvBridge()
        self.yolo_model = YOLO(YOLO_LOCATION)  # Uncomment when YOLO/model is ready
        self.camera_view = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10
        )
        self.model_view = self.create_publisher(
            Image, MODEL_VIEW_TOPIC, 10
        )
        self.image = None
        self.tf_broadcaster = TransformBroadcaster(self)
        #self.timer = self.create_timer(0.1, self.timer_callback)

    def tf_helper(self, robot_id, x, y, roll, pitch, yaw):
        t = TransformStamped()

        qx, qy, qz, qw = tf_transformations.quaternion_from_euler(roll, pitch, yaw) #roll, pitch, yaw = radians
        t.header.stamp = self.get_logger().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = f"robot{robot_id}"
        t.transform.translation.x = x 
        t.transform.translation.y = y
        t.transform.translation.z = 0 #check if true
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

        
    def image_callback(self, data):
        """Callback to receive image from camera"""
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')


    def model_use(self):
        if self.image is None:
            self.get_logger().warn("No image received yet")
            return None
        frame = self.image
        results = self.yolo_model.track(frame, verbose=False, classes=0, show=True, tracker="bytetrack.yaml")
        
        for result in results:
            for box in result.boxes:
                x, y, w, h = [round(i) for i in box.xywh[0].tolist()]
                confidence = box.conf.item

                if confidence > CONF_THRESH:
                    id = box.id

                    #GET ORIENTATION
                    angle_degrees = 180 #DUMMY ANGLE, IMPLEMENT LOGIC FOR ANGLES
                    yaw = math.radians(angle_degrees)
                    roll, pitch = 0.0, 0.0
                    #------------------------------------------------------------------------

                    self.tf_helper(id, x, y, roll, pitch, yaw)

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