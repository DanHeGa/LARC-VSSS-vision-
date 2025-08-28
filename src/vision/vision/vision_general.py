#!/usr/bin/env python3

import cv2
from .vision_constants import (
    YOLO_LOCATION,
    CONF_THRESH,
    MODEL_VIEW_TOPIC,
    WARPED_VIEW_TOPIC
)

import rclpy 
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import transforms3d.euler as euler
import math
import numpy as np

colors = {
    "green": np.load("src\vision/utils/LUTs/lut_green.npy"),
    "blue": np.load("src\vision/utils/LUTs/lut_blue.npy"),
    "pink": np.load("src\vision/utils/LUTs/lut_pink.npy"),
    "red": np.load("src\vision/utils/LUTs/lut_red.npy")
}

class CameraDetections(Node):
    def __init__(self):
        super().__init__('camera_detections')
        self.bridge = CvBridge()
        self.yolo_model = YOLO(YOLO_LOCATION)  # Uncomment when YOLO/model is ready
        self.camera_view = self.create_subscription(
            Image, WARPED_VIEW_TOPIC, self.image_callback, 10
        )
        self.model_view = self.create_publisher(
            Image, MODEL_VIEW_TOPIC, 10
        )
        self.image = None
        self.tf_broadcaster = TransformBroadcaster(self)
        self.homography = np.load("homography.npy")
        self.get_logger().info("Starting model node/general vision node")
        #self.timer = self.create_timer(0.1, self.timer_callback)

    def orientation(self, img):
        scale = 6
        img = cv2.resize(img, None, fx=scale, fy=scale)

        h, w, _ = img.shape
        img_center = (w // 2, h // 2)

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        u = img_yuv[:, :, 1]
        v = img_yuv[:, :, 2]

        centers = []

        for color_name, lut in colors.items():
            mask = lut[v, u]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1800:  # ignorar ruido pequeño
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy, color_name, area))
                    
        filtered_centers = sorted(centers, key=lambda x: x[3], reverse=True)[:2]

        angle = None
        print(len(filtered_centers))
        if len(filtered_centers) >= 2:
            (x1, y1, c1, area), (x2, y2, c2, area) = filtered_centers

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            dx = mid_x - img_center[0]
            dy = img_center[1] - mid_y
            angle = math.degrees(math.atan2(dy, dx))

            return angle

    def image_callback(self, data):
        """Callback to receive image from camera"""
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.model_use()

    def tf_helper(self, robot_id, x, y, roll, pitch, yaw):
        t = TransformStamped()

        qx, qy, qz, qw = euler.euler2quat(roll, pitch, yaw) #roll, pitch, yaw = radians
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = f"robot{robot_id}"
        t.transform.translation.x = float(x) 
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0 #check if true
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

    def image_to_field(self, x_img, y_img, H):
        pt_img = np.array([[[x_img, y_img]]])
        pt_field = cv2.perspectiveTransform(pt_img, H)
        x_field, y_field = pt_field[0][0]
        return x_field, y_field

    def model_use(self):
        if self.image is None:
            self.get_logger().warn("No image received yet")
            return None
        frame = self.image.copy()
        results = self.yolo_model(frame, verbose=False, classes=0)
        id_track = -1
        for result in results:
            for box in result.boxes:
                x, y, w, h = [round(i) for i in box.xywh[0].tolist()]
                confidence = box.conf.item()
                if confidence > CONF_THRESH:
                    id = id_track + 1
                    #Robot position ---------------------------------------------------------
                    x_center = (x + w) / 2 #this is a problem
                    y_center = (y + h) / 2

                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    # Dibuja el bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Opcional: muestra la confianza
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (int(x_center), int(y_center)), 8, (0, 0, 255), -1)
                    #Convert to field coordinates - this may be the problem !!
                    x_field, y_field = self.image_to_field(x_center, y_center, self.homography)
                    text = f"({x_field:.1f}, {y_field:.1f})"
                    cv2.putText(frame, text, (int(x_center), int(y_center)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #GET ORIENTATION --------------------------------------------------------
                    angle_degrees = self.orientation(roi)#DUMMY ANGLE, IMPLEMENT LOGIC FOR ANGLES
                    yaw = math.radians(angle_degrees)
                    roll, pitch = 0.0, 0.0
                    #------------------------------------------------------------------------
                    #Send robot transforms
                    self.tf_helper(id, x_center, y_center, roll, pitch, yaw)
        # Mostrar el frame anotado siempre, aunque no haya detecciones
        cv2.imshow("Model", frame)
        cv2.waitKey(1)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.model_view.publish(msg)

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