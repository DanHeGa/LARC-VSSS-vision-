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
import torch

device = 'cuda' if torch.cuda.is_available() else  'cpu'

colors = {
    "green": np.load("/home/dany/ros2_vision_ws/src/vision/utils/LUTs/lut_green.npy"),
    "blue": np.load("/home/dany/ros2_vision_ws/src/vision/utils/LUTs/lut_blue.npy"),
    "pink": np.load("/home/dany/ros2_vision_ws/src/vision/utils/LUTs/lut_pink.npy"),
    "red": np.load("/home/dany/ros2_vision_ws/src/vision/utils/LUTs/lut_red.npy")
}

kernel_size = 10
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

orange = np.load("/home/dany/ros2_vision_ws/src/vision/utils/LUTs/lut_orange.npy")


class CameraDetections(Node):
    def __init__(self):
        super().__init__('camera_detections')
        self.bridge = CvBridge()
        self.yolo_model = YOLO(YOLO_LOCATION)  # Uncomment when YOLO/model is ready
        self.yolo_model.to(device)
        self.camera_view = self.create_subscription(
            Image, WARPED_VIEW_TOPIC, self.image_callback, 10
        )
        self.model_view = self.create_publisher(
            Image, MODEL_VIEW_TOPIC, 10
        )
        self.image = None
        self.tf_broadcaster = TransformBroadcaster(self)
        self.homography = np.load("homography.npy")
        self.perspectiveMatrix = np.load("persMatrix.npy")
        self.get_logger().info("Starting model node/general vision node")
        self.last_center = None
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
        # print(len(filtered_centers))
        if len(filtered_centers) >= 2:
            (x1, y1, c1, area), (x2, y2, c2, area) = filtered_centers

            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            cv2.line(img, img_center, (mid_x, mid_y), (255, 255, 255), 2)
            cv2.circle(img, (mid_x, mid_y), 6, (255, 255, 255), -1)

            dx = mid_x - img_center[0]
            dy = img_center[1] - mid_y
            angle = math.degrees(math.atan2(dy, dx))
            self.get_logger().info(angle)
            cv2.imshow("Orientacion", img)
            cv2.waitKey(1)

            return angle

    def image_callback(self, data):
        """Callback to receive image from camera"""
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        self.model_use()
        self.ball_detection(self.image)

    def tf_helper(self, id, x, y, roll, pitch, yaw):
        t = TransformStamped()

        qx, qy, qz, qw = euler.euler2quat(roll, pitch, yaw) #roll, pitch, yaw = radians
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "upper_left_corner"
        t.child_frame_id = id
        t.transform.translation.x = float(x) 
        t.transform.translation.y = float(y)
        t.transform.translation.z = 0.0
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


    def image_to_field(self, x_img, y_img, H, perspectiveMatrix=None):
        pt_img = np.array([[[x_img, y_img]]], dtype=np.float32)
        if perspectiveMatrix is not None:
            inverse_perspective = np.linalg.inv(perspectiveMatrix)
            pt_original = cv2.perspectiveTransform(pt_img, inverse_perspective)
        else:
            pt_original = pt_img
        pt_field = cv2.perspectiveTransform(pt_original, H)
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
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)
                    roi = frame[y1:y2, x1:x2]
                    
                    x_center = (x1 + x2) / 2 
                    y_center = (y1 + y2) / 2

                    # Dibuja el bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Opcional: muestra la confianza
                    conf_text = f"{confidence:.2f}"
                    cv2.putText(frame, conf_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (int(x_center), int(y_center)), 4, (0, 0, 255), -1)
                    
                    #Convert to field coordinates - this may be the problem !!
                    x_field, y_field = self.image_to_field(x_center, y_center, self.homography)
                    
                    text = f"({x_field:.1f}, {y_field:.1f})"
                    cv2.putText(frame, text, (int(x_center), int(y_center)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #GET ORIENTATION --------------------------------------------------------
                    angle_degrees = self.orientation(roi)
                    if angle_degrees is not None:
                        yaw = math.radians(angle_degrees)
                    else:
                        yaw = 0.0
                    roll, pitch = 0.0, 0.0
                    #------------------------------------------------------------------------
                    #Send robot transforms
                    x_cm = x_center / 100
                    y_cm = y_center / 100

                    self.tf_helper(f"robot_{id}", x_cm, y_cm, roll, pitch, yaw)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.model_view.publish(msg)

    def ball_detection(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Bajar brillo
        v = np.clip(v - 40, 0, 255)

        hsv_darker = cv2.merge([h, s, v])
        frame = cv2.cvtColor(hsv_darker, cv2.COLOR_HSV2BGR)

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        U = yuv[:, :, 1]
        V = yuv[:, :, 2]

        mask = orange[V, U]

        # Filtrado
        # quita puntitos de ruido
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # rellena agujeros pequeños dentro de la pelota
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        # suavizado pequeño para bordes más lisos
        mask = cv2.medianBlur(mask, 5)
        
        # encontrar la pelota
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        possible_ellipses = []

        for cnt in contours:
            if len(cnt) >= 4: 
                area = cv2.contourArea(cnt)
                if area > 10 and area < 10000:  # se ajusta dependiendo del tamaño esperado
                    ellipse = cv2.fitEllipse(cnt)
                    possible_ellipses.append(ellipse)
                    
        chosen_ellipse = None

        if possible_ellipses:
            if self.last_center is None:
                chosen_ellipse = max(possible_ellipses, key=lambda e: np.pi * (e[1][0] / 2) * (e[1][1] / 2))
            else:
                def distance(e):
                    center = e[0]
                    return np.linalg.norm(np.array(center) - np.array(self.last_center))
                
                chosen_ellipse = min(possible_ellipses, key=distance)
            
        if chosen_ellipse is not None:
            cv2.ellipse(frame, chosen_ellipse, (0, 255, 0), 2)
            self.last_center = chosen_ellipse[0] 
            (x, y) = self.last_center
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            self.get_logger().info("Ball center: " + str(self.last_center))
            real_x, real_y = self.image_to_field(x, y, self.homography)
            self.tf_helper("Ball", real_x, real_y, 0.0, 0.0, 0.0)
        else:
            self.last_center = None
            self.get_logger().info("Ball not detected")
        
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