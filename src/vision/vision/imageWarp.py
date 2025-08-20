import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from .vision_constants import (
    CAMERA_TOPIC
)

width = 640
height = 480
objectivePoints = np.float32([[0, height], [0,0], [width, 0], [width, height]])
real_field_coors = [[0,0],
                    [130, 0],
                    [150, 130],
                    [0, 130]]

clicked_points = []
def mouse_callback(event, x, y, _, __):
    """
    Callback function to capture mouse clicks and store the clicked points.
    
    Args:
        event (int): Type of mouse event (e.g., left button click).
        x (int): X-coordinate of the mouse click.
        y (int): Y-coordinate of the mouse click.
        _ (int): Unused parameter.
        __ (int): Unused parameter.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: {x}, {y}")
        clicked_points.append((x, y))

def getHomography(img, realCoor):
    """
    Computes the homography matrix based on user-clicked points and real-world coordinates.
    
    Args:
        cap (cv2.VideoCapture): Video capture object for live feed.
        realCoor (list): List of real-world coordinates corresponding to the clicked points.
    
    Returns:
        ndarray: Homography matrix mapping pixel coordinates to real-world coordinates.
    """
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)

    while(len(clicked_points) < 4):
        cv2.putText(img, f"Click on Point {len(clicked_points) + 1} out of 4", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        for pt in clicked_points:
            cv2.circle(img, pt, 2, (0, 0, 255), -1)
        cv2.imshow("Calibration", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise Exception("Calibration aborted")
    
    cv2.destroyWindow("Calibration")

    pxCoors = np.array(clicked_points, dtype=np.float32)
    realCoors = np.array(realCoor, dtype=np.float32)

    H, _ = cv2.findHomography(pxCoors, realCoors, cv2.RANSAC, 5.0)
    
    return H

class ImageWarpChange(Node):
    def __init__(self):
        super().__init__('image_warp_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10
        )
        self.publisher = self.create_publisher(
            Image, 
            'camera/warped_img',
            10
        )
        self.homography = None

    def image_callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        if self.homography is not None:
            print(f"Matrix {self.homography}")
            warped_img = cv2.warpPerspective(cv_img, self.homography, (640, 480)) #see if it's better to have 640, 480
            cv2.imshow("Warped", warped_img)
            warped_img = self.bridge.cv2_to_imgmsg(warped_img, encoding='bgr8')
            self.publisher.publish(warped_img)
        else:
            self.homography = getHomography(cv_img, real_field_coors)


def main(args=None):
    rclpy.init(args=args)
    node = ImageWarpChange()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
