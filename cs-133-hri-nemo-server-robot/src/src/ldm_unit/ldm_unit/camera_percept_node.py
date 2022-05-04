'''
This is the main decision making node.
'''
import glob
from typing import List
# Third party imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPerceptNode(Node):

    def __init__(self):
        super().__init__('camera_percept_node')
        # --  Vars.
        self.publish_rate = 0.1  # Seconds
        self.frame_count = 0  # Count of frame being published
        # TODO: The video source need not be hard-coded.
        self.video_source = 0  # This is the video source.
        self.vid = None
    
        # -- Objects
        self.publisher = self.create_publisher(Image, "percept/video", 10)
        self.timer = self.create_timer(self.publish_rate, self._image_callback)
        self.bridge = CvBridge()
        # -- Initalizing
        # TODO: Enable this for actual source
        self._initialize_video_source()


    def _initialize_video_source(self):
        # Get the image from the source - For example, may be a webcam.
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            self.get_logger().error(
                "Unable to capture video device: {}".format(
                    self.video_source))
        else:
            self.get_logger().error(
                "Capture video device: {}".format(self.video_source))

    def _image_callback(self):
        # TODO: Uncomment for real data
        if self.vid == None:
            return
        ret, frame = self.vid.read()
        # ret = True  # TODO: Change for for real data
        # Publish the OpenCV image as a ROS image.
        if ret:

            self.publisher.publish(self.bridge.cv2_to_imgmsg(frame))
            self.get_logger().info(
                    "Published frame with count: {}".format(self.frame_count))
            self.frame_count += 1
            # cv2.imshow('frame',frame)
            # # cv2.waitKey(0)
            # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CameraPerceptNode()
    rclpy.spin(node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
