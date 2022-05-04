'''
This is the main decision making node.
'''
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import random
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2 
import numpy as np
from fer import FER


class VideoDecisionNode(Node):

	def __init__(self):
		super().__init__('video_decision_node')
		# TODO: Message type should change for audio.
		# NOTE: These callbacks fire every time there is a message.
		self.subscriber = self.create_subscription(
			Image, 'percept/video', self._percept_callback, 10)
		self.publisher = self.create_publisher(
			Bool, 'decision/video', 10)
		self.bridge = CvBridge()
		self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		self.emotion_detector = FER(mtcnn=True)
		self.timer = self.create_timer(2, self._decision_callback)
		self.process_timer = self.create_timer(0.5, self._process_callback)
		self.decision = {
			"frame" : None, 
			"did_find_face" : None, 
			"num_faces" : None, 
			"emotion" : None, 
			"emotion_confidence" : None 
		}

	def _percept_callback(self, msg):
		

		self.frame = self.bridge.imgmsg_to_cv2(msg) 
		try:
			numpy_horizontal_concat = np.concatenate((self.frame, self.decision["frame"]), axis=1)
			cv2.imshow('Live',numpy_horizontal_concat)
			cv2.waitKey(1)
		except:
			self.decision["frame"] = self.frame

	def _process_callback(self):
		try:
			frame = self.frame
			emotion = emotion_confidence = None 
			frame, did_find_face, num_faces = self._detect_face(frame, True)
			if did_find_face:
				frame, emotion, emotion_confidence = self._detect_emotion(frame, True)
			self.decision = {
				"frame" : frame, 
				"did_find_face" : did_find_face, 
				"num_faces" : num_faces, 
				"emotion" : emotion, 
				"emotion_confidence" : emotion_confidence}
		except:
			pass 
			

	def _decision_callback(self):
		"""
		Make the decision based on audio and publish the result.
		"""
		msg = Bool()
		if self.decision["did_find_face"]:
			msg.data = True 
		else:
		
			msg.data = False
			# msg.data = 0.5 > random.random()
		self.publisher.publish(msg)
		self.get_logger().info(
                    "Video decision: {}".format(msg.data))

	def _detect_face(self, frame, draw : bool = False ):
		"""
		Detect a face in  araw image and give new frame back. 
		"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = self.faceCascade.detectMultiScale(
			gray, 1.1,4
		)
		if draw:
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		return frame, len(faces) > 0, len(faces)

	def _detect_emotion(self, frame, draw : bool = False):
		
		res = self.emotion_detector.detect_emotions(frame)
		if len(res) == 0:
			return frame, None, None
		emotion_probs = res[0]['emotions']
		x , y, w, h  = res[0]['box']
		k = max(emotion_probs , key=emotion_probs.get)
		# Draw 
		if draw:
			cv2.putText(frame, '{}: {}'.format(k, emotion_probs[k]), (x,y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
		return frame, k, emotion_probs[k]




def main(args=None):
	rclpy.init(args=args)
	node = VideoDecisionNode()
	rclpy.spin(node)
	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	node.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
