'''
This is the main decision making node.
'''
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String
import random
import speech_recognition as sr


class AudioDecisionNode(Node):

    def __init__(self):
        super().__init__('audio_decision_node')
        # -- Vars.
        # -- Objects
        # TODO: Message type should change for audio.
        # NOTE: These callbacks fire every time there is a message.
        self.publisher = self.create_publisher(
            Bool, 'decision/audio', 10)
        self.timer = self.create_timer(2, self._decision_callback)

    def _decision_callback(self):
        """
        Make the decision based on audio and publish the result.
        """
        # Import some scripts to make the decision
        # Publish the decision, which is a bool.
        # NOTE: Randomly sending flag for testing

        # r = sr.Recognizer()
        # with sr.Microphone(device_index=0) as source:
        #     print("Listening...")
        #     r.pause_threshold = 1
        #     r.energy_threshold = 200
        #     audio = r.listen(source)
        # try:
        #     print("Recognizing...")
        #     query = r.recognize_google(audio, language='en-in')
        #     print(f"User Said: {query}\n")
        # except Exception as e:
        #     print(e)
        #     speak("Please Say that Again")
        #     print("Say that Again...")



        self.get_logger().info("audio")
        msg = Bool()
        msg.data = 0.5 > random.random()
        self.publisher.publish(msg)

    


def main(args=None):
    rclpy.init(args=args)
    node = AudioDecisionNode()
    rclpy.spin(node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
