'''
This is the main decision making node.
'''
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import redis
import serial
import time

class LDMNode(Node):

    def __init__(self):
        super().__init__('ldm_node')
        # -- Vars.
        self.publish_rate = 3  # Seconds
        self.audio_decision = False
        self.video_decision = False
        # -- Objects
        self.subscriber_audio = self.create_subscription(
            Bool, 'decision/audio', self._set_audio_decision_callback, 10)
        self.subscriber_video = self.create_subscription(
            Bool, 'decision/video', self._set_video_decision_callback, 10)
        self.publisher = self.create_publisher(
            Bool, 'ldm_unit/should_serve', 10)
        self.timer = self.create_timer(
            self.publish_rate, self._trigger_callback)
        self.ser = serial.Serial(
            port='/dev/ttyUSB0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
            baudrate = 115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1)
        ## Robot vars. 
        self.previous_decision = True 

    def _set_audio_decision_callback(self, msg):
        self.audio_decision = msg.data

    def _set_video_decision_callback(self, msg):
        self.video_decision = msg.data

    def _trigger_callback(self):
        msg = Bool()
        # decision = self.audio_decision and self.video_decision
        decision = self.video_decision
        msg.data = decision
        #if decision != self.previous_decision:
        self._serial_transmit(self.previous_decision)
        self.previous_decision = decision
        self.publisher.publish(msg)

    def _serial_transmit(self, decision : bool):
        ser = self.ser
        if decision:
    
            ser.write(str.encode("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY\n"))
            # time.sleep(1)
            # ser.write(str.encode("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"))
            # time.sleep(.25)
            # ser.write(str.encode("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"))
            # time.sleep(1)
            # ser.write(str.encode("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"))
            # time.sleep(.25)
            self.get_logger().info(
            "SENDING SHOULD SERVE --------> YES")
        else:

            ser.write(str.encode("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN\n"))
            # time.sleep(1)
            # ser.write(str.encode("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"))
            # time.sleep(.25)
            # ser.write(str.encode("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"))
            # time.sleep(1)
            # ser.write(str.encode("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"))
            # time.sleep(.25)
            self.get_logger().info(
            "SENDING SHOULD SERVE --------> NO")


def main(args=None):
    rclpy.init(args=args)
    node = LDMNode()
    rclpy.spin(node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
