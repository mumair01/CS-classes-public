'''
This is the main decision making node.
'''
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import speech_recognition as sr
import pyaudio
import wave
import sys 



class MicPerceptNodeNode(Node):

    def __init__(self):
        super().__init__('mic_percept_node')
        # -- Vars.
        self.publish_rate = 0.5
        self.can_read = True  # TODO: Change to False for real data.
        # -- Objects
        # TODO: Determine how to publish audio.
        self.publisher = self.create_publisher(String, "percept/audio", 10)
        self.timer = self.create_timer(self.publish_rate, self._audio_callback)
        self._initialize_mic()
        self.mic = None  # TODO: Remove for real data

    def _initialize_mic(self):
        try:

            self.r = sr.Recognizer()
            self.can_read = True
            self.get_logger().info("Ready to publish audio")
        except:
            self.get_logger().error("Unable to read microohone")

    def _audio_callback(self):
        pass 

        # msg = String()
        # # TODO: Uncomment for real data
        # if self.can_read:
        #     self._record_audio()
            # file = sr.AudioFile('/home/umair/Documents/repos/hri_ros/src/src/ldm_unit/ldm_unit/output.wav')
        
            # with file as source:
            #     # TODO: Need to determine how to send audio over ros2
            #     # self.r.adjust_for_ambient_noise(source)
            #     audio = self.r.record(source)
            #     # audio = self.r.listen(source, timeout = 3)
            #     try:
            #         data = self.r.recognize_google(audio)
            #         msg.data = data 
            #     except sr.UnknownValueError : 
            #         msg.data = ""
            #     self.publisher.publish(msg)
            #     self.get_logger().info(msg.data)

    # def _record_audio(self):
    #     p = pyaudio.PyAudio()
    #     info = p.get_host_api_info_by_index(0)
    #     numdevices = info.get('deviceCount')
    #     #for each audio device, determine if is an input or an output and add it to the appropriate list and dictionary
    #     for i in range (0,numdevices):
    #         if p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels')>0:
    #                 print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name'))

    #         if p.get_device_info_by_host_api_device_index(0,i).get('maxOutputChannels')>0:
    #                 print("Output Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0,i).get('name'))

    #     devinfo = p.get_device_info_by_index(0)
    #     print(devinfo["index"],devinfo["maxInputChannels"])
    #     print("Selected device is ",devinfo.get('name'))
    #     if p.is_format_supported(44100.0,  # Sample rate
    #                             input_device=devinfo["index"],
    #                             input_channels=devinfo['maxInputChannels'],
    #                             input_format=pyaudio.paInt16):
    #         print('Yay!')
    #     p.terminate()

    def _record_audio(self):
        chunk = 16  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 48000  # Record at 44100 samples per second
        seconds = 30
        filename = "/home/umair/Documents/repos/hri_ros/src/src/ldm_unit/ldm_unit/output.wav"
    
        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        for i in range(p.get_device_count()):#list all available audio devices
            dev = p.get_device_info_by_index(i)
            print((i,dev['name'],dev['maxInputChannels']))
            print(p.get_device_count())

        self.get_logger().info("HERE!")

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True, 
                        input_device_index = 0)
        print(stream)
        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        print("Closefd")
        # Terminate the PortAudio interface
        p.terminate()


        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()



def main(args=None):
    rclpy.init(args=args)
    node = MicPerceptNodeNode()
    rclpy.spin(node)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
