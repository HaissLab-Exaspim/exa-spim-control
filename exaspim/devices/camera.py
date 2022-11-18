import numpy
from egrabber import *


class Camera:

	def __init__(self):

		self.gentl = EGenTL() # instantiate egentl
		self.grabber = EGrabber(self.gentl) # instantiate egrabber

	def configure(self, cfg):

		# TODO: we should pass in params individually.
		self.cfg = cfg
		self.grabber.realloc_buffers(self.cfg.ram_buffer) # allocate RAM buffer N frames
		self.grabber.stream.set("UnpackingMode", "Msb") # msb packing of 12-bit data
		self.grabber.remote.set("AcquisitionFrameRate", self.cfg.frame_rate) # set camera exposure fps
		self.grabber.remote.set("ExposureTime", self.cfg.dwell_time*1.0e6) # set exposure time us, i.e. slit width
		if self.grabber.remote.get("TriggerMode") != "On": # set camera to external trigger mode
			self.grabber.remote.set("TriggerMode", "On") 
		self.grabber.remote.set("Gain", self.cfg.digital_gain) # set digital gain to 1

	def start(self, live=False):

		if live:
			self.grabber.start()
		else:
			self.grabber.start(self.cfg.n_frames*self.cfg.n_channels)

	def grab_frame(self):

		buffer = Buffer(self.grabber, timeout = int(1.0e7))
		ptr = buffer.get_info(BUFFER_INFO_BASE, INFO_DATATYPE_PTR) # grab pointer to new frame
		data = ct.cast(ptr, ct.POINTER(ct.c_ubyte*self.cfg.cam_x*self.cfg.cam_y*2)).contents # grab frame data
		image = numpy.frombuffer(data, count=int(self.cfg.cam_x*self.cfg.cam_y), dtype=numpy.uint16).reshape((self.cfg.cam_y,self.cfg.cam_x)) # cast data to numpy array of correct size/datatype, push to numpy buffer
		self.tstamp = buffer.get_info(BUFFER_INFO_TIMESTAMP, INFO_DATATYPE_SIZET) # grab new frame time stamp
		buffer.push()

		return image

	def stop(self):

		self.grabber.stop()

	def print_statistics(self, ch):

		num_frame = self.grabber.stream.get_info(STREAM_INFO_NUM_DELIVERED, INFO_DATATYPE_SIZET)
		num_queued = self.grabber.stream.get_info(STREAM_INFO_NUM_QUEUED, INFO_DATATYPE_SIZET) # number of available frames in ram buffer
		num_dropped = self.grabber.stream.get_info(STREAM_INFO_NUM_UNDERRUN, INFO_DATATYPE_SIZET) # number of underrun, i.e. dropped frames
		data_rate = self.grabber.stream.get('StatisticsDataRate') # stream data rate
		frame_rate = self.grabber.stream.get('StatisticsFrameRate') # stream frame rate

		print(('frame: {}, channel: ' + ch + ', size: {}x{}, time: {:.2f} ms, speed: {:.2f} MB/s, rate: {:.2f} fps, queue: {}/{}, dropped: {}').format(num_frame, self.cfg.cam_x, self.cfg.cam_y, (self.tstamp)/1000.0, data_rate, frame_rate, num_queued, self.cfg.ram_buffer, num_dropped))

