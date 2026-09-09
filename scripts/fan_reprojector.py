#!/usr/bin/python3
import rospy
import math
import cv2
import numpy as np
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from dynamic_reconfigure.server import Server
from oculus_sonar.cfg import SonarReprojectorConfig
from oculus_sonar.msg import SonarConfig
from sonar_geometry import FOV_DEG, polar_remap, strip_gains, suppress_center_beam

class SonarFanToOccupancyGrid:
	def __init__(self):
		rospy.init_node('fan_reprojector_node')
		self.bridge = CvBridge()
		self.fov_degrees = FOV_DEG
		self.max_range_m = 40.0
		self.downsample_factor = 1.0
		self.sonar_frame = 'oculus_link'

		self.use_compressed = rospy.get_param('~use_compressed', False)
		self.gain_normalise = rospy.get_param('~gain_normalise', False)
		self.image_topic = rospy.get_param('~image_topic','/oculus_sonar/image')
		self.compressed_topic = self.image_topic+"/compressed"

		self.grid_topic = rospy.get_param('~grid_topic', '/oculus_sonar/grid')
		self.config_topic = rospy.get_param('~config_topic', '/oculus_sonar/config')

		# cached remap state
		self._map_x = None
		self._map_y = None
		self._invalid_mask = None
		self._last_remap_key = None

		if self.use_compressed:
			self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)
		else:
			self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

		self.grid_pub = rospy.Publisher(self.grid_topic, OccupancyGrid, queue_size=1)
		self.config_sub = rospy.Subscriber(self.config_topic, SonarConfig, self.sonar_config_callback, queue_size=1)
		self.server = Server(SonarReprojectorConfig, self.reconfig_callback)

	def sonar_config_callback(self, msg):
		self.max_range_m = msg.range

	def reconfig_callback(self, config, level):
		self.downsample_factor = config.downsample_factor
		self.sonar_frame = config.sonar_frame
		return config

	def _rebuild_maps(self, h, w):
		half_fov = np.deg2rad(self.fov_degrees) / 2.0

		# output is only the bottom half (the fan), so height = h, width = 2*h
		out_h = h
		out_w = 2 * h

		map_x, map_y, valid = polar_remap(h, w, out_h, out_w, half_fov, 1.0, 1.0)

		if self.downsample_factor != 1.0:
			new_h = int(round(out_h * self.downsample_factor))
			new_w = int(round(out_w * self.downsample_factor))
			map_x = cv2.resize(map_x, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
			map_y = cv2.resize(map_y, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
			valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)

		self._map_x = map_x
		self._map_y = map_y
		self._invalid_mask = ~valid
		self._last_remap_key = (h, w, self.fov_degrees, self.downsample_factor)


	def _process_image(self, img, stamp):
		img = suppress_center_beam(img)
		h, w = img.shape
		img = cv2.flip(img, 1)

		remap_key = (h, w, self.fov_degrees, self.downsample_factor)
		if remap_key != self._last_remap_key:
			self._rebuild_maps(h, w)

		canvas = cv2.remap(img, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		canvas[self._invalid_mask] = 0

		res_m_per_pixel = self.max_range_m / canvas.shape[0]
		height, width = canvas.shape

		grid = OccupancyGrid()
		grid.header.stamp = stamp
		grid.header.frame_id = self.sonar_frame
		grid.info.resolution = res_m_per_pixel
		grid.info.width = width
		grid.info.height = height

		origin_x = width / 2.0 * res_m_per_pixel
		origin_y = -height * res_m_per_pixel
		grid.info.origin.position = Point(x=origin_x, y=origin_y, z=0.0)

		quat = tf.transformations.quaternion_from_euler(0, 0, math.radians(90))
		grid.info.origin.orientation = Quaternion(*quat)

		grid.data = np.asarray(canvas, dtype=np.int8).ravel()

		self.grid_pub.publish(grid)

	def image_callback(self, msg):
		self.image_sub.unregister()
		try:
			img = strip_gains(self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8'), self.gain_normalise)
			self._process_image(img, msg.header.stamp)

		except Exception as e:
			rospy.logerr("Error in sonar fan to occupancy grid: %s", str(e))

		self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

	def image_callback_compressed(self, msg):
		self.image_sub.unregister()
		try:
			np_arr = np.frombuffer(msg.data, np.uint8)
			img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

			if img is None:
				raise RuntimeError("Failed to decode compressed sonar image")

			self._process_image(strip_gains(img, self.gain_normalise), msg.header.stamp)

		except Exception as e:
			rospy.logerr("Error in compressed sonar fan to occupancy grid: %s", str(e))

		self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)


if __name__ == '__main__':
	SonarFanToOccupancyGrid()
	rospy.spin()