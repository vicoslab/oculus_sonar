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

class SonarFanToOccupancyGrid:
	def __init__(self):
		rospy.init_node('fan_reprojector_node')
		self.bridge = CvBridge()
		self.fov_degrees = 130.0
		self.max_range_m = 40.0
		self.downsample_factor = 1.0
		self.sonar_frame = 'oculus_link'

		self.use_compressed = rospy.get_param('~use_compressed', False)
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
		angle_range_rad = np.deg2rad(self.fov_degrees)
		half_fov = angle_range_rad / 2.0
		
		# output is only the bottom half (the fan), so height = h, width = 2*h
		out_h = h
		out_w = 2 * h
		cx = out_w // 2
		cy = out_h - 1  # bottom of the half-canvas maps to range=0

		ys, xs = np.indices((out_h, out_w))
		dx = xs - cx
		dy = cy - ys  # y increases upward in sonar space
		r = np.sqrt(dx**2 + dy**2)
		
		# True real-world geometric angle of this specific pixel on the grid
		angle = np.arctan2(dx, dy)

		# Acoustic arrays are usually uniform in sine-space: sin(theta) = k * index
		# Map physical grid angle back into unrectified sonar sensor space
		sin_max = np.sin(half_fov)
		# Normalize sine value between -1.0 and 1.0 across the FOV
		norm_sin = np.sin(angle) / sin_max
		# Remap linearly to raw image columns (0 to w-1)
		col_map = ((norm_sin + 1.0) / 2.0) * (w - 1)

		edge_noise_cut = 4
		valid = (
			(col_map >= 0) & (col_map < w - edge_noise_cut) &
			(r >= 0) & (r < h) &
			(np.abs(angle) <= half_fov)
		)

		map_x = np.where(valid, col_map, 0).astype(np.float32)
		map_y = np.where(valid, r, 0).astype(np.float32)

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
		h, w = img.shape
		img = cv2.flip(img, 1)

		remap_key = (h, w, self.fov_degrees, self.downsample_factor)
		if remap_key != self._last_remap_key:
			self._rebuild_maps(h, w)

		canvas = cv2.remap(img, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		canvas[self._invalid_mask] = 0

		res_m_per_pixel = self.max_range_m / canvas.shape[0]
		height, width = canvas.shape

		# percentile stretch over fan pixels only, so hot pixels or the zeroed
		# corners outside the fan don't swing the exposure between frames
		lo, hi = np.percentile(canvas[~self._invalid_mask], (1.0, 99.9))
		if hi - lo >= 1.0:
			canvas = np.clip((canvas.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
			canvas[self._invalid_mask] = 0

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
			img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
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

			self._process_image(img, msg.header.stamp)

		except Exception as e:
			rospy.logerr("Error in compressed sonar fan to occupancy grid: %s", str(e))

		self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)


if __name__ == '__main__':
	SonarFanToOccupancyGrid()
	rospy.spin()