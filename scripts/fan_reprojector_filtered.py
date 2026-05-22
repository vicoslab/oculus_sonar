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
from oculus_sonar.cfg import SonarReprojectorFilteredConfig
from oculus_sonar.msg import SonarConfig

class SonarFanToFilteredOccupancyGrid:
	def __init__(self):
		rospy.init_node('sonar_occupancy_node')
		self.bridge = CvBridge()
		self.fov_degrees = 130.0
		self.max_range_m = 40.0
		self.downsample_factor = 1.0
		self.sonar_frame = 'oculus_link'
		self.spatial_median_kernel = 1
		self.temporal_frames = 2
		self.temporal_mode = 'median'

		self.use_compressed = rospy.get_param('~use_compressed', True)
		self.image_topic = rospy.get_param('~image_topic', '/oculus_sonar/image')
		self.compressed_topic = self.image_topic + "/compressed"
		self.grid_topic = rospy.get_param('~grid_topic', '/oculus_sonar/grid_filtered')
		self.config_topic = rospy.get_param('~config_topic', '/oculus_sonar/config')

		self._map_x = None
		self._map_y = None
		self._invalid_mask = None
		self._last_remap_key = None

		self._temporal_buf = None
		self._temporal_idx = 0
		self._temporal_count = 0

		if self.use_compressed:
			self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)
		else:
			self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
		self.grid_pub = rospy.Publisher(self.grid_topic, OccupancyGrid, queue_size=1)
		self.config_sub = rospy.Subscriber(self.config_topic, SonarConfig, self.sonar_config_callback, queue_size=1)
		self.server = Server(SonarReprojectorFilteredConfig, self.reconfig_callback)

	def sonar_config_callback(self, msg):
		self.max_range_m = msg.range

	def reconfig_callback(self, config, level):
		self.downsample_factor = config.downsample_factor
		self.sonar_frame = config.sonar_frame

		kernel = config.spatial_median_kernel
		if kernel > 1 and kernel % 2 == 0:
			kernel += 1
		self.spatial_median_kernel = kernel

		new_n = max(1, config.temporal_frames)
		if new_n != self.temporal_frames:
			self.temporal_frames = new_n
			self._reset_temporal_buffer()

		self.temporal_mode = config.temporal_mode
		return config

	def _reset_temporal_buffer(self):
		self._temporal_buf = None
		self._temporal_idx = 0
		self._temporal_count = 0

	def _rebuild_maps(self, h, w):
		angle_range_rad = np.deg2rad(self.fov_degrees)
		out_h = h
		out_w = 2 * h
		cx = out_w // 2
		cy = out_h - 1

		ys, xs = np.indices((out_h, out_w))
		dx = xs - cx
		dy = cy - ys
		r = np.sqrt(dx**2 + dy**2)
		angle = np.arctan2(dx, dy)

		col_map = ((angle + angle_range_rad / 2.0) / angle_range_rad) * (w - 1)
		edge_noise_cut = 4
		valid = (
			(col_map >= 0) & (col_map < w - edge_noise_cut) &
			(r >= 0) & (r < h) &
			(np.abs(angle) <= angle_range_rad / 2.0)
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
		self._reset_temporal_buffer()

	def _correct_row_dropouts(self, img, window=30, threshold=0.8, min_mean=0.1):
		img = img.astype(np.float32)
		row_means = img.mean(axis=1)  # shape (H,)
		# median filter is robust to clustered dropouts as long as window >> cluster size
		from scipy.ndimage import median_filter
		reference = median_filter(row_means, size=window, mode='reflect')
		ratio = np.where(reference > min_mean, row_means / np.maximum(row_means, 1e-6), 1.0)
		bad = (ratio < threshold) & (row_means > min_mean * 0.1)  # don't try to fix truly dead rows
		scale = np.where(bad, reference / np.maximum(row_means, 1e-6), 1.0)
		img *= scale[:, np.newaxis]
		return np.clip(img, 0, 255).astype(np.uint8)

	def _push_temporal(self, frame):
		n = self.temporal_frames
		h, w = frame.shape

		if self._temporal_buf is None or self._temporal_buf.shape != (n, h, w):
			self._temporal_buf = np.empty((n, h, w), dtype=np.float32)
			self._temporal_idx = 0
			self._temporal_count = 0

		self._temporal_buf[self._temporal_idx] = frame
		self._temporal_idx = (self._temporal_idx + 1) % n
		self._temporal_count = min(self._temporal_count + 1, n)

		filled = self._temporal_buf[:self._temporal_count]
		if self.temporal_mode == 'mean':
			return np.mean(filled, axis=0).astype(np.uint8)
		else:
			return np.median(filled, axis=0).astype(np.uint8)

	def _process_image(self, img, stamp):
		h, w = img.shape
		img = cv2.flip(img, 1)
		img = self._correct_row_dropouts(img) 

		remap_key = (h, w, self.fov_degrees, self.downsample_factor)
		if remap_key != self._last_remap_key:
			self._rebuild_maps(h, w)

		canvas = cv2.remap(img, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		canvas[self._invalid_mask] = 0

		if self.temporal_frames > 1:
			canvas = self._push_temporal(canvas)
		if self.spatial_median_kernel > 1:
			canvas = cv2.medianBlur(canvas, self.spatial_median_kernel)

		res_m_per_pixel = self.max_range_m / canvas.shape[0]
		height, width = canvas.shape

		lo, hi = int(canvas.min()), int(canvas.max())
		if hi > lo:
			cv2.normalize(canvas, canvas, 0, 255, cv2.NORM_MINMAX)

		grid = OccupancyGrid()
		grid.header.stamp = stamp + rospy.Duration(0.5)
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
	SonarFanToFilteredOccupancyGrid()
	rospy.spin()