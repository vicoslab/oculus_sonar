#!/usr/bin/python3
import rospy
import math
import cv2
import numpy as np
import tf2_ros
import tf.transformations as tft
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from dynamic_reconfigure.server import Server
from oculus_sonar.cfg import SonarReprojectorConfig
from oculus_sonar.msg import SonarConfig
from sonar_geometry import FOV_DEG, plane_det, polar_remap, strip_gains

class SonarFanToProjectedImage:
	def __init__(self):
		rospy.init_node('fan_reprojector_image_node')
		self.bridge = CvBridge()
		self.fov_degrees = FOV_DEG
		self.max_range_m = 25.0
		self.downsample_factor = 1.0
		self.sonar_frame = 'oculus_link'

		self.base_frame = rospy.get_param('~base_frame', 'base_link')
		self.meters_per_pixel = rospy.get_param('~meters_per_pixel', 0.01)
		self.edge_noise_cut_px = rospy.get_param('~edge_noise_cut_px', 4)
		self.tilt_quantum_deg = rospy.get_param('~tilt_quantum_deg', 0.5)

		self.use_compressed = rospy.get_param('~use_compressed', True)
		self.gain_normalise = rospy.get_param('~gain_normalise', False)
		self.image_topic = rospy.get_param('~image_topic', '/oculus_sonar/image')
		self.compressed_topic = self.image_topic + "/compressed"

		self.projected_topic = rospy.get_param('~projected_topic', '/oculus_sonar/projected_image')
		self.config_topic = rospy.get_param('~config_topic', '/oculus_sonar/config')

		# cached remap state
		self._map_x = None
		self._map_y = None
		self._invalid_mask = None
		self._last_remap_key = None

		self.reconfig_updated = False

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		if self.use_compressed:
			self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)
		else:
			self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

		self.image_pub = rospy.Publisher(self.projected_topic + "/compressed", CompressedImage, queue_size=1)
		self.config_sub = rospy.Subscriber(self.config_topic, SonarConfig, self.sonar_config_callback, queue_size=1)
		self.server = Server(SonarReprojectorConfig, self.reconfig_callback)

	def sonar_config_callback(self, msg):
		self.max_range_m = msg.range
		self.reconfig_updated = True

	def reconfig_callback(self, config, level):
		self.downsample_factor = config.downsample_factor
		self.sonar_frame = config.sonar_frame
		self.reconfig_updated = True
		return config

	def _lookup_mount_tilt(self):
		try:
			tf_stamped = self.tf_buffer.lookup_transform(self.base_frame, self.sonar_frame, rospy.Time(0), rospy.Duration(1.0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			rospy.logwarn_throttle(5.0, f"TF {self.base_frame} -> {self.sonar_frame} unavailable, retrying: {e}")
			return None

		q = tf_stamped.transform.rotation
		roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
		return pitch, roll

	def _ground_from_plane(self, pitch, roll):
		# orthographic projection of the tilted fan plane onto horizontal ground.
		# with R = Rz*Ry*Rx and yaw dropped, an in-plane point (fwd, cross) lands at
		# (fwd*cos(pitch) + cross*sin(pitch)*sin(roll), cross*cos(roll)), so roll both
		# foreshortens the cross axis and shears forward
		cp = math.cos(pitch)
		sp = math.sin(pitch)
		cr = math.cos(roll)
		sr = math.sin(roll)
		return np.array([[cp, sp * sr], [0.0, cr]], dtype=np.float64)

	def _rebuild_maps(self, h, w, pitch, roll, remap_key):
		scale = self.meters_per_pixel / self.downsample_factor
		half_fov = np.deg2rad(self.fov_degrees) / 2.0
		sin_max = np.sin(half_fov)

		ground = self._ground_from_plane(pitch, roll)
		if abs(plane_det(ground)) < 0.2:
			rospy.logwarn("Sonar mount near-vertical (pitch=%.1f deg), ignoring tilt", np.degrees(pitch))
			ground = np.eye(2)

		# output canvas is deliberately sized to the untilted worst case rather than
		# to the reprojected extent, so the pixel grid is identical at every tilt and
		# the fan visibly retracts toward the bottom instead of the image being
		# resized (which any scale-to-fit viewer would undo). bottom-centre is range 0
		half_w_px = int(np.ceil(self.max_range_m * sin_max / scale))
		out_h = max(1, int(np.ceil(self.max_range_m / scale)))
		out_w = 2 * half_w_px + 1

		self._map_x, self._map_y, valid = polar_remap(h, w, out_h, out_w, half_fov, scale, self.max_range_m / h, ground, self.edge_noise_cut_px)
		self._invalid_mask = ~valid
		self._last_remap_key = remap_key

		self.reconfig_updated = False

	def _process_image(self, img, stamp):
		h, w = img.shape

		tilt = self._lookup_mount_tilt()
		if tilt is None:
			return

		pitch, roll = tilt

		# quantise the tilt so platform attitude noise doesn't rebuild every frame
		tilt_key = (round(math.degrees(pitch) / self.tilt_quantum_deg), round(math.degrees(roll) / self.tilt_quantum_deg))
		remap_key = (h, w, self.fov_degrees, self.downsample_factor, self.max_range_m, self.meters_per_pixel, self.sonar_frame, tilt_key)

		if remap_key != self._last_remap_key or self.reconfig_updated:
			self._rebuild_maps(h, w, pitch, roll, remap_key)

		canvas = cv2.remap(img, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		canvas[self._invalid_mask] = 0

		# percentile stretch over fan pixels only, so hot pixels or the zeroed
		# corners outside the fan don't swing the exposure between frames
		if np.any(~self._invalid_mask):
			lo, hi = np.percentile(canvas[~self._invalid_mask], (1.0, 99.9))
			if hi - lo >= 1.0:
				canvas = np.clip((canvas.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
				canvas[self._invalid_mask] = 0

		msg = self.bridge.cv2_to_compressed_imgmsg(canvas, dst_format='png')
		msg.header.stamp = stamp
		msg.header.frame_id = self.sonar_frame
		self.image_pub.publish(msg)

	def image_callback(self, msg):
		self.image_sub.unregister()
		try:
			img = strip_gains(self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8'), self.gain_normalise)
			self._process_image(img, msg.header.stamp)

		except Exception as e:
			rospy.logerr("Error in sonar fan reprojection: %s", str(e))

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
			rospy.logerr("Error in compressed sonar fan reprojection: %s", str(e))

		self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)


if __name__ == '__main__':
	SonarFanToProjectedImage()
	rospy.spin()