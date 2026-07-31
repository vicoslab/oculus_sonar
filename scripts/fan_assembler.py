#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import cv2

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, CompressedImage, Imu
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
import tf.transformations as tft

from oculus_sonar.cfg import FanAssemblerConfig
from oculus_sonar.msg import SonarConfig

FOV_DEG = 130.0
GRID_MARGIN = 10

# auto exposure percentiles: low bound rejects dropouts, high bound rejects isolated
# hot pixels but must stay high enough to reach sparse targets covering <1% of the image
NORM_PCT_LO = 1.0
NORM_PCT_HI = 99.9

class FanAssembler:
	def __init__(self):
		rospy.init_node("fan_assembler")
		self.bridge = CvBridge()

		self.fixed_frame = rospy.get_param("~fixed_frame", "local")
		self.sonar_frame = rospy.get_param("~sonar_frame", "oculus_link")
		self.cell_size = rospy.get_param("~cell_size", 0.05)
		self.use_compressed = rospy.get_param("~use_compressed", True)
		self.image_topic = rospy.get_param("~image_topic", "/oculus_sonar/image")
		self.range_m = rospy.get_param("~default_range", 25.0)

		self.cfg = None
		self.cfg_server = Server(FanAssemblerConfig, self.reconfig_cb)

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.half_fov = np.radians(FOV_DEG / 2.0)
		self.sin_max = np.sin(self.half_fov)

		self.log_odds = None
		self.grid_origin_x = 0.0
		self.grid_origin_y = 0.0
		self.grid_w = 0
		self.grid_h = 0

		self.do_mapping = True
		self.direction_yaw = None
		self.frame_counter = 0

		self.pub = rospy.Publisher("/oculus_sonar/stacked_grid", OccupancyGrid, queue_size=1, latch=True)
		self.config_sub = rospy.Subscriber("/oculus_sonar/config", SonarConfig, self.sonar_config_cb, queue_size=1)
		self.imu_sub = rospy.Subscriber("/imu/data", Imu, self.imu_cb, queue_size=1)
		self.direction_sub = rospy.Subscriber("/oculus_stacker/direction", PoseStamped, self.direction_cb, queue_size=1)

		self.enabled = False
		self.enabled_sub = rospy.Subscriber("/oculus_stacker/enabled", Bool, self.enabled_cb)
		self.enabled_pub = rospy.Publisher("/oculus_stacker/enabled", Bool, queue_size=1, latch=True)
		self.enabled_pub.publish(self.enabled)

		self.image_sub = self.subscribe_image()

	def subscribe_image(self):
		if self.use_compressed:
			return rospy.Subscriber(self.image_topic + "/compressed", CompressedImage, self.image_cb, queue_size=1)
		return rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1)

	def reconfig_cb(self, config, level):
		self.cfg = config
		return config

	def sonar_config_cb(self, msg):
		self.range_m = msg.range

	def direction_cb(self, msg):
		q = msg.pose.orientation
		_, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
		self.direction_yaw = yaw

	def imu_cb(self, msg: Imu):
		self.do_mapping = abs(msg.angular_velocity.z) < self.cfg.max_yaw_rate

	def enabled_cb(self, msg: Bool):
		if msg.data != self.enabled:
			self.enabled = msg.data
			self.enabled_pub.publish(self.enabled)

	def image_cb(self, msg):
		if not (self.enabled and self.do_mapping):
			return

		self.image_sub.unregister()

		try:
			if self.use_compressed:
				img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
				if img is None:
					raise RuntimeError("failed to decode compressed sonar image")
			else:
				img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

			self.process_frame(img.astype(np.float32), msg.header.stamp)
		except Exception as e:
			rospy.logwarn_throttle(5.0, f"Frame processing failed: {e}")

		self.image_sub = self.subscribe_image()

	def process_frame(self, img, stamp):
		try:
			tf_stamped = self.tf_buffer.lookup_transform(self.fixed_frame, self.sonar_frame, stamp, rospy.Duration(0.2))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
			return

		t = tf_stamped.transform.translation
		r = tf_stamped.transform.rotation
		_, _, yaw = tft.euler_from_quaternion([r.x, r.y, r.z, r.w])

		if self.direction_yaw is not None:
			diff = abs(np.arctan2(np.sin(yaw - self.direction_yaw), np.cos(yaw - self.direction_yaw)))
			if diff > np.radians(self.cfg.direction_gate_deg):
				return

		# full rotation, the sonar may be mounted with a substantial tilt
		rot = tft.quaternion_matrix([r.x, r.y, r.z, r.w])[:2, :2]

		img = cv2.flip(img, 1)  # match the beam order convention of the display reprojector
		evidence = self.polar_evidence(img)
		if evidence is None:
			return

		self.integrate(evidence, rot, t.x, t.y)

		self.frame_counter = (self.frame_counter + 1) % self.cfg.publish_every
		if self.frame_counter == 0:
			self.publish(stamp)

	def polar_evidence(self, img):
		c = self.cfg

		# per-frame auto exposure, percentiles so single hot pixels or dropouts don't rescale everything
		lo, hi = np.percentile(img, (NORM_PCT_LO, NORM_PCT_HI))
		if hi - lo < 1e-3:
			return None
		intensity = np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

		occupied = intensity >= c.t_occ
		evidence = np.zeros_like(intensity)
		evidence[occupied] = c.l_occ * (intensity[occupied] - c.t_occ) / max(1.0 - c.t_occ, 1e-3)
		evidence[intensity <= c.t_free] = -c.l_free

		h, w = intensity.shape
		bin_m = self.range_m / h
		ranges = (np.arange(h) + 0.5) * bin_m

		# acoustic shadow is occlusion, not free space: cells beyond the first strong
		# return in a beam carry no information and must not erase anything
		margin_bins = max(1, int(round(c.occlusion_margin / bin_m)))
		strong = np.maximum.accumulate(occupied, axis=0)
		shadowed = np.zeros_like(occupied)
		shadowed[margin_bins:] = strong[:-margin_bins]
		evidence[shadowed & ~occupied] = 0.0

		# confidence: ramp up over the nadir/fish clutter zone, taper off at far range
		w_range = np.clip(ranges / max(c.near_range_full, 1e-3), 0.0, 1.0)
		far_start = c.far_taper_start * self.range_m
		far = np.clip((ranges - far_start) / max(self.range_m - far_start, 1e-3), 0.0, 1.0)
		w_range *= 1.0 - (1.0 - c.far_taper_floor) * far

		# columns are uniform in sine space across the FOV, taper the weak edge beams
		sin_bearing = (2.0 * np.arange(w) / (w - 1) - 1.0) * self.sin_max
		bearing = np.arcsin(np.clip(sin_bearing, -1.0, 1.0))
		w_bearing = np.clip((self.half_fov - np.abs(bearing)) / np.radians(c.edge_taper_deg), 0.0, 1.0)

		evidence *= w_range[:, None].astype(np.float32) * w_bearing[None, :].astype(np.float32)
		return evidence

	def ensure_capacity(self, min_x, max_x, min_y, max_y):
		if self.log_odds is None:
			self.grid_origin_x = (np.floor(min_x / self.cell_size) - GRID_MARGIN) * self.cell_size
			self.grid_origin_y = (np.floor(min_y / self.cell_size) - GRID_MARGIN) * self.cell_size
			self.grid_w = int(np.ceil((max_x - self.grid_origin_x) / self.cell_size)) + GRID_MARGIN
			self.grid_h = int(np.ceil((max_y - self.grid_origin_y) / self.cell_size)) + GRID_MARGIN
			self.log_odds = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
			return

		ix_min = int(np.floor((min_x - self.grid_origin_x) / self.cell_size))
		ix_max = int(np.ceil((max_x - self.grid_origin_x) / self.cell_size))
		iy_min = int(np.floor((min_y - self.grid_origin_y) / self.cell_size))
		iy_max = int(np.ceil((max_y - self.grid_origin_y) / self.cell_size))

		pad_left = -ix_min + GRID_MARGIN if ix_min < 0 else 0
		pad_right = ix_max - (self.grid_w - 1) + GRID_MARGIN if ix_max > self.grid_w - 1 else 0
		pad_down = -iy_min + GRID_MARGIN if iy_min < 0 else 0
		pad_up = iy_max - (self.grid_h - 1) + GRID_MARGIN if iy_max > self.grid_h - 1 else 0

		if pad_left or pad_right or pad_down or pad_up:
			new_w = self.grid_w + pad_left + pad_right
			new_h = self.grid_h + pad_down + pad_up
			new_data = np.zeros((new_h, new_w), dtype=np.float32)
			new_data[pad_down:pad_down + self.grid_h, pad_left:pad_left + self.grid_w] = self.log_odds
			self.log_odds = new_data
			self.grid_origin_x -= pad_left * self.cell_size
			self.grid_origin_y -= pad_down * self.cell_size
			self.grid_w = new_w
			self.grid_h = new_h

	def integrate(self, evidence, rot, sx, sy):
		h, w = evidence.shape
		bin_m = self.range_m / h

		det = rot[0, 0] * rot[1, 1] - rot[0, 1] * rot[1, 0]
		if abs(det) < 0.2:
			rospy.logwarn_throttle(5.0, "Sonar plane near-vertical, skipping frame")
			return

		# the projected fan is contained in the slant range disc regardless of tilt
		min_x, max_x = sx - self.range_m, sx + self.range_m
		min_y, max_y = sy - self.range_m, sy + self.range_m
		self.ensure_capacity(min_x, max_x, min_y, max_y)

		ix0 = max(0, int((min_x - self.grid_origin_x) / self.cell_size))
		ix1 = min(self.grid_w, int((max_x - self.grid_origin_x) / self.cell_size) + 1)
		iy0 = max(0, int((min_y - self.grid_origin_y) / self.cell_size))
		iy1 = min(self.grid_h, int((max_y - self.grid_origin_y) / self.cell_size) + 1)

		cell_x = self.grid_origin_x + (np.arange(ix0, ix1) + 0.5) * self.cell_size
		cell_y = self.grid_origin_y + (np.arange(iy0, iy1) + 0.5) * self.cell_size
		wx, wy = np.meshgrid(cell_x, cell_y)

		dx = wx - sx
		dy = wy - sy

		# invert the world XY projection of the sonar beam plane: solve
		# rot @ (a, b) = (dx, dy) for in-plane coordinates, where returns are
		# assumed to lie on the plane (centre of the vertical aperture)
		a = (rot[1, 1] * dx - rot[0, 1] * dy) / det
		b = (rot[0, 0] * dy - rot[1, 0] * dx) / det
		r = np.sqrt(a * a + b * b)
		bearing = np.arctan2(b, a)

		inside = (r < self.range_m) & (np.abs(bearing) < self.half_fov)

		rows = r / bin_m - 0.5
		cols = ((np.sin(bearing) / self.sin_max) + 1.0) / 2.0 * (w - 1)
		rows = np.where(inside, rows, -10.0).astype(np.float32)
		cols = np.where(inside, cols, -10.0).astype(np.float32)

		update = cv2.remap(evidence, cols, rows, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

		region = self.log_odds[iy0:iy1, ix0:ix1]
		np.clip(region + update, self.cfg.l_min, self.cfg.l_max, out=region)

	def publish(self, stamp):
		display = np.clip(self.log_odds / self.cfg.l_max, 0.0, 1.0) ** self.cfg.display_gamma
		data = (display * 255.0).astype(np.uint8).astype(np.int8)

		out = OccupancyGrid()
		out.header.stamp = stamp
		out.header.frame_id = self.fixed_frame
		out.info.resolution = self.cell_size
		out.info.width = self.grid_w
		out.info.height = self.grid_h
		out.info.origin.position.x = self.grid_origin_x
		out.info.origin.position.y = self.grid_origin_y
		out.info.origin.orientation.w = 1.0
		out.data = data.ravel().tolist()
		self.pub.publish(out)

if __name__ == "__main__":
	node = FanAssembler()
	rospy.spin()
