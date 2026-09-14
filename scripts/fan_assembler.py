#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import cv2

from collections import deque

from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, CompressedImage, Imu, Range
from cv_bridge import CvBridge
from dynamic_reconfigure.server import Server
import tf.transformations as tft

from oculus_sonar.cfg import FanAssemblerConfig
from oculus_sonar.msg import SonarConfig
from sonar_geometry import FOV_DEG, bearing_to_column, beam_bearings, plane_det, plane_inverse, strip_gains, suppress_center_beam

VERTICAL_FOV_DEG = 20.0
GRID_MARGIN = 10
SAT_COUNT = 255
UNOBSERVED = 1

# backwards jumps smaller than this are just out-of-order frames, not a replay restart
TIME_JUMP_TOLERANCE = 1.0

NADIR_MEDIAN_WINDOW = 5
VIEWPOINT_RESET_LEVEL = 1

PUBLISH_BAND_CELLS = 1 << 22

class FanAssembler:
	def __init__(self):
		rospy.init_node("fan_assembler")
		self.bridge = CvBridge()

		self.fixed_frame = rospy.get_param("~fixed_frame", "local")
		self.sonar_frame = rospy.get_param("~sonar_frame", "oculus_stabilized")
		self.cell_size = rospy.get_param("~cell_size", 0.05)
		self.use_compressed = rospy.get_param("~use_compressed", True)
		self.gain_normalise = rospy.get_param("~gain_normalise", False)
		self.image_topic = rospy.get_param("~image_topic", "/oculus_sonar/image")
		self.range_m = rospy.get_param("~default_range", 25.0)

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.half_fov = np.radians(FOV_DEG / 2.0)

		self.sum = None
		self.count = None
		self.grid_origin_x = 0.0
		self.grid_origin_y = 0.0
		self.grid_w = 0
		self.grid_h = 0

		self.do_mapping = True
		self.direction_yaw = None
		self.frame_counter = 0
		self.last_stamp = None
		self.viewpoints = set()

		self.range_window = deque(maxlen=NADIR_MEDIAN_WINDOW)
		self.altitude = None

		self.cfg = None
		self.cfg_server = Server(FanAssemblerConfig, self.reconfig_cb)

		self.pub = rospy.Publisher("/oculus_sonar/stacked_grid", OccupancyGrid, queue_size=1, latch=True)
		self.config_sub = rospy.Subscriber("/oculus_sonar/config", SonarConfig, self.sonar_config_cb, queue_size=1)
		self.imu_sub = rospy.Subscriber("/imu/data", Imu, self.imu_cb, queue_size=1)
		self.echosounder_sub = rospy.Subscriber("/echosounder/range", Range, self.echosounder_cb, queue_size=1)
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

		# the gate keys encode the quantisation, so old entries mean nothing once it changes
		if level & VIEWPOINT_RESET_LEVEL:
			self.viewpoints.clear()

		return config

	def sonar_config_cb(self, msg):
		self.range_m = msg.range

	def direction_cb(self, msg):
		q = msg.pose.orientation
		_, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
		self.direction_yaw = yaw

	def imu_cb(self, msg: Imu):
		self.do_mapping = abs(msg.angular_velocity.z) < self.cfg.max_yaw_rate

	def echosounder_cb(self, msg: Range):
		if not np.isfinite(msg.range) or msg.range < msg.min_range or msg.range > msg.max_range:
			return

		self.range_window.append(msg.range)

		if len(self.range_window) == NADIR_MEDIAN_WINDOW:
			self.altitude = float(np.median(self.range_window))

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

			self.process_frame(strip_gains(img, self.gain_normalise), msg.header.stamp)
		except Exception as e:
			rospy.logwarn_throttle(5.0, f"Frame processing failed: {e}")

		self.image_sub = self.subscribe_image()

	def time_jumped(self, stamp):
		if self.last_stamp is None or stamp >= self.last_stamp - rospy.Duration(TIME_JUMP_TOLERANCE):
			self.last_stamp = stamp
			return False

		# tf2 rejects every transform older than what it already holds, so a replay
		# restart wedges lookups permanently until the buffer is emptied
		rospy.logwarn(f"Time jumped back {(self.last_stamp - stamp).to_sec():.1f} s, clearing TF buffer")
		self.tf_buffer.clear()
		self.last_stamp = None
		self.viewpoints.clear()
		self.range_window.clear()
		self.altitude = None
		return True

	def seen_viewpoint(self, sx, sy, yaw):
		bins = self.cfg.viewpoint_yaw_bins
		cell = self.cfg.viewpoint_cell_m

		sector = int(yaw % (2.0 * np.pi) / (2.0 * np.pi / bins)) % bins
		key = (int(np.floor(sx / cell)), int(np.floor(sy / cell)), sector)

		if key in self.viewpoints:
			return True

		self.viewpoints.add(key)
		return False

	def nadir_cut(self, mat):
		if self.altitude is None:
			return 0.0

		tilt = np.arcsin(np.clip(-mat[2, 0], -1.0, 1.0)) + np.radians(VERTICAL_FOV_DEG / 2.0)
		if tilt <= np.radians(1.0):
			return 0.0

		cut = 0.5 * self.altitude / np.sin(tilt)

		return float(np.clip(cut, 0.0, max(0.0, self.range_m - self.cfg.nadir_margin_m)))

	def process_frame(self, img, stamp):
		if self.time_jumped(stamp):
			return

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

		if self.seen_viewpoint(t.x, t.y, yaw):
			return

		# full rotation, the sonar may be mounted with a substantial tilt
		mat = tft.quaternion_matrix([r.x, r.y, r.z, r.w])
		rot = mat[:2, :2]

		img = suppress_center_beam(img, self.cfg.beam_suppress_width, threshold=self.cfg.beam_suppress_threshold)

		img = cv2.flip(img, 1)  # match the beam order convention of the display reprojector
		intensity, valid = self.polar_frame(img)
		self.integrate(intensity, valid, rot, t.x, t.y, self.nadir_cut(mat))

		self.frame_counter = (self.frame_counter + 1) % self.cfg.publish_every
		if self.frame_counter == 0:
			self.publish(stamp)

	def polar_frame(self, img):
		w = img.shape[1]

		bearing = beam_bearings(w, self.half_fov)

		# the outermost beams have too little array gain to average usefully, and
		# attenuating them would only feed the mean darker samples, so discard them
		edge_ok = self.half_fov - np.abs(bearing) >= np.radians(self.cfg.edge_exclude_deg)

		return img, np.broadcast_to(edge_ok, img.shape).astype(np.uint8) * 255

	def ensure_capacity(self, min_x, max_x, min_y, max_y):
		if self.count is None:
			self.grid_origin_x = (np.floor(min_x / self.cell_size) - GRID_MARGIN) * self.cell_size
			self.grid_origin_y = (np.floor(min_y / self.cell_size) - GRID_MARGIN) * self.cell_size
			self.grid_w = int(np.ceil((max_x - self.grid_origin_x) / self.cell_size)) + GRID_MARGIN
			self.grid_h = int(np.ceil((max_y - self.grid_origin_y) / self.cell_size)) + GRID_MARGIN
			self.sum = np.zeros((self.grid_h, self.grid_w), dtype=np.uint16)
			self.count = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
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

			for name in ("sum", "count"):
				old = getattr(self, name)
				grown = np.zeros((new_h, new_w), dtype=old.dtype)
				grown[pad_down:pad_down + self.grid_h, pad_left:pad_left + self.grid_w] = old
				setattr(self, name, grown)

			self.grid_origin_x -= pad_left * self.cell_size
			self.grid_origin_y -= pad_down * self.cell_size
			self.grid_w = new_w
			self.grid_h = new_h

	def integrate(self, intensity, valid, rot, sx, sy, nadir_cut):
		h, w = intensity.shape
		bin_m = self.range_m / h

		det = plane_det(rot)
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

		a, b = plane_inverse(dx, dy, rot, det)
		r = np.sqrt(a * a + b * b)
		bearing = np.arctan2(b, a)

		inside = (r < self.range_m) & (np.abs(bearing) < self.half_fov)

		rows = r / bin_m - 0.5
		cols = bearing_to_column(bearing, w, self.half_fov)
		rows = np.where(inside, rows, -10.0).astype(np.float32)
		cols = np.where(inside, cols, -10.0).astype(np.float32)

		patch = cv2.remap(intensity, cols, rows, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		cover = cv2.remap(valid, cols, rows, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

		sum_region = self.sum[iy0:iy1, ix0:ix1]
		count_region = self.count[iy0:iy1, ix0:ix1]

		fresh = (patch > 0) & (cover > 0) & (count_region < SAT_COUNT) & ((r >= nadir_cut) | (count_region == 0))

		# uint16 sums can never overflow: capping the count at 255 bounds any cell at 255 * 255
		contrib = patch.astype(np.uint16)
		contrib *= fresh
		sum_region += contrib
		count_region += fresh

	def publish(self, stamp):
		if self.count is None:
			return

		display = np.empty((self.grid_h, self.grid_w), dtype=np.uint8)
		band = max(1, PUBLISH_BAND_CELLS // max(self.grid_w, 1))

		display_min = self.cfg.display_min
		display_max = max(self.cfg.display_max, display_min)

		for y0 in range(0, self.grid_h, band):
			y1 = min(y0 + band, self.grid_h)
			counts = self.count[y0:y1]

			mean = self.sum[y0:y1].astype(np.float32) / np.maximum(counts, 1)
			mean += display_min
			np.clip(mean, display_min, display_max, out=mean)

			np.copyto(display[y0:y1], mean.astype(np.uint8))
			np.copyto(display[y0:y1], np.uint8(UNOBSERVED), where=counts == 0)

		out = OccupancyGrid()
		out.header.stamp = stamp
		out.header.frame_id = self.fixed_frame
		out.info.resolution = self.cell_size
		out.info.width = self.grid_w
		out.info.height = self.grid_h
		out.info.origin.position.x = self.grid_origin_x
		out.info.origin.position.y = self.grid_origin_y
		out.info.origin.orientation.w = 1.0
		out.data = display.view(np.int8).ravel().tolist()
		self.pub.publish(out)

if __name__ == "__main__":
	node = FanAssembler()
	rospy.spin()