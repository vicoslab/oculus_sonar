#!/usr/bin/python3
import rospy
import math
import cv2
import numpy as np
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, PointField
from oculus_sonar.msg import SonarConfig


class SonarBottomDetector:
	def __init__(self):
		rospy.init_node('sonar_bottom_detector_node')
		self.bridge = CvBridge()

		self.use_compressed = rospy.get_param('~use_compressed', True)
		self.image_topic = rospy.get_param('~image_topic', '/oculus_sonar/image')
		self.compressed_topic = self.image_topic + '/compressed'
		self.config_topic = rospy.get_param('~config_topic', '/oculus_sonar/config')
		self.cloud_topic = rospy.get_param('~cloud_topic', '/oculus_sonar/cloud')

		self.sonar_frame = rospy.get_param('~sonar_frame', 'oculus_link')
		self.max_range_m = rospy.get_param('~initial_max_range', 30.0)

		self.hfov_deg = rospy.get_param('~horizontal_fov_deg', 130.0)
		self.vfov_deg = rospy.get_param('~vertical_fov_deg', 20.0)
		self.elevation_offset_deg = rospy.get_param('~elevation_offset_deg', -10.0)
		self.beam_spacing = rospy.get_param('~beam_spacing', 'sine')
		self.invert_bearing = rospy.get_param('~invert_bearing', True)
		self.range_bias_m = rospy.get_param('~range_bias_m', 0.0)

		self.near_blank = rospy.get_param('~near_blank_bins', 14)
		self.smooth_range = rospy.get_param('~smooth_range_bins', 9)
		self.smooth_beams = rospy.get_param('~smooth_beam_bins', 5)
		self.w_before = rospy.get_param('~window_before_bins', 20)
		self.w_after = rospy.get_param('~window_after_bins', 60)
		self.gate_t = rospy.get_param('~gate_t', 3.0)
		self.gate_ratio = rospy.get_param('~gate_ratio', 1.35)
		self.dp_lambda = rospy.get_param('~dp_lambda', 0.02)
		self.dp_cap = rospy.get_param('~dp_cap', 0.5)
		self.early_bias = rospy.get_param('~early_bias', 0.15)
		self.max_beam_step = rospy.get_param('~max_beam_step', 6.0)
		self.min_anchor = rospy.get_param('~min_anchor_beams', 20)
		self.max_gap = rospy.get_param('~max_gap_beams', 10)
		self.beam_decimation = max(1, int(rospy.get_param('~beam_decimation', 1)))

		self.suppress_width = rospy.get_param('~beam_suppress_width', 10)
		self.suppress_pad = rospy.get_param('~beam_suppress_pad', 2)
		self.suppress_bands = rospy.get_param('~beam_suppress_bands', 8)
		self.suppress_threshold = rospy.get_param('~beam_suppress_threshold', 2.0)

		self._bearings = None
		self._bearing_key = None

		self.cloud_pub = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1)
		self.config_sub = rospy.Subscriber(self.config_topic, SonarConfig, self.sonar_config_callback, queue_size=1)

		if self.use_compressed:
			self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)
		else:
			self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

	def sonar_config_callback(self, msg):
		self.max_range_m = msg.range

	def _suppress_center_beam(self, f):
		if self.suppress_width <= 0:
			return f

		h, w = f.shape
		c = w // 2
		half = int(self.suppress_width)
		pad = int(self.suppress_pad)

		lo = max(c - half, 0)
		hi = min(c + half + 1, w)
		rlo = max(c - half * (1 + pad), 0)
		rhi = min(c + half * (1 + pad) + 1, w)

		win = f[:, rlo:rhi]
		il = lo - rlo
		ir = hi - rlo

		nbands = max(1, min(int(self.suppress_bands), h // 8))
		edges = np.linspace(0, h, nbands + 1).astype(int)
		prof = np.stack([np.median(win[a:b], axis=0) for a, b in zip(edges[:-1], edges[1:])])

		ref = np.median(np.concatenate([prof[:, :il], prof[:, ir:]], axis=1), axis=1, keepdims=True)
		excess = prof - ref
		excess[:, :il] = 0.0
		excess[:, ir:] = 0.0
		excess[excess < self.suppress_threshold] = 0.0

		if nbands > 1:
			excess = cv2.resize(excess, (excess.shape[1], h), interpolation=cv2.INTER_LINEAR)
		else:
			excess = np.repeat(excess, h, axis=0)

		out = f.copy()
		out[:, rlo:rhi] = np.clip(win - excess, 0, 255)
		return out

	def _prepare(self, img):
		f = self._suppress_center_beam(img.astype(np.float32))
		f = cv2.boxFilter(f, -1, (self.smooth_beams, self.smooth_range), normalize=True, borderType=cv2.BORDER_REPLICATE)
		f[:self.near_blank] = 0.0
		return f

	def _step_stats(self, s):
		rows, beams = s.shape
		wb = self.w_before
		wa = self.w_after

		c1 = np.vstack([np.zeros((1, beams)), np.cumsum(s, axis=0)])
		c2 = np.vstack([np.zeros((1, beams)), np.cumsum(s.astype(np.float64) ** 2, axis=0)])

		t = np.zeros((rows, beams), np.float32)
		ratio = np.zeros((rows, beams), np.float32)

		lo = self.near_blank + wb
		hi = rows - wa
		if hi <= lo:
			return t, ratio

		r = np.arange(lo, hi)
		m0 = (c1[r] - c1[r - wb]) / float(wb)
		v0 = np.maximum((c2[r] - c2[r - wb]) / float(wb) - m0 ** 2, 1e-6)
		m1 = (c1[r + wa] - c1[r]) / float(wa)
		v1 = np.maximum((c2[r + wa] - c2[r]) / float(wa) - m1 ** 2, 1e-6)

		t[lo:hi] = (m1 - m0) / np.sqrt(v0 / wb + v1 / wa)
		ratio[lo:hi] = m1 / np.maximum(m0, 1e-3)
		return t, ratio

	def _lower_envelope(self, d, lam):
		n = d.shape[0]
		i = np.arange(n, dtype=np.float32)
		fwd = lam * i + np.minimum.accumulate(d - lam * i)
		rev = -lam * i + np.minimum.accumulate((d + lam * i)[::-1])[::-1]
		return np.minimum(fwd, rev)

	def _track(self, score):
		rows, beams = score.shape
		lam = self.dp_lambda
		cap = self.dp_cap
		cost = -score

		dp = np.empty((rows, beams), np.float32)
		dp[:, 0] = cost[:, 0]
		for b in range(1, beams):
			prev = dp[:, b - 1]
			dp[:, b] = np.minimum(self._lower_envelope(prev, lam), prev.min() + cap) + cost[:, b]

		path = np.empty(beams, np.int32)
		path[-1] = int(np.argmin(dp[:, -1]))
		idx = np.arange(rows)
		for b in range(beams - 1, 0, -1):
			path[b - 1] = int(np.argmin(dp[:, b - 1] + np.minimum(lam * np.abs(idx - path[b]), cap)))
		return path

	def _grow(self, path, ok):
		beams = path.size
		step = self.max_beam_step
		runs = []

		i = 0
		while i < beams:
			if not ok[i]:
				i += 1
				continue
			j = i + 1
			while j < beams and ok[j] and abs(int(path[j]) - int(path[j - 1])) <= step:
				j += 1
			runs.append((j - i, i, j))
			i = j

		keep = np.zeros(beams, bool)
		if not runs:
			return keep

		n, a, z = max(runs)
		if n < self.min_anchor:
			return keep

		keep[a:z] = True

		last = z - 1
		for b in range(z, beams):
			gap = b - last
			if gap > self.max_gap:
				break
			if ok[b] and abs(int(path[b]) - int(path[last])) <= step * gap:
				keep[b] = True
				last = b

		last = a
		for b in range(a - 1, -1, -1):
			gap = last - b
			if gap > self.max_gap:
				break
			if ok[b] and abs(int(path[b]) - int(path[last])) <= step * gap:
				keep[b] = True
				last = b

		return keep

	def _detect_bottom(self, img):
		s = self._prepare(img)
		t, ratio = self._step_stats(s)

		ok = (t > self.gate_t) & (ratio > self.gate_ratio)
		score = np.where(ok, t, 0.0)

		peak = np.percentile(score, 99.9)
		if peak <= 0.0:
			return None, None
		score = score / peak

		rows = s.shape[0]
		score = score - self.early_bias * (np.arange(rows, dtype=np.float32) / rows)[:, None] * (score > 0)

		path = self._track(score)
		hit = ok[path, np.arange(s.shape[1])]
		keep = self._grow(path, hit)

		intensity = s[path, np.arange(s.shape[1])]
		return np.where(keep, path.astype(np.float32), np.nan), intensity

	def _bearings_for(self, beams):
		key = (beams, self.hfov_deg, self.beam_spacing, self.invert_bearing)
		if key == self._bearing_key:
			return self._bearings

		u = 2.0 * np.arange(beams, dtype=np.float64) / max(1, beams - 1) - 1.0
		half = math.radians(self.hfov_deg) / 2.0

		if self.beam_spacing == 'angle':
			psi = u * half
		else:
			psi = np.arcsin(np.clip(u * math.sin(half), -1.0, 1.0))

		if self.invert_bearing:
			psi = -psi

		self._bearings = psi
		self._bearing_key = key
		return psi

	def _make_cloud(self, det, intensity, rows, stamp):
		beams = det.size
		psi = self._bearings_for(beams)
		eps = math.radians(self.elevation_offset_deg)

		sel = np.isfinite(det)
		if self.beam_decimation > 1:
			mask = np.zeros(beams, bool)
			mask[::self.beam_decimation] = True
			sel = sel & mask

		if not sel.any():
			return None

		bins = det[sel]
		rng = (bins + 0.5) * (self.max_range_m / float(rows)) + self.range_bias_m
		ang = psi[sel]

		x = rng * math.cos(eps) * np.cos(ang)
		y = rng * math.cos(eps) * np.sin(ang)
		z = rng * math.sin(eps) * np.ones_like(rng)
		inten = intensity[sel]

		pts = np.stack([x, y, z, inten], axis=1).astype(np.float32)

		msg = PointCloud2()
		msg.header.stamp = stamp
		msg.header.frame_id = self.sonar_frame
		msg.height = 1
		msg.width = pts.shape[0]
		msg.fields = [
			PointField('x', 0, PointField.FLOAT32, 1),
			PointField('y', 4, PointField.FLOAT32, 1),
			PointField('z', 8, PointField.FLOAT32, 1),
			PointField('intensity', 12, PointField.FLOAT32, 1),
		]
		msg.is_bigendian = False
		msg.point_step = 16
		msg.row_step = msg.point_step * msg.width
		msg.is_dense = True
		msg.data = pts.tobytes()
		return msg

	def _process_image(self, img, stamp):
		det, intensity = self._detect_bottom(img)
		if det is None:
			return

		msg = self._make_cloud(det, intensity, img.shape[0], stamp)
		if msg is not None:
			self.cloud_pub.publish(msg)

	def image_callback(self, msg):
		self.image_sub.unregister()
		try:
			img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
			self._process_image(img, msg.header.stamp)

		except Exception as e:
			rospy.logerr("Error in sonar bottom detector: %s", str(e))

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
			rospy.logerr("Error in compressed sonar bottom detector: %s", str(e))

		self.image_sub = rospy.Subscriber(self.compressed_topic, CompressedImage, self.image_callback_compressed, queue_size=1)


if __name__ == '__main__':
	SonarBottomDetector()
	rospy.spin()