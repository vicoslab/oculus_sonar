#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import cv2

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Range, Imu
from collections import deque
import tf.transformations as tft

CELL_SIZE = 0.10
INTENSITY_THRESHOLD = 90
OVERLAP_GATE = 1.0
NADIR_ANGLE_DEG = 19  # steepest ray of the swath, first to hit bottom
NADIR_TAN = np.tan(np.radians(NADIR_ANGLE_DEG))
ECHO_WINDOW = 10
DIRECTION_THRESHOLD_DEG = 30

class SonarStitcher:
	def __init__(self):
		rospy.init_node("sonar_stitcher")
		self.fixed_frame = rospy.get_param("~fixed_frame", "local")
		self.intensity_threshold = rospy.get_param("~intensity_threshold", INTENSITY_THRESHOLD)
		self.overlap_gate = rospy.get_param("~overlap_gate", OVERLAP_GATE)

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.echo_window = deque(maxlen=ECHO_WINDOW)
		self.nadir_radius = None  # metres, updated from echosounder

		self.do_mapping = True
		self.grid_data = None
		self.grid_origin_x = 0.0
		self.grid_origin_y = 0.0
		self.grid_w = 0
		self.grid_h = 0
		self.direction_yaw = None
		self.pub_prescaler = 0

		self.pub = rospy.Publisher("/oculus_sonar/stacked_grid", OccupancyGrid, queue_size=1)
		self.echosounder_sub = rospy.Subscriber("/echosounder/range", Range, self.echo_cb,  queue_size=1)
		self.cmd_vel_sub = rospy.Subscriber("/imu/data", Imu, self.imu_cb,  queue_size=1)
		self.direction_sub = rospy.Subscriber("/oculus_stacker/direction", PoseStamped, self.direction_cb, queue_size=1)

		self.enabled = False
		self.enabled_sub = rospy.Subscriber("/oculus_stacker/enabled", Bool, self.enabled_callback)
		self.enabled_pub = rospy.Publisher("/oculus_stacker/enabled", Bool, queue_size=1, latch=True)
		self.enabled_pub.publish(self.enabled)

		self.grid_sub = rospy.Subscriber("/oculus_sonar/grid", OccupancyGrid, self.grid_cb, queue_size=1)

	def direction_cb(self, msg: PoseStamped):
		q = msg.pose.orientation
		_, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
		self.direction_yaw = yaw

	def enabled_callback(self, msg: Bool):
		if msg.data != self.enabled:
			self.enabled = msg.data
			self.enabled_pub.publish(self.enabled)

	def imu_cb(self, msg: Twist):
		# discard big turns
		self.do_mapping = abs(msg.angular_velocity.z) < 0.2

	def echo_cb(self, msg: Range):
		self.echo_window.append(msg.range)
		depth = float(np.median(self.echo_window))
		self.nadir_radius = depth / NADIR_TAN

	def ensure_capacity(self, world_xs, world_ys):
		margin = 10
		if self.grid_data is None:
			ox = np.floor(world_xs.min() / CELL_SIZE) * CELL_SIZE - margin * CELL_SIZE
			oy = np.floor(world_ys.min() / CELL_SIZE) * CELL_SIZE - margin * CELL_SIZE
			w = int(np.ceil((world_xs.max() - ox) / CELL_SIZE)) + margin * 2
			h = int(np.ceil((world_ys.max() - oy) / CELL_SIZE)) + margin * 2
			self.grid_origin_x = ox
			self.grid_origin_y = oy
			self.grid_w = w
			self.grid_h = h
			self.grid_data = np.zeros((h, w), dtype=np.float32)
			return

		ix = ((world_xs - self.grid_origin_x) / CELL_SIZE).astype(int)
		iy = ((world_ys - self.grid_origin_y) / CELL_SIZE).astype(int)
		pad_left  = max(0, -ix.min() + margin)
		pad_right = max(0,  ix.max() - (self.grid_w - 1) + margin)
		pad_down  = max(0, -iy.min() + margin)
		pad_up    = max(0,  iy.max() - (self.grid_h - 1) + margin)

		if pad_left or pad_right or pad_down or pad_up:
			new_w = self.grid_w + pad_left + pad_right
			new_h = self.grid_h + pad_down + pad_up
			new_data = np.zeros((new_h, new_w), dtype=np.float32)
			new_data[pad_down:pad_down + self.grid_h, pad_left:pad_left + self.grid_w] = self.grid_data
			self.grid_data = new_data
			self.grid_origin_x -= pad_left * CELL_SIZE
			self.grid_origin_y -= pad_down * CELL_SIZE
			self.grid_w = new_w
			self.grid_h = new_h

	def process_new_grid(self, msg: OccupancyGrid):
		try:
			tf_stamped = self.tf_buffer.lookup_transform(self.fixed_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(0.5))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			rospy.logwarn_throttle(5.0, f"TF lookup failed: {e}")
			self.tf_buffer = tf2_ros.Buffer()
			self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
			return

		t = tf_stamped.transform.translation
		r = tf_stamped.transform.rotation
		mat = tft.quaternion_matrix([r.x, r.y, r.z, r.w])
		mat[0, 3] = t.x
		mat[1, 3] = t.y
		mat[2, 3] = t.z

		if self.direction_yaw is not None:
			_, _, vehicle_yaw = tft.euler_from_quaternion([r.x, r.y, r.z, r.w])
			diff = abs(np.arctan2(np.sin(vehicle_yaw - self.direction_yaw), np.cos(vehicle_yaw - self.direction_yaw)))
			if diff > np.radians(DIRECTION_THRESHOLD_DEG):
				return

		res = msg.info.resolution
		cols = msg.info.width
		rows = msg.info.height

		oq = msg.info.origin.orientation
		origin_mat = tft.quaternion_matrix([oq.x, oq.y, oq.z, oq.w])
		origin_mat[0, 3] = msg.info.origin.position.x
		origin_mat[1, 3] = msg.info.origin.position.y

		ci, ri = np.meshgrid(np.arange(cols), np.arange(rows))
		cell_x = (ci.ravel() + 0.5) * res
		cell_y = (ri.ravel() + 0.5) * res

		cell_pts = origin_mat @ np.vstack([cell_x, cell_y, np.zeros(len(cell_x)), np.ones(len(cell_x))])
		pts = mat @ cell_pts
		world_x = pts[0]
		world_y = pts[1]

		raw = np.array(msg.data, dtype=np.int16)
		raw[raw < 0] += 256
		raw = raw.astype(np.float32)

		# nadir mask: drop cells within the blind zone radius of the sonar XY position
		if self.nadir_radius is not None:
			sonar_wx = mat[0, 3]
			sonar_wy = mat[1, 3]
			xy_dist = np.sqrt((world_x - sonar_wx) ** 2 + (world_y - sonar_wy) ** 2)
			raw[xy_dist < self.nadir_radius] = 0
			raw[xy_dist > 18] = 0

		#suppress middle
		sonar_wx = mat[0, 3]
		sonar_wy = mat[1, 3]
		sonar_yaw = np.arctan2(mat[1, 0], mat[0, 0])
		bearing = np.arctan2(world_y - sonar_wy, world_x - sonar_wx) - sonar_yaw
		bearing = np.arctan2(np.sin(bearing), np.cos(bearing))  # wrap to [-pi, pi]
		raw[np.abs(bearing) < np.radians(3.0)] *= 0.86

		keep = raw >= self.intensity_threshold
		world_x = world_x[keep]
		world_y = world_y[keep]
		intensities = raw[keep]

		if len(intensities) == 0:
			return

		self.ensure_capacity(world_x, world_y)

		ix = ((world_x - self.grid_origin_x) / CELL_SIZE).astype(int)
		iy = ((world_y - self.grid_origin_y) / CELL_SIZE).astype(int)
		valid = (ix >= 0) & (ix < self.grid_w) & (iy >= 0) & (iy < self.grid_h)
		ix, iy, intensities = ix[valid], iy[valid], intensities[valid]

		if len(ix) == 0:
			return
		
		intensities = 255.0 * (intensities / 255.0) ** 2.0

		self.grid_data[iy, ix] = self.grid_data[iy, ix] * 0.9 + intensities * 0.25

		self.pub_prescaler = (self.pub_prescaler + 1 ) % 5
		if self.pub_prescaler == 0:
			self.publish(msg.header.stamp)

	def grid_cb(self, msg: OccupancyGrid):
		if self.do_mapping and self.enabled:
			self.grid_sub.unregister()
			try:
				self.process_new_grid(msg)
			except Exception as e:
				rospy.logwarn_throttle(5.0, f"Callback failed: {e}")

			self.grid_sub = rospy.Subscriber("/oculus_sonar/grid", OccupancyGrid, self.grid_cb, queue_size=1)

	def publish(self, stamp):
		out = OccupancyGrid()
		out.header.stamp = stamp
		out.header.frame_id = self.fixed_frame
		out.info.resolution = CELL_SIZE
		out.info.width = self.grid_w
		out.info.height = self.grid_h
		out.info.origin.position.x = self.grid_origin_x
		out.info.origin.position.y = self.grid_origin_y
		out.info.origin.orientation.w = 1.0
		out.data = self.grid_data.ravel().astype(np.int8).tolist()
		self.pub.publish(out)

if __name__ == "__main__":
	node = SonarStitcher()
	rospy.spin()