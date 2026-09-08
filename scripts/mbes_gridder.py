#!/usr/bin/env python

import math
import threading
import numpy as np

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2, PointField, Range
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped, Point, Vector3, Quaternion
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

class BathymetryAccumulator(object):
	def __init__(self):
		self.fixed_frame = rospy.get_param("~fixed_frame", "local")
		self.cloud_topic = rospy.get_param("~cloud_topic", "/oculus_sonar/cloud")
		self.range_topic = rospy.get_param("~range_topic", "/echosounder/range")
		self.output_topic = rospy.get_param("~output_topic", "/bathymetry/cloud")
		self.marker_topic = rospy.get_param("~marker_topic", "/bathymetry/markers")
		self.publish_rate = rospy.get_param("~publish_rate", 1.0)
		self.transform_timeout = rospy.Duration(rospy.get_param("~transform_timeout", 0.2))

		self.resolution = rospy.get_param("~resolution", 0.5)
		self.cell_quantile = rospy.get_param("~cell_quantile", 0.5)
		self.min_hits = rospy.get_param("~min_hits", 3)
		self.neighbour_radius = int(rospy.get_param("~neighbour_radius", 2))
		self.mad_gain = rospy.get_param("~mad_gain", 3.0)
		self.min_deviation = rospy.get_param("~min_deviation", 0.05)
		self.min_neighbours = rospy.get_param("~min_neighbours", 4)

		self.colour_min_z = rospy.get_param("~colour_min_z", -30.0)
		self.colour_max_z = rospy.get_param("~colour_max_z", 0.0)

		self.lock = threading.Lock()

		self.samples = {}
		self.medians = {}
		self.dirty = set()

		self.index = {}
		self.keys = []
		self.xyz = np.zeros((4096, 3), dtype=np.float32)
		self.count = 0
		self.marker_points = []
		self.marker_colours = []

		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(rospy.get_param("~tf_cache_time", 30.0)))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.cloud_msg = PointCloud2()
		self.cloud_msg.header.frame_id = self.fixed_frame
		self.cloud_msg.height = 1
		self.cloud_msg.fields = [
			PointField("x", 0, PointField.FLOAT32, 1),
			PointField("y", 4, PointField.FLOAT32, 1),
			PointField("z", 8, PointField.FLOAT32, 1),
		]
		self.cloud_msg.is_bigendian = False
		self.cloud_msg.point_step = 12
		self.cloud_msg.is_dense = True

		self.marker = Marker()
		self.marker.header.frame_id = self.fixed_frame
		self.marker.ns = "bathymetry"
		self.marker.id = 0
		self.marker.type = Marker.CUBE_LIST
		self.marker.action = Marker.ADD
		self.marker.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
		self.marker.scale = Vector3(self.resolution, self.resolution, self.resolution)
		self.marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
		self.marker.points = self.marker_points
		self.marker.colors = self.marker_colours

		self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1, latch=True)
		self.marker_pub = rospy.Publisher(self.marker_topic, MarkerArray, queue_size=1, latch=True)
		self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=5)
		self.range_sub = rospy.Subscriber(self.range_topic, Range, self.range_callback, queue_size=50)
		self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_callback)

	def lookup(self, frame, stamp):
		try:
			return self.tf_buffer.lookup_transform(self.fixed_frame, frame, stamp, self.transform_timeout)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
			rospy.logwarn_throttle(5.0, "transform %s -> %s failed: %s" % (frame, self.fixed_frame, exc))
			return None

	def insert(self, x, y, z):
		key = (int(math.floor(x / self.resolution)), int(math.floor(y / self.resolution)))
		bucket = self.samples.get(key)
		if bucket is None:
			bucket = []
			self.samples[key] = bucket

		bucket.append(z)
		self.dirty.add(key)

	def cloud_callback(self, msg):
		transform = self.lookup(msg.header.frame_id, msg.header.stamp)
		if transform is None:
			return

		transformed = do_transform_cloud(msg, transform)

		with self.lock:
			for point in point_cloud2.read_points(transformed, field_names=("x", "y", "z"), skip_nans=True):
				self.insert(point[0], point[1], point[2])

	def range_callback(self, msg):
		if msg.range < msg.min_range or msg.range > msg.max_range:
			return

		transform = self.lookup(msg.header.frame_id, msg.header.stamp)
		if transform is None:
			return

		local = PointStamped()
		local.header = msg.header
		local.point.x = msg.range
		local.point.y = 0.0
		local.point.z = 0.0

		world = tf2_geometry_msgs.do_transform_point(local, transform)

		with self.lock:
			self.insert(world.point.x, world.point.y, world.point.z)

	def quantile(self, values, q):
		ordered = sorted(values)
		return ordered[int(q * (len(ordered) - 1))]

	def evaluate(self, key):
		entry = self.medians.get(key)
		if entry is None:
			return None

		median, hits = entry
		if hits < self.min_hits:
			return None

		neighbours = []
		for dx in range(-self.neighbour_radius, self.neighbour_radius + 1):
			for dy in range(-self.neighbour_radius, self.neighbour_radius + 1):
				if dx == 0 and dy == 0:
					continue

				other = self.medians.get((key[0] + dx, key[1] + dy))
				if other is not None and other[1] >= self.min_hits:
					neighbours.append(other[0])

		if len(neighbours) < self.min_neighbours:
			return median

		reference = self.quantile(neighbours, 0.5)
		mad = self.quantile([abs(value - reference) for value in neighbours], 0.5)
		if abs(median - reference) > self.mad_gain * max(mad, self.min_deviation):
			return None

		return median

	def colour(self, z):
		span = self.colour_max_z - self.colour_min_z
		t = 0.0 if span <= 0.0 else (z - self.colour_min_z) / span
		t = max(0.0, min(1.0, t))

		return ColorRGBA(t, 0.4 * (1.0 - abs(2.0 * t - 1.0)) + 0.3, 1.0 - t, 1.0)

	def add_cell(self, key, z):
		if self.count == self.xyz.shape[0]:
			grown = np.zeros((self.count * 2, 3), dtype=np.float32)
			grown[:self.count] = self.xyz
			self.xyz = grown

		row = self.count
		self.index[key] = row
		self.keys.append(key)
		self.xyz[row, 0] = (key[0] + 0.5) * self.resolution
		self.xyz[row, 1] = (key[1] + 0.5) * self.resolution
		self.xyz[row, 2] = z
		self.marker_points.append(Point(self.xyz[row, 0], self.xyz[row, 1], z))
		self.marker_colours.append(self.colour(z))
		self.count += 1

	def remove_cell(self, key):
		row = self.index.pop(key)
		last = self.count - 1

		if row != last:
			moved = self.keys[last]
			self.keys[row] = moved
			self.index[moved] = row
			self.xyz[row] = self.xyz[last]
			self.marker_points[row] = self.marker_points[last]
			self.marker_colours[row] = self.marker_colours[last]

		self.keys.pop()
		self.marker_points.pop()
		self.marker_colours.pop()
		self.count -= 1

	def update_cell(self, key, z):
		row = self.index[key]
		self.xyz[row, 2] = z
		self.marker_points[row].z = z
		self.marker_colours[row] = self.colour(z)

	def refresh(self):
		dirty = self.dirty
		self.dirty = set()
		if not dirty:
			return False

		for key in dirty:
			bucket = self.samples[key]
			self.medians[key] = (self.quantile(bucket, self.cell_quantile), len(bucket))

		stale = set()
		for key in dirty:
			for dx in range(-self.neighbour_radius, self.neighbour_radius + 1):
				for dy in range(-self.neighbour_radius, self.neighbour_radius + 1):
					stale.add((key[0] + dx, key[1] + dy))

		for key in stale:
			z = self.evaluate(key)
			present = key in self.index

			if z is None:
				if present:
					self.remove_cell(key)
			elif present:
				self.update_cell(key, z)
			else:
				self.add_cell(key, z)

		return True

	def publish_callback(self, event):
		with self.lock:
			self.refresh()
			if self.count == 0:
				return

			stamp = rospy.Time.now()
			#self.cloud_msg.header.stamp = stamp
			#self.cloud_msg.width = self.count
			#self.cloud_msg.row_step = self.count * self.cloud_msg.point_step
			#self.cloud_msg.data = self.xyz[:self.count].tobytes()
			self.marker.header.stamp = stamp

			array = MarkerArray()
			array.markers.append(self.marker)

		#self.pub.publish(self.cloud_msg)
		self.marker_pub.publish(array)

def main():
	rospy.init_node("bathymetry_accumulator")
	BathymetryAccumulator()
	rospy.spin()

if __name__ == "__main__":
	main()