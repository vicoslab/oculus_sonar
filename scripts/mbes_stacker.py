#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2, PointField, Range
from sensor_msgs import point_cloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

class BathymetryAccumulator(object):
	def __init__(self):
		self.fixed_frame = rospy.get_param("~fixed_frame", "local")
		self.cloud_topic = rospy.get_param("~cloud_topic", "/oculus_sonar/cloud")
		self.range_topic = rospy.get_param("~range_topic", "/echosounder/range")
		self.output_topic = rospy.get_param("~output_topic", "/bathymetry/cloud")
		self.publish_rate = rospy.get_param("~publish_rate", 1.0)
		self.transform_timeout = rospy.Duration(rospy.get_param("~transform_timeout", 0.2))

		self.points = []

		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(rospy.get_param("~tf_cache_time", 30.0)))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.fields = [
			PointField("x", 0, PointField.FLOAT32, 1),
			PointField("y", 4, PointField.FLOAT32, 1),
			PointField("z", 8, PointField.FLOAT32, 1),
		]

		self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1, latch=True)
		self.cloud_sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cloud_callback, queue_size=5)
		self.range_sub = rospy.Subscriber(self.range_topic, Range, self.range_callback, queue_size=50)
		self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.publish_callback)

	def lookup(self, frame, stamp):
		try:
			return self.tf_buffer.lookup_transform(self.fixed_frame, frame, stamp, self.transform_timeout)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
			rospy.logwarn_throttle(5.0, "transform %s -> %s failed: %s" % (frame, self.fixed_frame, exc))
			return None

	def cloud_callback(self, msg):
		transform = self.lookup(msg.header.frame_id, msg.header.stamp)
		if transform is None:
			return

		transformed = do_transform_cloud(msg, transform)
		for point in point_cloud2.read_points(transformed, field_names=("x", "y", "z"), skip_nans=True):
			self.points.append(point)

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
		self.points.append((world.point.x, world.point.y, world.point.z))

	def publish_callback(self, event):
		points = list(self.points)
		if not points:
			return

		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id = self.fixed_frame

		self.pub.publish(point_cloud2.create_cloud(header, self.fields, points))

def main():
	rospy.init_node("bathymetry_accumulator")
	BathymetryAccumulator()
	rospy.spin()

if __name__ == "__main__":
	main()