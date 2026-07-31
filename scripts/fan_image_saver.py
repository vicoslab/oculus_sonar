#!/usr/bin/env python3

import os
import json
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import CompressedImage


class SonarSaver:
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic","/oculus_sonar/projected_image/compressed")
        self.local_frame = rospy.get_param("~fixed_frame", "local")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        self.output_dir = os.path.expanduser(rospy.get_param("~output_dir", "~/sonar_dataset"))
        self.output_dir = os.path.abspath(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(120.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.sub = rospy.Subscriber(self.image_topic, CompressedImage, self.image_callback, queue_size=10)

        rospy.loginfo("Saving sonar images to %s", self.output_dir)

    def image_callback(self, msg):
        seq = msg.header.seq

        try:
            tf = self.tf_buffer.lookup_transform(self.local_frame, self.base_frame, msg.header.stamp, rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(5.0, "TF lookup failed: %s", str(e))
            return

        t = tf.transform.translation
        q = tf.transform.rotation

        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)

        image_path = os.path.join(self.output_dir, f"{seq}.png")
        json_path = os.path.join(self.output_dir, f"{seq}.json")

        # CompressedImage already contains PNG bytes
        with open(image_path, "wb") as f:
            f.write(msg.data)

        pose = {
            "seq": seq,
            "stamp": {
                "secs": msg.header.stamp.secs,
                "nsecs": msg.header.stamp.nsecs,
            },
            "frame_id": self.local_frame,
            "position": {
                "x": t.x,
                "y": t.y,
            },
            "yaw": yaw,
            "quaternion": {
                "x": q.x,
                "y": q.y,
                "z": q.z,
                "w": q.w,
            },
        }

        with open(json_path, "w") as f:
            json.dump(pose, f, indent=2)


if __name__ == "__main__":
    rospy.init_node("sonar_pose_saver")
    SonarSaver()
    rospy.spin() 
