#!/usr/bin/env python3

import rospy
import numpy as np
import cv2

from nav_msgs.msg import OccupancyGrid

class MapSaver:
    def __init__(self):
        self.topic = rospy.get_param("~topic", "/oculus_sonar/stacked_grid")
        self.filename = rospy.get_param("~filename", "raw_map.png")

        rospy.Subscriber(self.topic, OccupancyGrid, self.cb, queue_size=1)
        rospy.loginfo(f"Waiting for map on {self.topic}")

    def cb(self, msg):
        width = msg.info.width
        height = msg.info.height
        
        img = np.array(msg.data, dtype=np.int8).reshape(height, width).astype(np.uint8)
        cv2.imwrite(self.filename, np.flipud(img))

        rospy.loginfo(f"Saved map to {self.filename}")
        rospy.signal_shutdown("Map saved")

if __name__ == "__main__":
    rospy.init_node("raw_map_png_saver")
    MapSaver()
    rospy.spin()