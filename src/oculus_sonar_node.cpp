#include <ros/ros.h>

#include "OculusNode.h"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "oculus_sonar");

	OculusNode sonarNode("oculus_sonar");
	ros::spin();

	return 0;
}
