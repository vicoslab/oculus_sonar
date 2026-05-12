#ifndef _OCULUS_SONAR_OCULUS_NODE_H_
#define _OCULUS_SONAR_OCULUS_NODE_H_

#include <ros/ros.h>

#include <oculus_driver/AsyncService.h>
#include <oculus_driver/SonarDriver.h>
#include <oculus_driver/OculusMessage.h>

#include <oculus_sonar/OculusStatus.h>
#include <oculus_sonar/SonarConfig.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Temperature.h>
#include <sensor_msgs/FluidPressure.h>
#include <opencv2/imgcodecs.hpp>

#include <dynamic_reconfigure/server.h>
#include <oculus_sonar/OculusSonarConfig.h>

class OculusNode
{
	protected:

	bool publishWithoutSubs_;

	ros::NodeHandle node_;

	ros::Publisher imagePublisher_;
	ros::Publisher compressedImagePublisher_;
	ros::Publisher configPublisher_;
	ros::Publisher temperaturePublisher_;
	ros::Publisher pressurePublisher_;

	dynamic_reconfigure::Server<oculus_sonar::OculusSonarConfig> configServer_;

	oculus::AsyncService service_;
	oculus::SonarDriver sonar_;

	public:

	OculusNode(const std::string& nodeName);
	~OculusNode();

	void start();
	void stop();

	void ping_callback(const oculus::PingMessage::ConstPtr& msg);
	void status_callback(const OculusStatusMsg& status);
	void dummy_callback(const OculusMessageHeader& msg);

	void reconfigure_callback(oculus_sonar::OculusSonarConfig& config, int32_t level);

	bool has_ping_subscribers() const;
};

#endif //_OCULUS_SONAR_OCULUS_NODE_H_