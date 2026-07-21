#ifndef _OCULUS_SONAR_OCULUS_NODE_H_
#define _OCULUS_SONAR_OCULUS_NODE_H_

#include <atomic>

#include <ros/ros.h>

#include <oculus_driver/AsyncService.h>
#include <oculus_driver/SonarDriver.h>
#include <oculus_driver/OculusMessage.h>

#include <oculus_sonar/OculusStatus.h>
#include <oculus_sonar/SonarConfig.h>

#include <std_msgs/Bool.h>
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

	ros::NodeHandle node_;
	ros::NodeHandle privateNode_;

	ros::Publisher imagePublisher_;
	ros::Publisher compressedImagePublisher_;
	ros::Publisher configPublisher_;
	ros::Publisher temperaturePublisher_;
	ros::Publisher pressurePublisher_;
	ros::Publisher enabledPublisher_;
	ros::Subscriber enabledSubscriber_;
	ros::Timer standbyTimer_;

	dynamic_reconfigure::Server<oculus_sonar::OculusSonarConfig> configServer_;

	oculus::AsyncService service_;
	oculus::SonarDriver sonar_;

	// ping callbacks run on the driver io thread, everything else on the ROS spinner
	std::atomic<bool> enabled_;
	oculus_sonar::OculusSonarConfig currentConfig_;

	public:

	OculusNode(const std::string& nodeName);
	~OculusNode();

	void start();
	void stop();

	void ping_callback(const oculus::PingMessage::ConstPtr& msg);
	void status_callback(const OculusStatusMsg& status);
	void enabled_callback(const std_msgs::Bool& msg);
	void standby_timer_callback(const ros::TimerEvent& event);
	void reconfigure_callback(oculus_sonar::OculusSonarConfig& config, int32_t level);
	void apply_config();
	void publish_enabled();
};

#endif //_OCULUS_SONAR_OCULUS_NODE_H_
