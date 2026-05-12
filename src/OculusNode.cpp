#include "OculusNode.h"
#include "conversions.h"
#include <algorithm>
#include <vector>
#include <opencv2/imgcodecs.hpp>

OculusNode::OculusNode(const std::string& nodeName): node_(nodeName), configServer_(node_),	sonar_(service_.io_service()){
	node_.param<bool>("publish_without_subs", publishWithoutSubs_, false);

	imagePublisher_ = node_.advertise<sensor_msgs::Image>("image", 100);
	compressedImagePublisher_ = node_.advertise<sensor_msgs::CompressedImage>("image/compressed", 100);
	configPublisher_ = node_.advertise<oculus_sonar::SonarConfig>("config", 1, true); // latched
	temperaturePublisher_ = node_.advertise<sensor_msgs::Temperature>("temperature", 1);
	pressurePublisher_ = node_.advertise<sensor_msgs::FluidPressure>("pressure", 1);

	sonar_.add_ping_callback(std::bind(&OculusNode::ping_callback, this, std::placeholders::_1));
	sonar_.add_status_callback(std::bind(&OculusNode::status_callback, this, std::placeholders::_1));
	sonar_.add_dummy_callback(std::bind(&OculusNode::dummy_callback, this, std::placeholders::_1));
	this->start();

	configServer_.setCallback(std::bind(&OculusNode::reconfigure_callback, this, std::placeholders::_1, std::placeholders::_2));
}

OculusNode::~OculusNode(){
	this->stop();
}

void OculusNode::start(){
	service_.start();
	if(!sonar_.wait_next_message()) {
		throw std::runtime_error("Timeout reached while waiting for sonar. Is it plugged in ?");
	}

	oculus_sonar::OculusSonarConfig initialConfig;
	node_.param<int>("frequency_mode", initialConfig.frequency_mode, 1);
	node_.param<int>("ping_rate", initialConfig.ping_rate, 0);
	node_.param<int>("data_depth", initialConfig.data_depth, 0);
	node_.param<int>("nbeams", initialConfig.nbeams, 0);
	node_.param<bool>("send_gain", initialConfig.send_gain, false);
	node_.param<bool>("gain_assist", initialConfig.gain_assist, false);
	node_.param<double>("range", initialConfig.range, 3.0);
	node_.param<int>("gamma_correction", initialConfig.gamma_correction, 127);
	node_.param<double>("gain_percent", initialConfig.gain_percent, 50.0);
	node_.param<double>("sound_speed", initialConfig.sound_speed, 1500.0);
	node_.param<bool>("use_salinity", initialConfig.use_salinity, true);
	node_.param<double>("salinity", initialConfig.salinity, 35.0);
	this->reconfigure_callback(initialConfig, 0);
}

void OculusNode::stop(){
	service_.stop();
}

void OculusNode::ping_callback(const oculus::PingMessage::ConstPtr& ping){
	if(!publishWithoutSubs_ && !this->has_ping_subscribers()) {
		std::cout << "Going to standby mode" << std::endl;
		sonar_.standby();
		return;
	}

	if(imagePublisher_.getNumSubscribers() > 0) {
		sensor_msgs::Image img;
		oculus::copy_to_ros(img, ping);
		imagePublisher_.publish(img);
	}

	if(compressedImagePublisher_.getNumSubscribers() > 0) {
		auto sampleSize = ping->sample_size();
		int cvType = (sampleSize == 2) ? CV_16UC1 : CV_8UC1;
		// step is in bytes; divide by sampleSize to get columns for cv::Mat
		cv::Mat mat(ping->range_count(), ping->step() / sampleSize, cvType, const_cast<uint8_t*>(ping->ping_data()));

		sensor_msgs::CompressedImage img;
		img.header.stamp = oculus::to_ros_stamp(ping->timestamp());
		img.header.frame_id = "oculus_sonar";
		img.format = "png"; // lossless; sonar data doesn't compress well with jpeg
		cv::imencode(".png", mat, img.data);
		compressedImagePublisher_.publish(img);
	}
}

void OculusNode::status_callback(const OculusStatusMsg& status){
	ros::Time stamp = ros::Time::now();

	double max_temp = 0.0;
	for(double t : {status.temperature0, status.temperature1, status.temperature2, status.temperature3, status.temperature4, status.temperature5, status.temperature6, status.temperature7}){
		if(t > max_temp){
            max_temp = t;
        }
    }

	sensor_msgs::Temperature tempMsg;
	tempMsg.header.stamp = stamp;
	tempMsg.header.frame_id = "oculus_link";
	tempMsg.temperature = max_temp;
	tempMsg.variance = 0.0;
	temperaturePublisher_.publish(tempMsg);

	// Blueprint docs don't specify pressure units. Empirically the values are consistent with gauge bar
	// at shallow depth (0.017 bar expected at 17cm fresh water, observed ~-0.057 with apparent zero offset).
	// Assumed bar; if wrong, scale factor here is the only thing that needs changing.
	sensor_msgs::FluidPressure pressMsg;
	pressMsg.header.stamp = stamp;
	pressMsg.header.frame_id = "oculus_link";
	pressMsg.fluid_pressure = status.pressure * 1.0e5; // assumed bar -> Pa
	pressMsg.variance = 0.0;
	pressurePublisher_.publish(pressMsg);
}

void OculusNode::reconfigure_callback(oculus_sonar::OculusSonarConfig& config, int32_t level){
    ROS_INFO("Reconfigure callback triggered, level=%d", level);

	oculus::SonarDriver::PingConfig currentConfig;
	std::memset(&currentConfig, 0, sizeof(currentConfig));

	currentConfig.masterMode = config.frequency_mode;
	switch(config.ping_rate)	{
		case 0: currentConfig.pingRate = pingRateNormal;  break;
		case 1: currentConfig.pingRate = pingRateHigh;    break;
		case 2: currentConfig.pingRate = pingRateHighest; break;
		case 3: currentConfig.pingRate = pingRateLow;     break;
		case 4: currentConfig.pingRate = pingRateLowest;  break;
		case 5: currentConfig.pingRate = pingRateStandby; break;
		default: break;
	}

	currentConfig.flags = 0x01  // always in meters
	                    | 0x04  // force send gain to true
	                    | 0x08; // use simple ping

	switch(config.data_depth)	{
		case oculus_sonar::OculusSonar_8bits:
			break;
		case oculus_sonar::OculusSonar_16bits:
			currentConfig.flags |= 0x02;
			break;
		default: break;
	}

	switch(config.nbeams)	{
		case oculus_sonar::OculusSonar_256beams:
			break;
		case oculus_sonar::OculusSonar_512beams:
			currentConfig.flags |= 0x40;
			break;
		default: break;
	}

	if(config.gain_assist)
		currentConfig.flags |= 0x10;

	currentConfig.range = config.range;
	currentConfig.gammaCorrection = config.gamma_correction;
	currentConfig.gainPercent = config.gain_percent;

	if(config.use_salinity)
		currentConfig.speedOfSound = 0.0;
	else
		currentConfig.speedOfSound = config.sound_speed;
	currentConfig.salinity = config.salinity;

    
    ROS_INFO("Requesting ping config from sonar...");

	auto feedback = sonar_.request_ping_config(currentConfig);
	config.frequency_mode = feedback.masterMode;
	config.data_depth = (feedback.flags & 0x02) ? 1 : 0;
	config.send_gain = (feedback.flags & 0x04) ? 1 : 0;
	config.gain_assist = (feedback.flags & 0x10) ? 1 : 0;
	config.nbeams = (feedback.flags & 0x40) ? 1 : 0;
	config.range = feedback.range;
	config.gamma_correction = feedback.gammaCorrection;
	config.gain_percent = feedback.gainPercent;
	config.sound_speed = feedback.speedOfSound;
	config.salinity = feedback.salinity;

    ROS_INFO("Got feedback, publishing config");

	oculus_sonar::SonarConfig configMsg;
	configMsg.frequency_mode = config.frequency_mode;
	configMsg.ping_rate = config.ping_rate;
	configMsg.data_depth = config.data_depth;
	configMsg.nbeams = config.nbeams;
	configMsg.send_gain = config.send_gain;
	configMsg.gain_assist = config.gain_assist;
	configMsg.range = config.range;
	configMsg.gamma_correction = config.gamma_correction;
	configMsg.gain_percent = config.gain_percent;
	configMsg.sound_speed = config.sound_speed;
	configMsg.use_salinity = config.use_salinity;
	configMsg.salinity = config.salinity;
	configPublisher_.publish(configMsg);

	ROS_INFO("Config published to sonar_config topic");
}

bool OculusNode::has_ping_subscribers() const{
	return imagePublisher_.getNumSubscribers() > 0
		|| compressedImagePublisher_.getNumSubscribers() > 0;
}

void OculusNode::dummy_callback(const OculusMessageHeader& msg){
	if(publishWithoutSubs_ || this->has_ping_subscribers()) {
		std::cout << "Exiting standby mode" << std::endl;
		sonar_.resume();
	}
}