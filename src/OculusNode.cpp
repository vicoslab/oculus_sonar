#include "OculusNode.h"
#include "conversions.h"
#include <algorithm>
#include <vector>
#include <opencv2/imgcodecs.hpp>

OculusNode::OculusNode(const std::string& nodeName): node_(nodeName), privateNode_("~"), configServer_(privateNode_), sonar_(service_.io_service()), enabled_(false){
	imagePublisher_ = node_.advertise<sensor_msgs::Image>("image", 100);
	compressedImagePublisher_ = node_.advertise<sensor_msgs::CompressedImage>("image/compressed", 100);
	configPublisher_ = node_.advertise<oculus_sonar::SonarConfig>("config", 1, true); // latched
	temperaturePublisher_ = node_.advertise<sensor_msgs::Temperature>("temperature", 1);
	pressurePublisher_ = node_.advertise<sensor_msgs::FluidPressure>("pressure", 1);

	enabledPublisher_ = node_.advertise<std_msgs::Bool>("enabled", 1, true); // latched
	enabledSubscriber_ = node_.subscribe("enabled", 1, &OculusNode::enabled_callback, this);
	this->publish_enabled();

	sonar_.add_ping_callback(std::bind(&OculusNode::ping_callback, this, std::placeholders::_1));
	sonar_.add_status_callback(std::bind(&OculusNode::status_callback, this, std::placeholders::_1));
	this->start();

	// fires the callback once with the current private params, caching them until first enable
	configServer_.setCallback(std::bind(&OculusNode::reconfigure_callback, this, std::placeholders::_1, std::placeholders::_2));

	standbyTimer_ = node_.createTimer(ros::Duration(5.0), &OculusNode::standby_timer_callback, this);
}

OculusNode::~OculusNode(){
	this->stop();
}

void OculusNode::start(){
	service_.start();

	while(!sonar_.wait_next_message()) {
		if(!ros::ok()) {
			return;
		}
		ROS_WARN("Timeout reached while waiting for sonar, retrying. Is it plugged in ?");
	}

	ROS_INFO("Sonar connected");
}

void OculusNode::stop(){
	service_.stop();
}

void OculusNode::publish_enabled(){
	std_msgs::Bool msg;
	msg.data = enabled_;
	enabledPublisher_.publish(msg);
}

void OculusNode::enabled_callback(const std_msgs::Bool& msg){
	if(msg.data == enabled_) {
		return;
	}

	enabled_ = msg.data;
	this->publish_enabled();

	if(enabled_) {
		ROS_INFO("Sonar enabled, applying configuration");
		this->apply_config();
		configServer_.updateConfig(currentConfig_);
	} else {
		ROS_INFO("Sonar disabled, requesting standby");
		sonar_.standby();
	}
}

void OculusNode::standby_timer_callback(const ros::TimerEvent& event){
	if(!enabled_) {
		sonar_.standby();
	}
}

void OculusNode::ping_callback(const oculus::PingMessage::ConstPtr& ping){
	if(!enabled_) {
		// the driver fires the sonar on every (re)connection, push it back down
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
	currentConfig_ = config;

	if(!enabled_) {
		// cache only, applied on the next rising edge so the sonar stays in standby
		return;
	}

	this->apply_config();
	config = currentConfig_;
}

void OculusNode::apply_config(){
	oculus::SonarDriver::PingConfig request;
	std::memset(&request, 0, sizeof(request));

	request.masterMode = currentConfig_.frequency_mode;
	switch(currentConfig_.ping_rate) {
		case 0: request.pingRate = pingRateNormal;  break;
		case 1: request.pingRate = pingRateHigh;    break;
		case 2: request.pingRate = pingRateHighest; break;
		case 3: request.pingRate = pingRateLow;     break;
		case 4: request.pingRate = pingRateLowest;  break;
		case 5: request.pingRate = pingRateStandby; break;
		default: break;
	}

	request.flags = 0x01  // always in meters
	              | 0x04  // force send gain to true
	              | 0x08; // use simple ping

	switch(currentConfig_.data_depth) {
		case oculus_sonar::OculusSonar_8bits:
			break;
		case oculus_sonar::OculusSonar_16bits:
			request.flags |= 0x02;
			break;
		default: break;
	}

	switch(currentConfig_.nbeams) {
		case oculus_sonar::OculusSonar_256beams:
			break;
		case oculus_sonar::OculusSonar_512beams:
			request.flags |= 0x40;
			break;
		default: break;
	}

	if(currentConfig_.gain_assist)
		request.flags |= 0x10;

	request.range = currentConfig_.range;
	request.gammaCorrection = currentConfig_.gamma_correction;
	request.gainPercent = currentConfig_.gain_percent;

	if(currentConfig_.use_salinity)
		request.speedOfSound = 0.0;
	else
		request.speedOfSound = currentConfig_.sound_speed;
	request.salinity = currentConfig_.salinity;

	ROS_INFO("Requesting ping config from sonar...");

	auto feedback = sonar_.request_ping_config(request);
	currentConfig_.frequency_mode = feedback.masterMode;
	currentConfig_.data_depth = (feedback.flags & 0x02) ? 1 : 0;
	currentConfig_.send_gain = (feedback.flags & 0x04) ? 1 : 0;
	currentConfig_.gain_assist = (feedback.flags & 0x10) ? 1 : 0;
	currentConfig_.nbeams = (feedback.flags & 0x40) ? 1 : 0;
	currentConfig_.range = feedback.range;
	currentConfig_.gamma_correction = feedback.gammaCorrection;
	currentConfig_.gain_percent = feedback.gainPercent;
	currentConfig_.sound_speed = feedback.speedOfSound;
	currentConfig_.salinity = feedback.salinity;

	oculus_sonar::SonarConfig configMsg;
	configMsg.frequency_mode = currentConfig_.frequency_mode;
	configMsg.ping_rate = currentConfig_.ping_rate;
	configMsg.data_depth = currentConfig_.data_depth;
	configMsg.nbeams = currentConfig_.nbeams;
	configMsg.send_gain = currentConfig_.send_gain;
	configMsg.gain_assist = currentConfig_.gain_assist;
	configMsg.range = currentConfig_.range;
	configMsg.gamma_correction = currentConfig_.gamma_correction;
	configMsg.gain_percent = currentConfig_.gain_percent;
	configMsg.sound_speed = currentConfig_.sound_speed;
	configMsg.use_salinity = currentConfig_.use_salinity;
	configMsg.salinity = currentConfig_.salinity;
	configPublisher_.publish(configMsg);

	ROS_INFO("Sonar configuration applied");
}
