#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/Range.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <climits>

static constexpr double CELL_SIZE = 0.05;
static constexpr int INTENSITY_THRESHOLD = 50;
static constexpr double OVERLAP_GATE = 1.0;
static constexpr double NADIR_ANGLE_DEG = 19.0; // steepest ray of the swath, first to hit bottom
static const double NADIR_TAN = std::tan(NADIR_ANGLE_DEG * M_PI / 180.0);
static constexpr int ECHO_WINDOW = 10;
static constexpr double DIRECTION_THRESHOLD_DEG = 30.0;

static Eigen::Matrix4d transformMatrix(double qx, double qy, double qz, double qw, double tx, double ty, double tz) {
	Eigen::Quaterniond q(qw, qx, qy, qz);
	q.normalize();
	Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
	m.block<3, 3>(0, 0) = q.toRotationMatrix();
	m(0, 3) = tx;
	m(1, 3) = ty;
	m(2, 3) = tz;
	return m;
}

// matches tf.transformations euler_from_quaternion yaw, which equals atan2(R10, R00)
static double yawFromQuat(double x, double y, double z, double w) {
	return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

class SonarStitcher {
public:
	SonarStitcher() {
		ros::NodeHandle nh;
		ros::NodeHandle pnh("~");
		pnh.param("fixed_frame", fixed_frame_, std::string("local"));
		pnh.param("intensity_threshold", intensity_threshold_, (double)INTENSITY_THRESHOLD);
		pnh.param("overlap_gate", overlap_gate_, OVERLAP_GATE);
		tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));
		clahe_ = cv::createCLAHE(20.0, cv::Size(8, 8));
		pub_ = nh.advertise<nav_msgs::OccupancyGrid>("/oculus_sonar/stacked_grid", 1);
		echosounder_sub_ = nh.subscribe("/echosounder/range", 1, &SonarStitcher::echoCb, this);
		cmd_vel_sub_ = nh.subscribe("/cmd_vel", 1, &SonarStitcher::cmdVelCb, this);
		direction_sub_ = nh.subscribe("/oculus_stacker/direction", 1, &SonarStitcher::directionCb, this);
		enabled_sub_ = nh.subscribe("/oculus_stacker/enabled", 1, &SonarStitcher::enabledCb, this);
		enabled_pub_ = nh.advertise<std_msgs::Bool>("/oculus_stacker/enabled", 1, true);
		publishEnabled();
		grid_sub_ = nh.subscribe("/oculus_sonar/grid", 1, &SonarStitcher::gridCb, this);
	}

private:
	void directionCb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
		const auto& q = msg->pose.orientation;
		direction_yaw_ = yawFromQuat(q.x, q.y, q.z, q.w);
		has_direction_ = true;
	}

	void enabledCb(const std_msgs::Bool::ConstPtr& msg) {
		if (msg->data != enabled_) {
			enabled_ = msg->data;
			publishEnabled();
		}
	}

	void cmdVelCb(const geometry_msgs::Twist::ConstPtr& msg) {
		do_mapping_ = std::abs(msg->angular.z) < 0.5; // discard big turns
	}

	void echoCb(const sensor_msgs::Range::ConstPtr& msg) {
		echo_window_.push_back(msg->range);
		if ((int)echo_window_.size() > ECHO_WINDOW) echo_window_.pop_front();
		std::vector<double> tmp(echo_window_.begin(), echo_window_.end());
		std::sort(tmp.begin(), tmp.end());
		size_t m = tmp.size();
		double depth = (m % 2) ? tmp[m / 2] : 0.5 * (tmp[m / 2 - 1] + tmp[m / 2]);
		nadir_radius_ = depth / NADIR_TAN;
		has_nadir_ = true;
	}

	void publishEnabled() {
		std_msgs::Bool b;
		b.data = enabled_;
		enabled_pub_.publish(b);
	}

	void ensureCapacity(const std::vector<float>& xs, const std::vector<float>& ys) {
		const int margin = 10;
		if (grid_data_.empty()) {
			double minx = *std::min_element(xs.begin(), xs.end());
			double maxx = *std::max_element(xs.begin(), xs.end());
			double miny = *std::min_element(ys.begin(), ys.end());
			double maxy = *std::max_element(ys.begin(), ys.end());
			double ox = std::floor(minx / CELL_SIZE) * CELL_SIZE - margin * CELL_SIZE;
			double oy = std::floor(miny / CELL_SIZE) * CELL_SIZE - margin * CELL_SIZE;
			grid_origin_x_ = ox;
			grid_origin_y_ = oy;
			grid_w_ = (int)std::ceil((maxx - ox) / CELL_SIZE) + margin * 2;
			grid_h_ = (int)std::ceil((maxy - oy) / CELL_SIZE) + margin * 2;
			grid_data_.assign((size_t)grid_w_ * grid_h_, 0.0f);
			return;
		}
		int ixmin = INT_MAX, ixmax = INT_MIN, iymin = INT_MAX, iymax = INT_MIN;
		for (size_t i = 0; i < xs.size(); ++i) {
			int ix = (int)((xs[i] - grid_origin_x_) / CELL_SIZE);
			int iy = (int)((ys[i] - grid_origin_y_) / CELL_SIZE);
			ixmin = std::min(ixmin, ix);
			ixmax = std::max(ixmax, ix);
			iymin = std::min(iymin, iy);
			iymax = std::max(iymax, iy);
		}
		int pad_left = std::max(0, -ixmin + margin);
		int pad_right = std::max(0, ixmax - (grid_w_ - 1) + margin);
		int pad_down = std::max(0, -iymin + margin);
		int pad_up = std::max(0, iymax - (grid_h_ - 1) + margin);
		if (pad_left || pad_right || pad_down || pad_up) {
			int new_w = grid_w_ + pad_left + pad_right;
			int new_h = grid_h_ + pad_down + pad_up;
			std::vector<float> nd((size_t)new_w * new_h, 0.0f);
			for (int r = 0; r < grid_h_; ++r)
				std::copy(&grid_data_[(size_t)r * grid_w_], &grid_data_[(size_t)r * grid_w_] + grid_w_, &nd[(size_t)(r + pad_down) * new_w + pad_left]);
			grid_data_.swap(nd);
			grid_origin_x_ -= pad_left * CELL_SIZE;
			grid_origin_y_ -= pad_down * CELL_SIZE;
			grid_w_ = new_w;
			grid_h_ = new_h;
		}
	}

	void processNewGrid(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
		geometry_msgs::TransformStamped tf_stamped;
		try {
			tf_stamped = tf_buffer_.lookupTransform(fixed_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.5));
		} catch (const tf2::TransformException& e) {
			ROS_WARN_THROTTLE(5.0, "TF lookup failed: %s", e.what());
			tf_buffer_.clear();
			tf_listener_.reset(new tf2_ros::TransformListener(tf_buffer_));
			return;
		}
		const auto& t = tf_stamped.transform.translation;
		const auto& r = tf_stamped.transform.rotation;
		Eigen::Matrix4d mat = transformMatrix(r.x, r.y, r.z, r.w, t.x, t.y, t.z);
		double sonar_yaw = yawFromQuat(r.x, r.y, r.z, r.w);
		if (has_direction_) {
			double d = sonar_yaw - direction_yaw_;
			double diff = std::abs(std::atan2(std::sin(d), std::cos(d)));
			if (diff > DIRECTION_THRESHOLD_DEG * M_PI / 180.0) return;
		}
		double res = msg->info.resolution;
		int cols = msg->info.width;
		int rows = msg->info.height;
		const auto& oq = msg->info.origin.orientation;
		const auto& op = msg->info.origin.position;
		Eigen::Matrix4d origin_mat = transformMatrix(oq.x, oq.y, oq.z, oq.w, op.x, op.y, 0.0);
		Eigen::Matrix4d M = mat * origin_mat;
		double M00 = M(0, 0), M01 = M(0, 1), M03 = M(0, 3);
		double M10 = M(1, 0), M11 = M(1, 1), M13 = M(1, 3);
		double sonar_wx = t.x, sonar_wy = t.y;
		const double mid_band = 3.0 * M_PI / 180.0;
		const int n = rows * cols;

		// CLAHE on the raw intensities, negatives clamped to zero first
		cv::Mat img(rows, cols, CV_8UC1);
		for (int k = 0; k < n; ++k) {
			int v = msg->data[k];
			img.data[k] = v < 0 ? 0 : (uchar)v;
		}
		clahe_->apply(img, img);

		std::vector<double> lx(cols), ry(rows);
		for (int c = 0; c < cols; ++c) lx[c] = (c + 0.5) * res;
		for (int rr = 0; rr < rows; ++rr) ry[rr] = (rr + 0.5) * res;

		std::vector<float> kx, ky, ki;
		kx.reserve(n);
		ky.reserve(n);
		ki.reserve(n);
		for (int rr = 0; rr < rows; ++rr) {
			double yr = ry[rr];
			const uchar* rowp = img.data + (size_t)rr * cols;
			for (int c = 0; c < cols; ++c) {
				float intensity = (float)rowp[c];
				double wx = M00 * lx[c] + M01 * yr + M03;
				double wy = M10 * lx[c] + M11 * yr + M13;
				if (has_nadir_) {
					double dx = wx - sonar_wx, dy = wy - sonar_wy;
					double dist = std::sqrt(dx * dx + dy * dy);
					if (dist < nadir_radius_) intensity = 0.0f; // blind zone + far cutoff
				}
				double bearing = std::atan2(wy - sonar_wy, wx - sonar_wx) - sonar_yaw;
				bearing = std::atan2(std::sin(bearing), std::cos(bearing));
				if (std::abs(bearing) < mid_band) intensity *= 0.86f;
				if (intensity >= intensity_threshold_) {
					kx.push_back((float)wx);
					ky.push_back((float)wy);
					ki.push_back(intensity);
				}
			}
		}
		if (ki.empty()) return;
		ensureCapacity(kx, ky);

		// last-write-wins per cell against the pre-update grid, matching numpy fancy indexing
		std::unordered_map<size_t, float> cell_intensity;
		cell_intensity.reserve(ki.size());
		for (size_t i = 0; i < ki.size(); ++i) {
			int ix = (int)((kx[i] - grid_origin_x_) / CELL_SIZE);
			int iy = (int)((ky[i] - grid_origin_y_) / CELL_SIZE);
			if (ix < 0 || ix >= grid_w_ || iy < 0 || iy >= grid_h_) continue;
			double gi = 255.0 * std::pow(ki[i] / 255.0, 5.0);
			cell_intensity[(size_t)iy * grid_w_ + ix] = (float)gi;
		}
		for (const auto& kv : cell_intensity) {
			float& cell = grid_data_[kv.first];
			cell = cell * 0.96f + kv.second * 0.04f;
		}
		pub_prescaler_ = (pub_prescaler_ + 1) % 5;
		if (pub_prescaler_ == 0)
			publish(msg->header.stamp);
	}

	void gridCb(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
		if (!(do_mapping_ && enabled_)) return;
		try {
			processNewGrid(msg);
		} catch (const std::exception& e) {
			ROS_WARN_THROTTLE(5.0, "Callback failed: %s", e.what());
		}
	}

	void publish(const ros::Time& stamp) {
		nav_msgs::OccupancyGrid out;
		out.header.stamp = stamp;
		out.header.frame_id = fixed_frame_;
		out.info.resolution = CELL_SIZE;
		out.info.width = grid_w_;
		out.info.height = grid_h_;
		out.info.origin.position.x = grid_origin_x_;
		out.info.origin.position.y = grid_origin_y_;
		out.info.origin.orientation.w = 1.0;
		out.data.resize(grid_data_.size());
		for (size_t i = 0; i < grid_data_.size(); ++i)
			out.data[i] = (int8_t)(long long)grid_data_[i]; // truncate then wrap mod 256, matching numpy int8 cast
		pub_.publish(out);
	}

	std::string fixed_frame_;
	double intensity_threshold_ = INTENSITY_THRESHOLD;
	double overlap_gate_ = OVERLAP_GATE;
	tf2_ros::Buffer tf_buffer_;
	std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
	std::deque<double> echo_window_;
	double nadir_radius_ = 0.0;
	bool has_nadir_ = false;
	bool do_mapping_ = true;
	std::vector<float> grid_data_;
	double grid_origin_x_ = 0.0;
	double grid_origin_y_ = 0.0;
	int grid_w_ = 0;
	int grid_h_ = 0;
	double direction_yaw_ = 0.0;
	bool has_direction_ = false;
	int pub_prescaler_ = 0;
	bool enabled_ = false;
	cv::Ptr<cv::CLAHE> clahe_;
	ros::Publisher pub_;
	ros::Publisher enabled_pub_;
	ros::Subscriber echosounder_sub_;
	ros::Subscriber cmd_vel_sub_;
	ros::Subscriber direction_sub_;
	ros::Subscriber enabled_sub_;
	ros::Subscriber grid_sub_;
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "sonar_stitcher");
	SonarStitcher node;
	ros::spin();
	return 0;
}
