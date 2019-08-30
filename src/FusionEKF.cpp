#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  // https://www.youtube.com/watch?v=udsB-13ntY8
  noise_ax_           = 9.0;
  noise_ay_           = 9.0;
  is_initialized_     = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_      = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  /**
   * Finish initializing the FusionEKF.
   * Set the process and measurement noises
   */
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  VectorXd x_in = VectorXd(4);
  x_in << 0, 0, 0, 0;

  MatrixXd P_in = MatrixXd(4, 4);
  P_in << 1, 0,   0,     0,
          0, 1,   0,     0,
          0, 0, 1000,    0,
          0, 0,    0, 1000;

  // 13 Laser Measurements Part 3: Helpful Equations
  // F = (
  // 1  0  Δt 0
  // 0  1  0  Δt
  // 0  0  1  0
  // 0  0  0  1
  // )
  MatrixXd F_in = MatrixXd(4, 4);
  F_in <<  1, 0, 1, 0,
           0, 1, 0, 1,
           0, 0, 1, 0,
           0, 0, 0, 1;

  MatrixXd Q_in = MatrixXd(4, 4);
  Q_in.setZero();

  ekf_.Init(x_in, P_in, F_in, H_laser_, R_laser_, Q_in);
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     * Convert radar from polar to cartesian coordinates.
     */
    // first measurement
    cout << "EKF:\n";
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      float rho     = measurement_pack.raw_measurements_[0];
      float theta   = measurement_pack.raw_measurements_[1];
      float rho_dot = measurement_pack.raw_measurements_[2];
      ekf_.x_(0)    = rho     * cos(theta);
      ekf_.x_(1)    = rho     * sin(theta);
      ekf_.x_(2)    = rho_dot * cos(theta);
      ekf_.x_(3)    = rho_dot * sin(theta);
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      float px      = measurement_pack.raw_measurements_[0];
      float py      = measurement_pack.raw_measurements_[1];

      ekf_.x_(0) = px;
      ekf_.x_(1) = py;
      ekf_.x_(2) = 0.0;
      ekf_.x_(3) = 0.0;
    }

    ekf_.F_ <<   1,   0,   1,   0,
                 0,   1,   0,   1,
                 0,   0,   1,   0,
                 0,   0,   0,   1;

    ekf_.P_ <<   1,   0,   0,   0,
                 0,   1,   0,   0,
                 0,   0, 500,   0,
                 0,   0,   0, 500;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    cout << "Init FusionEKF done.!\n";
    return;
  }
  previous_timestamp_ = measurement_pack.timestamp_;
  /**
   * Prediction
   */
   float dt  = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
   float dt2 = dt  * dt;
   float dt3 = dt2 * dt;
   float dt4 = dt3 * dt;

   ekf_.F_(0, 2) = dt;
   ekf_.F_(1, 3) = dt;
  /**
   * Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   *
   * Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
   ekf_.Q_ <<
   dt4 / 4 * noise_ax_,                  0, dt3 / 2 * noise_ax_,                   0,
                     0, dt4 / 4 * noise_ay_,                  0, dt3 / 2 * noise_ay_,
   dt3 / 2 * noise_ax_,                  0,     dt2 * noise_ax_,                   0,
                     0, dt3 / 2 * noise_ay_,                  0,      dt2 * noise_ay_;
   ekf_.Predict();

  /**
   * Update
   */

  /**
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // Tool Jacobian
    Hj_     = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
