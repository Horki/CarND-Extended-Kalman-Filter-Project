#include <iostream>
#include <cmath>
#include "kalman_filter.h"

// #define PI 3.14159265

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  //Update the covariance matrix using the process noise and state transition matrix
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;

  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  //Update State
  x_ = x_ + (K * y);
  //Update covariance matrix
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);  
  P_ = (I - K*H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];
  
  float rho_p     = sqrt(px * px + py * py);
  float theta_p   = atan2(py,px);
  float rho_dot_p = (px * vx + py * vy) / rho_p;

  VectorXd z_pred = VectorXd(3);
  z_pred << rho_p, theta_p, rho_dot_p;

  VectorXd y = z - z_pred;
  
  if (y[1] > M_PI){
    y[1] = y[1] - (2 * M_PI);
  } else if (y[1] < -M_PI){
    y[1] = y[1] + (2 * M_PI);
  }

  MatrixXd Ht  = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S   = H_ * PHt + R_;
  MatrixXd K   = PHt * S.inverse();

  // Update State
  x_          += (K * y);
  // Update covariance matrix
  long x_size  = x_.size();
  MatrixXd I   = MatrixXd::Identity(x_size, x_size);  
  P_           = (I - (K * H_)) * P_;
}
