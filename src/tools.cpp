#include "tools.h"
#include <iostream>
#include <cmath>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  VectorXd rmse = VectorXd(4);
  // init "empty" RMSE
  rmse << 0, 0, 0, 0;

  // TODO: throw exception
  // Check validity of two inputs
  if (estimations.empty() || ground_truth.size() != estimations.size()) {
    std::cerr << "Invalid estimations or ground truth data\n";
    return rmse;
  }

  // TODO: use c++ std for this
  // Accumulate squared residuals
  for (size_t i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    // Coefficent-Wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse /= estimations.size();

  // Calculate the squared root
  return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  // State params
  MatrixXd Hj(3, 4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  float c1 = pow(px, 2.0) + pow(px, 2.0);
  float c2 = sqrt(c1);
  float c3 = c1 * c2;

  // check division by zero
  // TODO: throw exception
  if (fabs(c1) < 0.0001) {
    std::cerr << "Tools::CalculateJacobian(): Err - Division by Zero\n";
    return Hj;
  }

  // compute the Jacobian matrix
  Hj <<              (px/c2),               (py/c2),     0,     0,
                    -(py/c1),               (px/c1),     0,     0,
       py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
