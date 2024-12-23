#ifndef _INCLUDE_EVAL_HPP
#define _INCLUDE_EVAL_HPP

#include <string>
// Eigen
#include <Eigen/Core>

class Eval {
private:
    /* data */
public:
    Eval(){};
    ~Eval(){};

    void compute_adj_rpe(Eigen::Matrix4d& gt,
                         Eigen::Matrix4d& lo,
                         double& t_e,
                         double& r_e);
};

// Eval::eval(/* args */)
// {
// }

// Eval::~eval()
// {
// }

void Eval::compute_adj_rpe(Eigen::Matrix4d& gt,
                           Eigen::Matrix4d& lo,
                           double& t_e,
                           double& r_e) {
    Eigen::Matrix4d delta_T = lo.inverse() * gt;

    t_e = delta_T.topRightCorner(3, 1).norm();

    r_e = std::abs(std::acos(
                  fmin(fmax((delta_T.block<3, 3>(0, 0).trace() - 1) / 2, -1.0),
                       1.0))) /
          M_PI * 180;
}

#endif