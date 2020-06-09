#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/joint/joints.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/geometry.hpp"

#include "hpp/fcl/distance.h"
#include "hpp/fcl/collision.h"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include <pinocchio/algorithm/model.hpp>
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/spatial/act-on-set.hpp"


#include "pinocchio/utils/timer.hpp"

#include <Eigen/Sparse>

#include <fstream>
#include <math.h>
#include <ctime>
#include <iostream>
// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own directory here.
#ifndef PINOCCHIO_MODEL_DIR
  #define PINOCCHIO_MODEL_DIR "/home/mzwang/catkin_ws/src/trajectory_my/model"
#endif
typedef Eigen::Triplet<double> T;


double quaternion_disp(Eigen::Quaterniond q1, Eigen::Quaterniond q2){
  Eigen::Quaterniond q = q1.inverse() * q2;
  // std::vector<double> rot_vec(3);
  // rot_vec[0] = q.vec()[0];
  // rot_vec[1] = q.vec()[1];
  // rot_vec[2] = q.vec()[2];
  if (abs(q.w()) < 1.0){
    double a = acos(q.w());
    double sina = sin(a);
    if (abs(sina) >= 0.05){
      double c = a/sina;
      q.vec()[0] *= c;
      q.vec()[1] *= c;
      q.vec()[2] *= c;
    }
  }
  return q.vec().norm();
}

void setRootJointBounds(pinocchio::Model& model,
    const pinocchio::JointIndex& rtIdx,
    const std::string& rootType)
{
  double b = 100;
  if (rootType == "freeflyer") {
    const std::size_t idx = model.joints[rtIdx].idx_q();
    model.upperPositionLimit.segment<3>(idx).setConstant(+b);
    model.lowerPositionLimit.segment<3>(idx).setConstant(-b);
    // Quaternion bounds
    b = 1.01;
    const std::size_t quat_idx = idx + 3;
    model.upperPositionLimit.segment<4>(quat_idx).setConstant(+b);
    model.lowerPositionLimit.segment<4>(quat_idx).setConstant(-b);
  } else if (rootType == "planar") {
    const std::size_t idx = model.joints[rtIdx].idx_q();
    model.upperPositionLimit.segment<2>(idx).setConstant(+b);
    model.lowerPositionLimit.segment<2>(idx).setConstant(-b);
    // Unit complex bounds
    b = 1.01;
    const std::size_t cplx_idx = idx + 2;
    model.upperPositionLimit.segment<2>(cplx_idx).setConstant(+b);
    model.lowerPositionLimit.segment<2>(cplx_idx).setConstant(-b);
  }
}

void setPrefix (const std::string& prefix,
    pinocchio::Model& model, pinocchio::GeometryModel& geomModel,
    const pinocchio::JointIndex& idFirstJoint,
    const pinocchio::FrameIndex& idFirstFrame)
{
  for (pinocchio::JointIndex i = idFirstJoint; i < model.joints.size(); ++i) {
    model.names[i] = prefix + model.names[i];
  }
  for (pinocchio::FrameIndex i = idFirstFrame; i < model.frames.size(); ++i) {
    pinocchio::Frame& f = model.frames[i];
    f.name = prefix + f.name;
  }
  BOOST_FOREACH(pinocchio::GeometryObject& go, geomModel.geometryObjects) {
    go.name = prefix + go.name;
  }
}

struct Quaternion
{
    double w, x, y, z;
};

Quaternion ToQuaternion(double yaw, double pitch, double roll) // yaw (Z), pitch (Y), roll (X)
{
    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    Quaternion q;
    q.w = cr * cp * cy + sr * sp * sy;
    q.x = sr * cp * cy - cr * sp * sy;
    q.y = cr * sp * cy + sr * cp * sy;
    q.z = cr * cp * sy - sr * sp * cy;

    return q;
}

int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  const std::string urdf_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/ur3e_robot_abs.urdf");
  Model urmodel;
  pinocchio::urdf::buildModel(urdf_filename, urmodel);
  Eigen::VectorXd urq = randomConfiguration(urmodel);
  std::cout << urmodel << "\n";
  Data urdata(urmodel);
  Eigen::VectorXd urv(6);
  //derivative test
  computeAllTerms(urmodel, urdata, urq, urv);
  Eigen::VectorXd urtau(6);
  urq << 0.5, -0.47, -0.32, -1.97, 0.97, -0.47;
  urv << 0.1,0.1,0.1,0.1,0.1,0.1;
  urtau << 0,0,0,0,0,0;
  std::cout << "check joints: " << urmodel.njoints << "\n";
  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) urfext((size_t)urmodel.njoints, pinocchio::Force::Zero());
  pinocchio::Force::Vector3 tz = pinocchio::Force::Vector3::Zero();
  pinocchio::Force::Vector3 ty = pinocchio::Force::Vector3::Zero();
  ty[0] = 0.1;
  tz[2] = 0.8;
  std::cout << "check force: \n";
  for (pinocchio::Force f : urfext) std::cout << f << "\n"; 
  computeMinverse(urmodel, urdata, urq);
  Eigen::MatrixXd M = urdata.M;
  Eigen::MatrixXd Minv = urdata.Minv;
  M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  Minv.triangularView<Eigen::StrictlyLower>() = Minv.transpose().triangularView<Eigen::StrictlyLower>();
  std::cout << "check m1: \n" << M << "\n";
  std::cout << "check minv1 : \n" << Minv << "\n";
  std::cout << "multiplication: \n " << M * Minv << "\n";
  aba(urmodel, urdata, urq, urv, urtau);
  std::cout << "check acceleration: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  // std::cout << "check if nle: " << urdata.nle << "\n";
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";
  // std::cout << "check m: \n" << urdata.M << "\n";
  // std::cout << "check derivative minv: \n" << urdata.Minv << "\n";
  // std::cout << "multiplication: \n " << M * urdata.Minv << "\n";

  urtau << 0.0,0.0,0.0,0.0,0.0,0.1;
  std::cout << "tau: " << urtau.transpose() << "\n";
  aba(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "acceleration with tau: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";

  // urfext[1].angular(tz);
  // urfext[2].angular(ty);
  urfext[6].angular(ty);
  for (pinocchio::Force f : urfext) std::cout << f << "\n"; 
  urtau << 0,0,0,0,0,0;
  aba(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "acceleration with fext: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";


}