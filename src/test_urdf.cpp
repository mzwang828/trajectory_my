#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include <Eigen/Sparse>


#include <ctime>
#include <iostream>
// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own directory here.
#ifndef PINOCCHIO_MODEL_DIR
  #define PINOCCHIO_MODEL_DIR "/home/mzwang/catkin_ws/src/trajectory_my/model"
#endif
int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  // You should change here to set up your own URDF file or just pass it as an argument of this example.
  const std::string urdf_filename = PINOCCHIO_MODEL_DIR + std::string("/ur5_robot.urdf");
  
  // Load the urdf model
  Model model;
  pinocchio::urdf::buildModel(urdf_filename,model);
  
  // Create data required by the algorithms
  Data data(model);
  
  // Sample a random configuration
  Eigen::VectorXd q = randomConfiguration(model);
  Eigen::VectorXd v(6);
  /*
  std::cout << "q: " << q.transpose() << std::endl;
  // Perform the forward kinematics over the kinematic tree
  forwardKinematics(model,data,q);
  // Print out the placement of each joint of the kinematic tree
  for(JointIndex joint_id = 1; joint_id < (JointIndex)model.njoints; ++joint_id)
    std::cout << model.names[joint_id] << "\t\t: "
              << data.oMi[joint_id].translation().transpose()
              << std::endl;
  */

  // kinematics test
  // pinocchio::FrameIndex frameID = model.getFrameId("ee_fixed_joint");
  // q << 0, 0, 0, 0, 0, 0;
  // forwardKinematics(model,data,q);
  // pinocchio::updateFramePlacement(model, data, frameID);
  // pinocchio::GeometryData::SE3 ini_pose = data.oMf[frameID];
  // Eigen::Quaterniond ini_quater(ini_pose.rotation());
  // std::cout << "initial pose: \n" << ini_pose << "\n";
  // // std::cout << "initial pose: " << ini_pose.rotation() << "\n";
  // // std::cout << "initial pose: " << ini_pose.rotation().inverse() << "\n";
  // // std::cout << "initial pose: " << ini_pose.translation() << "\n";
  // std::cout << "initial pose quaternion: " << ini_quater.w() << "\n";
  // std::cout << "initial pose quaternion: " << ini_quater.vec().transpose() << "\n";

  // q << 0.5,0,0.5,0,0.5,0;
  // q << 0.58, 0.92, -1.06, -0.67, 1.39, 0.16;
  // forwardKinematics(model,data,q);
  // pinocchio::updateFramePlacement(model, data, frameID);
  // pinocchio::GeometryData::SE3 final_pose = data.oMf[frameID];
  // Eigen::Quaterniond end_quater(final_pose.rotation());
  // std::cout << "q: " << q.transpose() << std::endl;
  // std::cout << "final pose: \n" << final_pose << "\n";
  // std::cout << "final pose quaternion: " << end_quater.w() << "\n";
  // std::cout << "final pose quaternion: " << end_quater.vec().transpose() << "\n";
  // std::cout << "inverse w: " << end_quater.inverse().w() << "\n";
  // std::cout << "inverse vec: " << end_quater.inverse().vec().transpose() << "\n";
  // std::cout << "quaternion multi w: " << log((end_quater * end_quater.inverse()).norm()) << "\n";

  // derivative test
  // computeAllTerms(model, data, q, v);
  // Eigen::VectorXd tau(6);
  // q << 0, 0, 0, 0, 0, 0;
  // v << 1,1,1,1,1,1;
  // tau << 2,2,2,2,2,2;
  // computeABADerivatives(model, data, q, v, tau);
  // std::cout << "check derivative ddq_dq: " << data.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: " << data.ddq_dv << "\n";
  // std::cout << "check derivative minv: " << data.Minv << "\n";
  // std::cout << "----------\n";
  // v << 2,2,2,2,2,2;
  // tau << 2,2,2,2,2,2;
  // computeABADerivatives(model, data, q, v, tau);
  // std::cout << "check derivative ddq_dq: " << data.ddq_dq << "\n";
  // std::cout << "check derivative ddq_dv: " << data.ddq_dv << "\n";
  // std::cout << "check derivative minv: " << data.Minv << "\n";


  /*
  // dynmaics test
  q << 0, 0, 0, 0, 0, 0;
  v << 0, 0, 0, 0, 0, 0;
  // computeAllTerms(model, data, q, v);
  // time_t tstart, tend; 
  // tstart = time(0);
  crba(model,data,q);
  nonLinearEffects(model,data,q,v);
  // tend = time(0); 
  // std::cout << "It took "<< tend-tstart << std::endl;

  std::cout << data.M << "\n";
  std::cout << data.Minv << "\n";
  std::cout << data.nle << "\n";
  
  // tstart = time(0);
  q = randomConfiguration(model);
  v << 0.3, 0, 0, 0, 0, 0;
  // tstart = time(0);
  computeAllTerms(model, data, q, v);
  // tend = time(0); 
  // std::cout << "It took "<< tend << std::endl;
  std::cout << "------" << "\n";
  std::cout << data.M << "\n";
  std::cout << data.Minv << "\n";
  std::cout << data.nle << "\n";
*/

  // eigen test
  Eigen::MatrixXd Mat(3,3);
  Eigen::VectorXd vec(3);
  Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
  Mat << 1,0,0,2,1,0,0,0,1;
  vec << 0,0,0;
  std::cout << Mat << "\n";
  SpMat = Mat.sparseView();
  SpMat.coeffRef(0,0) = 0;
  SpMat.coeffRef(0,1) = 0;
  SpMat.coeffRef(0,2) = 0;
  SpMat.prune(0,0);
  std::cout << SpMat << "\n";
}