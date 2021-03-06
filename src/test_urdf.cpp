#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/joint/joints.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/geometry.hpp"

#include "hpp/fcl/distance.h"
#include "hpp/fcl/math/transform.h"
#include "hpp/fcl/shape/geometric_shapes.h"

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
  
  // You should change here to set up your own URDF file or just pass it as an argument of this example.
  const std::string urdf_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/ur3e_robot_abs.urdf");
  const std::string box_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/box.urdf");
  
  // Load the robot urdf model
  Model robot_model, box_model, model;
  pinocchio::urdf::buildModel(urdf_filename, robot_model);
  // pinocchio::Model::Index contactId =  model.addFrame(
  //                                   pinocchio::Frame("contactPoint", 
  //                                   model.getJointId("base_to_pusher"), 
  //                                   -1, 
  //                                   pinocchio::SE3::Identity(), 
  //                                   pinocchio::OP_FRAME));
  // pinocchio::JointIndex joint_index = model.joints.size();
  // pinocchio::FrameIndex frame_index = model.nframes;
  // load the box urdf model
  pinocchio::urdf::buildModel(box_filename, JointModelPlanar(), box_model);
  // pinocchio::urdf::buildModel(box_filename, JointModelFreeFlyer(), box_model);
  setRootJointBounds(box_model, 1, "planar"); // root joint is with joint_index 1
  box_model.frames[1].name = "box_root_joint"; // index of root joint is 1. 0 = universe
  box_model.names[1] = "box_root_joint";
  pinocchio::appendModel(box_model, robot_model, 0, pinocchio::SE3::Identity(), model);
  // for (std::string& n : model.names) std::cout << n << "\n" ;
  for (Frame& f : model.frames) std::cout << f.name << "\n";
  pinocchio::Model::Index contactId =  model.addFrame(
                                    pinocchio::Frame("contactPoint", 
                                    model.getJointId("wrist_3_joint"), 
                                    -1, 
                                    pinocchio::SE3::Identity(), 
                                    pinocchio::OP_FRAME));
  pinocchio::Model::Index object_contactId =  model.addFrame(
                                  pinocchio::Frame("object_contactPoint", 
                                  model.getJointId("box_root_joint"), 
                                  -1, 
                                  pinocchio::SE3::Identity(), 
                                  pinocchio::OP_FRAME));
  std::cout << "------\n";
  std::cout << model;

  GeometryModel geom_model, box_geom_model;
  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model, PINOCCHIO_MODEL_DIR);
  pinocchio::urdf::buildGeom(model, box_filename, pinocchio::COLLISION, box_geom_model, PINOCCHIO_MODEL_DIR);
  // setPrefix("box", ground_model, ground_geom_model, 0, 0);
  // pinocchio::appendModel(model, ground_model, geom_model, ground_geom_model, 0, pinocchio::SE3::Identity(), fused_model, fused_geom_model);
  pinocchio::appendGeometryModel(geom_model, box_geom_model);
  // std::cout << geom_model << "\n";
  // // Add all possible collision pairs and remove the ones collected in the SRDF file
  // geom_model.addAllCollisionPairs();
  // pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
  GeomIndex tip_id = geom_model.getGeometryId("ee_link_0");
  GeomIndex box_id = geom_model.getGeometryId("obj_front_0");
  CollisionPair cp = CollisionPair(tip_id, box_id);
  geom_model.addCollisionPair(cp);
  PairIndex cp_index = geom_model.findCollisionPair(cp);
  
  Data data(model);
  GeometryData geom_data(geom_model);
  Eigen::VectorXd q = randomConfiguration(model);
  
  Eigen::VectorXd q_test(model.nq);
  double theta = 0.6;
  q_test << -0.05,-0.80,1.55,-0.96,1.57,-0.00, 0.50, 0.13, cos(theta), sin(theta);
  pinocchio::computeJointJacobians(model, data, q_test);
  pinocchio::framesForwardKinematics(model, data, q_test);
  Eigen::Matrix3d ee_rotation, box_root_rotation;
  Eigen::Vector3d ee_translation, box_front_translation, box_root_translation,
                  box_left_translation, box_right_translation;
  ee_rotation = data.oMf[model.getFrameId("ee_link")].rotation();
  ee_translation = data.oMf[model.getFrameId("ee_link")].translation();
  box_root_rotation = data.oMf[model.getFrameId("box")].rotation();
  box_root_translation = data.oMf[model.getFrameId("box")].translation();
  box_front_translation = data.oMf[model.getFrameId("obj_front")].translation();
  box_left_translation = data.oMf[model.getFrameId("obj_left")].translation();
  box_right_translation = data.oMf[model.getFrameId("obj_right")].translation();

  boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_geom (new hpp::fcl::Box (0.1,0.1,0.1));
  boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_ee_geom (new hpp::fcl::Cylinder (0.03, 0.00010));
  boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_front_geom (new hpp::fcl::Box (0.00010,0.08,0.08));
  boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_left_geom (new hpp::fcl::Box (0.08,0.00010,0.08));
  boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_right_geom (new hpp::fcl::Box (0.08,0.00010,0.08));

  hpp::fcl::CollisionObject fcl_box(fcl_box_geom, box_root_rotation, box_root_translation);
  hpp::fcl::CollisionObject fcl_ee(fcl_ee_geom, ee_rotation, ee_translation);
  hpp::fcl::CollisionObject fcl_box_front(fcl_box_front_geom, box_root_rotation, box_front_translation);
  hpp::fcl::CollisionObject fcl_box_left(fcl_box_left_geom, box_root_rotation, box_left_translation);
  hpp::fcl::CollisionObject fcl_box_right(fcl_box_right_geom, box_root_rotation, box_right_translation);

  hpp::fcl::DistanceRequest distReq;
  hpp::fcl::DistanceResult distRes;

  distReq.enable_nearest_points = true;

  // distance between EE and box, needs to be constrained as positive (no penetration)
  distRes.clear();
  hpp::fcl::distance(&fcl_ee, &fcl_box, distReq, distRes);
  double distance_box = distRes.min_distance;
  // distance between EE and front plane, used in force constraints
  distRes.clear();
  hpp::fcl::distance(&fcl_ee, &fcl_box_front, distReq, distRes);
  double distance_front = distRes.min_distance;
  // distance between EE and left plane, used in force constraints
  distRes.clear();
  hpp::fcl::distance(&fcl_ee, &fcl_box_left, distReq, distRes);
  double distance_left = distRes.min_distance;
  // distance between EE and right plane, used in force constraints
  distRes.clear();
  hpp::fcl::distance(&fcl_ee, &fcl_box_right, distReq, distRes);
  double distance_right = distRes.min_distance;

  std::cout << "---------------\n";
  pinocchio::Data::Matrix6x J_box(6, model.nv);
  J_box.setZero();
  getFrameJacobian(model, data, model.getFrameId("box"), LOCAL_WORLD_ALIGNED, J_box);
  std::cout << "box jacobian:\n " << J_box << "\n";

  pinocchio::SE3 root_joint_frame_placement = data.oMi[model.getJointId("box_root_joint")];
  Eigen::Vector3d goal_global(0.53, 0.00, 0.00);
  Eigen::Vector3d goal_local = (root_joint_frame_placement.inverse().act(goal_global));

  std::cout << "goal local: " << goal_local.transpose() << "\n";

  // numerical difference to get dDistance_dq
  Eigen::MatrixXd dgoal_dx;
  dgoal_dx.resize(9,3);
  dgoal_dx.setZero();

  double alpha = 1e-6;
  Eigen::VectorXd pos_eps(model.nv);
  pos_eps << -0.05,-0.80,1.55,-0.96,1.57,-0.00, 0.50, 0.13, theta;
  for(int k = 0; k < model.nv; ++k)
  {
    pos_eps[k] += alpha;
    Eigen::VectorXd q_eps(model.nq);
    q_eps.segment(0, 8) = pos_eps.segment(0, 8);
    q_eps(model.nq - 2) = cos(pos_eps(9 - 1));
    q_eps(model.nq - 1) = sin(pos_eps(9 - 1));

    pinocchio::framesForwardKinematics(model, data, q_eps);
    root_joint_frame_placement = data.oMi[model.getJointId("box_root_joint")];
    Eigen::Vector3d goal_local_plus = (root_joint_frame_placement.inverse().act(goal_global));
    // std::cout << "goal plus : " << goal_local_plus.transpose() << "\n";
    dgoal_dx.row(k) = (goal_local_plus - goal_local) / alpha;
  
    pos_eps[k] -= alpha;
  }

  std::cout << "dgoal_dx : \n" << dgoal_dx << "\n";

}