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
  const std::string point_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/point.urdf");

  Model boxmodel, pmodel, urmodel;
  GeometryModel geom_model, pgeom_model;

  Eigen::Matrix<double, 3, 1> point_translation;
  point_translation << 0,0, 0;
  Eigen::Vector3d force;
  force << 5, 0, 0;
  pinocchio::urdf::buildModel(point_filename, pmodel);
  pinocchio::urdf::buildModel(box_filename, JointModelFreeFlyer(), boxmodel);
  setRootJointBounds(boxmodel, 1, "freeflyer");
  boxmodel.frames[1].name = "box_root_joint"; // index of root joint is 1. 0 = universe
  boxmodel.names[1] = "box_root_joint";
  pinocchio::SE3 point_pose;
  point_pose = pinocchio::SE3::Identity();
  point_pose.translation() = point_translation;

  pinocchio::appendModel(boxmodel, pmodel, 0, point_pose, urmodel);
  
  pinocchio::Model::Index object_contactId =  urmodel.addFrame(
                                  pinocchio::Frame("contactPoint", 
                                  urmodel.getJointId("box_root_joint"), 
                                  -1, 
                                  pinocchio::SE3::Identity(), 
                                  pinocchio::OP_FRAME));

  pinocchio::urdf::buildGeom(urmodel, box_filename, pinocchio::COLLISION, geom_model, PINOCCHIO_MODEL_DIR);
  pinocchio::urdf::buildGeom(urmodel, point_filename, pinocchio::COLLISION, pgeom_model, PINOCCHIO_MODEL_DIR);
  pinocchio::appendGeometryModel(geom_model, pgeom_model);

  GeomIndex point_id = geom_model.getGeometryId("point_0");
  GeomIndex box_id = geom_model.getGeometryId("obj_front_0");
  CollisionPair cp = CollisionPair(box_id, point_id);
  geom_model.addCollisionPair(cp);
  PairIndex cp_index = geom_model.findCollisionPair(cp);

  std::cout << "check model: " << urmodel << "\n";

  Data urdata(urmodel);
  GeometryData geom_data(geom_model);
  geom_data.collisionRequest.enable_contact = true;
  Eigen::VectorXd urq = randomConfiguration(urmodel);
  Eigen::VectorXd urv(urmodel.nv);
  //derivative test
  // computeAllTerms(urmodel, urdata, urq, urv);
  Eigen::VectorXd urtau(urmodel.nv);
  Quaternion qt;
  qt = ToQuaternion(0.0, 0, 0);
  urq << 1, 0.01, 0, qt.x, qt.y, qt.z, qt.w;
  urv << 1,1,1,1,1,1;
  urtau << 0,0,0,0,0,0;

  computeCollisions(urmodel,urdata,geom_model,geom_data,urq);
  computeDistances(urmodel, urdata, geom_model, geom_data, urq);
  hpp::fcl::DistanceResult dr = geom_data.distanceResults[cp_index];
  std::cout << " - nearest point: " << dr.nearest_points[0].transpose() << "," << dr.nearest_points[1].transpose() << "\n";
  std::cout << "Pose of point: \n" << geom_data.oMg[geom_model.getGeometryId("point_0")] << "\n";
  std::cout << "Pose of front plane: \n" << geom_data.oMg[geom_model.getGeometryId("obj_front_0")] << "\n";
  pinocchio::computeJointJacobians(urmodel, urdata, urq);
  pinocchio::framesForwardKinematics(urmodel, urdata, urq);
  pinocchio::SE3 root_joint_frame_placement = urdata.oMf[urmodel.getFrameId("box_root_joint")];
  Eigen::Vector3d object_r_j2c = root_joint_frame_placement.inverse().act(dr.nearest_points[0]);
  Eigen::Vector3d object_r_j2c_true = object_r_j2c;
  Eigen::VectorXd force_ext(6);
  force_ext.head(3) = force;
  force_ext(3) = -object_r_j2c(2) * force(1) + object_r_j2c(1) * force(2);
  force_ext(4) = object_r_j2c(2) * force(0) - object_r_j2c(0) * force(2);
  force_ext(5) = -object_r_j2c(1) * force(0) + object_r_j2c(0) * force(1);
  pinocchio::Force::Vector6 fext_ref = force_ext;
  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext((size_t)urmodel.njoints, pinocchio::Force::Zero());
  fext[1] = pinocchio::ForceRef<pinocchio::Force::Vector6>(fext_ref);
  pinocchio::aba(urmodel, urdata, urq, urv, urtau, fext);
  Eigen::VectorXd a0 = urdata.ddq;

  Eigen::VectorXd q_plus(urmodel.nq), a_plus(urmodel.nv);
  Eigen::VectorXd v_eps(Eigen::VectorXd::Zero(urmodel.nv));
  Eigen::MatrixXd aba_partial_dq_fd(urmodel.nv,urmodel.nv); aba_partial_dq_fd.setZero();
  double alpha = 1e-8;
  for(int k = 0; k < urmodel.nv; ++k)
  {
    v_eps[k] += alpha;
    q_plus = integrate(urmodel,urq,v_eps);

    computeCollisions(urmodel,urdata,geom_model,geom_data,q_plus);
    computeDistances(urmodel, urdata, geom_model, geom_data, q_plus);
    dr = geom_data.distanceResults[cp_index];

    pinocchio::computeJointJacobians(urmodel, urdata, q_plus);
    pinocchio::framesForwardKinematics(urmodel, urdata, q_plus);
    root_joint_frame_placement = urdata.oMf[urmodel.getFrameId("box_root_joint")];
    object_r_j2c = root_joint_frame_placement.inverse().act(dr.nearest_points[0]);
    Eigen::VectorXd force_ext_plus(6);
    force_ext_plus.head(3) = force;
    force_ext_plus(3) = -object_r_j2c(2) * force(1) + object_r_j2c(1) * force(2);
    force_ext_plus(4) = object_r_j2c(2) * force(0) - object_r_j2c(0) * force(2);
    force_ext_plus(5) = -object_r_j2c(1) * force(0) + object_r_j2c(0) * force(1);
    pinocchio::Force::Vector6 fext_ref_plus = force_ext_plus;
    PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext_plus((size_t)urmodel.njoints, pinocchio::Force::Zero());
    fext_plus[1] = pinocchio::ForceRef<pinocchio::Force::Vector6>(fext_ref_plus);

    a_plus = aba(urmodel,urdata,q_plus,urv,urtau,fext_plus);

    aba_partial_dq_fd.col(k) = (a_plus - a0)/alpha;
    v_eps[k] -= alpha;
  }
  


  pinocchio::computeCollisions(urmodel, urdata, geom_model, geom_data,
                               urq);
  pinocchio::computeDistances(urmodel, urdata, geom_model, geom_data,
                              urq);
  Eigen::Vector3d force_world =
      geom_data.oMg[geom_model.getGeometryId("box_0")].rotation() *
      force;
  // std::cout << "check transform: " << geom_data.oMg[geom_model.getGeometryId("box_0")].rotation() << "\n";
  double fx = force_world(0);
  double fy = force_world(1);
  double fz = force_world(2);
  double mx = 0;
  double my = 0;
  double mz = 0;
  double th = 0.0;
  double lx = object_r_j2c_true(0);
  double ly = object_r_j2c_true(1);
  std::cout << "r_j2c: " << object_r_j2c_true << "\n";
  Eigen::VectorXd force_with_moment(6);
  force_with_moment.setZero();
  force_with_moment.head(3) = force_world;

  Eigen::MatrixXd dJdtheta(6,6);
  dJdtheta << -sin(0.6), -cos(0.6), 0, 0,  0, -lx*cos(0.6)+ly*sin(0.6),
          cos(0.6), -sin(0.6), 0, 0, 0,  -lx*sin(0.6)-ly*cos(0.6),
          0, 0, 0,  0,   0, 0,
          0, 0, 0, -sin(0.6), -cos(0.6), 0,
          0, 0, 0, cos(0.6), -sin(0.6), 0,
          0,0,0,0,0,0;
  Eigen::MatrixXd dJdqf(6,6);
  dJdqf.setZero();
  dJdqf.rightCols(1) = dJdtheta.transpose() * force_with_moment;
  // dJdqf.bottomRows(1) = (dJdtheta.transpose() * force_with_moment).transpose();

  std::cout << "force in world: " << force_world.transpose() << "\n";

  std::cout << "dJ^T/dq : \n" << dJdtheta.transpose() << "\n";
  std::cout << " dJ^T/dq*f : \n" << dJdqf << "\n";

  
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);

  Eigen::MatrixXd Minv = urdata.Minv;

  Minv.triangularView<Eigen::StrictlyLower>() =
      Minv.transpose().triangularView<Eigen::StrictlyLower>();
  Eigen::MatrixXd analy(6,6);
  analy = Minv * dJdqf + urdata.ddq_dq;

  std:: cout << "Minv * dJ^T/dq * f: \n" << Minv * dJdqf << "\n";
  // std::cout << "ddq_dq: \n" << u rdata.ddq_dq << "\n";
  std::cout << "analytical: \n " << analy << "\n";

  std::cout << "numerical: \n" << aba_partial_dq_fd << "\n";


  // compare aba results
  // aba(urmodel, urdata, urq, urv, urtau, fext);
  // std::cout << "acceleration with fext: " << urdata.ddq.transpose() << "\n";


  // urmodel.frames[object_contactId].placement.translation() = contact_point;
  // pinocchio::computeJointJacobians(urmodel, urdata, urq);
  // pinocchio::framesForwardKinematics(urmodel, urdata, urq);
  // pinocchio::Data::Matrix6x w_J_contact(6, urmodel.nv);
  // std::cout << "get frame pose: " << urdata.oMf[object_contactId] << "\n";
  // getFrameJacobian(urmodel, urdata, object_contactId, pinocchio::LOCAL_WORLD_ALIGNED, w_J_contact);
  // Eigen::VectorXd tau_from_fext(6);
  
  // tau_from_fext = w_J_contact.transpose() * force_with_moment;
  // std::cout << "Jacobian: " << w_J_contact << "\n";
  // std::cout << "force: " << force_with_moment << "\n";
  // aba(urmodel, urdata, urq, urv, urtau+tau_from_fext);
  // std::cout << "acceleration with tau: \n " << urdata.ddq << "\n";


  // Eigen::MatrixXd J_lw(6,6);
  // J_lw.setZero();
  // J_lw << cos(0.6), -sin(0.6), 0, 0,  0, -lx*sin(0.6)-ly*cos(0.6),
  //         sin(0.6), cos(0.6), 0, 0, 0,  lx*cos(0.6)-ly*sin(0.6),
  //         0, 0, 1,  ly,   -lx, 0,
  //         0, 0, 0, cos(0.6), -sin(0.6), 0,
  //         0, 0, 0, sin(0.6), cos(0.6), 0,
  //         0,0,0,0,0,1;

  // Eigen::VectorXd a_from_force(6);
  // a_from_force = Minv * J_lw.transpose() * force_with_moment;
  // aba(urmodel, urdata, urq, urv, urtau);
  // std::cout << "acceleration with analytical Jacobian: " << urdata.ddq.transpose() +  a_from_force.transpose() << "\n";

  // std::cout << "Local_world_aligned Jacobian: \n" << J_lw << "\n";
  // std::cout << "Pinocchio's Jacobian: \n " << w_J_contact << "\n";
  // std::cout << "fext: " << fext[1] << "\n";
  // std::cout << "world force: " << force_with_moment << "\n";
  // std::cout << "theta: " << 0.6 << ", lx: " << lx << ", ly: " << ly << "\n";
  

}