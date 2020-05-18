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



int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  // You should change here to set up your own URDF file or just pass it as an argument of this example.
  const std::string urdf_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/pusher.urdf");
  const std::string srdf_filename =
    PINOCCHIO_MODEL_DIR + std::string("/srdf/pusher.srdf");
  const std::string box_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/box.urdf");
  const std::string ur_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/ur3e_robot.urdf");
  
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
  for (std::string& n : model.names) std::cout << n << "\n" ;
  std::cout << "------\n";
  for (Frame& f : model.frames) std::cout << f.name << "\n";
  pinocchio::Model::Index contactId =  model.addFrame(
                                    pinocchio::Frame("contactPoint", 
                                    model.getJointId("base_to_pusher"), 
                                    -1, 
                                    pinocchio::SE3::Identity(), 
                                    pinocchio::OP_FRAME));
  pinocchio::Model::Index object_contactId =  model.addFrame(
                                  pinocchio::Frame("object_contactPoint", 
                                  model.getJointId("box_root_joint"), 
                                  -1, 
                                  pinocchio::SE3::Identity(), 
                                  pinocchio::OP_FRAME));
  std::cout << model;

  GeometryModel geom_model, box_geom_model;
  pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model, PINOCCHIO_MODEL_DIR);
  pinocchio::urdf::buildGeom(model, box_filename, pinocchio::COLLISION, box_geom_model, PINOCCHIO_MODEL_DIR);
  // setPrefix("box", ground_model, ground_geom_model, 0, 0);
  // pinocchio::appendModel(model, ground_model, geom_model, ground_geom_model, 0, pinocchio::SE3::Identity(), fused_model, fused_geom_model);
  pinocchio::appendGeometryModel(geom_model, box_geom_model);
  // // Add all possible collision pairs and remove the ones collected in the SRDF file
  // geom_model.addAllCollisionPairs();
  // pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
  GeomIndex tip_id = geom_model.getGeometryId("tip_0");
  GeomIndex box_id = geom_model.getGeometryId("obj_front_0");
  CollisionPair cp = CollisionPair(tip_id, box_id);
  geom_model.addCollisionPair(cp);
  PairIndex cp_index = geom_model.findCollisionPair(cp);
  
  Data data(model);
  GeometryData geom_data(geom_model);
  Eigen::VectorXd q = randomConfiguration(model);

  q << 0, 1.0, 0, cos(0.0), sin(0.0);
  // forwardKinematics(model,data,q);
  // updateFramePlacements(model,data);
  // pinocchio::SE3 joint_frame_placement = data.oMf[model.getFrameId("base_to_pusher")];
  // pinocchio::SE3 root_joint_frame_placement = data.oMf[model.getFrameId("box_root_joint")];
  geom_data.collisionRequest.enable_contact = true;

  computeCollisions(model,data,geom_model,geom_data,q);
  computeDistances(model, data, geom_model, geom_data, q);

  hpp::fcl::CollisionResult cr = geom_data.collisionResults[cp_index];
  hpp::fcl::DistanceResult dr = geom_data.distanceResults[cp_index];

  std::cout << "collision pair: " << geom_model.geometryObjects[cp.first].name << " , " << geom_model.geometryObjects[cp.second].name << " - collision: ";
  std::cout << (cr.isCollision() ? "yes" : "no");
  std::cout << " - nearest point: " << dr.nearest_points[0].transpose() << "," << dr.nearest_points[1].transpose() << "\n";
  std::cout << " - distance: " << dr.min_distance << std::endl;
  std::cout << " - contacts number: " << cr.numContacts() << "\n";
  // std::cout << " - normal: " << cr.getContact(0).normal.transpose() << ", " << cr.getContact(0).pos.transpose() << "\n";

  // DEBUG
  std::cout << "Collision pair exist: " << geom_model.existCollisionPair(cp) << "\n";
  // std::cout << "Penetration depth: " << cr.getContact(0).penetration_depth << "\n";
  std::cout << "Pose of tip: \n" << geom_data.oMg[geom_model.getGeometryId("tip_0")] << "\n";
  std::cout << "Pose of front plane: \n" << geom_data.oMg[geom_model.getGeometryId("obj_front_0")] << "\n";
  std::cout << "Pose of box: \n" << geom_data.oMg[geom_model.getGeometryId("box_0")] << "\n";
  std::cout << "Normal from DistanceResult: " << dr.normal.transpose() << "\n";

  std::cout << dr.normal.normalized().transpose() << "\n";

  // Jacobian
  // forwardKinematics(model,data,q);
  // updateFramePlacements(model,data);
  framesForwardKinematics(model, data, q);
  pinocchio::SE3 joint_frame_placement = data.oMf[model.getFrameId("base_to_pusher")];
  pinocchio::SE3 root_joint_frame_placement = data.oMf[model.getFrameId("box_root_joint")];
  model.frames[contactId].placement.translation() = joint_frame_placement.inverse().act(dr.nearest_points[0]);
  model.frames[object_contactId].placement.translation() = root_joint_frame_placement.inverse().act(dr.nearest_points[1]);
  Eigen::Vector3d r_com_contact = root_joint_frame_placement.inverse().act(dr.nearest_points[1]);
  std::cout << "point on box: " << root_joint_frame_placement.inverse().act(dr.nearest_points[1]).transpose() << "\n";

  pinocchio::Data::Matrix6x w_J_contact(6, model.nv), w_J_object(6, model.nv), J_local(6, model.nv), J_wl(6, model.nv);
  w_J_contact.fill(0);
  w_J_object.fill(0);
  J_local.fill(0);
  J_wl.fill(0);
  computeJointJacobians(model, data, q);
  framesForwardKinematics(model, data, q);
  getFrameJacobian(model, data, contactId, LOCAL_WORLD_ALIGNED, w_J_contact);
  getFrameJacobian(model, data, object_contactId, WORLD, w_J_object);
  getFrameJacobian(model, data, object_contactId, LOCAL, J_local);
  getFrameJacobian(model, data, object_contactId, LOCAL_WORLD_ALIGNED, J_wl);

  pinocchio::Data::Matrix6x J_final = -1 * w_J_contact + w_J_object;

  Eigen::VectorXd J_remapped(model.nv);
  Eigen::Vector3d front_normal(1.0,0,0);
  Eigen::Vector3d front_normal_transformed =
          geom_data.oMg[geom_model.getGeometryId("box_0")].rotation() *
          front_normal;
  std::cout << "rotated normal: " << front_normal_transformed << "\n";

 
  J_remapped = -1 * w_J_contact.topRows(3).transpose() * 
                  geom_data.oMg[geom_model.getGeometryId("tip_0")].rotation().transpose() * 
                  front_normal;

  std::cout << "kanyikan " << geom_data.oMg[geom_model.getGeometryId("tip_0")].rotation().transpose() << "\n";

  std::cout << "contact jacobian: " << w_J_contact << "\n";
  std::cout << "object jacobian: " << w_J_object << "\n";
  std::cout << "J_all" << J_remapped << "\n";

  std::cout << "Local: \n" << J_local << "\n";
  std::cout << "World: \n" << w_J_object << "\n";
  std::cout << "Aligend: \n" << J_wl << "\n";

  // Eigen::VectorXd v(4), tau(4);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Random(model.nv);

  computeABADerivatives(model, data, q, v, tau);

  std::cout << "check Minv: " <<  data.Minv << "\n";

  typedef PINOCCHIO_ALIGNED_STD_VECTOR(Force) ForceVector;

  std::cout << "joint number: " << model.njoints << "\n";

  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext((size_t)model.njoints, pinocchio::Force::Zero());

  Eigen::MatrixXd screw_transform;
  screw_transform.resize(6,3);
  screw_transform.topRows(3).setIdentity();
  screw_transform.bottomRows(3) << 0, -r_com_contact(2), r_com_contact(1), 
                                   r_com_contact(2), 0, -r_com_contact(0),
                                  -r_com_contact(1), r_com_contact(0), 0;
  Eigen::Vector3d normal(1,0,0);
  std::cout << "check screw: \n" << screw_transform << "\n";
  std::cout << "check force:  " << screw_transform * normal << "\n";

  pinocchio::Force::Vector3 f1 = pinocchio::Force::Vector3::Zero();
  pinocchio::Force::Vector3 f2 = pinocchio::Force::Vector3::Zero();
  f1[0] = -1;
  f2[0] = 1;
  fext[1].linear(f1);
  fext[2].linear(f2);
  std::cout << "fext 1: " << fext[1] << "\n";
  std::cout << "fext 2: " << fext[2] << "\n"; 
  // q << 0,1,0,cos(0), sin(0);
  // std::cout << "q : " << q.transpose() << "\n";
  // v << 0,0,0,0;
  std::cout << "v: " << v.transpose() << "\n";
  tau << -1,1,0,0;
  aba(model, data, q, v, tau, fext);
  std::cout << "check acceleration: " << data.ddq.transpose() << "\n";
  pinocchio::computeABADerivatives(model, data, q, v, tau, fext);
  std::cout << "ddq_dq: \n" << data.ddq_dq << "\n";
  std::cout << "ddq_dv: \n" << data.ddq_dv << "\n";
  computeJointKinematicHessians(model, data, q);
  std::cout << "hessian: \n" << getJointKinematicHessian(model, data, model.getJointId("box_root_joint"), LOCAL) << "\n";
  std::cout << "hessian: \n" << getJointKinematicHessian(model, data, model.getJointId("base_to_pusher"), LOCAL) << "\n";

  std::cout << "joint1 check: \n" << model.joints[1];
  std::cout << "joint2 check: \n" << model.joints[2];
  

  std::cout << "------------------------------\n";
  Model urmodel;
  pinocchio::urdf::buildModel(box_filename, JointModelFreeFlyer(), urmodel);
  setRootJointBounds(urmodel, 1, "freeflyer");
  Eigen::VectorXd urq = randomConfiguration(urmodel);
  Data urdata(urmodel);
  Eigen::VectorXd urv(urmodel.nv);
  //derivative test
  computeAllTerms(urmodel, urdata, urq, urv);
  Eigen::VectorXd urtau(urmodel.nv);
  // urq << 0.5, 0.2, 0.5, 0.2, 0.2, 0.3;
  urv << 1,1,1,1,1,1;
  urtau << 2,2,2,2,2,2;
  std::cout << "check joints: " << urmodel.njoints << "\n";
  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) urfext((size_t)urmodel.njoints, pinocchio::Force::Zero());
  pinocchio::Force::Vector3 t = pinocchio::Force::Vector3::Zero();
  pinocchio::Force::Vector3 f = pinocchio::Force::Vector3::Zero();
  f << 10,10,10;
  t << 10,10,10;
  urfext[2].linear(f);
  urfext[2].angular(t);
  computeMinverse(urmodel, urdata, urq);
  Eigen::MatrixXd M = urdata.M;
  Eigen::MatrixXd Minv = urdata.Minv;
  M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  Minv.triangularView<Eigen::StrictlyLower>() = Minv.transpose().triangularView<Eigen::StrictlyLower>();
  std::cout << "check m1: \n" << M << "\n";
  std::cout << "check minv1 : \n" << Minv << "\n";
  std::cout << "multiplication: \n " << M * Minv << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  std::cout << "check if nle: " << urdata.nle << "\n";
  std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";
  std::cout << "check m: \n" << urdata.M << "\n";
  std::cout << "check derivative minv: \n" << urdata.Minv << "\n";
  std::cout << "multiplication: \n " << M * urdata.Minv << "\n";

  computeABADerivatives(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";

}