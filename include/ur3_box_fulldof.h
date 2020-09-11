#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pinocchio/algorithm/model.hpp>

#include "hpp/fcl/distance.h"
#include "hpp/fcl/math/transform.h"
#include "hpp/fcl/shape/geometric_shapes.h"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/geometry.hpp"
#include "pinocchio/multibody/joint/joints.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/act-on-set.hpp"
// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own
// directory here.
#ifndef PINOCCHIO_MODEL_DIR
#define PINOCCHIO_MODEL_DIR "/home/mzwang/catkin_ws/src/trajectory_my/model"
#endif
typedef Eigen::Triplet<double> T;

namespace ifopt {

class ExVariables : public VariableSet {
public:
  // Every variable set has a name, here "var_set1". this allows the constraints
  // and costs to define values and Jacobians specifically w.r.t this variable
  // set.
  ExVariables(int n) : ExVariables(n, "var_set1"){};
  ExVariables(int n, const std::string &name) : VariableSet(n, name) {
    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_dof = params["n_dof"].as<int>();
    n_control = params["n_control"].as<int>();
    n_exforce = params["n_exforce"].as<int>();
    n_step = params["n_step"].as<int>();
    t_step = params["t_step"].as<double>();
    goal_x = params["box_goal_x"].as<double>();
    
    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    if (name == "position") {
      // for (int i = 0; i < n_step/2; i++){
      //   xvar(i*n_dof) =  0;
      //   xvar(i*n_dof+1) = -1.0 + i * (0.42*2/n_step);
      //   xvar(i*n_dof+2) = 1.86 - i * (0.43*2/n_step);
      //   xvar(i*n_dof+3) = -0.8 - i * (0.3*2/n_step);
      //   xvar(i*n_dof+4) = 1.57;
      //   xvar(i*n_dof+5) = 0;
      //   xvar(i*n_dof+6) = 0.49;
      //   xvar(i*n_dof+7) = 0.131073;
      //   xvar(i*n_dof+8) = 0;
      // }
      // for (int i = n_step/2; i < n_step; i++){

      //   xvar(i*n_dof) =  0;
      //   xvar(i*n_dof+1) = -0.58;
      //   xvar(i*n_dof+2) = 1.43;
      //   xvar(i*n_dof+3) = -0.83;
      //   xvar(i*n_dof+4) = 1.57;
      //   xvar(i*n_dof+5) = 0;
      //   xvar(i*n_dof+6) = 0.49 + (i - n_step/2) * (0.14*2/n_step);
      //   xvar(i*n_dof+7) = 0.131073;
      //   xvar(i*n_dof+8) = 0;
      // }
      for (int i = 0; i < n; i++)
        xvar(i) = 0.3;
    } else if (name == "velocity") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0.0;
    } else if (name == "effort") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0;
    } else if (name == "exforce") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0;
    } else if (name == "slack") {
      for (int i = 0; i < n; i++) {
        xvar(i) = 1e-2;
      }
    }
  }

  ExVariables(int n, const std::string &name, Eigen::VectorXd& init_values) : VariableSet(n, name) {
    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_dof = params["n_dof"].as<int>();
    n_control = params["n_control"].as<int>();
    n_exforce = params["n_exforce"].as<int>();
    n_step = params["n_step"].as<int>();
    t_step = params["t_step"].as<double>();
    goal_x = params["box_goal_x"].as<double>();

    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    xvar = init_values;
  }

  // Here is where you can transform the Eigen::Vector into whatever
  // internal representation of your variables you have (here two doubles, but
  // can also be complex classes such as splines, etc..
  void SetVariables(const VectorXd &x) override {
    for (int i = 0; i < x.size(); i++)
      xvar(i) = x(i);
  };

  // Here is the reverse transformation from the internal representation to
  // to the Eigen::Vector
  VectorXd GetValues() const override { return xvar; };

  // Each variable has an upper and lower bound set here
  VecBound GetBounds() const override {
    VecBound bounds(GetRows());
    if (GetName() == "position") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(-position_lim, position_lim);
      bounds.at(0) = Bounds(0, 0);
      bounds.at(1) = Bounds(-1.0, -1.0);
      bounds.at(2) = Bounds(1.86, 1.86);
      bounds.at(3) = Bounds(-0.8, -0.8);
      bounds.at(4) = Bounds(1.57, 1.57);
      bounds.at(5) = Bounds(0, 0);
      bounds.at(6) = Bounds(0.49, 0.49);
      bounds.at(7) = Bounds(0.131073, 0.131073);
      bounds.at(8) = Bounds(0, 0);

      bounds.at(GetRows() - 3) = Bounds(goal_x, goal_x);
      // bounds.at(GetRows() - 2) = Bounds(0.3, 0.3);
    } else if (GetName() == "velocity") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(-velocity_lim, velocity_lim);
      for (int i = 0; i < n_dof; i++)
        bounds.at(i) = Bounds(0, 0);
      for (int i = GetRows() - n_dof; i < GetRows(); i++)
        bounds.at(i) = Bounds(0, 0);
    } else if (GetName() == "effort") {
      for (int i = 0; i < n_step; i ++){
        bounds.at(i*n_control) = Bounds(-330, 330);
        bounds.at(i*n_control+1) = Bounds(-330, 330);
        bounds.at(i*n_control+2) = Bounds(-150, 150);
        bounds.at(i*n_control+3) = Bounds(-54, 54);
        bounds.at(i*n_control+4) = Bounds(-54, 54);
        bounds.at(i*n_control+5) = Bounds(-54, 54);
      }
    } else if (GetName() == "exforce") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(0, force_lim); // NOTE
    } else if (GetName() == "slack") {
      for (int i = 0; i < GetRows()/2; i++) {
        bounds.at(i) = Bounds(0, inf);                        // distance 
        bounds.at(GetRows()/2 + i) = Bounds(0, 1e-2);                       // phi*gamma < slack
      }
    }
    return bounds;
  }

private:
  Eigen::VectorXd xvar;
  int n_dof;                    // number of freedom
  int n_control;                // number of control
  int n_exforce;                // number of external force
  int n_step;                   // number of steps or (knot points - 1)
  double t_step;                // length of each step
  double position_lim = 3.14;
  double velocity_lim = 5;
  double effort_lim = 200;
  double force_lim = 100;
  double goal_x;
};

// system dynamics constraints
class ExConstraint : public ConstraintSet {
public:
  const std::string robot_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/ur3e_robot_abs.urdf");
  const std::string box_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/box.urdf");

  mutable pinocchio::Model robot_model, box_model, model;
  pinocchio::GeometryModel geom_model, box_geom_model;
  pinocchio::PairIndex cp_index;
  pinocchio::FrameIndex contactId, object_contactId;
  Eigen::Vector3d front_normal; // normal vector point to the front plane
  Eigen::Vector3d robot_contact_normal;
  mutable Eigen::VectorXd distance_cache; 
  mutable Eigen::MatrixXd J_remapped; // used to save calculated Jacobians for exforce
  mutable Eigen::MatrixXd fext_robot, fext_object; // used to save fext values for each joint
  mutable Eigen::MatrixXd dDistance_box_dq, dDistance_front_dq; // numerical gradients for distance
  // Input Mapping & Friction
  Eigen::MatrixXd B;
  Eigen::VectorXd f;
  int n_dof;                    // number of freedom
  int n_control;                // number of control
  int n_exforce;                // number of external force
  int n_step;                   // number of steps or (knot points - 1)
  double t_step;                // length of each step

  ExConstraint(int n) : ExConstraint(n, "constraint1") {}
  ExConstraint(int n, const std::string &name) : ConstraintSet(n, name) {
    front_normal << 1, 0, 0;
    robot_contact_normal << 0, -1, 0;
    // build the pusher model
    pinocchio::urdf::buildModel(robot_filename, robot_model);
    // build the box model
    pinocchio::urdf::buildModel(box_filename, pinocchio::JointModelPlanar(),
                                box_model);
    // set planar joint bounds, root joint is with joint_index 1
    setRootJointBounds(box_model, 1);
    // change box root joint name, otherwise duplicated with robot root joint
    box_model.frames[1].name = "box_root_joint"; // index of root joint is 1. 0 = universe
    box_model.names[1] = "box_root_joint";
    pinocchio::SE3 robot_base_pose = pinocchio::SE3::Identity();
    robot_base_pose.translation() << 0, 0, -0.05;
    pinocchio::appendModel(box_model, robot_model, 0, robot_base_pose, model);
    // add virtual contact point frame for Jacobian calculation
    // add as many as needed
    contactId = model.addFrame(
        pinocchio::Frame("contactPoint", model.getJointId("wrist_3_joint"), -1,
                         pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    object_contactId = model.addFrame(pinocchio::Frame(
        "object_contactPoint", model.getJointId("box_root_joint"), -1,
        pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    // build the geometry model
    pinocchio::urdf::buildGeom(model, robot_filename, pinocchio::COLLISION,
                               geom_model, PINOCCHIO_MODEL_DIR);
    pinocchio::urdf::buildGeom(model, box_filename, pinocchio::COLLISION,
                               box_geom_model, PINOCCHIO_MODEL_DIR);
    pinocchio::appendGeometryModel(geom_model, box_geom_model);

    // define the potential collision pair, as many as needed
    pinocchio::GeomIndex tip_id = geom_model.getGeometryId("ee_link_0");
    pinocchio::GeomIndex front_id = geom_model.getGeometryId("obj_front_0");
    pinocchio::CollisionPair cp = pinocchio::CollisionPair(tip_id, front_id);
    geom_model.addCollisionPair(cp);
    cp_index = geom_model.findCollisionPair(cp);

    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_dof = params["n_dof"].as<int>();
    n_control = params["n_control"].as<int>();
    n_exforce = params["n_exforce"].as<int>();
    n_step = params["n_step"].as<int>();
    t_step = params["t_step"].as<double>();
    J_remapped.resize(n_dof, n_step - 1);
    J_remapped.setZero();
    fext_robot.resize(6, n_step - 1); // fext for robot
    fext_robot.setZero();
    fext_object.resize(6, n_step - 1); // fext for box
    fext_object.setZero();

    B.resize(n_dof, n_control);
    B.setZero();
    B.topRows(n_control).setIdentity();
    f.resize(n_dof);
    f.setZero();
    f.tail(3) << 1.0, 0.0, 0.0;  

    distance_cache.resize(n_step);
    distance_cache.setZero();

    dDistance_box_dq.resize(n_step, n_dof);
    dDistance_box_dq.setZero();
    dDistance_front_dq.resize(n_step, n_dof);
    dDistance_front_dq.setZero();
  }

  void setRootJointBounds(pinocchio::Model &model,
                          const pinocchio::JointIndex &rtIdx) {
    double b = 5;
    const std::size_t idx = model.joints[rtIdx].idx_q();
    model.upperPositionLimit.segment<2>(idx).setConstant(+b);
    model.lowerPositionLimit.segment<2>(idx).setConstant(-b);
    // Unit complex bounds
    b = 1.01;
    const std::size_t cplx_idx = idx + 2;
    model.upperPositionLimit.segment<2>(cplx_idx).setConstant(+b);
    model.lowerPositionLimit.segment<2>(cplx_idx).setConstant(-b);
  }

  VectorXd GetValues() const override {
    pinocchio::Data data(model);
    pinocchio::Data data_next(model);
    pinocchio::GeometryData geom_data(geom_model);
    VectorXd g(GetRows());
    VectorXd pos = GetVariables()->GetComponent("position")->GetValues();
    VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
    VectorXd effort = GetVariables()->GetComponent("effort")->GetValues();
    VectorXd exforce = GetVariables()->GetComponent("exforce")->GetValues();
    VectorXd slack = GetVariables()->GetComponent("slack")->GetValues();
    // set contraint for each knot point
    // constraints shape
    /////////////////////////////////////////*  *////////
    // [
    //   ndof * (n_steps - 1) constraints for q_dot
    //   --------------------------------
    //   ndof * (n_steps - 1) constraints for q_ddot
    //   --------------------------------
    //   n_steps - 1 constraints for distance (with slack)
    //   --------------------------------
    //   n_steps - 1 constraints for exforce (with slack)
    //   --------------------------------
    //   n_steps - 1 constraints for complementary (slack)
    //   --------------------------------
    // ]
    ///////////////////////////////////////////////
    for (int i = 0; i < n_step - 1; i++) {
      // construct model configuration. q[-2] = c_theta, q[-1] = s_theta
      Eigen::VectorXd q(model.nq), q_next(model.nq);
      q.segment(0, n_dof - 1) = pos.segment(n_dof * i, n_dof - 1);
      q(model.nq - 2) = cos(pos(n_dof * i + n_dof - 1));
      q(model.nq - 1) = sin(pos(n_dof * i + n_dof - 1));
      q_next.segment(0, n_dof - 1) = pos.segment(n_dof * (i + 1), n_dof - 1);
      q_next(model.nq - 2) = cos(pos(n_dof * (i + 1) + n_dof - 1));
      q_next(model.nq - 1) = sin(pos(n_dof * (i + 1) + n_dof - 1));

      // calculate signed distance
      // pinocchio::framesForwardKinematics(model, data, q);
      pinocchio::framesForwardKinematics(model, data, q_next);
      Eigen::Matrix3d ee_rotation, box_front_rotation, box_root_rotation;
      Eigen::Vector3d ee_translation, box_front_translation, box_root_translation;
      ee_rotation = data.oMf[model.getFrameId("ee_link")].rotation();
      ee_translation = data.oMf[model.getFrameId("ee_link")].translation();
      box_front_rotation = data.oMf[model.getFrameId("obj_front")].rotation();
      box_front_translation = data.oMf[model.getFrameId("obj_front")].translation();
      box_root_rotation = data.oMf[model.getFrameId("box")].rotation();
      box_root_translation = data.oMf[model.getFrameId("box")].translation();
      

      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_geom (new hpp::fcl::Box (0.1,0.1,0.1));
      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_ee_geom (new hpp::fcl::Sphere (0.005));
      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_front_geom (new hpp::fcl::Sphere (0.005));

      hpp::fcl::CollisionObject fcl_box(fcl_box_geom, box_root_rotation, box_root_translation);
      hpp::fcl::CollisionObject fcl_ee(fcl_ee_geom, ee_rotation, ee_translation);
      hpp::fcl::CollisionObject fcl_box_front(fcl_box_front_geom, box_front_rotation, box_front_translation);
      
      hpp::fcl::DistanceRequest distReq;
      hpp::fcl::DistanceResult distRes;

      distReq.enable_nearest_points = true;

      // distance between EE and box, needs to be constrained as positive (no penetration)
      hpp::fcl::distance(&fcl_ee, &fcl_box, distReq, distRes);
      double distance_box = distRes.min_distance;
      // distance between EE and front plane, used in force constraints
      hpp::fcl::distance(&fcl_ee, &fcl_box_front, distReq, distRes);
      double distance_front = distRes.min_distance;

      // numerical difference to get dDistance_dq
      double alpha = 1e-8;
      Eigen::VectorXd pos_eps(model.nv);
      pos_eps = pos.segment(n_dof * (i+1) , n_dof);
      for(int k = 0; k < model.nv; ++k)
      {
        pos_eps[k] += alpha;
        Eigen::VectorXd q_eps(model.nq);
        q_eps.segment(0, n_dof-1) = pos_eps.segment(0, n_dof - 1);
        q_eps(model.nq - 2) = cos(pos_eps(n_dof - 1));
        q_eps(model.nq - 1) = sin(pos_eps(n_dof - 1));

        pinocchio::framesForwardKinematics(model, data, q_eps);
        ee_rotation = data.oMf[model.getFrameId("ee_link")].rotation();
        ee_translation = data.oMf[model.getFrameId("ee_link")].translation();
        box_front_rotation = data.oMf[model.getFrameId("obj_front")].rotation();
        box_front_translation = data.oMf[model.getFrameId("obj_front")].translation();
        box_root_rotation = data.oMf[model.getFrameId("box")].rotation();
        box_root_translation = data.oMf[model.getFrameId("box")].translation();
        fcl_ee.setTransform(ee_rotation, ee_translation); 
        fcl_box.setTransform(box_root_rotation, box_root_translation);
        fcl_box_front.setTransform(box_front_rotation, box_front_translation);
        hpp::fcl::distance(&fcl_ee, &fcl_box, distReq, distRes);
        double distance_box_plus = distRes.min_distance;
        hpp::fcl::distance(&fcl_ee, &fcl_box_front, distReq, distRes);
        double distance_front_plus = distRes.min_distance;
        dDistance_box_dq(i, k) = (distance_box_plus - distance_box) / alpha;
        dDistance_front_dq(i, k) = (distance_front_plus - distance_front) / alpha;
      
        pos_eps[k] -= alpha;
      }



      // Update the contact point frame
      // pinocchio::computeCollisions(model, data_next, geom_model, geom_data,
      //                              q_next);
      // pinocchio::computeDistances(model, data_next, geom_model, geom_data,
      //                             q_next);
      // dr = geom_data.distanceResults[cp_index];
      // front_normal_world =
      //     geom_data.oMg[geom_model.getGeometryId("box_0")].rotation() *
      //     front_normal;

      // Get Contact Point Jacobian
      pinocchio::computeJointJacobians(model, data_next, q_next);
      pinocchio::framesForwardKinematics(model, data_next, q_next);
      
      pinocchio::SE3 joint_frame_placement =
          data_next.oMf[model.getFrameId("wrist_3_joint")];
      pinocchio::SE3 root_joint_frame_placement =
          data_next.oMf[model.getFrameId("box_root_joint")];

      // Eigen::Vector3d robot_r_j2c = joint_frame_placement.inverse().act(dr.nearest_points[0]);
      // Eigen::Vector3d object_r_j2c = root_joint_frame_placement.inverse().act(dr.nearest_points[1]);
      Eigen::Vector3d robot_r_j2c(0.0, 0.092, 0.0);
      Eigen::Vector3d object_r_j2c(-0.05, 0, 0);
      model.frames[contactId].placement.translation() = robot_r_j2c;
      model.frames[object_contactId].placement.translation() = object_r_j2c;
      
      pinocchio::Data::Matrix6x w_J_contact(6, model.nv),
          w_J_contact_aligned(6, model.nv), w_J_object_aligned(6, model.nv),
          w_J_object(6, model.nv);
      w_J_contact.fill(0);
      w_J_object.fill(0);
      w_J_contact_aligned.fill(0);
      w_J_object_aligned.fill(0);
      pinocchio::getFrameJacobian(model, data_next, contactId, pinocchio::LOCAL,
                                  w_J_contact);
      pinocchio::getFrameJacobian(model, data_next, object_contactId,
                                  pinocchio::LOCAL, w_J_object);
      // J_remapped.col(i) = w_J_contact.topRows(3).transpose() * 
      //       data_next.oMi[model.getJointId("wrist_1_joint")].rotation().transpose() * 
      //       front_normal_world + w_J_object.topRows(3).transpose() * front_normal;

      J_remapped.col(i) = w_J_contact.topRows(3).transpose() * robot_contact_normal + w_J_object.topRows(3).transpose() * front_normal;
      // Calculate NLE, inertial matrix
      pinocchio::nonLinearEffects(model, data_next, q_next,
                                  vel.segment(n_dof * (i + 1), n_dof));
      pinocchio::computeMinverse(model, data_next, q_next);
      Eigen::MatrixXd Minv = data_next.Minv;
      Minv.triangularView<Eigen::StrictlyLower>() =
          Minv.transpose().triangularView<Eigen::StrictlyLower>();

      // Get external force in local joint frame
      Eigen::VectorXd force_cp(3), force_ocp(3);
      // contact force in [wrist_3_joint] frame at [contact] point
      // force_cp = -(data_next.oMi[model.getJointId("wrist_1_joint")].rotation().transpose() *
      //              front_normal_world * exforce(i + 1));
      force_cp = robot_contact_normal * exforce(i+1);
      // Get force and moment at [joint_origin] point
      fext_robot.col(i).head(3) = force_cp;
      fext_robot.col(i)(3) = -robot_r_j2c(2) * force_cp(1) + robot_r_j2c(1) * force_cp(2);
      fext_robot.col(i)(4) = robot_r_j2c(2) * force_cp(0) - robot_r_j2c(0) * force_cp(2);
      fext_robot.col(i)(5) = -robot_r_j2c(1) * force_cp(0) + robot_r_j2c(0) * force_cp(1);
      // contact force in [object] frame at [contact] point
      force_ocp = front_normal * exforce(i+1);
      fext_object.col(i).head(3) = force_ocp;
      fext_object.col(i)(3) = -object_r_j2c(2) * force_ocp(1) + object_r_j2c(1) * force_ocp(2);
      fext_object.col(i)(4) = object_r_j2c(2) * force_ocp(0) - object_r_j2c(0) * force_ocp(2);
      fext_object.col(i)(5) = -object_r_j2c(1) * force_ocp(0) + object_r_j2c(0) * force_ocp(1);
      // Calculate acceleration using Aba
      Eigen::VectorXd effort_remap(model.nv);
      effort_remap.setZero();
      effort_remap = B * effort.segment(n_control * (i + 1), n_control);
      pinocchio::Force::Vector6 fext_robot_ref = fext_robot.col(i);
      pinocchio::Force::Vector6 fext_object_ref = fext_object.col(i);
      PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext((size_t)model.njoints, pinocchio::Force::Zero());
      fext[6] = pinocchio::ForceRef<pinocchio::Force::Vector6>(fext_robot_ref);
      fext[7] = pinocchio::ForceRef<pinocchio::Force::Vector6>(fext_object_ref);

      pinocchio::aba(model, data_next, q_next, vel.segment(n_dof * (i + 1), n_dof),
                     effort_remap, fext);

      //////////////////////////////////////////////////////////////////////////
      pinocchio::forwardKinematics(model, data_next, q_next, vel.segment(n_dof * (i + 1), n_dof), data_next.ddq);

      Eigen::VectorXd a_classical = data_next.ddq;
      pinocchio::MotionTpl<double, 0> box_acceleration = pinocchio::getClassicalAcceleration(model, data_next, 7, pinocchio::LOCAL);
      a_classical(model.nv - 3) = box_acceleration.linear()(0);
      a_classical(model.nv - 2) = box_acceleration.linear()(1);
      a_classical(model.nv - 1) = box_acceleration.angular()(2);
      //////////////////////////////////////////////////////////////////////////
      
    
      // convert box velocity from local frame to world frame
      Eigen::MatrixXd v_l2w(n_dof, n_dof); 
      v_l2w.setIdentity(); 
      v_l2w.bottomRightCorner(3,3) << cos(pos(n_dof*(i+2) - 1)), -sin(pos(n_dof*(i+2) - 1)), 0,
                                      sin(pos(n_dof*(i+2) - 1)), cos(pos(n_dof*(i+2) - 1)), 0,
                                      0, 0, 1;

      // backward integration
      g.segment(n_dof * i, n_dof) =
          pos.segment(n_dof * i, n_dof) - pos.segment(n_dof * (i + 1), n_dof) +
          t_step * v_l2w * vel.segment(n_dof * (i + 1), n_dof);

      // smoothed if condition
      g.segment(n_dof * (n_step - 1 + i), n_dof) =
          1 / t_step *
              (vel.segment(n_dof * (i + 1), n_dof) -
               vel.segment(n_dof * i, n_dof)) - a_classical +
          Minv * f * tanh(20 * vel(n_dof * (i+1) - 3));

      // g.segment(n_dof * (n_step - 1 + i), n_dof) =
      //     1 / t_step *
      //         (vel.segment(n_dof * (i + 1), n_dof) -
      //          vel.segment(n_dof * i, n_dof)) +
      //     Minv * (data_next.nle - effort_remap -
      //     J_remapped.col(i) * exforce(i + 1) + f * tanh(20 * vel(n_dof * (i) + 3)));

      // Contact constraints, 3 constraints for each step
      g(n_dof * 2 * (n_step - 1) + i) = distance_front - slack(i);
      g(n_dof * 2 * (n_step - 1) + n_step - 1 + i) =
          exforce(i + 1) * slack(i) - slack(n_step - 1 + i);
      // state-trigger
      // g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) =
      //    -std::min(-slack(i), 0.0) * slack(n_step - 1 + i);
      // complimentray
      // g(n_dof * 2 * (n_step - 1) + 1 * (n_step - 1) + i) =
      //     slack(i) * exforce(i + 1) - slack(n_step - 1 + i);

      g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) =
          distance_box;

      distance_cache(i) = distance_front;
    }
    return g;
  };

  // // Constant values should always be put into GetBounds(), not GetValues().
  // // For inequality constraints (<,>), use Bounds(x, inf) or Bounds(-inf, x).
  VecBound GetBounds() const override {
    VecBound bounds(GetRows());
    for (int i = 0; i < n_dof * 2 * (n_step - 1); i++)
      bounds.at(i) = Bounds(0.0, 0.0);
    for (int i = 0; i < n_step - 1; i++) {
      bounds.at(n_dof * 2 * (n_step - 1) + i) = Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = Bounds(-inf, 0);
      bounds.at(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) =
          Bounds(0, inf);
    }
    return bounds;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {
    pinocchio::Data data_next(model);
    VectorXd pos = GetVariables()->GetComponent("position")->GetValues();
    VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
    VectorXd effort = GetVariables()->GetComponent("effort")->GetValues();
    VectorXd exforce = GetVariables()->GetComponent("exforce")->GetValues();
    VectorXd slack = GetVariables()->GetComponent("slack")->GetValues();
    std::vector<T> triplet_pos, triplet_vel, triplet_tau, triplet_exforce, triplet_slack;
    for (int i = 0; i < n_step - 1; i++) {
      Eigen::VectorXd q(model.nq), q_next(model.nq);
      q.segment(0, n_dof - 1) = pos.segment(n_dof * i, n_dof - 1);
      q(model.nq - 2) = cos(pos(n_dof * i + n_dof - 1));
      q(model.nq - 1) = sin(pos(n_dof * i + n_dof - 1));
      q_next.segment(0, n_dof - 1) = pos.segment(n_dof * (i + 1), n_dof - 1);
      q_next(model.nq - 2) = cos(pos(n_dof * (i + 1) + n_dof - 1));
      q_next(model.nq - 1) = sin(pos(n_dof * (i + 1) + n_dof - 1));
      
      // Get fext
      Eigen::VectorXd effort_remap(model.nv);
      effort_remap.setZero();
      effort_remap = B * effort.segment(n_control * (i + 1), n_control);
      pinocchio::Force::Vector6 fext_robot_ref = fext_robot.col(i);
      pinocchio::Force::Vector6 fext_object_ref = fext_object.col(i);
      PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext((size_t)model.njoints, pinocchio::Force::Zero());
      fext[6] = pinocchio::ForceRef<pinocchio::Force::Vector6>(fext_robot_ref);
      fext[7] = pinocchio::ForceRef<pinocchio::Force::Vector6>(fext_object_ref);

      pinocchio::computeABADerivatives(model, data_next,
                                       q_next,
                                       vel.segment(n_dof * (i + 1), n_dof),
                                       effort_remap,
                                       fext);
      Eigen::MatrixXd Minv = data_next.Minv;
      Minv.triangularView<Eigen::StrictlyLower>() =
          Minv.transpose().triangularView<Eigen::StrictlyLower>();
      
      Eigen::MatrixXd v_l2w(n_dof, n_dof), dv_l2w(n_dof, n_dof);  
      v_l2w.setIdentity(); 
      v_l2w.bottomRightCorner(3,3) << cos(pos(n_dof*(i+2) - 1)), -sin(pos(n_dof*(i+2) - 1)), 0,
                                      sin(pos(n_dof*(i+2) - 1)), cos(pos(n_dof*(i+2) - 1)), 0,
                                      0, 0, 1;
      dv_l2w.setZero();
      dv_l2w.bottomRightCorner(3,3) << -sin(pos(n_dof*(i+2) - 1)), -cos(pos(n_dof*(i+2) - 1)), 0,
                                        cos(pos(n_dof*(i+2) - 1)), -sin(pos(n_dof*(i+2) - 1)), 0,
                                        0, 0, 0;
      // dq_dtheta_k+1 due to converting box velocity from local to world
      Eigen::VectorXd dq_dtheta_plus = t_step * dv_l2w * vel.segment(n_dof * (i + 1), n_dof);

      // construct the triplet list for 5 sparse matrix (the 5 Jacobian,
      // corresponding to 5 constraint sets)
      for (int j = 0; j < n_dof; j++) {
        if (var_set == "position") {
          // Triplet for position
          triplet_pos.push_back(T(n_dof * i + j, n_dof * i + j, 1)); // dq_dq_k
          triplet_pos.push_back(
              T(n_dof * i + j, n_dof * (i+1) + j, -1)); // dq_dq_k+1
          triplet_pos.push_back(
              T(n_dof * i + j, n_dof * (i+1) + n_dof - 1, dq_dtheta_plus(j))); // dq_dtheta_k+1
          for (int k = 0; k < n_dof; k++) {
            triplet_pos.push_back(T(n_dof * (n_step - 1 + i) + j,
                                    n_dof * (i + 1) + k,
                                    -data_next.ddq_dq(j, k))); // ddq_dq_k+1
          }
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_front_dq(i, j))); //dDistance_front_k+1_dq_k+1
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_box_dq(i, j))); //dDistance_box_k+1_dq_k+1
        }
        if (var_set == "velocity") {
          // Triplet for velocity
          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j, n_dof * i + j,
                                  -1.0 / t_step)); // ddq_dv_k
          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j, n_dof * (i+1) - 3,
                                  (Minv * f)(j) * 20 * 
                                      (1.0 - tanh(20 * vel(n_dof * (i+1) - 3)) *
                                                 tanh(20 * vel(n_dof * (i+1) - 3))))); // ddq_dv_k from smoothed friction term
          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j,
                                  n_dof * i + j + n_dof,
                                  1.0 / t_step)); // ddq_dv_k+1
          for (int k = 0; k < n_dof; k++) {
            triplet_vel.push_back(
              T(n_dof * i + j, n_dof * (i+1) + k, t_step * v_l2w(j, k))); // dq_dv_k+1
            triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j,
                                    n_dof * (i + 1) + k,
                                    -data_next.ddq_dv(j, k))); // ddq_dv_k+1
          }
        }
        if (var_set == "effort") {
          // Triplet for torque
          for (int k = 0; k < n_control; k++) {
            triplet_tau.push_back(T(n_dof * (n_step - 1 + i) + j,
                                    n_control * i + n_control + k,
                                    (-Minv * B)(j, k))); // ddq_dt_k+1
          }
        }
        if (var_set == "exforce") {
          for (int k = 0; k < n_exforce; k++) {
            triplet_exforce.push_back(T(n_dof * (n_step - 1 + i) + j,
                                        n_exforce * i + n_exforce + k,
                                        (-Minv * J_remapped.col(i))(j)));
          }
        }
      }
      if (var_set == "exforce") {
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + n_step - 1 + i, i + 1, slack(i)));
      }
      if (var_set == "slack") {
        triplet_slack.push_back(T(n_dof * 2 * (n_step - 1) + i, i, -1));
        triplet_slack.push_back(T(n_dof * 2 * (n_step - 1) + n_step - 1 + i, i, exforce(i + 1)));
        triplet_slack.push_back(T(n_dof * 2 * (n_step - 1) + n_step - 1 + i, n_step - 1 + i, -1));
        // triplet_slack.push_back(T(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i, i, slack(n_step - 1 + i)));
        // triplet_slack.push_back(T(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i, n_step - 1 + i, slack(i)));
        // triplet_slack.push_back(T(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i, 2 * (n_step - 1) + i, -1));
      }
    } 
    if (var_set == "position") {
      jac_block.setFromTriplets(triplet_pos.begin(), triplet_pos.end());
    }
    if (var_set == "velocity") {
      jac_block.setFromTriplets(triplet_vel.begin(), triplet_vel.end());
    }
    if (var_set == "effort") {
      jac_block.setFromTriplets(triplet_tau.begin(), triplet_tau.end());
    }
    if (var_set == "exforce") {
      jac_block.setFromTriplets(triplet_exforce.begin(), triplet_exforce.end());
    }
    if (var_set == "slack") {
      jac_block.setFromTriplets(triplet_slack.begin(), triplet_slack.end());
    }
  }
};

class ExCost : public CostTerm {
public:
  ExCost() : ExCost("cost_term1") {}
  ExCost(const std::string &name) : CostTerm(name) {
    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    cost_func_type = params["cost_func_type"].as<int>();
    n_step = params["n_step"].as<int>();
    }

  double GetCost() const override {
    VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
    VectorXd torque = GetVariables()->GetComponent("effort")->GetValues();
    VectorXd slack = GetVariables()->GetComponent("slack")->GetValues();

    double cost = vel.squaredNorm();
    
    // penalty on slack
    float slack_weight = 1e4;
    switch (cost_func_type)
    {
    case 0:
      cost = cost + slack_weight * slack.tail(n_step-1).squaredNorm();
      break;
    case 1:
      cost = cost + slack_weight * slack.tail(n_step-1).lpNorm<1>();
      break;
    case 2:
      cost = cost + slack_weight * slack.tail(n_step-1).norm();
      break;
    }

    return cost;
  };

  void FillJacobianBlock(std::string var_set, Jacobian &jac) const override {
    if (var_set == "velocity"){
      VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
      int n = GetVariables()->GetComponent("velocity")->GetRows();
      std::vector<T> triplet_cost;
      for(int i = 0; i < n; i++){
        triplet_cost.push_back(T(0,i,2*vel(i)));
      }
      jac.setFromTriplets(triplet_cost.begin(), triplet_cost.end());
    }
    if (var_set == "slack"){
      VectorXd slack = GetVariables()->GetComponent("slack")->GetValues();
      std::vector<T> triplet_slack;
      switch (cost_func_type)
      {
      case 0:
        for (int i = 0; i < n_step - 1; i++)
        {
          triplet_slack.push_back(T(0, n_step-1 + i, 1e4 * 2 * slack(n_step - 1 + i)));
        }
        break;
      case 1:
        for (int i = 0; i < n_step - 1; i++)
        {
          triplet_slack.push_back(T(0, n_step-1 + i, 1e4));
        }
        break;
      case 2:
        for (int i = 0; i < n_step - 1; i++)
        {
          triplet_slack.push_back(T(0, n_step-1 + i, 1e4 * slack(n_step - 1 + i) / slack.tail(n_step - 1).norm()));
        }
        break;
      }
      jac.setFromTriplets(triplet_slack.begin(), triplet_slack.end());
    }
  }


private:
  int cost_func_type;
  int n_step;
};

} // namespace ifopt