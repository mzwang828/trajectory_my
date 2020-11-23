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
    goal = params["box_goal"].as<std::vector<double>>();
    
    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    if (name == "position") {
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
    } else if (name == "d_slack") {
      for (int i = 0; i < n; i++) {
        xvar(i) = 1e-2;
      }
    } else if (name == "df_slack") {
      for (int i = 0; i < n; i++) {
        xvar(i) = 1e-2;
      }
    }
  }

  ExVariables(int n, const std::string &name, Eigen::VectorXd &init_values) : VariableSet(n, name) {
    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_dof = params["n_dof"].as<int>();
    n_control = params["n_control"].as<int>();
    n_exforce = params["n_exforce"].as<int>();
    n_step = params["n_step"].as<int>();
    t_step = params["t_step"].as<double>();
    goal = params["box_goal"].as<std::vector<double>>();

    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    xvar = init_values;
  }

  ExVariables(int n, const std::string &name, std::vector<double> &goal_in) : VariableSet(n, name) {
    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_dof = params["n_dof"].as<int>();
    n_control = params["n_control"].as<int>();
    n_exforce = params["n_exforce"].as<int>();
    n_step = params["n_step"].as<int>();
    t_step = params["t_step"].as<double>();
    goal = goal_in;

    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    if (name == "position") {
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
    } else if (name == "d_slack") {
      for (int i = 0; i < n; i++) {
        xvar(i) = 1e-2;
      }
    } else if (name == "df_slack") {
      for (int i = 0; i < n; i++) {
        xvar(i) = 1e-2;
      }
    }
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
      bounds.at(7) = Bounds(0.0, 0.0);
      bounds.at(8) = Bounds(0, 0);

      bounds.at(GetRows() - 3) = Bounds(goal[0], goal[0]);
      bounds.at(GetRows() - 2) = Bounds(goal[1], goal[1]);
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
    } else if (GetName() == "d_slack") {
      for (int i = 0; i < GetRows(); i++) {
        bounds.at(i) = Bounds(0, inf);                        // distance 
      }
    } else if (GetName() == "df_slack") {
      for (int i = 0; i < GetRows(); i++) {
        bounds.at(i) = Bounds(0, 1e-2);                       // phi*gamma < slack
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
  std::vector<double> goal;
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
  pinocchio::FrameIndex contactId, object_frontId, object_backId,
                        object_leftId, object_rightId;
  Eigen::Vector3d front_normal, back_normal, left_normal, right_normal; // normal vector point to the front plane
  Eigen::Vector3d robot_contact_normal;
  Eigen::Vector3d goal_v3d; // 3d vector goal in world frame
  mutable Eigen::MatrixXd J_front_remapped,
                          J_back_remapped,
                          J_left_remapped,
                          J_right_remapped; // used to save calculated Jacobians for exforce
  mutable Eigen::MatrixXd fext_robot, fext_box; // used to save fext values for each joint
  mutable Eigen::MatrixXd dDistance_box_dq, dDistance_front_dq, dDistance_back_dq,
                          dDistance_left_dq, dDistance_right_dq; // numerical gradients for distance
  mutable Eigen::MatrixXd goal_local, dgx_dq, dgy_dq; // goal in box local frame

  // Input Mapping & Friction
  Eigen::MatrixXd B, f;
  int n_dof;                    // number of freedom
  int n_control;                // number of control
  int n_exforce;                // number of external force
  int n_step;                   // number of steps or (knot points - 1)
  double t_step;                // length of each step
  int constraint_type;          // constraint type
  std::vector<double> goal;     // goal

  ExConstraint(int n) : ExConstraint(n, "constraint1") {}
  ExConstraint(int n, std::vector<double> &goal_in) : ExConstraint(n, "constraint1"){
    goal_v3d << goal_in[0], goal_in[1], 0;
  }
  ExConstraint(int n, const std::string &name) : ConstraintSet(n, name) {
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
    object_frontId = model.addFrame(
        pinocchio::Frame("object_front", model.getJointId("box_root_joint"), -1,
        pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    object_leftId = model.addFrame(
        pinocchio::Frame("object_left", model.getJointId("box_root_joint"), -1,
        pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    object_rightId = model.addFrame(
        pinocchio::Frame("object_right", model.getJointId("box_root_joint"), -1,
        pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    object_backId = model.addFrame(
        pinocchio::Frame("object_back", model.getJointId("box_root_joint"), -1,
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
    constraint_type = params["constraint_type"].as<int>();
    goal = params["box_goal"].as<std::vector<double>>();

    front_normal << 1, 0, 0;
    back_normal << -1, 0, 0;
    left_normal << 0, -1, 0;
    right_normal << 0, 1, 0;
    robot_contact_normal << 0, -1, 0;
    goal_v3d << goal[0], goal[1], 0;

    goal_local.resize(3, n_step - 1);
    goal_local.setZero();

    J_front_remapped.resize(n_dof, n_step - 1);
    J_front_remapped.setZero();
    J_left_remapped.resize(n_dof, n_step - 1);
    J_left_remapped.setZero();
    J_right_remapped.resize(n_dof, n_step - 1);
    J_right_remapped.setZero();
    J_back_remapped.resize(n_dof, n_step - 1);
    J_back_remapped.setZero();
    
    fext_robot.resize(6, n_step - 1); // fext for robot
    fext_robot.setZero();
    fext_box.resize(6, n_step - 1); // fext for box
    fext_box.setZero();

    B.resize(n_dof, n_control);
    B.setZero();
    B.topRows(n_control).setIdentity();

    f.resize(n_dof, 3);
    f.setZero();
    f(6, 0) = 1.0;
    f(7, 1) = 1.0;
    f(8, 2) = 0.01;

    dDistance_box_dq.resize(n_step, n_dof);
    dDistance_box_dq.setZero();
    dDistance_front_dq.resize(n_step, n_dof);
    dDistance_front_dq.setZero();
    dDistance_left_dq.resize(n_step, n_dof);
    dDistance_left_dq.setZero();
    dDistance_right_dq.resize(n_step, n_dof);
    dDistance_right_dq.setZero();
    dDistance_back_dq.resize(n_step, n_dof);
    dDistance_back_dq.setZero();

    dgx_dq.resize(n_step, n_dof);
    dgx_dq.setZero();
    dgy_dq.resize(n_step, n_dof);
    dgy_dq.setZero();
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
    VectorXd d_slack = GetVariables()->GetComponent("d_slack")->GetValues();
    VectorXd df_slack = GetVariables()->GetComponent("df_slack")->GetValues();
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
      Eigen::Matrix3d ee_rotation, box_root_rotation;
      Eigen::Vector3d ee_translation, box_front_translation, box_root_translation,
                      box_left_translation, box_right_translation, box_back_translation;
      ee_rotation = data.oMf[model.getFrameId("ee_link")].rotation();
      ee_translation = data.oMf[model.getFrameId("ee_link")].translation();
      box_root_rotation = data.oMf[model.getFrameId("box")].rotation();
      box_root_translation = data.oMf[model.getFrameId("box")].translation();
      box_front_translation = data.oMf[model.getFrameId("obj_front")].translation();
      box_left_translation = data.oMf[model.getFrameId("obj_left")].translation();
      box_right_translation = data.oMf[model.getFrameId("obj_right")].translation();
      box_back_translation = data.oMf[model.getFrameId("obj_back")].translation();      

      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_geom (new hpp::fcl::Box (0.1,0.1,0.1));
      // boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_ee_geom (new hpp::fcl::Cylinder (0.03, 0.00010));
      // boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_front_geom (new hpp::fcl::Box (0.00010,0.08,0.08));
      // boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_left_geom (new hpp::fcl::Box (0.08,0.00010,0.08));
      // boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_right_geom (new hpp::fcl::Box (0.08,0.00010,0.08));
      // boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_back_geom (new hpp::fcl::Box (0.00010,0.08,0.08));

      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_ee_geom (new hpp::fcl::Sphere (0.005));
      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_front_geom (new hpp::fcl::Sphere (0.005));
      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_left_geom (new hpp::fcl::Sphere (0.005));
      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_right_geom (new hpp::fcl::Sphere (0.005));
      boost::shared_ptr<hpp::fcl::CollisionGeometry> fcl_box_back_geom (new hpp::fcl::Sphere (0.005));

      hpp::fcl::CollisionObject fcl_box(fcl_box_geom, box_root_rotation, box_root_translation);
      hpp::fcl::CollisionObject fcl_ee(fcl_ee_geom, ee_rotation, ee_translation);
      hpp::fcl::CollisionObject fcl_box_front(fcl_box_front_geom, box_root_rotation, box_front_translation);
      hpp::fcl::CollisionObject fcl_box_left(fcl_box_left_geom, box_root_rotation, box_left_translation);
      hpp::fcl::CollisionObject fcl_box_right(fcl_box_right_geom, box_root_rotation, box_right_translation);
      hpp::fcl::CollisionObject fcl_box_back(fcl_box_back_geom, box_root_rotation, box_back_translation);      

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
      // distance between EE and back plane, used in force constraints
      distRes.clear();
      hpp::fcl::distance(&fcl_ee, &fcl_box_back, distReq, distRes);
      double distance_back = distRes.min_distance;


      //////////////Variable contact points////////////////////////////////
      // Eigen::Vector3d contact_point_ee = distRes.nearest_points[0];
      // Eigen::Vector3d contact_point_front = distRes.nearest_points[1];
      /////////////////////////////////////////////////////////////////////

      // numerical difference to get dDistance_dq
      double alpha = 1e-6;
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
        box_root_rotation = data.oMf[model.getFrameId("box")].rotation();
        box_root_translation = data.oMf[model.getFrameId("box")].translation();
        box_front_translation = data.oMf[model.getFrameId("obj_front")].translation();
        box_left_translation = data.oMf[model.getFrameId("obj_left")].translation();
        box_right_translation = data.oMf[model.getFrameId("obj_right")].translation();
        box_back_translation = data.oMf[model.getFrameId("obj_back")].translation();
        fcl_ee.setTransform(ee_rotation, ee_translation); 
        fcl_box.setTransform(box_root_rotation, box_root_translation);
        fcl_box_front.setTransform(box_root_rotation, box_front_translation);
        fcl_box_left.setTransform(box_root_rotation, box_left_translation);
        fcl_box_right.setTransform(box_root_rotation, box_right_translation);
        fcl_box_back.setTransform(box_root_rotation, box_back_translation);
        distRes.clear();
        hpp::fcl::distance(&fcl_ee, &fcl_box, distReq, distRes);
        double distance_box_plus = distRes.min_distance;
        distRes.clear();
        hpp::fcl::distance(&fcl_ee, &fcl_box_front, distReq, distRes);
        double distance_front_plus = distRes.min_distance;
        distRes.clear();
        hpp::fcl::distance(&fcl_ee, &fcl_box_left, distReq, distRes);
        double distance_left_plus = distRes.min_distance;
        distRes.clear();
        hpp::fcl::distance(&fcl_ee, &fcl_box_right, distReq, distRes);
        double distance_right_plus = distRes.min_distance;
        distRes.clear();
        hpp::fcl::distance(&fcl_ee, &fcl_box_back, distReq, distRes);
        double distance_back_plus = distRes.min_distance;
        
        dDistance_box_dq(i, k) = (distance_box_plus - distance_box) / alpha;
        dDistance_front_dq(i, k) = (distance_front_plus - distance_front) / alpha;
        dDistance_left_dq(i, k) = (distance_left_plus - distance_left) / alpha;
        dDistance_right_dq(i, k) = (distance_right_plus - distance_right) / alpha;
        dDistance_back_dq(i, k) = (distance_back_plus - distance_back) / alpha;
      
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
      pinocchio::forwardKinematics(model, data_next, q_next);
      
      pinocchio::SE3 joint_frame_placement =
          data_next.oMi[model.getJointId("wrist_3_joint")];
      pinocchio::SE3 root_joint_frame_placement =
          data_next.oMi[model.getJointId("box_root_joint")];

      // Eigen::Vector3d robot_r_j2c = joint_frame_placement.inverse().act(contact_point_ee);
      // Eigen::Vector3d object_r_j2c = root_joint_frame_placement.inverse().act(contact_point_front);
      Eigen::Vector3d robot_r_j2c(0.0, 0.092, 0.0);
      Eigen::Vector3d front_r_j2c(-0.05, 0, 0), left_r_j2c(0.0, 0.05, 0.0), right_r_j2c(0.0, -0.05, 0.0), back_r_j2c(0.05, 0.0, 0.0);
      model.frames[contactId].placement.translation() = robot_r_j2c;
      model.frames[object_frontId].placement.translation() = front_r_j2c;
      model.frames[object_leftId].placement.translation() = left_r_j2c;
      model.frames[object_rightId].placement.translation() = right_r_j2c;
      model.frames[object_backId].placement.translation() = back_r_j2c;

      goal_local.col(i) = root_joint_frame_placement.inverse().act(goal_v3d);

      dgx_dq(i, 6) = -cos(pos(n_dof * (i + 1) + n_dof - 1));
      dgx_dq(i, 7) = sin(pos(n_dof * (i + 1) + n_dof - 1));
      dgx_dq(i, 8) = -sin(pos(n_dof * (i + 1) + n_dof - 1)) * (goal_v3d(0) - pos(n_dof * (i + 1) + n_dof - 3))
                     -cos(pos(n_dof * (i + 1) + n_dof - 1)) * (goal_v3d(1) - pos(n_dof * (i + 1) + n_dof - 2));

      dgy_dq(i, 6) = -sin(pos(n_dof * (i + 1) + n_dof - 1));
      dgy_dq(i, 7) = -cos(pos(n_dof * (i + 1) + n_dof - 1));
      dgy_dq(i, 8) = cos(pos(n_dof * (i + 1) + n_dof - 1)) * (goal_v3d(0) - pos(n_dof * (i + 1) + n_dof - 3))
                    -sin(pos(n_dof * (i + 1) + n_dof - 1)) * (goal_v3d(1) - pos(n_dof * (i + 1) + n_dof - 2));

      pinocchio::computeJointJacobians(model, data_next, q_next);
      pinocchio::framesForwardKinematics(model, data_next, q_next);
      
      pinocchio::Data::Matrix6x w_J_robot(6, model.nv),
                                w_J_front(6, model.nv),
                                w_J_left(6, model.nv),
                                w_J_right(6, model.nv),
                                w_J_back(6, model.nv);
      w_J_robot.setZero();
      w_J_front.setZero();
      w_J_left.setZero();
      w_J_right.setZero();
      w_J_back.setZero();

      pinocchio::getFrameJacobian(model, data_next, contactId, 
                                  pinocchio::LOCAL, w_J_robot);
      pinocchio::getFrameJacobian(model, data_next, object_frontId,
                                  pinocchio::LOCAL, w_J_front);
      pinocchio::getFrameJacobian(model, data_next, object_leftId,
                                  pinocchio::LOCAL, w_J_left);
      pinocchio::getFrameJacobian(model, data_next, object_rightId,
                                  pinocchio::LOCAL, w_J_right);
      pinocchio::getFrameJacobian(model, data_next, object_backId,
                                  pinocchio::LOCAL, w_J_back);                                  

      J_front_remapped.col(i) = w_J_robot.topRows(3).transpose() * robot_contact_normal + w_J_front.topRows(3).transpose() * front_normal;
      J_left_remapped.col(i) = w_J_robot.topRows(3).transpose() * robot_contact_normal + w_J_left.topRows(3).transpose() * left_normal;
      J_right_remapped.col(i) = w_J_robot.topRows(3).transpose() * robot_contact_normal + w_J_right.topRows(3).transpose() * right_normal;
      J_back_remapped.col(i) = w_J_robot.topRows(3).transpose() * robot_contact_normal + w_J_back.topRows(3).transpose() * back_normal;

      // Calculate NLE, inertial matrix
      pinocchio::nonLinearEffects(model, data_next, q_next,
                                  vel.segment(n_dof * (i + 1), n_dof));
      pinocchio::computeMinverse(model, data_next, q_next);
      Eigen::MatrixXd Minv = data_next.Minv;
      Minv.triangularView<Eigen::StrictlyLower>() =
          Minv.transpose().triangularView<Eigen::StrictlyLower>();

      // Get external force in local joint frame
      Eigen::VectorXd force_robot(3), force_front(3), force_left(3), force_right(3), force_back(3);
      // contact force in [wrist_3_joint] frame at [contact] point
      force_robot = robot_contact_normal * (exforce(i+1) + exforce(n_step + i+1) + exforce(2*n_step + i+1) + exforce(3*n_step + i+1));
      // Get force and moment at [joint_origin] point
      fext_robot.col(i).head(3) = force_robot;
      fext_robot.col(i)(3) = -robot_r_j2c(2) * force_robot(1) + robot_r_j2c(1) * force_robot(2);
      fext_robot.col(i)(4) = robot_r_j2c(2) * force_robot(0) - robot_r_j2c(0) * force_robot(2);
      fext_robot.col(i)(5) = -robot_r_j2c(1) * force_robot(0) + robot_r_j2c(0) * force_robot(1);
      // contact force in [object] frame at [contact] point of front surface
      force_front = front_normal * exforce(i+1);
      fext_box.col(i).head(3) = force_front;
      fext_box.col(i)(3) = -front_r_j2c(2) * force_front(1) + front_r_j2c(1) * force_front(2);
      fext_box.col(i)(4) = front_r_j2c(2) * force_front(0) - front_r_j2c(0) * force_front(2);
      fext_box.col(i)(5) = -front_r_j2c(1) * force_front(0) + front_r_j2c(0) * force_front(1);
      // contact force in [object] frame at [contact] point of left surface
      force_left = left_normal * exforce(n_step + i+1);
      fext_box.col(i).head(3) += force_left;
      fext_box.col(i)(3) += -left_r_j2c(2) * force_left(1) + left_r_j2c(1) * force_left(2);
      fext_box.col(i)(4) += left_r_j2c(2) * force_left(0) - left_r_j2c(0) * force_left(2);
      fext_box.col(i)(5) += -left_r_j2c(1) * force_left(0) + left_r_j2c(0) * force_left(1);
      // contact force in [object] frame at [contact] point of right surface
      force_right = right_normal * exforce(2 * n_step + i+1);
      fext_box.col(i).head(3) += force_right;
      fext_box.col(i)(3) += -right_r_j2c(2) * force_right(1) + right_r_j2c(1) * force_right(2);
      fext_box.col(i)(4) += right_r_j2c(2) * force_right(0) - right_r_j2c(0) * force_right(2);
      fext_box.col(i)(5) += -right_r_j2c(1) * force_right(0) + right_r_j2c(0) * force_right(1);
      // contact force in [object] frame at [contact] point of back surface
      force_back = back_normal * exforce(3 * n_step + i+1);
      fext_box.col(i).head(3) += force_back;
      fext_box.col(i)(3) += -back_r_j2c(2) * force_back(1) + back_r_j2c(1) * force_back(2);
      fext_box.col(i)(4) += back_r_j2c(2) * force_back(0) - back_r_j2c(0) * force_back(2);
      fext_box.col(i)(5) += -back_r_j2c(1) * force_back(0) + back_r_j2c(0) * force_back(1);      
      // Calculate acceleration using Aba
      Eigen::VectorXd effort_remap(model.nv);
      effort_remap.setZero();
      effort_remap = B * effort.segment(n_control * (i + 1), n_control);
      pinocchio::Force::Vector6 fext_robot_ref = fext_robot.col(i);
      pinocchio::Force::Vector6 fext_object_ref = fext_box.col(i);
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

      // box velocity vector for friction calculation
      Eigen::Vector3d vf_box;
      vf_box << tanh(20 * vel(n_dof * (i+1) - 3)), tanh(20 * vel(n_dof * (i+1) - 2)), tanh(20 * vel(n_dof * (i+1) - 1));

      // backward integration
      g.segment(n_dof * i, n_dof) =
          pos.segment(n_dof * i, n_dof) - pos.segment(n_dof * (i + 1), n_dof) +
          t_step * v_l2w * vel.segment(n_dof * (i + 1), n_dof);

      // smoothed if condition
      g.segment(n_dof * (n_step - 1 + i), n_dof) =
          1 / t_step *
              (vel.segment(n_dof * (i + 1), n_dof) -
               vel.segment(n_dof * i, n_dof)) - a_classical +
              Minv * f * vf_box;

      // distance box constraint
      g(n_dof * 2 * (n_step - 1) + 0 * (n_step - 1) + i) =
          distance_box;

      // Contact constraints
      // front
      g(n_dof * 2 * (n_step - 1) + 1 * (n_step - 1) + i) = 
          distance_front - d_slack(i);
      g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = 
          -std::min(front_normal.dot(goal_local.col(i))-(1e-3), 0.0) * exforce(i + 1);
      g(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) = 
          -std::min(-front_normal.dot(goal_local.col(i)), 0.0) * 
          (exforce(i + 1) * d_slack(i) - df_slack(i));
      // left
      g(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i) = 
          distance_left - d_slack(n_step - 1 + i);
      g(n_dof * 2 * (n_step - 1) + 5 * (n_step - 1) + i) = 
          -std::min(left_normal.dot(goal_local.col(i))-(1e-3), 0.0) * exforce(n_step + i + 1);
      g(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i) = 
          -std::min(-left_normal.dot(goal_local.col(i)), 0.0) * 
          (exforce(n_step + i + 1) * d_slack(n_step - 1 + i)
          - df_slack(1 * (n_step - 1) + i));
      // right
      g(n_dof * 2 * (n_step - 1) + 7 * (n_step - 1) + i) = 
          distance_right - d_slack(2 * (n_step - 1) + i);
      g(n_dof * 2 * (n_step - 1) + 8 * (n_step - 1) + i) = 
          -std::min(right_normal.dot(goal_local.col(i))-(1e-3), 0.0) * exforce(2 * n_step + i + 1);
      g(n_dof * 2 * (n_step - 1) + 9 * (n_step - 1) + i) = 
          -std::min(-right_normal.dot(goal_local.col(i)), 0.0) * 
          (exforce(2 * n_step + i + 1) * d_slack(2 * (n_step - 1) + i)
          - df_slack(2 * (n_step - 1) + i));
      // back
      g(n_dof * 2 * (n_step - 1) + 10 * (n_step - 1) + i) = 
          distance_back - d_slack(3 * (n_step - 1) + i);
      g(n_dof * 2 * (n_step - 1) + 11 * (n_step - 1) + i) = 
          -std::min(back_normal.dot(goal_local.col(i))-(1e-3), 0.0) * exforce(3 * n_step + i + 1);
      g(n_dof * 2 * (n_step - 1) + 12 * (n_step - 1) + i) = 
          -std::min(-back_normal.dot(goal_local.col(i)), 0.0) * 
          (exforce(3 * n_step + i + 1) * d_slack(3 * (n_step - 1) + i)
          - df_slack(3 * (n_step - 1) + i));
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
      bounds.at(n_dof * 2 * (n_step - 1) + i) = Bounds(0.0, inf);
      bounds.at(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = 
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) = 
          Bounds(-inf, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i) = 
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 5 * (n_step - 1) + i) = 
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i) =
          Bounds(-inf, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 7 * (n_step - 1) + i) =
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 8 * (n_step - 1) + i) =
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 9 * (n_step - 1) + i) =
          Bounds(-inf, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 10 * (n_step - 1) + i) =
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 11 * (n_step - 1) + i) =
          Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 12 * (n_step - 1) + i) =
          Bounds(-inf, 0.0);
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
    VectorXd d_slack = GetVariables()->GetComponent("d_slack")->GetValues();
    VectorXd df_slack = GetVariables()->GetComponent("df_slack")->GetValues();
    std::vector<T> triplet_pos, triplet_vel, triplet_tau, triplet_exforce, 
                   triplet_d_slack, triplet_df_slack;
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
      pinocchio::Force::Vector6 fext_object_ref = fext_box.col(i);
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
          triplet_pos.push_back(T(n_dof * i + j, n_dof * i + j, 1)); // dq_dq_t
          triplet_pos.push_back(
              T(n_dof * i + j, n_dof * (i+1) + j, -1)); // dq_dq_t+1
          triplet_pos.push_back(
              T(n_dof * i + j, n_dof * (i+1) + n_dof - 1, dq_dtheta_plus(j))); // dq_dtheta_t+1
          for (int k = 0; k < n_dof; k++) {
            triplet_pos.push_back(T(n_dof * (n_step - 1 + i) + j,
                                    n_dof * (i + 1) + k,
                                    -data_next.ddq_dq(j, k))); // ddq_dq_k+1
          }
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 0 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_box_dq(i, j))); //dDistance_box_k+1_dq_k+1
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 1 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_front_dq(i, j))); //dDistance_front_k+1_dq_k+1
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_left_dq(i, j))); //dDistance_left_k+1_dq_k+1
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 7 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_right_dq(i, j))); //dDistance_right_k+1_dq_k+1
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 10 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j, 
                                  dDistance_back_dq(i, j))); //dDistance_right_k+1_dq_k+1
          // dGoal_Local_dq_t+1
          Eigen::Vector3d dgl_dq(dgx_dq(i, j), dgy_dq(i, j), 0.0);
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  front_normal.dot(goal_local.col(i))-(1e-3) < 0.0?
                                  front_normal.dot(dgl_dq) * exforce(i + 1) : 0.0));
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  -front_normal.dot(goal_local.col(i)) < 0.0?
                                  -front_normal.dot(dgl_dq) * (exforce(i + 1) * d_slack(i) - df_slack(i)) : 0.0));                       
          
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 5 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  left_normal.dot(goal_local.col(i))-(1e-3) < 0.0?
                                  left_normal.dot(dgl_dq) * exforce(n_step + i + 1) : 0.0));
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  -left_normal.dot(goal_local.col(i)) < 0.0?
                                  -left_normal.dot(dgl_dq) * 
                                  (exforce(n_step + i + 1) * d_slack(n_step - 1 + i) 
                                  - df_slack(1 * (n_step - 1) + i)) : 0.0));

          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 8 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  right_normal.dot(goal_local.col(i))-(1e-3) < 0.0?
                                  right_normal.dot(dgl_dq) * exforce(2 * n_step + i + 1) : 0.0));
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 9 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  -right_normal.dot(goal_local.col(i)) < 0.0?
                                  -right_normal.dot(dgl_dq) * 
                                  (exforce(2 * n_step + i + 1) * d_slack(2 * (n_step - 1) + i)
                                  - df_slack(2 * (n_step - 1) + i)) : 0.0));
                            
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 11 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  back_normal.dot(goal_local.col(i))-(1e-3) < 0.0?
                                  back_normal.dot(dgl_dq) * exforce(3 * n_step + i + 1) : 0.0));
          triplet_pos.push_back(T(n_dof * 2 * (n_step - 1) + 12 * (n_step - 1) + i, 
                                  n_dof * (i+1) + j,
                                  -back_normal.dot(goal_local.col(i)) < 0.0?
                                  -back_normal.dot(dgl_dq) * 
                                  (exforce(3 * n_step + i + 1) * d_slack(3 * (n_step - 1) + i)
                                  - df_slack(3 * (n_step - 1) + i)) : 0.0));
        }
        if (var_set == "velocity") {
          // Triplet for velocity
          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j, n_dof * i + j,
                                  -1.0 / t_step)); // ddq_dv_k

          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j, n_dof * (i+1) - 3,
                                  (Minv * f)(j, 0) * 20 * 
                                      (1.0 - tanh(20 * vel(n_dof * (i+1) - 3)) *
                                                 tanh(20 * vel(n_dof * (i+1) - 3))))); // ddq_dv_k from smoothed friction term x
          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j, n_dof * (i+1) - 2,
                                  (Minv * f)(j, 1) * 20 * 
                                      (1.0 - tanh(20 * vel(n_dof * (i+1) - 2)) *
                                                 tanh(20 * vel(n_dof * (i+1) - 2))))); // ddq_dv_k from smoothed friction term y
          triplet_vel.push_back(T(n_dof * (n_step - 1 + i) + j, n_dof * (i+1) - 1,
                                  (Minv * f)(j, 2) * 20 * 
                                      (1.0 - tanh(20 * vel(n_dof * (i+1) - 1)) *
                                                 tanh(20 * vel(n_dof * (i+1) - 1))))); // ddq_dv_k from smoothed friction term rotation

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
          // front
          triplet_exforce.push_back(T(n_dof * (n_step - 1 + i) + j, i+1,
                                    (-Minv * J_front_remapped.col(i))(j)));
          // left
          triplet_exforce.push_back(T(n_dof * (n_step - 1 + i) + j, 
                                    n_step + i + 1,
                                    (-Minv * J_left_remapped.col(i))(j)));
          // right
          triplet_exforce.push_back(T(n_dof * (n_step - 1 + i) + j, 
                                    2 * n_step + i + 1,
                                    (-Minv * J_right_remapped.col(i))(j)));
          // back
          triplet_exforce.push_back(T(n_dof * (n_step - 1 + i) + j, 
                                    3 * n_step + i + 1,
                                    (-Minv * J_back_remapped.col(i))(j)));                                    
        }
      }
      if (var_set == "exforce") {
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i, 
                                  i + 1, -std::min(front_normal.dot(goal_local.col(i))-(1e-3), 0.0)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i, 
                                  i + 1, -std::min(-front_normal.dot(goal_local.col(i)), 0.0) * d_slack(i)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 5 * (n_step - 1) + i, 
                                  n_step + i + 1, -std::min(left_normal.dot(goal_local.col(i))-(1e-3), 0.0)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i, 
                                  n_step + i + 1, -std::min(-left_normal.dot(goal_local.col(i)), 0.0) * d_slack(n_step - 1 + i)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 8 * (n_step - 1) + i, 
                                  2 * n_step + i + 1, -std::min(right_normal.dot(goal_local.col(i))-(1e-3), 0.0)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 9 * (n_step - 1) + i, 
                                  2 * n_step + i + 1, 
                                  -std::min(-right_normal.dot(goal_local.col(i)), 0.0) * d_slack(2 * (n_step - 1) + i)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 11 * (n_step - 1) + i, 
                                  3 * n_step + i + 1, -std::min(back_normal.dot(goal_local.col(i))-(1e-3), 0.0)));
        triplet_exforce.push_back(T(n_dof * 2 * (n_step - 1) + 12 * (n_step - 1) + i, 
                                  3 * n_step + i + 1, 
                                  -std::min(-back_normal.dot(goal_local.col(i)), 0.0) * d_slack(3 * (n_step - 1) + i)));                                  
      }
      if (var_set == "d_slack") {
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 1 * (n_step - 1) + i, 
                                  i, -1));
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i, 
                                  i, -std::min(-front_normal.dot(goal_local.col(i)), 0.0) *  exforce(i + 1)));
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i, 
                                  n_step - 1 + i, -1));
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i, 
                                  n_step - 1 + i, 
                                  -std::min(-left_normal.dot(goal_local.col(i)), 0.0) * exforce(n_step + i + 1)));                                
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 7 * (n_step - 1) + i, 
                                  2 * (n_step - 1) + i, -1));
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 9 * (n_step - 1) + i, 
                                  2 * (n_step - 1) + i, 
                                  -std::min(-right_normal.dot(goal_local.col(i)), 0.0) * exforce(2 * n_step + i + 1)));
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 10 * (n_step - 1) + i, 
                                  3 * (n_step - 1) + i, -1));
        triplet_d_slack.push_back(T(n_dof * 2 * (n_step - 1) + 12 * (n_step - 1) + i, 
                                  3 * (n_step - 1) + i, 
                                  -std::min(-back_normal.dot(goal_local.col(i)), 0.0) * exforce(3 * n_step + i + 1)));                                  
      }
      if (var_set == "df_slack") {
        triplet_df_slack.push_back(T(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i, 
                                   i, std::min(-front_normal.dot(goal_local.col(i)), 0.0)));
        triplet_df_slack.push_back(T(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i, 
                                   n_step - 1 + i, std::min(-left_normal.dot(goal_local.col(i)), 0.0)));
        triplet_df_slack.push_back(T(n_dof * 2 * (n_step - 1) + 9 * (n_step - 1) + i, 
                                   2 * (n_step - 1) + i, std::min(-right_normal.dot(goal_local.col(i)), 0.0)));
        triplet_df_slack.push_back(T(n_dof * 2 * (n_step - 1) + 12 * (n_step - 1) + i, 
                                   3 * (n_step - 1) + i, std::min(-back_normal.dot(goal_local.col(i)), 0.0)));                                   
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
    if (var_set == "d_slack") {
      jac_block.setFromTriplets(triplet_d_slack.begin(), triplet_d_slack.end());
    }
    if (var_set == "df_slack") {
      jac_block.setFromTriplets(triplet_df_slack.begin(), triplet_df_slack.end());
    }
  }
};

class ExCost : public CostTerm {
public:
  ExCost() : ExCost("cost_term1") {}
  ExCost(const std::string &name) : CostTerm(name) {
    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_step = params["n_step"].as<int>();
    }

  double GetCost() const override {
    VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
    VectorXd torque = GetVariables()->GetComponent("effort")->GetValues();
    VectorXd df_slack = GetVariables()->GetComponent("df_slack")->GetValues();

    double cost = vel.squaredNorm();

    cost = cost + 1e4 * df_slack.lpNorm<1>();

    return cost;
  }

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
    if (var_set == "df_slack"){
      VectorXd df_slack = GetVariables()->GetComponent("df_slack")->GetValues();
      std::vector<T> triplet_slack;
      
      for (int i = 0; i < 4 * (n_step - 1); i++)
      {
        triplet_slack.push_back(T(0, i, 1e4));
      }
      
      jac.setFromTriplets(triplet_slack.begin(), triplet_slack.end());
    }
  }


private:
  int n_step;
};

} // namespace ifopt