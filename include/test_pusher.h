#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/parsers/srdf.hpp"
#include "pinocchio/parsers/sample-models.hpp"

#include "pinocchio/multibody/model.hpp"
#include "pinocchio/multibody/joint/joints.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/geometry.hpp"

#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/geometry.hpp"
#include <pinocchio/algorithm/model.hpp>
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/spatial/act-on-set.hpp"

#include "hpp/fcl/distance.h"
#include "hpp/fcl/collision.h"
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
#include <Eigen/Geometry> 
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <yaml-cpp/yaml.h>
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
    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    if (name == "position") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0;
    } else if (name == "velocity") {
      for (int i = 0; i < n; i++)
        xvar(i) = 1;
    } else if (name == "effort") {
      for (int i = 0; i < n; i++)
        xvar(i) = 1;
    } else if (name == "exforce") {
      for (int i = 0; i < n; i++)
        xvar(i) = 1;
    } else if (name == "slack") {
      for (int i = 0; i < n/2; i++){
        xvar(i) = 0.1;
        xvar(n/2 + i) = 1;
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
        bounds.at(i) = Bounds(-5, 5);
      for (int i = 0; i < GetRows(); i = i+4)
        bounds.at(i) = Bounds(-0.05, 0.2);
      bounds.at(0) = Bounds(0, 0);
      bounds.at(1) = Bounds(0.35, 0.35);
      bounds.at(2) = Bounds(0, 0);
      bounds.at(3) = Bounds(0, 0);
      // bounds.at(GetRows()-3) = Bounds(0.43, 0.43);
      // bounds.at(GetRows()-2) = Bounds(0, 0);
    } else if (GetName() == "velocity") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(-velocity_lim, velocity_lim);
      for (int i = 0; i < 4; i++)
        bounds.at(i) = Bounds(0, 0);
      for (int i = GetRows() - 4; i < GetRows(); i++)
        bounds.at(i) = Bounds(0, 0);
    } else if (GetName() == "effort") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(-effort_lim, effort_lim);
    } else if (GetName() == "exforce") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(-force_lim, force_lim);  // NOTE
    } else if (GetName() == "slack") {
      for (int i = 0; i < GetRows()/2; i++) {
        bounds.at(i) = Bounds(0, inf);  // distance slack
        bounds.at(GetRows()/2 + i) = Bounds(-force_lim, force_lim); // force slack
      }        
    }
    return bounds;
  }

private:
  Eigen::VectorXd xvar;
  double position_lim = 3.14;
  double velocity_lim = 0.5;
  double effort_lim = 10;
  double force_lim = 10;
};

// system dynamics constraints
class ExConstraint : public ConstraintSet {
public:
  const std::string urdf_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/pusher.urdf");
  const std::string srdf_filename =
    PINOCCHIO_MODEL_DIR + std::string("/srdf/pusher.srdf");
  const std::string box_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/box.urdf");

  mutable pinocchio::Model model;
  pinocchio::GeometryModel geom_model, box_geom_model;
  pinocchio::PairIndex cp_index;
  pinocchio::FrameIndex contactId, object_contactId;
  int n_dof;       // number of freedom
  int n_control;     // number of control
  int n_exforce;   // number of external force
  int n_step;     // number of steps or (knot points - 1)
  double t_step;   // length of each step

  ExConstraint(int n) : ExConstraint(n, "constraint1") {}
  ExConstraint(int n, const std::string &name) : ConstraintSet(n, name) {
    pinocchio::urdf::buildModel(urdf_filename, model); // build the pusher model
    pinocchio::JointIndex joint_index = model.joints.size();
    pinocchio::FrameIndex frame_index = model.nframes;
    // add virtual contact point frame for Jacobian calculation
    // add as many as needed
    contactId =  model.addFrame(pinocchio::Frame("contactPoint", 
                                model.getJointId("base_to_pusher"), 
                                -1, 
                                pinocchio::SE3::Identity(), 
                                pinocchio::OP_FRAME));
    // build the box into the model
    pinocchio::urdf::buildModel(box_filename, pinocchio::JointModelPlanar(), model); 
    // set planar joint bounds
    setRootJointBounds(model, joint_index); 
    // change box root joint name, otherwise duplicated with robot root joint
    model.names[joint_index] = "box_root_joint";
    model.frames[frame_index].name = "box_root_joint";
    // add virtual contact point frame on box
    object_contactId =  model.addFrame(
                                pinocchio::Frame("object_contactPoint", 
                                model.getJointId("box_root_joint"), 
                                -1, 
                                pinocchio::SE3::Identity(), 
                                pinocchio::OP_FRAME));
    // build the geometry model
    pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION, geom_model, PINOCCHIO_MODEL_DIR);
    pinocchio::urdf::buildGeom(model, box_filename, pinocchio::COLLISION, box_geom_model, PINOCCHIO_MODEL_DIR);
    pinocchio::appendGeometryModel(geom_model, box_geom_model);
    geom_model.addAllCollisionPairs();
    pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
    
    // define the potential collision pair, as many as needed
    pinocchio::GeomIndex tip_id = geom_model.getGeometryId("tip_0");
    pinocchio::GeomIndex box_id = geom_model.getGeometryId("box_0");
    pinocchio::CollisionPair cp = pinocchio::CollisionPair(tip_id, box_id);
    cp_index = geom_model.findCollisionPair(cp);

    YAML::Node params = YAML::LoadFile("/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    n_dof = params["n_dof"].as<int>();
    n_control = params["n_control"].as<int>();
    n_exforce = params["n_exforce"].as<int>();
    n_step = params["n_step"].as<int>();
    t_step = params["t_step"].as<double>();
  }

  void setRootJointBounds(pinocchio::Model &model,
                          const pinocchio::JointIndex &rtIdx)
  {
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
    pinocchio::GeometryData geom_data_next(geom_model);
    VectorXd g(GetRows());
    VectorXd pos = GetVariables()->GetComponent("position")->GetValues();
    VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
    VectorXd effort = GetVariables()->GetComponent("effort")->GetValues();
    VectorXd exforce = GetVariables()->GetComponent("exforce")->GetValues();
    VectorXd slack = GetVariables()->GetComponent("slack")->GetValues();

    // set contraint for each knot point
    // constraints shape
    ///////////////////////////////////////////////
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
      Eigen::VectorXd q(model.nq);
      q.segment(0, n_dof - 1) = pos.segment(n_dof * (i + 1), n_dof - 1);
      q(model.nq - 2) = cos(pos(n_dof * (i + 1) + n_dof - 1));
      q(model.nq - 1) = sin(pos(n_dof * (i + 1) + n_dof - 1));
      // Update the contact point frame
      pinocchio::framesForwardKinematics(model, data_next, q);
      pinocchio::SE3 joint_frame_placement =
          data.oMf[model.getFrameId("base_to_pusher")];
      pinocchio::SE3 object_joint_frame_placement =
          data.oMf[model.getFrameId("box_root_joint")];
      pinocchio::computeCollisions(model, data_next, geom_model, geom_data_next,
                                   q);
      pinocchio::computeDistances(model, data_next, geom_model, geom_data_next,
                                  q);
      hpp::fcl::DistanceResult dr = geom_data_next.distanceResults[cp_index];
      model.frames[contactId].placement.translation() =
          joint_frame_placement.inverse().act(dr.nearest_points[0]);
      model.frames[object_contactId].placement.translation() =
          object_joint_frame_placement.inverse().act(dr.nearest_points[1]);

      // Calculate NLE, inertial matrix and Jacobian
      pinocchio::nonLinearEffects(model, data_next, q,
                                  vel.segment(n_dof * (i + 1), n_dof));

      pinocchio::computeMinverse(model, data_next, q);
      Eigen::MatrixXd Minv = data_next.Minv;
      Minv.triangularView<Eigen::StrictlyLower>() =
          Minv.transpose().triangularView<Eigen::StrictlyLower>();

      pinocchio::Data::Matrix6x w_J_contact(6, model.nv),
          w_J_object(6, model.nv);
      w_J_contact.fill(0);
      w_J_object.fill(0);
      pinocchio::computeJointJacobians(model, data_next, q);
      pinocchio::framesForwardKinematics(model, data_next, q);
      pinocchio::getFrameJacobian(model, data_next, contactId, pinocchio::WORLD,
                                  w_J_contact);
      pinocchio::getFrameJacobian(model, data_next, object_contactId, pinocchio::WORLD,
                                  w_J_object);
      // TODO: normal force normal vector. fixed for now
      Eigen::Vector3d normal_f(1, 0, 0);
      normal_f.normalize();
      // Jacobian mapping exforce to both robot and object
      // Be careful to the sign!!! Different for pusher and box!
      pinocchio::Data::Matrix6x J_final = -1 * w_J_contact + w_J_object;
      
      Eigen::VectorXd J_remapped(model.nv);
      J_remapped = (normal_f.transpose() * J_final.topRows(3)).transpose();
      // std::cout << J_remapped << "\n";
      // Input Mapping
      Eigen::Vector4d B(1.0, 0, 0, 0);
      Eigen::Vector4d f(0.0, 0.05, 0, 0);
      // backward integration
      g.segment(n_dof * i, n_dof) =
          pos.segment(n_dof * i, n_dof) - pos.segment(n_dof * (i + 1), n_dof) +
          t_step * (vel.segment(n_dof * (i + 1), n_dof));

      g.segment(n_dof * (n_step - 1 + i), n_dof) =
          1 / t_step *
              (vel.segment(n_dof * (i + 1), n_dof) -
               vel.segment(n_dof * i, n_dof)) +
          Minv * (data_next.nle - B * effort(i + 1)) -
          J_remapped * exforce(i+1)/* + f * vel(n_dof * (i + 1)+1)*/;

      // Complementary constraints, 3 constraints for each step
      g(n_dof * 2 * (n_step - 1) + i) = dr.min_distance - slack(i);
      g(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = exforce(i+1) - slack(n_step - 1 + i);
      // g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = dr.min_distance * exforce(i);
      g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = slack(i) * slack(n_step - 1 + i);
      // slack
      // g(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) = slack(i);
      // g(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i) = slack(n_step - 1 + i);
    }
    return g;
  };


  // // Constant values should always be put into GetBounds(), not GetValues().
  // // For inequality constraints (<,>), use Bounds(x, inf) or Bounds(-inf, x).
  VecBound GetBounds() const override {
    VecBound bounds(GetRows());
    for (int i = 0; i < n_dof * 2 * (n_step - 1); i++)
      bounds.at(i) = Bounds(0.0, 0.0);
    // for (int i = 0; i < n_step - 1; i++){
    //   bounds.at(n_dof * 2 * (n_step - 1) + i) = Bounds(0.0, inf);
    //   bounds.at(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = Bounds(0.0, inf);
    //   bounds.at(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = Bounds(0.0, 0.0);
    // }
    for (int i = 0; i < n_step - 1; i++){
      bounds.at(n_dof * 2 * (n_step - 1) + i) = Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = Bounds(0.0, 0.0);
      // bounds.at(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) = Bounds(0.0, inf);
      // bounds.at(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i) = Bounds(0.0, inf);
    }
    return bounds;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override { }
  
};

class ExCost : public CostTerm {
public:
  ExCost() : ExCost("cost_term1") {}
  ExCost(const std::string &name) : CostTerm(name) {}

  double GetCost() const override {
    VectorXd pos = GetVariables()->GetComponent("position")->GetValues();
    VectorXd torque = GetVariables()->GetComponent("effort")->GetValues();
    int n = GetVariables()->GetComponent("effort")->GetRows();
    Eigen::VectorXd vec(n);
    for (int i = 0; i < n; i++) {
      vec(i) = 1;
    }
    vec(0) = 0.5;
    vec(vec.size() - 1) = 0.5;
    Eigen::MatrixXd weight(n, n);
    weight = vec.asDiagonal();
    // double cost = (torque.transpose() * weight) * torque;

    YAML::Node params = YAML::LoadFile("/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    std::vector<double> box_goal = params["box_goal"].as< std::vector<double> >();
    // double cost = 50*(pow((box_goal[0] - pos[pos.size() - 3]), 2) + pow((box_goal[1] - pos[pos.size() - 2]), 2));
    double cost = abs(box_goal[0] - pos[pos.size() - 3]) + abs(box_goal[1] - pos[pos.size() - 2]);

    return cost;
  };

  void FillJacobianBlock(std::string var_set, Jacobian &jac) const override {
  }
  
};

} // namespace ifopt