#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
#include <Eigen/Geometry> 
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own
// directory here.
#ifndef PINOCCHIO_MODEL_DIR
#define PINOCCHIO_MODEL_DIR "/home/mzwang/catkin_ws/src/trajectory_my/model"
#endif

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
        xvar(i) = 0;
    } else if (name == "torque") {
      for (int i = 0; i < n; i++)
        xvar(i) = 1;
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
      for (int i = 0; i < 6; i++)
        bounds.at(i) = Bounds(0, 0);
      // for (int i = GetRows() - 6; i < GetRows(); i++)
      //   bounds.at(i) = Bounds(0, 0);

      // bounds.at(GetRows() - 6) = Bounds(0.5, 0.5);
      // bounds.at(GetRows() - 5) = Bounds(0.0, 0.0);
      // bounds.at(GetRows() - 4) = Bounds(0.5, 0.5);
      // bounds.at(GetRows() - 3) = Bounds(0.0, 0.0);
      // bounds.at(GetRows() - 2) = Bounds(0.5, 0.5);
      // bounds.at(GetRows() - 1) = Bounds(0.0, 0.0);
      
    } else if (GetName() == "velocity") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = NoBound;
      for (int i = 0; i < 6; i++)
        bounds.at(i) = Bounds(0, 0);
      for (int i = GetRows() - 6; i < GetRows(); i++)
        bounds.at(i) = Bounds(0, 0);
    } else {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(-torque_lim, torque_lim);
    }
    return bounds;
  }

private:
  Eigen::VectorXd xvar;
  double position_lim = 3.14;
  double torque_lim = 100;
};

class ExConstraint : public ConstraintSet {
public:
  const std::string urdf_filename =
      PINOCCHIO_MODEL_DIR + std::string("/ur5_robot.urdf");
  pinocchio::Model model;
  int ndof = 6;        // number of freedom
  int nsteps = 20;     // number of steps or (knot points - 1)
  double tstep = 0.05; // length of each step
  Eigen::Matrix3d goal_rotation; // goal rotation matrix
  Eigen::Quaterniond goal_quaternion, goal_quaternion_ap; // goal quaternion
  Eigen::Vector3d goal_translation; // goal translation

  ExConstraint(int n) : ExConstraint(n, "constraint1") {}
  ExConstraint(int n, const std::string &name) : ConstraintSet(n, name) {
    pinocchio::urdf::buildModel(urdf_filename, model);
    // set start and end pose
    goal_rotation <<   -0.0515054, 0.90572, -0.420735,
                       0.971862, -0.0515054, -0.229849,
                      -0.229849, -0.420735, -0.877583;
    goal_quaternion = goal_rotation;
    goal_quaternion_ap = this->GetAP(goal_quaternion);
    goal_translation << 0.578674, 0.522807, -0.200875;
  }

  Eigen::Quaterniond GetAP(Eigen::Quaterniond &a){
    Eigen::Quaterniond quater_ap(-a.w(), -a.x(), -a.y(), -a.z());
    return quater_ap;
  }

  VectorXd GetValues() const override {
    pinocchio::Data data(model);
    pinocchio::Data data_next(model);
    VectorXd g(GetRows());
    VectorXd pos = GetVariables()->GetComponent("position")->GetValues();
    VectorXd vel = GetVariables()->GetComponent("velocity")->GetValues();
    VectorXd torque = GetVariables()->GetComponent("torque")->GetValues();

    // set contraint for each knot point
    for (int i = 0; i < nsteps - 1; i++) {
      pinocchio::computeAllTerms(model, data, pos.segment(ndof * i, ndof),
                                 vel.segment(ndof * i, ndof));
      pinocchio::computeAllTerms(model, data_next,
                                 pos.segment(ndof * (i + 1), ndof),
                                 vel.segment(ndof * (i + 1), ndof));
      // trapezoidal collocation. Minv needed but not correctly provided
      // by Pinocchio
      // g.segment(ndof * (2 * i), ndof) =
      //     pos.segment(ndof * i, ndof) - pos.segment(ndof * (i + 1), ndof) +
      //     0.5 * tstep *
      //         (vel.segment(ndof * i, ndof) + vel.segment(ndof * (i + 1),
      //         ndof));
      // g.segment(ndof * (2 * i + 1), ndof) =
      //     vel.segment(ndof * i, ndof) - vel.segment(ndof * (i + 1), ndof) +
      //     0.5 * tstep *
      //         (data_next.Minv *
      //              (torque.segment(ndof * (i + 1), ndof) - data_next.nle) +
      //          data.Minv * (torque.segment(ndof * (i), ndof) - data.nle));

      // backward integration
      g.segment(ndof * (2 * i), ndof) =
          pos.segment(ndof * i, ndof) - pos.segment(ndof * (i + 1), ndof) +
          tstep * (vel.segment(ndof * (i + 1), ndof));

      g.segment(ndof * (2 * i + 1), ndof) =
          data_next.M * (vel.segment(ndof * (i + 1), ndof) -
                         vel.segment(ndof * i, ndof)) +
          tstep * (data_next.nle - torque.segment(ndof * (i + 1), ndof));
    }

    // set constraint for final pose
    // calculate final pose error
    pinocchio::FrameIndex frameID = model.getFrameId("ee_fixed_joint");
    pinocchio::forwardKinematics(model, data, pos.segment(ndof*(nsteps-1), ndof));
    pinocchio::updateFramePlacement(model, data, frameID);
    pinocchio::GeometryData::SE3 end_pose = data.oMf[frameID];
    Eigen::Matrix3d end_rotation = end_pose.rotation();
    Eigen::Vector3d end_translation = end_pose.translation();

    // Eigen::Quaterniond end_quaternion(end_rotation);
    Eigen::Vector3d end_translation_error = end_translation - goal_translation;

    // g(GetRows() - 4) = 0*
    //     log((end_quaternion.inverse() * goal_quaternion).norm()) *
    //     log((end_quaternion.inverse() * goal_quaternion_ap).norm());

    g.segment(GetRows() - 3, 3) = end_translation_error;

    return g;
  };

  // // Constant values should always be put into GetBounds(), not GetValues().
  // // For inequality constraints (<,>), use Bounds(x, inf) or Bounds(-inf, x).
  VecBound GetBounds() const override {
    VecBound bounds(GetRows());
    for (int i = 0; i < GetRows(); i++)
      bounds.at(i) = Bounds(0.0, 0.0);
    return bounds;
  }

  void FillJacobianBlock(std::string var_set,
                         Jacobian &jac_block) const override {}
};

class ExCost : public CostTerm {
public:
  ExCost() : ExCost("cost_term1") {}
  ExCost(const std::string &name) : CostTerm(name) {}

  double GetCost() const override {
    VectorXd torque = GetVariables()->GetComponent("torque")->GetValues();
    int n = GetVariables()->GetComponent("torque")->GetRows();
    Eigen::VectorXd vec(n);
    for (int i; i < n; i++) {
      vec(i) = 1;
    }
    vec(0) = 0.5;
    vec(vec.size() - 1) = 0.5;
    Eigen::MatrixXd weight(n, n);
    weight = vec.asDiagonal();
    auto cost = (torque.transpose() * weight) * torque;
    return cost;
  };

  void FillJacobianBlock(std::string var_set, Jacobian &jac) const override {}
};

} // namespace ifopt