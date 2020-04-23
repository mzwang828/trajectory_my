#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>
#include <ifopt/variable_set.h>
#include <math.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <pinocchio/algorithm/model.hpp>

#include "hpp/fcl/collision.h"
#include "hpp/fcl/distance.h"
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
    xvar = Eigen::VectorXd::Zero(n);
    // the initial values where the NLP starts iterating from
    if (name == "position") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0.0;
    } else if (name == "velocity") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0.0;
    } else if (name == "effort") {
      for (int i = 0; i < n; i++)
        xvar(i) = 1;
    } else if (name == "exforce") {
      for (int i = 0; i < n; i++)
        xvar(i) = 1;
    } else if (name == "slack") {
      for (int i = 0; i < n / 2; i++) {
        xvar(i) = 0.2;
        xvar(n / 2 + i) = 1;
      }
    }
    // FOR FRICTION
    else if (name == "friction") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0;
    } else if (name == "v_slack") {
      for (int i = 0; i < n; i++)
        xvar(i) = 0;
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
      for (int i = 0; i < GetRows(); i = i + 4)
        bounds.at(i) = Bounds(-0.05, 1.0);
      bounds.at(0) = Bounds(0, 0);
      bounds.at(1) = Bounds(0.35, 0.35);
      bounds.at(2) = Bounds(0, 0);
      bounds.at(3) = Bounds(0, 0);
      bounds.at(GetRows() - 3) = Bounds(0.53, 0.53);
      bounds.at(GetRows() - 2) = Bounds(0, 0);
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
        bounds.at(i) = Bounds(0, force_lim); // NOTE
    } else if (GetName() == "slack") {
      for (int i = 0; i < GetRows() / 2; i++) {
        bounds.at(i) = Bounds(0, inf);                       // distance slack
        bounds.at(GetRows() / 2 + i) = Bounds(0, force_lim); // force slack
      }
    }
    // FOR FRICTION
    else if (GetName() == "friction") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(0, inf);
    } else if (GetName() == "v_slack") {
      for (int i = 0; i < GetRows(); i++)
        bounds.at(i) = Bounds(0, inf);
    }
    return bounds;
  }

private:
  Eigen::VectorXd xvar;
  double position_lim = 5;
  double velocity_lim = 5;
  double effort_lim = 100;
  double force_lim = 100;
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

  mutable pinocchio::Model robot_model, box_model, model;
  pinocchio::GeometryModel geom_model, box_geom_model;
  pinocchio::PairIndex cp_index;
  pinocchio::FrameIndex contactId, object_contactId;
  Eigen::Vector3d front_normal; // normal vector point to the front plane
  int n_dof;                    // number of freedom
  int n_control;                // number of control
  int n_exforce;                // number of external force
  int n_step;                   // number of steps or (knot points - 1)
  double t_step;                // length of each step

  ExConstraint(int n) : ExConstraint(n, "constraint1") {}
  ExConstraint(int n, const std::string &name) : ConstraintSet(n, name) {
    front_normal << 1, 0, 0;
    // build the pusher model
    pinocchio::urdf::buildModel(urdf_filename, robot_model);
    // build the box model
    pinocchio::urdf::buildModel(box_filename, pinocchio::JointModelPlanar(),
                                box_model);
    // set planar joint bounds, root joint is with joint_index 1
    setRootJointBounds(box_model, 1);
    // change box root joint name, otherwise duplicated with robot root joint
    box_model.frames[1].name =
        "box_root_joint"; // index of root joint is 1. 0 = universe
    box_model.names[1] = "box_root_joint";
    pinocchio::appendModel(box_model, robot_model, 0,
                           pinocchio::SE3::Identity(), model);
    // add virtual contact point frame for Jacobian calculation
    // add as many as needed
    contactId = model.addFrame(
        pinocchio::Frame("contactPoint", model.getJointId("base_to_pusher"), -1,
                         pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    object_contactId = model.addFrame(pinocchio::Frame(
        "object_contactPoint", model.getJointId("box_root_joint"), -1,
        pinocchio::SE3::Identity(), pinocchio::OP_FRAME));
    // build the geometry model
    pinocchio::urdf::buildGeom(model, urdf_filename, pinocchio::COLLISION,
                               geom_model, PINOCCHIO_MODEL_DIR);
    pinocchio::urdf::buildGeom(model, box_filename, pinocchio::COLLISION,
                               box_geom_model, PINOCCHIO_MODEL_DIR);
    pinocchio::appendGeometryModel(geom_model, box_geom_model);

    // define the potential collision pair, as many as needed
    pinocchio::GeomIndex tip_id = geom_model.getGeometryId("tip_0");
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
    // FOR FRICTION
    // VectorXd friction =
    // GetVariables()->GetComponent("friction")->GetValues(); VectorXd v_slack =
    // GetVariables()->GetComponent("v_slack")->GetValues();

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
      pinocchio::computeCollisions(model, data, geom_model, geom_data, q);
      pinocchio::computeDistances(model, data, geom_model, geom_data, q);
      hpp::fcl::DistanceResult dr = geom_data.distanceResults[cp_index];
      Eigen::Vector3d front_normal_transformed =
          geom_data.oMg[geom_model.getGeometryId("box_0")].rotation() *
          front_normal;
      // get sign from distance normal & surface normal vector
      double distance =
          signbit(dr.normal.transpose() * front_normal_transformed)
              ? (-1) * dr.min_distance
              : dr.min_distance;

      // Update the contact point frame
      pinocchio::computeCollisions(model, data_next, geom_model, geom_data,
                                   q_next);
      pinocchio::computeDistances(model, data_next, geom_model, geom_data,
                                  q_next);
      dr = geom_data.distanceResults[cp_index];

      // Get Contact Point Jacobian
      pinocchio::framesForwardKinematics(model, data_next, q_next);
      pinocchio::SE3 joint_frame_placement =
          data_next.oMf[model.getFrameId("base_to_pusher")];
      pinocchio::SE3 root_joint_frame_placement =
          data_next.oMf[model.getFrameId("box_root_joint")];
      model.frames[contactId].placement.translation() =
          joint_frame_placement.inverse().act(dr.nearest_points[0]);
      model.frames[object_contactId].placement.translation() =
          root_joint_frame_placement.inverse().act(dr.nearest_points[1]);

      pinocchio::Data::Matrix6x w_J_contact(6, model.nv),
          w_J_contact_aligned(6, model.nv), w_J_object_aligned(6, model.nv),
          w_J_object(6, model.nv);
      w_J_contact.fill(0);
      w_J_object.fill(0);
      w_J_contact_aligned.fill(0);
      w_J_object_aligned.fill(0);
      pinocchio::computeJointJacobians(model, data_next, q_next);
      pinocchio::framesForwardKinematics(model, data_next, q_next);
      pinocchio::getFrameJacobian(model, data_next, contactId, pinocchio::LOCAL,
                                  w_J_contact);
      pinocchio::getFrameJacobian(model, data_next, object_contactId,
                                  pinocchio::LOCAL, w_J_object);
      pinocchio::getFrameJacobian(model, data_next, contactId,
                                  pinocchio::LOCAL_WORLD_ALIGNED,
                                  w_J_contact_aligned);
      pinocchio::getFrameJacobian(model, data_next, object_contactId,
                                  pinocchio::LOCAL_WORLD_ALIGNED,
                                  w_J_object_aligned);
      w_J_contact.rightCols(1) = w_J_contact_aligned.rightCols(1);
      w_J_object.rightCols(1) = w_J_object_aligned.rightCols(1);
      // Jacobian mapping exforce to both robot and object
      // Be careful to the sign!!! Different for pusher and box!
      pinocchio::Data::Matrix6x J_final = -1 * w_J_contact + w_J_object;
      Eigen::VectorXd J_remapped(model.nv);
      J_remapped = J_final.topRows(3).transpose() * dr.normal;

      // Calculate NLE, inertial matrix
      pinocchio::nonLinearEffects(model, data_next, q_next,
                                  vel.segment(n_dof * (i + 1), n_dof));
      pinocchio::computeMinverse(model, data_next, q_next);
      Eigen::MatrixXd Minv = data_next.Minv;
      Minv.triangularView<Eigen::StrictlyLower>() =
          Minv.transpose().triangularView<Eigen::StrictlyLower>();

      // Input Mapping & Friction
      Eigen::Vector4d B(1.0, 0, 0, 0);
      Eigen::Vector4d f(0.0, 0.5, 0, 0);
      // backward integration
      g.segment(n_dof * i, n_dof) =
          pos.segment(n_dof * i, n_dof) - pos.segment(n_dof * (i + 1), n_dof) +
          t_step * (vel.segment(n_dof * (i + 1), n_dof));

      // smoothed if condition
      g.segment(n_dof * (n_step - 1 + i), n_dof) =
          1 / t_step *
              (vel.segment(n_dof * (i + 1), n_dof) -
               vel.segment(n_dof * i, n_dof)) +
          Minv * (data_next.nle - B * effort(i + 1)) -
          J_remapped * exforce(i + 1) + f * tanh(20 * vel(n_dof * (i) + 1));

      // Complimentary friction////////////////
      // Eigen::Vector4d friction_pos(0.0, friction(i), 0, 0);
      // Eigen::Vector4d friction_neg(0.0, -friction(n_step - 1 + i), 0, 0);
      // g.segment(n_dof * (n_step - 1 + i), n_dof) =
      //     1 / t_step *
      //         (vel.segment(n_dof * (i + 1), n_dof) -
      //           vel.segment(n_dof * i, n_dof)) +
      //     Minv * (data_next.nle - B * effort(i + 1)) -
      //     J_remapped * exforce(i + 1) + friction_pos + friction_neg;
      // g(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) = v_slack(i) +
      // vel(n_dof * (i) + 1); // Eq. (11) g(n_dof * 2 * (n_step - 1) + 4 *
      // (n_step - 1) + i) = v_slack(i) - vel(n_dof * (i) + 1); // Eq. (12)
      // g(n_dof * 2 * (n_step - 1) + 5 * (n_step - 1) + i) = 0.5 - friction(i)
      // - friction(n_step - 1 + i); // Eq. (10) g(n_dof * 2 * (n_step - 1) + 6
      // * (n_step - 1) + i) = (0.5 - friction(i) - friction(n_step - 1 + i)) *
      // v_slack(i); // Eq. (14) g(n_dof * 2 * (n_step - 1) + 7 * (n_step - 1) +
      // i) = (v_slack(i) + vel(n_dof * (i) + 1)) * friction(i); // Eq. (15)
      // g(n_dof * 2 * (n_step - 1) + 8 * (n_step - 1) + i) = (v_slack(i) -
      // vel(n_dof * (i) + 1)) * friction(n_step - 1 + i); // Eq. (16)
      /////////////////////////////////////////

      // Complementary constraints, 3 constraints for each step
      g(n_dof * 2 * (n_step - 1) + i) = distance - slack(i);
      g(n_dof * 2 * (n_step - 1) + n_step - 1 + i) =
          exforce(i + 1) - slack(n_step - 1 + i);
      // g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) = dr.min_distance *
      // exforce(i);
      g(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) =
          slack(i) * slack(n_step - 1 + i);
      // slack
      // g(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) = slack(i);
      // g(n_dof * 2 * (n_step - 1) + 4 * (n_step - 1) + i) = slack(n_step - 1 +
      // i);
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
    //   bounds.at(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = Bounds(0.0,
    //   inf); bounds.at(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) =
    //   Bounds(0.0, 0.0);
    // }
    for (int i = 0; i < n_step - 1; i++) {
      bounds.at(n_dof * 2 * (n_step - 1) + i) = Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + n_step - 1 + i) = Bounds(0.0, 0.0);
      bounds.at(n_dof * 2 * (n_step - 1) + 2 * (n_step - 1) + i) =
          Bounds(0.0, 0.0);
      // FRICTION
      // bounds.at(n_dof * 2 * (n_step - 1) + 3 * (n_step - 1) + i) =
      // Bounds(0.0, inf); // Eq. (11) bounds.at(n_dof * 2 * (n_step - 1) + 4 *
      // (n_step - 1) + i) = Bounds(0.0, inf); // Eq. (12) bounds.at(n_dof * 2 *
      // (n_step - 1) + 5 * (n_step - 1) + i) = Bounds(0.0, inf); // Eq. (10)
      // bounds.at(n_dof * 2 * (n_step - 1) + 6 * (n_step - 1) + i) =
      // Bounds(0.0, 0.0); // Eq. (14) bounds.at(n_dof * 2 * (n_step - 1) + 7 *
      // (n_step - 1) + i) = Bounds(0.0, 0.0); // Eq. (15) bounds.at(n_dof * 2 *
      // (n_step - 1) + 8 * (n_step - 1) + i) = Bounds(0.0, 0.0); // Eq. (16)
    }
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
    double cost = (torque.transpose() * weight) * torque;

    YAML::Node params = YAML::LoadFile(
        "/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
    std::vector<double> box_goal = params["box_goal"].as<std::vector<double>>();
    // double cost = 100*(pow((box_goal[0] - pos[pos.size() - 3]), 2) +
    // pow((box_goal[1] - pos[pos.size() - 2]), 2)); double cost =
    // abs(box_goal[0] - pos[pos.size() - 3]) + abs(box_goal[1] - pos[pos.size()
    // - 2]);

    return cost;
  };

  void FillJacobianBlock(std::string var_set, Jacobian &jac) const override {}
};

} // namespace ifopt