#include <iostream>
#include <fstream>
#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include "ur3_box_fulldof.h"
#include <yaml-cpp/yaml.h>

using namespace ifopt;

int main()
{
  YAML::Node params = YAML::LoadFile("/home/mzwang/catkin_ws/src/trajectory_my/Config/params.yaml");
  int ndof = params["n_dof"].as<int>();         // number of freedom
  int nsteps = params["n_step"].as<int>();      // number of steps or (knot points - 1)
  double tstep = params["t_step"].as<double>(); // length of each step
  int n_exforce = params["n_exforce"].as<int>();
  int n_control = params["n_control"].as<int>();

  IpoptSolver solver;
  solver.SetOption("linear_solver", "mumps");
  solver.SetOption("jacobian_approximation", "exact");
  solver.SetOption("max_cpu_time", 1e6);
  solver.SetOption("max_iter", 30000);

  std::string output_path = "/home/mzwang/catkin_ws/src/trajectory_my/logs/time_and_cost.txt";
  std::ofstream file;
  file.open(output_path, std::ios::app);

  double goal_r = 0.0;
  YAML::Node goals_yaml = YAML::LoadFile("/home/mzwang/catkin_ws/src/trajectory_my/Config/goals.yaml");
  const YAML::Node &goals = goals_yaml["goal"];

  for (YAML::const_iterator it = goals.begin(); it != goals.end(); ++it)
  {
    const YAML::Node &goal = *it;
    goal_r = goal.as<double>();
    Problem nlp;
    nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "position", goal_r));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "velocity"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_control*nsteps, "effort"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_exforce*nsteps, "exforce"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(2*(nsteps-1), "slack"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_exforce*nsteps, "friction"));

    nlp.AddConstraintSet(std::make_shared<ExConstraint>(2*ndof*(nsteps-1)+7*(nsteps-1)));
    nlp.AddCostSet      (std::make_shared<ExCost>());
    std::cout << "Goal: " << goal_r << "\n"; 
    nlp.PrintCurrent();
    solver.Solve(nlp);
    nlp.PrintCurrent();
    
    if (file.is_open()){
      file << solver.GetTotalWallclockTime() << ", " << nlp.GetCosts().GetValues() << "\n";
    }
    else{
      std::cout << " WARNING: Unable to open the trajectory file.\n";
    }
  }
  file.close();

}