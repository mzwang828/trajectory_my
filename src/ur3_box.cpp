#include <iostream>
#include <fstream>
#include <ifopt/problem.h>
#include <ifopt/snopt_solver.h>
#include "ur3_box_problem.h"
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

  Eigen::VectorXd q_init(ndof*nsteps);
  Eigen::VectorXd q_dot_init(ndof*nsteps);
  Eigen::VectorXd control_init(n_control*nsteps);
  Eigen::VectorXd exforce_init(n_exforce*nsteps);
  Eigen::VectorXd slack_init(2*nsteps-2);
  Problem nlp;

  //read in initial values
  std::ifstream init_trajfile;
  std::string trajPathStr = "/home/mzwang/catkin_ws/src/trajectory_my/logs/trajectory.txt";
  const char *trajPath = trajPathStr.c_str();
  init_trajfile.open(trajPath);
  if (!init_trajfile) {
    std::cout << "No trajectory file found, starting from user input guess.";
    nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "position"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "velocity"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_control*nsteps, "effort"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_exforce*nsteps, "exforce"));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(2*nsteps-2, "slack"));
  } else {
    std::cout << "Found trajectory from previous iteration.";
    std::string line;
    double value;
    getline(init_trajfile, line);
    {std::istringstream ss (line);
    for(int i = 0; i < ndof*nsteps; i++){
      ss >> value;
      q_init(i) = value;
    }}
    getline(init_trajfile, line);
    {std::istringstream ss (line);
    for(int i = 0; i < ndof*nsteps; i++){
      ss >> value;
      q_dot_init(i) = value;
    }}
    getline(init_trajfile, line);
    {std::istringstream ss (line);
    for(int i = 0; i < n_control*nsteps; i++){
      ss >> value;
      control_init(i) = value;
    }}
    getline(init_trajfile, line);
    {std::istringstream ss (line);
    for(int i = 0; i < n_exforce*nsteps; i++){
      ss >> value;
      exforce_init(i) = value;
    }}
    getline(init_trajfile, line);
    {std::istringstream ss (line);
    for(int i = 0; i < (2*nsteps-2); i++){
      ss >> value;
      slack_init(i) = value;
    }}
    nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "position", q_init));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "velocity", q_dot_init));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_control*nsteps, "effort", control_init));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(n_exforce*nsteps, "exforce", exforce_init));
    nlp.AddVariableSet  (std::make_shared<ExVariables>(2*nsteps-2, "slack", slack_init));
  }

  nlp.AddConstraintSet(std::make_shared<ExConstraint>(2*ndof*(nsteps-1)+3*(nsteps-1)));
  nlp.AddCostSet      (std::make_shared<ExCost>());
  nlp.PrintCurrent();

  SnoptSolver solver;
  solver.Solve(nlp);

  nlp.PrintCurrent();

  Eigen::VectorXd variables = nlp.GetOptVariables()->GetValues();


  Eigen::Map<Eigen::MatrixXd> Q(variables.segment(0, ndof*nsteps).data(), ndof, nsteps);
  Eigen::Map<Eigen::MatrixXd> Q_dot(variables.segment(ndof*nsteps, ndof*nsteps).data(), ndof, nsteps);
  Eigen::Map<Eigen::MatrixXd> C(variables.segment(2*ndof*nsteps, n_control*nsteps).data(), n_control, nsteps);
  Eigen::Map<Eigen::MatrixXd> F(variables.segment(2*ndof*nsteps+n_control*nsteps, n_exforce*nsteps).data(), n_exforce, nsteps);
  Eigen::Map<Eigen::MatrixXd> d_slack(variables.segment(2*ndof*nsteps+n_control*nsteps + n_exforce*nsteps, nsteps - 1).data(), 1, nsteps - 1);
  Eigen::Map<Eigen::MatrixXd> f_slack(variables.segment(2*ndof*nsteps+n_control*nsteps + n_exforce*nsteps + nsteps - 1, nsteps - 1).data(), 1, nsteps - 1);

  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "Q_dot: \n" << Q_dot << std::endl;
  std::cout << "C: \n" << C << std::endl;
  std::cout << "eF: \n" << F << std::endl;
  std::cout << "distance_slack: \n" << d_slack << std::endl;
  std::cout << "force_slack: \n" << f_slack << std::endl;

  Eigen::VectorXd Q_save = nlp.GetOptVariables()->GetComponent("position")->GetValues();
  Eigen::VectorXd Q_dot_save = nlp.GetOptVariables()->GetComponent("velocity")->GetValues();
  Eigen::VectorXd C_save = nlp.GetOptVariables()->GetComponent("effort")->GetValues();
  Eigen::VectorXd F_save = nlp.GetOptVariables()->GetComponent("exforce")->GetValues();
  Eigen::VectorXd slack_save = nlp.GetOptVariables()->GetComponent("slack")->GetValues();

  // write trajectory to file
  std::ofstream trajFile;
  trajFile.open(trajPath);
  if (trajFile.is_open()){
    trajFile << Q_save.transpose() << "\n";
    trajFile << Q_dot_save.transpose() << "\n";
    trajFile << C_save.transpose() << "\n";
    trajFile << F_save.transpose() << "\n";
    trajFile << slack_save.transpose() << "\n";
  }
  else{
    std::cout << " WARNING: Unable to open the trajectory file.\n";
  }
  trajFile.close();

}