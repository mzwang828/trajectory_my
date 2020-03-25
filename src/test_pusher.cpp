#include <iostream>
#include <fstream>
#include <ifopt/problem.h>
#include <ifopt/snopt_solver.h>
#include "test_pusher.h"
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

  Problem nlp;

  nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "position"));
  nlp.AddVariableSet  (std::make_shared<ExVariables>(ndof*nsteps, "velocity"));
  nlp.AddVariableSet  (std::make_shared<ExVariables>(n_control*nsteps, "effort"));
  nlp.AddVariableSet  (std::make_shared<ExVariables>(n_exforce*nsteps, "exforce"));

  nlp.AddConstraintSet(std::make_shared<ExConstraint>(2*ndof*(nsteps-1)+3*(nsteps-1)));
  nlp.AddCostSet      (std::make_shared<ExCost>());
  nlp.PrintCurrent();

  SnoptSolver solver;
  solver.Solve(nlp);

  nlp.PrintCurrent();

  Eigen::VectorXd variables = nlp.GetOptVariables()->GetValues();

  Eigen::Map<Eigen::MatrixXd> Q(variables.segment(0, ndof*nsteps).data(), ndof, nsteps);
  Eigen::Map<Eigen::MatrixXd> Q_dot(variables.segment(ndof*nsteps, ndof*nsteps).data(), ndof, nsteps);
  Eigen::Map<Eigen::MatrixXd> T(variables.segment(2*ndof*nsteps, ndof*nsteps).data(), ndof, nsteps);

  // write trajectory to file
  std::ofstream trajFile;
  std::string trajPathStr = "/home/mzwang/catkin_ws/src/trajectory_my/logs/trajectory.txt";
  const char *trajPath = trajPathStr.c_str();
  trajFile.open(trajPath);
  if (trajFile.is_open()){
    trajFile << "time, positions, velocities, controls \n";
    for(int i=0; i<nsteps; i++){
      trajFile << i*tstep << ",";
      for (int j = 0; j < ndof; j++){
        trajFile << Q(j, i) << ",";        
      }
      for (int j = 0; j < ndof; j++){
        trajFile << Q_dot(j, i) << ",";        
      }    
      for (int j = 0; j < ndof; j++){
        trajFile << T(j, i) << ",";        
      }
      trajFile << "\n";
    }
  }
  else{
    std::cout << " WARNING: Unable to open the trajectory file.\n";
  }
  trajFile.close();

  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "Q_dot: \n" << Q_dot << std::endl;
  std::cout << "T: \n" << T << std::endl;
  
}