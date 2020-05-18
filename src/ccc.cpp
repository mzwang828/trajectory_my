
  model.frames[frame_index].name = "box_root_joint";
  model.names[joint_index] = "box_root_joint";
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
                            
  // for (Frame& f : model.frames) std::cout << f.name << "\n";
  // Create data required by the algorithms

  // geometory model
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
  PairIndex cp_index = geom_model.findCollisionPair(cp);

  geom_model.addCollisionPair(cp);
  // ground_geom_model.addAllCollisionPairs();
  // pinocchio::appendGeometryModel(geom_model, ground_geom_model);
  
  std::cout << "fused_model: \n" << model << std::endl;
  // std::cout << "frame counts: " << model.nframes << "\n";
  Data data(model);
  GeometryData geom_data(geom_model);

  Eigen::VectorXd q = randomConfiguration(model);

  q << 1.0, 1.5, 0, 1, 0.45;
  forwardKinematics(model,data,q);
  updateFramePlacements(model,data);
  // std::cout << "Joint value: " << q.transpose()  << "\n";

  pinocchio::SE3 joint_frame_placement = data.oMf[model.getFrameId("base_to_pusher")];
  pinocchio::SE3 root_joint_frame_placement = data.oMf[model.getFrameId("box_root_joint")];

  // Eigen::Matrix4f w_T_j;
  // w_T_j.block<3,3>(0,0) = data.oM
  // std::cout << "box frame pose: \n" << data.oMf[model.getFrameId("box")] << "\n";
  // std::cout << "tip frame pose: \n" << data.oMf[model.getFrameId("tip")] << "\n";


  // q << 0, 0, 0, -0.5, -1;
  // forwardKinematics(model,data,q);
  // updateFramePlacements(model,data);
  // std::cout << "Joint value: " << q  << "\n";
  // std::cout << "box frame pose: \n" << data.oMf[model.getFrameId("box")] << "\n";
  geom_data.collisionRequest.enable_contact = true;

  computeCollisions(model,data,geom_model,geom_data,q);
  computeDistances(model, data, geom_model, geom_data, q);
  

  hpp::fcl::CollisionResult cr = geom_data.collisionResults[cp_index];
  hpp::fcl::DistanceResult dr = geom_data.distanceResults[cp_index];

  std::cout << "collision pair: " << cp.first << " , " << cp.second << " - collision: ";
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
  
  
  model.frames[contactId].placement.translation() = joint_frame_placement.inverse().act(dr.nearest_points[0]);
  model.frames[object_contactId].placement.translation() = root_joint_frame_placement.inverse().act(dr.nearest_points[1]);

  forwardKinematics(model, data, q);
  updateFramePlacements(model, data);
  std::cout << "contact frame pose: \n" << data.oMf[model.getFrameId("contactPoint")] << "\n";
  std::cout << "object contact frame pose: \n" << data.oMf[model.getFrameId("object_contactPoint")] << "\n";


  Data::Matrix6x w_J_contact(6,model.nv), w_J_object(6, model.nv);
  w_J_contact.fill(0);
  w_J_object.fill(0);
  computeJointJacobians(model, data, q);
  framesForwardKinematics(model, data, q);
  getFrameJacobian(model, data, contactId, WORLD, w_J_contact);
  getFrameJacobian(model, data, object_contactId, WORLD, w_J_object);

  std::cout << "contact jacobian: " << w_J_contact << "\n";
  std::cout << "object jacobian: " << w_J_object << "\n";

  pinocchio::Data::Matrix6x J_final = -1 * w_J_contact + w_J_object;
  Eigen::Vector3d normal_f(1, 0, 0);
  Eigen::VectorXd J_remapped(model.nv);
  J_remapped = (normal_f.transpose() * J_final.topRows(3)).transpose();

  std::cout << J_remapped << "\n";
  */
  // all collision pairs
  // for(size_t k = 0; k < geom_model.collisionPairs.size(); ++k)
  // {
  //   const CollisionPair & cp = geom_model.collisionPairs[k];
  //   const hpp::fcl::CollisionResult & cr = geom_data.collisionResults[k];
  //   const hpp::fcl::DistanceResult & dr = geom_data.distanceResults[k];
      
  //   std::cout << "collision pair: " << geom_model.geometryObjects[cp.first].name << " , " << geom_model.geometryObjects[cp.second].name << " - collision: ";
  //   std::cout << (cr.isCollision() ? "yes" : "no");
  //   std::cout << " - distance: " << dr.min_distance << std::endl;
  // }

  // check body
  // std::cout << "body number: " << model.nbodies << "\n";
  // std::cout << "box body id: " << model.getBodyId("box") << "\n";
  // std::cout << "box body id: " << model.getBodyId("table") << "\n"; 
  // std::cout << "force size: " << data.of.size() << "\n";
  // std::cout << "force 0: " << data.of[0] << "\n";
  // std::cout << "force 1: " << data.of[1] << "\n";
  // std::cout << "force 2: " << data.of[2] << "\n";
  // std::cout << "lambda_c: " << data.lambda_c << "\n";



  // Sample a random configuration
  // Eigen::VectorXd q = randomConfiguration(model);
  // Eigen::VectorXd v(6);
  // pinocchio::FrameIndex frameID = model.getFrameId("ee_fixed_joint");
  /*
  std::cout << "q: " << q.transpose() << std::endl;
  // Perform the forward kinematics over the kinematic tree
  forwardKinematics(model,data,q);
  // Print out the placement of each joint of the kinematic tree
  for(JointIndex joint_id = 1; joint_id < (JointIndex)model.njoints; ++joint_id)
    std::cout << model.names[joint_id] << "\t\t: "
              << data.oMi[joint_id].translation().transpose()
              << std::endl;
  */

  // Jacobian test
  // for (int i=0; i<model.nframes; i++){
  //   std::cout << model.frames[i].name << "\n";
  // }
  
  // Data::Matrix6x J_RF(6, model.nv);
  // J_RF.setZero()
  // computeJointJacobians(model, data, , WORLD, );

  // kinematics test
  // q << 0, 0, 0, 0, 0, 0;
  // forwardKinematics(model,data,q);
  // pinocchio::updateFramePlacement(model, data, frameID);
  // pinocchio::GeometryData::SE3 ini_pose = data.oMf[frameID];
  // Eigen::Quaterniond ini_quater(ini_pose.rotation());
  // std::cout << "initial pose: \n" << ini_pose << "\n";
  // // std::cout << "initial pose: " << ini_pose.rotation() << "\n";
  // // std::cout << "initial pose: " << ini_pose.rotation().inverse() << "\n";
  // // std::cout << "initial pose: " << ini_pose.translation() << "\n";
  // std::cout << "initial pose quaternion: " << ini_quater.w() << "\n";
  // std::cout << "initial pose quaternion: " << ini_quater.vec().transpose() << "\n";

  // q << 0.5,0,0.5,0,0.5,0;
  // q << 0.49,0.88,-0.92,-1.72,0.03,1.86;
  // forwardKinematics(model,data,q);
  // pinocchio::updateFramePlacement(model, data, frameID);
  // pinocchio::GeometryData::SE3 final_pose = data.oMf[frameID];
  // Eigen::Quaterniond end_quater(final_pose.rotation());
  // // std::cout << "q: " << q.transpose() << std::endl;
  // std::cout << "final pose: \n" << final_pose << "\n";
  // std::cout << "final pose quaternion: " << end_quater.w() << "\n";
  // std::cout << "final pose quaternion: " << end_quater.vec().transpose() << "\n";
  // std::cout << "inverse w: " << end_quater.inverse().w() << "\n";
  // std::cout << "inverse vec: " << end_quater.inverse().vec().transpose() << "\n";
  // std::cout << "quaternion multi w: " << end_quater.dot(end_quater) << "\n";
  // std::cout << "quaternion multi w: " << end_quater.dot(end_quater.inverse()) << "\n";
  // double test_d = quaternion_disp(end_quater, end_quater);
  // std::cout << "check: " << test_d << "\n";
  // Eigen::Quaterniond quater_ap(-end_quater.w(), -end_quater.x(), -end_quater.y(), -end_quater.z());
  // test_d = quaternion_disp(end_quater, quater_ap);
  // std::cout << "check: " << test_d << "\n";

  //derivative test
  // computeAllTerms(model, data, q, v);
  // Eigen::VectorXd tau(6);
  // q << 0.5, 0.2, 0.5, 0.2, 0.2, 0.3;
  // v << 0.1,0.1,0.1,0.1,0.1,0.1;
  // tau << 2,2,2,2,2,2;
  // computeMinverse(model, data, q);
  // Eigen::MatrixXd M = data.M;
  // Eigen::MatrixXd Minv = data.Minv;
  // M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  // Minv.triangularView<Eigen::StrictlyLower>() = Minv.transpose().triangularView<Eigen::StrictlyLower>();
  // std::cout << "check m1: \n" << M << "\n";
  // std::cout << "check minv1 : \n" << Minv << "\n";
  // std::cout << "multiplication: \n " << M * Minv << "\n";
  // computeABADerivatives(model, data, q, v, tau);
  // std::cout << "check if nle: " << data.nle << "\n";
  // std::cout << "check derivative ddq_dq: " << data.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: " << data.ddq_dv << "\n";
  // std::cout << "check m: \n" << data.M << "\n";
  // std::cout << "check derivative minv: \n" << data.Minv << "\n";
  // std::cout << "multiplication: \n " << M * data.Minv << "\n";
  // std::cout << "----------\n";
  // q << 0.5,0,0.5,0,0.5,0;
  // v << 2,2,2,2,2,2;
  // tau << 2,2,2,2,2,2;
  // computeAllTerms(model, data, q, v);
  // computeABADerivatives(model, data, q, v, tau);
  // std::cout << "check if nle: " << data.nle << "\n";
  // std::cout << "check derivative ddq_dq: " << data.ddq_dq << "\n";
  // std::cout << "check derivative ddq_dv: " << data.ddq_dv << "\n";
  // std::cout << "check derivative minv: " << data.Minv << "\n";


  /*
  // dynmaics test
  q << 0, 0, 0, 0, 0, 0;
  v << 0, 0, 0, 0, 0, 0;
  // computeAllTerms(model, data, q, v);
  // time_t tstart, tend; 
  // tstart = time(0);
  crba(model,data,q);
  nonLinearEffects(model,data,q,v);
  // tend = time(0); 
  // std::cout << "It took "<< tend-tstart << std::endl;

  std::cout << data.M << "\n";
  std::cout << data.Minv << "\n";
  std::cout << data.nle << "\n";
  
  // tstart = time(0);
  q = randomConfiguration(model);
  v << 0.3, 0, 0, 0, 0, 0;
  // tstart = time(0);
  computeAllTerms(model, data, q, v);
  // tend = time(0); 
  // std::cout << "It took "<< tend << std::endl;
  std::cout << "------" << "\n";
  std::cout << data.M << "\n";
  std::cout << data.Minv << "\n";
  std::cout << data.nle << "\n";
*/

  // eigen test
  // Eigen::MatrixXd Mat(3,3);
  // Eigen::VectorXd vec(3);
  // Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat(3,4);
  // Mat << 1,0,0,2,1,0,0,0,1;
  // vec << 0,0,0;
  // std::cout << Mat << "\n";
  // SpMat = Mat.sparseView();
  // SpMat.coeffRef(0,0) = 0;
  // SpMat.coeffRef(0,1) = 0;
  // SpMat.coeffRef(0,2) = 0;
  // SpMat.prune(0,0);
  // std::vector<T> triplet;
  // triplet.push_back(T(1,2,3));
  // triplet.push_back(T(1,2,5));
  // triplet.push_back(T(2,2,1));
  // SpMat.setFromTriplets(triplet.begin(), triplet.end());  
  // std::cout << SpMat << "\n";
  // std::cout <<  SpMat.cols() << "\n";



    // std::cout << "------------------------------\n";
  // Model urmodel;
  // pinocchio::urdf::buildModel(box_filename, JointModelFreeFlyer(), urmodel);
  // setRootJointBounds(urmodel, 1, "freeflyer");
  // Eigen::VectorXd urq = randomConfiguration(urmodel);
  // Data urdata(urmodel);
  // Eigen::VectorXd urv(urmodel.nv);
  // //derivative test
  // computeAllTerms(urmodel, urdata, urq, urv);
  // Eigen::VectorXd urtau(urmodel.nv);
  // // urq << 0.5, 0.2, 0.5, 0.2, 0.2, 0.3;
  // urv << 1,1,1,1,1,1;
  // urtau << 2,2,2,2,2,2;
  // std::cout << "check joints: " << urmodel.njoints << "\n";
  // PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) urfext((size_t)urmodel.njoints, pinocchio::Force::Zero());
  // pinocchio::Force::Vector3 t = pinocchio::Force::Vector3::Zero();
  // pinocchio::Force::Vector3 f = pinocchio::Force::Vector3::Zero();
  // f << 10,10,10;
  // t << 10,10,10;
  // urfext[1].linear(f);
  // urfext[1].angular(t);
  // computeMinverse(urmodel, urdata, urq);
  // Eigen::MatrixXd M = urdata.M;
  // Eigen::MatrixXd Minv = urdata.Minv;
  // M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  // Minv.triangularView<Eigen::StrictlyLower>() = Minv.transpose().triangularView<Eigen::StrictlyLower>();
  // std::cout << "check m1: \n" << M << "\n";
  // std::cout << "check minv1 : \n" << Minv << "\n";
  // std::cout << "multiplication: \n " << M * Minv << "\n";
  // computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  // std::cout << "check if nle: " << urdata.nle << "\n";
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";
  // std::cout << "check m: \n" << urdata.M << "\n";
  // std::cout << "check derivative minv: \n" << urdata.Minv << "\n";
  // std::cout << "multiplication: \n " << M * urdata.Minv << "\n";

  // computeABADerivatives(urmodel, urdata, urq, urv, urtau, urfext);
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";


  std::cout << "------------------------------\n";
  Model urmodel;
  pinocchio::urdf::buildModel(ur_filename, urmodel);
  Eigen::VectorXd urq = randomConfiguration(urmodel);
  std::cout << urmodel << "\n";
  Data urdata(urmodel);
  Eigen::VectorXd urv(6);
  //derivative test
  computeAllTerms(urmodel, urdata, urq, urv);
  Eigen::VectorXd urtau(6);
  urq << 0.5, -0.47, -0.32, -1.97, 0.97, -0.47;
  urv << 0.1,0.1,0.1,0.1,0.1,0.1;
  urtau << 0,0,0,0,0,0;
  std::cout << "check joints: " << urmodel.njoints << "\n";
  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) urfext((size_t)urmodel.njoints, pinocchio::Force::Zero());
  pinocchio::Force::Vector3 tz = pinocchio::Force::Vector3::Zero();
  pinocchio::Force::Vector3 ty = pinocchio::Force::Vector3::Zero();
  pinocchio::Force::Vector3 fx = pinocchio::Force::Vector3::Zero();
  ty[1] = 0.6;
  fx[0] = 0.2;
  tz[2] = 0.8;
  std::cout << "check force: \n";
  for (pinocchio::Force f : urfext) std::cout << f << "\n"; 
  computeMinverse(urmodel, urdata, urq);
  Eigen::MatrixXd M = urdata.M;
  Eigen::MatrixXd Minv = urdata.Minv;
  M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  Minv.triangularView<Eigen::StrictlyLower>() = Minv.transpose().triangularView<Eigen::StrictlyLower>();
  std::cout << "check m1: \n" << M << "\n";
  std::cout << "check minv1 : \n" << Minv << "\n";
  std::cout << "multiplication: \n " << M * Minv << "\n";
  aba(urmodel, urdata, urq, urv, urtau);
  std::cout << "check acceleration: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  // std::cout << "check if nle: " << urdata.nle << "\n";
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";
  // std::cout << "check m: \n" << urdata.M << "\n";
  // std::cout << "check derivative minv: \n" << urdata.Minv << "\n";
  // std::cout << "multiplication: \n " << M * urdata.Minv << "\n";

  urtau << 0.8,0.6,0.0,0.0,0.0,0.0;
  std::cout << "tau: " << urtau.transpose() << "\n";
  aba(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "acceleration with tau: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";

  urfext[1].angular(tz);
  urfext[2].linear(fx);
  urfext[2].angular(ty);
  // urfext[6].angular(ty);
  for (pinocchio::Force f : urfext) std::cout << f << "\n"; 
  urtau << 0,0,0,0,0,0;
  aba(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "acceleration with fext: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";


  ///////////////////////URDF-pusher-box////////////////////////////////////////////////////////////////
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

  // q << 0, 1.0, 1.2, cos(0.52), sin(0.52);
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
  // v << 1,1,1,1;
  // tau << 1,1,1,1;
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
  pinocchio::computeABADerivatives(model, data, q, v, tau, fext);
  std::cout << "ddq_dq: \n" << data.ddq_dq << "\n";
  std::cout << "ddq_dv: \n" << data.ddq_dv << "\n";

  std::cout << "joint1 check: \n" << model.joints[1];
  std::cout << "joint2 check: \n" << model.joints[2];
  

  std::cout << "------------------------------\n";
  Model urmodel;
  pinocchio::urdf::buildModel(ur_filename, urmodel);
  Eigen::VectorXd urq = randomConfiguration(urmodel);
  std::cout << urmodel << "\n";
  Data urdata(urmodel);
  Eigen::VectorXd urv(6);
  //derivative test
  computeAllTerms(urmodel, urdata, urq, urv);
  Eigen::VectorXd urtau(6);
  urq << 0.5, -0.47, -0.32, -1.97, 0.97, -0.47;
  urv << 0.1,0.1,0.1,0.1,0.1,0.1;
  urtau << 0,0,0,0,0,0;
  std::cout << "check joints: " << urmodel.njoints << "\n";
  PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) urfext((size_t)urmodel.njoints, pinocchio::Force::Zero());
  pinocchio::Force::Vector3 tz = pinocchio::Force::Vector3::Zero();
  pinocchio::Force::Vector3 ty = pinocchio::Force::Vector3::Zero();
  ty[0] = 0.1;
  tz[2] = 0.8;
  std::cout << "check force: \n";
  for (pinocchio::Force f : urfext) std::cout << f << "\n"; 
  computeMinverse(urmodel, urdata, urq);
  Eigen::MatrixXd M = urdata.M;
  Eigen::MatrixXd Minv = urdata.Minv;
  M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  Minv.triangularView<Eigen::StrictlyLower>() = Minv.transpose().triangularView<Eigen::StrictlyLower>();
  std::cout << "check m1: \n" << M << "\n";
  std::cout << "check minv1 : \n" << Minv << "\n";
  std::cout << "multiplication: \n " << M * Minv << "\n";
  aba(urmodel, urdata, urq, urv, urtau);
  std::cout << "check acceleration: " << urdata.ddq.transpose() << "\n";
  computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  // std::cout << "check if nle: " << urdata.nle << "\n";
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";
  // std::cout << "check m: \n" << urdata.M << "\n";
  // std::cout << "check derivative minv: \n" << urdata.Minv << "\n";
  // std::cout << "multiplication: \n " << M * urdata.Minv << "\n";

  urtau << 0.0,0.0,0.0,0.0,0.0,0.1;
  std::cout << "tau: " << urtau.transpose() << "\n";
  aba(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "acceleration with tau: " << urdata.ddq.transpose() << "\n";
  // computeABADerivatives(urmodel, urdata, urq, urv, urtau);
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";

  // urfext[1].angular(tz);
  // urfext[2].angular(ty);
  urfext[6].angular(ty);
  for (pinocchio::Force f : urfext) std::cout << f << "\n"; 
  urtau << 0,0,0,0,0,0;
  aba(urmodel, urdata, urq, urv, urtau, urfext);
  std::cout << "acceleration with fext: " << urdata.ddq.transpose() << "\n";
  // computeABADerivatives(urmodel, urdata, urq, urv, urtau, urfext);
  // std::cout << "check derivative ddq_dq: \n" << urdata.ddq_dq<< "\n";
  // std::cout << "check derivative ddq_dv: \n" << urdata.ddq_dv << "\n";
  ///////////////////////URDF-pusher-box////////////////////////////////////////////////////////////////




  //////////////////////// URDF - UR3 - BOX///////////////////////////////////////////////
  using namespace pinocchio;
  
  // You should change here to set up your own URDF file or just pass it as an argument of this example.
  const std::string urdf_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/ur3e_robot_abs.urdf");
  const std::string box_filename =
      PINOCCHIO_MODEL_DIR + std::string("/urdf/box.urdf");
  
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
  // for (std::string& n : model.names) std::cout << n << "\n" ;
  // std::cout << "------\n";
  // for (Frame& f : model.frames) std::cout << f.name << "\n";
  pinocchio::Model::Index contactId =  model.addFrame(
                                    pinocchio::Frame("contactPoint", 
                                    model.getJointId("wrist_3_joint"), 
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
  // std::cout << geom_model << "\n";
  // // Add all possible collision pairs and remove the ones collected in the SRDF file
  // geom_model.addAllCollisionPairs();
  // pinocchio::srdf::removeCollisionPairs(model, geom_model, srdf_filename);
  GeomIndex tip_id = geom_model.getGeometryId("ee_link_0");
  GeomIndex box_id = geom_model.getGeometryId("obj_front_0");
  CollisionPair cp = CollisionPair(tip_id, box_id);
  geom_model.addCollisionPair(cp);
  PairIndex cp_index = geom_model.findCollisionPair(cp);
  
  Data data(model);
  GeometryData geom_data(geom_model);
  Eigen::VectorXd q = randomConfiguration(model);

  q << 0,0,0,0,0,0, 1.0, 0, cos(0.52), sin(0.52);
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
  std::cout << "Pose of tip: \n" << geom_data.oMg[geom_model.getGeometryId("ee_link_0")] << "\n";
  std::cout << "Pose of front plane: \n" << geom_data.oMg[geom_model.getGeometryId("obj_front_0")] << "\n";
  std::cout << "Normal from DistanceResult: " << dr.normal.transpose() << "\n";

  std::cout << dr.normal.normalized().transpose() << "\n";

  // Jacobian
  // forwardKinematics(model,data,q);
  // updateFramePlacements(model,data);
  framesForwardKinematics(model, data, q);
  pinocchio::SE3 joint_frame_placement = data.oMf[model.getFrameId("wrist_3_joint")];
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
  // v << 1,1,1,1;
  // tau << 1,1,1,1;
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
  pinocchio::computeABADerivatives(model, data, q, v, tau, fext);
  std::cout << "ddq_dq: \n" << data.ddq_dq << "\n";
  std::cout << "ddq_dv: \n" << data.ddq_dv << "\n";

  std::cout << "joint1 check: \n" << model.joints[1];
  std::cout << "joint2 check: \n" << model.joints[2];
  //////////////////////// URDF - UR3 - BOX///////////////////////////////////////////////
