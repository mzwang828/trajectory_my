
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