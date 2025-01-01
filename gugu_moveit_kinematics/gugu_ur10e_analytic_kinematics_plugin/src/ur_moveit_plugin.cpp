/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Sachin Chitta, David Lu!!, Ugo Cupcic */

//#include <moveit/gugu_kdl_kinematics_plugin/kdl_kinematics_plugin.h>
// #include <moveit/gugu_kdl_kinematics_plugin/chainiksolver_vel_mimic_svd.hpp>

#include <tf2_kdl/tf2_kdl.hpp>
#include <tf2/transform_datatypes.h>

#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/kinfam_io.hpp>

#include <gugu_ur10e_analytic_kinematics_plugin/ur_moveit_plugin.h>
#include <gugu_ur10e_analytic_kinematics_plugin/ur_kin_gugu.h>

namespace ur_kinematics
{
static rclcpp::Logger LOGGER = rclcpp::get_logger("gugu_moveit_kdl_kinematics_plugin.kdl_kinematics_plugin");

rclcpp::Clock URKinematicsPlugin::steady_clock_{ RCL_STEADY_TIME };

URKinematicsPlugin::URKinematicsPlugin() : initialized_(false)
{
}

void URKinematicsPlugin::getRandomConfiguration(Eigen::VectorXd& jnt_array) const
{
  state_->setToRandomPositions(joint_model_group_);
  state_->copyJointGroupPositions(joint_model_group_, &jnt_array[0]);
}

void URKinematicsPlugin::getRandomConfiguration(const Eigen::VectorXd& seed_state,
                                                 const std::vector<double>& consistency_limits,
                                                 Eigen::VectorXd& jnt_array) const
{
  joint_model_group_->getVariableRandomPositionsNearBy(state_->getRandomNumberGenerator(), &jnt_array[0],
                                                       &seed_state[0], consistency_limits);
}

bool URKinematicsPlugin::checkConsistency(const Eigen::VectorXd& seed_state,
                                           const std::vector<double>& consistency_limits,
                                           const Eigen::VectorXd& solution) const
{
  for (std::size_t i = 0; i < dimension_; ++i)
    if (fabs(seed_state(i) - solution(i)) > consistency_limits[i])
      return false;
  return true;
}

void URKinematicsPlugin::getJointWeights()
{
  const std::vector<std::string>& active_names = joint_model_group_->getActiveJointModelNames();
  std::vector<std::string> names;
  std::vector<double> weights;
  if (lookupParam(node_, "joint_weights.weights", weights, weights))
  {
    if (!lookupParam(node_, "joint_weights.names", names, names) || (names.size() != weights.size()))
    {
      RCLCPP_ERROR(LOGGER, "Expecting list parameter joint_weights.names of same size as list joint_weights.weights");
      // fall back to default weights
      weights.clear();
    }
  }
  else if (lookupParam(node_, "joint_weights", weights,
                       weights))  // try reading weight lists (for all active joints) directly
  {
    std::size_t num_active = active_names.size();
    if (weights.size() == num_active)
    {
      joint_weights_ = weights;
      return;
    }
    else if (!weights.empty())
    {
      RCLCPP_ERROR(LOGGER, "Expecting parameter joint_weights to list weights for all active joints (%zu) in order",
                   num_active);
      // fall back to default weights
      weights.clear();
    }
  }

  // by default assign weights of 1.0 to all joints
  joint_weights_ = std::vector<double>(active_names.size(), 1.0);
  if (weights.empty())  // indicates default case
    return;

  // modify weights of listed joints
  assert(names.size() == weights.size());
  for (size_t i = 0; i != names.size(); ++i)
  {
    auto it = std::find(active_names.begin(), active_names.end(), names[i]);
    if (it == active_names.cend())
      RCLCPP_WARN(LOGGER, "Joint '%s' is not an active joint in group '%s'", names[i].c_str(),
                  joint_model_group_->getName().c_str());
    else if (weights[i] < 0.0)
      RCLCPP_WARN(LOGGER, "Negative weight %f for joint '%s' will be ignored", weights[i], names[i].c_str());
    else
      joint_weights_[it - active_names.begin()] = weights[i];
  }
  RCLCPP_INFO_STREAM(
      LOGGER, "Joint weights for group '"
                  << getGroupName() << "': \n"
                  << Eigen::Map<const Eigen::VectorXd>(joint_weights_.data(), joint_weights_.size()).transpose());
}

bool URKinematicsPlugin::initialize(const rclcpp::Node::SharedPtr& node, const moveit::core::RobotModel& robot_model,
                                     const std::string& group_name, const std::string& base_frame,
                                     const std::vector<std::string>& tip_frames, double search_discretization)
{
  node_ = node;
  storeValues(robot_model, group_name, base_frame, tip_frames, search_discretization);
  joint_model_group_ = robot_model_->getJointModelGroup(group_name);
  //add start
  //  const robot_model::JointModelGroup* joint_model_group = robot_model_->getJointModelGroup(group_name);
  const moveit::core::JointModelGroup* joint_model_group = robot_model_->getJointModelGroup(group_name);
  //add end
  if (!joint_model_group_)
    return false;

  if (!joint_model_group_->isChain())
  {
    RCLCPP_ERROR(LOGGER, "Group '%s' is not a chain", group_name.c_str());
    return false;
  }
  if (!joint_model_group_->isSingleDOFJoints())
  {
    RCLCPP_ERROR(LOGGER, "Group '%s' includes joints that have more than 1 DOF", group_name.c_str());
    return false;
  }

  KDL::Tree kdl_tree;

  if (!kdl_parser::treeFromUrdfModel(*robot_model.getURDF(), kdl_tree))
  {
    RCLCPP_ERROR(LOGGER, "Could not initialize tree object");
    return false;
  }
  if (!kdl_tree.getChain(base_frame_, getTipFrame(), kdl_chain_))
  {
    RCLCPP_ERROR(LOGGER, "Could not initialize chain object");
    return false;
  }

  dimension_ = joint_model_group_->getActiveJointModels().size() + joint_model_group_->getMimicJointModels().size();
  for (std::size_t i = 0; i < joint_model_group_->getJointModels().size(); ++i)
  {
    if (joint_model_group_->getJointModels()[i]->getType() == moveit::core::JointModel::REVOLUTE ||
        joint_model_group_->getJointModels()[i]->getType() == moveit::core::JointModel::PRISMATIC)
    {
      solver_info_.joint_names.push_back(joint_model_group_->getJointModelNames()[i]);
      const std::vector<moveit_msgs::msg::JointLimits>& jvec =
          joint_model_group_->getJointModels()[i]->getVariableBoundsMsg();
      solver_info_.limits.insert(solver_info_.limits.end(), jvec.begin(), jvec.end());
    }
  }

  //add start
  for (std::size_t i=0; i < joint_model_group->getJointModels().size(); ++i)
  {
    if(joint_model_group->getJointModels()[i]->getType() == moveit::core::JointModel::REVOLUTE || 
       joint_model_group->getJointModels()[i]->getType() == moveit::core::JointModel::PRISMATIC)
    {
      ik_chain_info_.joint_names.push_back(joint_model_group->getJointModelNames()[i]);
      const std::vector<moveit_msgs::msg::JointLimits> &jvec = joint_model_group->getJointModels()[i]->getVariableBoundsMsg();
      ik_chain_info_.limits.insert(ik_chain_info_.limits.end(), jvec.begin(), jvec.end());
    }
  }
  

  fk_chain_info_.joint_names = ik_chain_info_.joint_names;
  fk_chain_info_.limits = ik_chain_info_.limits;
  //add end

  if (!joint_model_group_->hasLinkModel(getTipFrame()))
  {
    RCLCPP_ERROR(LOGGER, "Could not find tip name in joint group '%s'", group_name.c_str());
    return false;
  }
  solver_info_.link_names.push_back(getTipFrame());

  // joint_min_.resize(solver_info_.limits.size());
  // joint_max_.resize(solver_info_.limits.size());

  // for (unsigned int i = 0; i < solver_info_.limits.size(); ++i)
  // {
  //   joint_min_(i) = solver_info_.limits[i].min_position;
  //   joint_max_(i) = solver_info_.limits[i].max_position;
  // }

  //add start
  ik_chain_info_.link_names.push_back(getTipFrame());
  fk_chain_info_.link_names = joint_model_group->getLinkModelNames();

  joint_min_.resize(ik_chain_info_.limits.size());
  joint_max_.resize(ik_chain_info_.limits.size());

  for(unsigned int i=0; i < ik_chain_info_.limits.size(); i++)
  {
    joint_min_(i) = ik_chain_info_.limits[i].min_position;
    joint_max_(i) = ik_chain_info_.limits[i].max_position;
  }
  //add end

  // Get Solver Parameters
  lookupParam(node_, "max_solver_iterations", max_solver_iterations_, 500);
  lookupParam(node_, "epsilon", epsilon_, 1e-5);
  lookupParam(node_, "orientation_vs_position", orientation_vs_position_weight_, 1.0);

  bool position_ik;
  lookupParam(node_, "position_only_ik", position_ik, false);
  if (position_ik)  // position_only_ik overrules orientation_vs_position
    orientation_vs_position_weight_ = 0.0;
  if (orientation_vs_position_weight_ == 0.0)
    RCLCPP_INFO(LOGGER, "Using position only ik");

  getJointWeights();

  // Check for mimic joints
  unsigned int joint_counter = 0;
  for (std::size_t i = 0; i < kdl_chain_.getNrOfSegments(); ++i)
  {
    const moveit::core::JointModel* jm = robot_model_->getJointModel(kdl_chain_.segments[i].getJoint().getName());

    // first check whether it belongs to the set of active joints in the group
    if (jm->getMimic() == nullptr && jm->getVariableCount() > 0)
    {
      JointMimic mimic_joint;
      mimic_joint.reset(joint_counter);
      mimic_joint.joint_name = kdl_chain_.segments[i].getJoint().getName();
      mimic_joint.active = true;
      mimic_joints_.push_back(mimic_joint);
      ++joint_counter;
      continue;
    }
    if (joint_model_group_->hasJointModel(jm->getName()))
    {
      if (jm->getMimic() && joint_model_group_->hasJointModel(jm->getMimic()->getName()))
      {
        JointMimic mimic_joint;
        mimic_joint.joint_name = kdl_chain_.segments[i].getJoint().getName();
        mimic_joint.offset = jm->getMimicOffset();
        mimic_joint.multiplier = jm->getMimicFactor();
        mimic_joints_.push_back(mimic_joint);
        continue;
      }
    }
  }
  for (JointMimic& mimic_joint : mimic_joints_)
  {
    if (!mimic_joint.active)
    {
      const moveit::core::JointModel* joint_model =
          joint_model_group_->getJointModel(mimic_joint.joint_name)->getMimic();
      for (JointMimic& mimic_joint_recal : mimic_joints_)
      {
        if (mimic_joint_recal.joint_name == joint_model->getName())
        {
          mimic_joint.map_index = mimic_joint_recal.map_index;
        }
      }
    }
  }

  // Setup the joint state groups that we need
  state_ = std::make_shared<moveit::core::RobotState>(robot_model_);

  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_chain_);

  //add start
  // Store things for when the set of redundant joints may change
  // position_ik_ = position_ik;
  // joint_model_group_ = joint_model_group;
  // max_solver_iterations_ = max_solver_iterations;
  // epsilon_ = epsilon;

  lookupParam(node_, "arm_prefix", arm_prefix_, std::string(""));
  // lookupParam("arm_prefix", arm_prefix_, std::string(""));

  ur_joint_names_.push_back(arm_prefix_ + "shoulder_pan_joint");
  ur_joint_names_.push_back(arm_prefix_ + "shoulder_lift_joint");
  ur_joint_names_.push_back(arm_prefix_ + "elbow_joint");
  ur_joint_names_.push_back(arm_prefix_ + "wrist_1_joint");
  ur_joint_names_.push_back(arm_prefix_ + "wrist_2_joint");
  ur_joint_names_.push_back(arm_prefix_ + "wrist_3_joint");

  // ur_link_names_.push_back(arm_prefix_ + "base_link");       // 0
  ur_link_names_.push_back(arm_prefix_ + "base_link_inertia");    // 1
  ur_link_names_.push_back(arm_prefix_ + "shoulder_link");   // 2
  ur_link_names_.push_back(arm_prefix_ + "upper_arm_link");  // 3
  ur_link_names_.push_back(arm_prefix_ + "forearm_link");    // 4
  ur_link_names_.push_back(arm_prefix_ + "wrist_1_link");    // 5
  ur_link_names_.push_back(arm_prefix_ + "wrist_2_link");    // 6
  ur_link_names_.push_back(arm_prefix_ + "wrist_3_link");    // 7
  // ur_link_names_.push_back(arm_prefix_ + "ee_link");         // 8

  ur_joint_inds_start_ = getJointIndex(ur_joint_names_[0]);

  // check to make sure the serial chain is properly defined in the model
  int cur_ur_joint_ind, last_ur_joint_ind = ur_joint_inds_start_;
  for(int i=1; i<6; i++) {
    cur_ur_joint_ind = getJointIndex(ur_joint_names_[i]);
    if(cur_ur_joint_ind < 0) {
      RCLCPP_ERROR(LOGGER,
        "Kin chain provided in model doesn't contain standard UR joint '%s'.",
        ur_joint_names_[i].c_str());
      return false;
    }
    if(cur_ur_joint_ind != last_ur_joint_ind + 1) {
      RCLCPP_ERROR(LOGGER,
        "Kin chain provided in model doesn't have proper serial joint order: '%s'.",
        ur_joint_names_[i].c_str());
      return false;
    }
    last_ur_joint_ind = cur_ur_joint_ind;
  }
  // if successful, the kinematic chain includes a serial chain of the UR joints

  kdl_tree.getChain(getBaseFrame(), ur_link_names_.front(), kdl_base_chain_);
  kdl_tree.getChain(ur_link_names_.back(), getTipFrame(), kdl_tip_chain_);

  RCLCPP_INFO_STREAM(LOGGER, "kdl_base_chain_:  "
                            << "NrOfSegments:  "<< kdl_base_chain_.getNrOfSegments() << " "
                            << "NrOfJoints:  "<<  kdl_base_chain_.getNrOfJoints() << " ");
                            // << kdl_base_chain_.segments );

  RCLCPP_INFO_STREAM(LOGGER, "kdl_tip_chain_:  "
                            << "NrOfSegments:  "<< kdl_tip_chain_.getNrOfSegments() << " "
                            << "NrOfJoints:  "<<  kdl_tip_chain_.getNrOfJoints() << " ");
                            // << kdl_tip_chain_.segments );

  // weights for redundant solution selection
  ik_weights_.resize(6);

  // if(!lookupParam("ik_weights", ik_weights_, ik_weights_)) {
  if(!lookupParam(node_, "ik_weights", ik_weights_, ik_weights_)) {
    ik_weights_[0] = 1.0;
    ik_weights_[1] = 1.0;
    ik_weights_[2] = 1.0;
    ik_weights_[3] = 1.0;
    ik_weights_[4] = 1.0;
    ik_weights_[5] = 1.0;
  }
  //add end
    
  initialized_ = true;
  // RCLCPP_DEBUG(LOGGER, "KDL solver initialized");
  RCLCPP_INFO(LOGGER, "gugu_KDL solver initialized");
  return true;
}

bool URKinematicsPlugin::timedOut(const rclcpp::Time& start_time, double duration) const
{
  return ((steady_clock_.now() - start_time).seconds() >= duration);
}

bool URKinematicsPlugin::getPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                        const std::vector<double>& ik_seed_state, std::vector<double>& solution,
                                        moveit_msgs::msg::MoveItErrorCodes& error_code,
                                        const kinematics::KinematicsQueryOptions& options) const
{
  std::vector<double> consistency_limits;

  // limit search to a single attempt by setting a timeout of zero
  return searchPositionIK(ik_pose, ik_seed_state, 0.0, consistency_limits, solution, IKCallbackFn(), error_code,
                          options);
}

bool URKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state, double timeout,
                                           std::vector<double>& solution,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
  std::vector<double> consistency_limits;

  return searchPositionIK(ik_pose, ik_seed_state, timeout, consistency_limits, solution, IKCallbackFn(), error_code,
                          options);
}

bool URKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state, double timeout,
                                           const std::vector<double>& consistency_limits, std::vector<double>& solution,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
  return searchPositionIK(ik_pose, ik_seed_state, timeout, consistency_limits, solution, IKCallbackFn(), error_code,
                          options);
}

bool URKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state, double timeout,
                                           std::vector<double>& solution, const IKCallbackFn& solution_callback,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
  std::vector<double> consistency_limits;
  return searchPositionIK(ik_pose, ik_seed_state, timeout, consistency_limits, solution, solution_callback, error_code,
                          options);
}

typedef std::pair<int, double> idx_double;
bool comparator(const idx_double& l, const idx_double& r)
{ return l.second < r.second; }

bool URKinematicsPlugin::searchPositionIK(const geometry_msgs::msg::Pose& ik_pose,
                                           const std::vector<double>& ik_seed_state, double timeout,
                                           const std::vector<double>& consistency_limits, std::vector<double>& solution,
                                           const IKCallbackFn& solution_callback,
                                           moveit_msgs::msg::MoveItErrorCodes& error_code,
                                           const kinematics::KinematicsQueryOptions& options) const
{
  const rclcpp::Time start_time = steady_clock_.now();
  if (!initialized_)
  {
    RCLCPP_ERROR(LOGGER, "kinematics solver not initialized");
    error_code.val = error_code.NO_IK_SOLUTION;
    return false;
  }

  if (ik_seed_state.size() != dimension_)
  {
    RCLCPP_ERROR(LOGGER, "Seed state must have size %d instead of size %zu\n", dimension_, ik_seed_state.size());
    error_code.val = error_code.NO_IK_SOLUTION;
    return false;
  }

  // Resize consistency limits to remove mimic joints
  std::vector<double> consistency_limits_mimic;
  if (!consistency_limits.empty())
  {
    if (consistency_limits.size() != dimension_)
    {
      RCLCPP_ERROR(LOGGER, "Consistency limits must be empty or have size %d instead of size %zu\n", dimension_,
                   consistency_limits.size());
      error_code.val = error_code.NO_IK_SOLUTION;
      return false;
    }

    for (std::size_t i = 0; i < dimension_; ++i)
    {
      if (mimic_joints_[i].active)
        consistency_limits_mimic.push_back(consistency_limits[i]);
    }
  }
  // Eigen::Matrix<double, 6, 1> cartesian_weights;
  // cartesian_weights.topRows<3>().setConstant(1.0);
  // cartesian_weights.bottomRows<3>().setConstant(orientation_vs_position_weight_);

  KDL::JntArray jnt_seed_state(dimension_);
  // KDL::JntArray jnt_pos_in(dimension_);
  // KDL::JntArray jnt_pos_out(dimension_);
  jnt_seed_state.data = Eigen::Map<const Eigen::VectorXd>(ik_seed_state.data(), ik_seed_state.size());
  // jnt_pos_in = jnt_seed_state;

  // KDL::ChainIkSolverVelMimicSVD ik_solver_vel(kdl_chain_, mimic_joints_, orientation_vs_position_weight_ == 0.0);
  solution.resize(dimension_);

  KDL::ChainFkSolverPos_recursive fk_solver_base(kdl_base_chain_);
  KDL::ChainFkSolverPos_recursive fk_solver_tip(kdl_tip_chain_);

  KDL::JntArray jnt_pos_test(jnt_seed_state);
  KDL::JntArray jnt_pos_base(ur_joint_inds_start_);
  KDL::JntArray jnt_pos_tip(dimension_ - 6 - ur_joint_inds_start_);
  KDL::Frame pose_base, pose_tip;

  KDL::Frame kdl_ik_pose;
  KDL::Frame kdl_ik_pose_ur_chain;
  double homo_ik_pose[4][4];
  double q_ik_sols[8][6]; // maximum of 8 IK solutions
  uint16_t num_sols;

  // RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: Position request pose is "
  //                                 << ik_pose.position.x << " " << ik_pose.position.y << " " << ik_pose.position.z << " "
  //                                 << ik_pose.orientation.x << " " << ik_pose.orientation.y << " "
  //                                 << ik_pose.orientation.z << " " << ik_pose.orientation.w);



  unsigned int attempt = 0;
  do
  {
    ++attempt;
    // if (attempt > 1)  // randomly re-seed after first attempt
    // {
    //   if (!consistency_limits_mimic.empty())
    //     getRandomConfiguration(jnt_seed_state.data, consistency_limits_mimic, jnt_pos_in.data);
    //   else
    //     getRandomConfiguration(jnt_pos_in.data);
    //   RCLCPP_DEBUG_STREAM(LOGGER, "New random configuration (" << attempt << "): " << jnt_pos_in);
    // }

    // int ik_valid =
    //     CartToJnt(ik_solver_vel, jnt_pos_in, pose_desired, jnt_pos_out, max_solver_iterations_,
    //               Eigen::Map<const Eigen::VectorXd>(joint_weights_.data(), joint_weights_.size()), cartesian_weights);
    // if (ik_valid == 0 || options.return_approximate_solution)  // found acceptable solution
    // {
    //   if (!consistency_limits_mimic.empty() &&
    //       !checkConsistency(jnt_seed_state.data, consistency_limits_mimic, jnt_pos_out.data))
    //     continue;

    //   Eigen::Map<Eigen::VectorXd>(solution.data(), solution.size()) = jnt_pos_out.data;
    //   if (solution_callback)
    //   {
    //     solution_callback(ik_pose, solution, error_code);
    //     if (error_code.val != error_code.SUCCESS)
    //       continue;
    //   }

    //   // solution passed consistency check and solution callback
    //   error_code.val = error_code.SUCCESS;
    //   // RCLCPP_DEBUG_STREAM(LOGGER, "Solved after " << (steady_clock_.now() - start_time).seconds() << " < " << timeout
    //   //                                             << "s and " << attempt << " attempts");
    //   RCLCPP_INFO_STREAM(LOGGER, "Solved after " << (steady_clock_.now() - start_time).seconds() << " < " << timeout
    //                                               << "s and " << attempt << " attempts");
      
    //   return true;
    // }

    /////////////////////////////////////////////////////////////////////////////
    // find transformation from robot base to UR base and UR tip to robot tip
    for(uint32_t i=0; i<jnt_pos_base.rows(); i++)
      jnt_pos_base(i) = jnt_pos_test(i);
    for(uint32_t i=0; i<jnt_pos_tip.rows(); i++)
      jnt_pos_tip(i) = jnt_pos_test(i + ur_joint_inds_start_ + 6);
    for(uint32_t i=0; i<jnt_seed_state.rows(); i++)
      solution[i] = jnt_pos_test(i);

    if(fk_solver_base.JntToCart(jnt_pos_base, pose_base) < 0) {
      RCLCPP_ERROR(LOGGER, "Could not compute FK for base chain");
      return false;
    }

    if(fk_solver_tip.JntToCart(jnt_pos_tip, pose_tip) < 0) {
      RCLCPP_ERROR(LOGGER, "Could not compute FK for tip chain");
      return false;
    }
    /////////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////
    // Convert into query for analytic solver
    tf2::fromMsg(ik_pose, kdl_ik_pose);
    // tf::poseMsgToKDL(ik_pose, kdl_ik_pose);

    //test start
    // double origin_ik_pose[4][4];
    // kdl_ik_pose.Make4x4((double*) origin_ik_pose);
    // RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: ur10e origin Position request pose is "
    //                   << origin_ik_pose[0][0] << " " << origin_ik_pose[0][1]  << " " << origin_ik_pose[0][2]  << " " << origin_ik_pose[0][3]  << " "
    //                   << origin_ik_pose[1][0] << " " << origin_ik_pose[1][1]  << " " << origin_ik_pose[1][2]  << " " << origin_ik_pose[1][3]  << " "
    //                   << origin_ik_pose[2][0] << " " << origin_ik_pose[2][1]  << " " << origin_ik_pose[2][2]  << " " << origin_ik_pose[2][3]  << " "
    //                   << origin_ik_pose[3][0] << " " << origin_ik_pose[3][1]  << " " << origin_ik_pose[3][2]  << " " << origin_ik_pose[3][3]  << " "
    //                   );

    // KDL::Rotation rotation_pose_base(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0);
    // KDL::Vector vector_pose_base(0.0,0.0,0.0);
    // KDL::Frame pose_base(rotation_pose_base, vector_pose_base);

    // double pose_base_matrix[4][4];
    // pose_base.Inverse().Make4x4((double*) pose_base_matrix);
    // RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: ur10e pose_base is "
    //                   << pose_base_matrix[0][0] << " " << pose_base_matrix[0][1]  << " " << pose_base_matrix[0][2]  << " " << pose_base_matrix[0][3]  << " "
    //                   << pose_base_matrix[1][0] << " " << pose_base_matrix[1][1]  << " " << pose_base_matrix[1][2]  << " " << pose_base_matrix[1][3]  << " "
    //                   << pose_base_matrix[2][0] << " " << pose_base_matrix[2][1]  << " " << pose_base_matrix[2][2]  << " " << pose_base_matrix[2][3]  << " "
    //                   << pose_base_matrix[3][0] << " " << pose_base_matrix[3][1]  << " " << pose_base_matrix[3][2]  << " " << pose_base_matrix[3][3]  << " "
    //                   );


    // KDL::Rotation rotation_pose_tip(0.0,-1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0);
    // KDL::Vector vector_pose_tip(0.0,0.0,0.05);
    // KDL::Frame pose_tip(rotation_pose_tip, vector_pose_tip);
    
    // double pose_tip_matrix[4][4];
    // pose_tip.Inverse().Make4x4((double*) pose_tip_matrix);
    // RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: ur10e pose_tip is "
    //                   << pose_tip_matrix[0][0] << " " << pose_tip_matrix[0][1]  << " " << pose_tip_matrix[0][2]  << " " << pose_tip_matrix[0][3]  << " "
    //                   << pose_tip_matrix[1][0] << " " << pose_tip_matrix[1][1]  << " " << pose_tip_matrix[1][2]  << " " << pose_tip_matrix[1][3]  << " "
    //                   << pose_tip_matrix[2][0] << " " << pose_tip_matrix[2][1]  << " " << pose_tip_matrix[2][2]  << " " << pose_tip_matrix[2][3]  << " "
    //                   << pose_tip_matrix[3][0] << " " << pose_tip_matrix[3][1]  << " " << pose_tip_matrix[3][2]  << " " << pose_tip_matrix[3][3]  << " "
    //                   );
    //test end

    kdl_ik_pose_ur_chain = pose_base.Inverse() * kdl_ik_pose * pose_tip.Inverse();

    kdl_ik_pose_ur_chain.Make4x4((double*) homo_ik_pose);
    
    // RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: ur10e Position request pose is "
    //                   << homo_ik_pose[0][0] << " " << homo_ik_pose[0][1]  << " " << homo_ik_pose[0][2]  << " " << homo_ik_pose[0][3]  << " "
    //                   << homo_ik_pose[1][0] << " " << homo_ik_pose[1][1]  << " " << homo_ik_pose[1][2]  << " " << homo_ik_pose[1][3]  << " "
    //                   << homo_ik_pose[2][0] << " " << homo_ik_pose[2][1]  << " " << homo_ik_pose[2][2]  << " " << homo_ik_pose[2][3]  << " "
    //                   << homo_ik_pose[3][0] << " " << homo_ik_pose[3][1]  << " " << homo_ik_pose[3][2]  << " " << homo_ik_pose[3][3]  << " "
    //                   );

#if KDL_OLD_BUG_FIX
    // in older versions of KDL, setting this flag might be necessary
    for(int i=0; i<3; i++) homo_ik_pose[i][3] *= 1000; // strange KDL fix
#endif
    /////////////////////////////////////////////////////////////////////////////

    // Do the analytic IK
    num_sols = inverse_gugu((double*) homo_ik_pose, (double*) q_ik_sols,
                           jnt_pos_test(ur_joint_inds_start_+5));


    // uint16_t num_valid_sols;
    std::vector< std::vector<double> > q_ik_valid_sols;
    for(uint16_t i=0; i<num_sols; i++)
    {
      bool valid = true;
      std::vector< double > valid_solution;
      valid_solution.assign(6,0.0);

      for(uint16_t j=0; j<6; j++)
      {
        if((q_ik_sols[i][j] <= ik_chain_info_.limits[j].max_position) && (q_ik_sols[i][j] >= ik_chain_info_.limits[j].min_position))
        {
          valid_solution[j] = q_ik_sols[i][j];
          valid = true;
          continue;
        }
        else if ((q_ik_sols[i][j] > ik_chain_info_.limits[j].max_position) && (q_ik_sols[i][j]-2*M_PI > ik_chain_info_.limits[j].min_position))
        {
          valid_solution[j] = q_ik_sols[i][j]-2*M_PI;
          valid = true;
          continue;
        }
        else if ((q_ik_sols[i][j] < ik_chain_info_.limits[j].min_position) && (q_ik_sols[i][j]+2*M_PI < ik_chain_info_.limits[j].max_position))
        {
          valid_solution[j] = q_ik_sols[i][j]+2*M_PI;
          valid = true;
          continue;
        }
        else
        {
          valid = false;
          break;
        }
      }

      if(valid)
      {
        q_ik_valid_sols.push_back(valid_solution);
      }
    }


    // use weighted absolute deviations to determine the solution closest the seed state
    std::vector<idx_double> weighted_diffs;
    for(uint16_t i=0; i<q_ik_valid_sols.size(); i++) {
      double cur_weighted_diff = 0;
      for(uint16_t j=0; j<6; j++) {
        // solution violates the consistency_limits, throw it out
        double abs_diff = std::fabs(ik_seed_state[ur_joint_inds_start_+j] - q_ik_valid_sols[i][j]);
        if(!consistency_limits.empty() && abs_diff > consistency_limits[ur_joint_inds_start_+j]) {
          cur_weighted_diff = std::numeric_limits<double>::infinity();
          break;
        }
        cur_weighted_diff += ik_weights_[j] * abs_diff;
      }
      weighted_diffs.push_back(idx_double(i, cur_weighted_diff));
    }

    std::sort(weighted_diffs.begin(), weighted_diffs.end(), comparator);

#if 0
    printf("start\n");
    printf("                     q %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f\n", ik_seed_state[1], ik_seed_state[2], ik_seed_state[3], ik_seed_state[4], ik_seed_state[5], ik_seed_state[6]);
    for(uint16_t i=0; i<weighted_diffs.size(); i++) {
      int cur_idx = weighted_diffs[i].first;
      printf("diff %f, i %d, q %1.2f, %1.2f, %1.2f, %1.2f, %1.2f, %1.2f\n", weighted_diffs[i].second, cur_idx, q_ik_valid_sols[cur_idx][0], q_ik_valid_sols[cur_idx][1], q_ik_valid_sols[cur_idx][2], q_ik_valid_sols[cur_idx][3], q_ik_valid_sols[cur_idx][4], q_ik_valid_sols[cur_idx][5]);
    }
    printf("end\n");
#endif

    for(uint16_t i=0; i<weighted_diffs.size(); i++) {
      if(weighted_diffs[i].second == std::numeric_limits<double>::infinity()) {
        // rest are infinity, no more feasible solutions
        break;
      }

      // copy the best solution to the output
      int cur_idx = weighted_diffs[i].first;
      solution = q_ik_valid_sols[cur_idx];

      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution is: "
                      << solution[0] << " " << solution[1] << " " 
                      << solution[2] << " " << solution[3] << " " 
                      << solution[4] << " " << solution[5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-1 is: "
                      << q_ik_sols[0][0] << " " << q_ik_sols[0][1] << " " 
                      << q_ik_sols[0][2] << " " << q_ik_sols[0][3] << " " 
                      << q_ik_sols[0][4] << " " << q_ik_sols[0][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-2 is: "
                      << q_ik_sols[1][0] << " " << q_ik_sols[1][1] << " " 
                      << q_ik_sols[1][2] << " " << q_ik_sols[1][3] << " " 
                      << q_ik_sols[1][4] << " " << q_ik_sols[1][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-3 is: "
                      << q_ik_sols[2][0] << " " << q_ik_sols[2][1] << " " 
                      << q_ik_sols[2][2] << " " << q_ik_sols[2][3] << " " 
                      << q_ik_sols[2][4] << " " << q_ik_sols[2][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-4 is: "
                      << q_ik_sols[3][0] << " " << q_ik_sols[3][1] << " " 
                      << q_ik_sols[3][2] << " " << q_ik_sols[3][3] << " " 
                      << q_ik_sols[3][4] << " " << q_ik_sols[3][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-5 is: "
                      << q_ik_sols[4][0] << " " << q_ik_sols[4][1] << " " 
                      << q_ik_sols[4][2] << " " << q_ik_sols[4][3] << " " 
                      << q_ik_sols[4][4] << " " << q_ik_sols[4][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-6 is: "
                      << q_ik_sols[5][0] << " " << q_ik_sols[5][1] << " " 
                      << q_ik_sols[5][2] << " " << q_ik_sols[5][3] << " " 
                      << q_ik_sols[5][4] << " " << q_ik_sols[5][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-7 is: "
                      << q_ik_sols[6][0] << " " << q_ik_sols[6][1] << " " 
                      << q_ik_sols[6][2] << " " << q_ik_sols[6][3] << " " 
                      << q_ik_sols[6][4] << " " << q_ik_sols[6][5]);
      RCLCPP_INFO_STREAM(LOGGER, "searchPositionIK: solution-8 is: "
                      << q_ik_sols[7][0] << " " << q_ik_sols[7][1] << " " 
                      << q_ik_sols[7][2] << " " << q_ik_sols[7][3] << " " 
                      << q_ik_sols[7][4] << " " << q_ik_sols[7][5]);


      // see if this solution passes the callback function test
      // if(!solution_callback.empty())
      if(solution_callback)
        solution_callback(ik_pose, solution, error_code);
      else
        error_code.val = error_code.SUCCESS;

      if(error_code.val == error_code.SUCCESS) {
#if 0
        std::vector<std::string> fk_link_names;
        fk_link_names.push_back(ur_link_names_.back());
        std::vector<geometry_msgs::Pose> fk_poses;
        getPositionFK(fk_link_names, solution, fk_poses);
        KDL::Frame kdl_fk_pose;
        tf::poseMsgToKDL(fk_poses[0], kdl_fk_pose);
        printf("FK(solution) - pose \n");
        for(int i=0; i<4; i++) {
          for(int j=0; j<4; j++)
            printf("%1.6f ", kdl_fk_pose(i, j)-kdl_ik_pose(i, j));
          printf("\n");
        }
#endif
        return true;
      }
    }
    // none of the solutions were both consistent and passed the solution callback

    // if(options.lock_redundant_joints) {
    //   ROS_DEBUG_NAMED("kdl","Will not pertubate redundant joints to find solution");
    //   break;
    // }

    // if(dimension_ == 6) {
    //   ROS_DEBUG_NAMED("kdl","No other joints to pertubate, cannot find solution");
    //   break;
    // }

    // randomly pertubate other joints and try again
    // if(!consistency_limits.empty()) {
    //   getRandomConfiguration(jnt_seed_state, consistency_limits, jnt_pos_test, false);
    // } else {
    //   getRandomConfiguration(jnt_pos_test, false);
    // }
    if (!consistency_limits_mimic.empty())
      getRandomConfiguration(jnt_seed_state.data, consistency_limits_mimic, jnt_pos_test.data);
    else
      getRandomConfiguration(jnt_pos_test.data);
    RCLCPP_DEBUG_STREAM(LOGGER, "New random configuration (" << attempt << "): " << jnt_pos_test);

  } while (!timedOut(start_time, timeout));

  // RCLCPP_DEBUG_STREAM(LOGGER, "IK timed out after " << (steady_clock_.now() - start_time).seconds() << " > " << timeout
  //                                                   << "s and " << attempt << " attempts");
  RCLCPP_INFO_STREAM(LOGGER, "IK timed out after " << (steady_clock_.now() - start_time).seconds() << " > " << timeout
                                                    << "s and " << attempt << " attempts");
  error_code.val = error_code.TIMED_OUT;
  return false;
}

// NOLINTNEXTLINE(readability-identifier-naming)
// int URKinematicsPlugin::CartToJnt(KDL::ChainIkSolverVelMimicSVD& ik_solver, const KDL::JntArray& q_init,
//                                    const KDL::Frame& p_in, KDL::JntArray& q_out, const unsigned int max_iter,
//                                    const Eigen::VectorXd& joint_weights, const Twist& cartesian_weights) const
// {
//   double last_delta_twist_norm = DBL_MAX;
//   double step_size = 1.0;
//   KDL::Frame f;
//   KDL::Twist delta_twist;
//   KDL::JntArray delta_q(q_out.rows()), q_backup(q_out.rows());
//   Eigen::ArrayXd extra_joint_weights(joint_weights.rows());
//   extra_joint_weights.setOnes();

//   q_out = q_init;
//   RCLCPP_DEBUG_STREAM(LOGGER, "Input: " << q_init);

//   unsigned int i;
//   bool success = false;
//   for (i = 0; i < max_iter; ++i)
//   {
//     fk_solver_->JntToCart(q_out, f);
//     delta_twist = diff(f, p_in);
//     RCLCPP_DEBUG_STREAM(LOGGER, "[" << std::setw(3) << i << "] delta_twist: " << delta_twist);

//     // check norms of position and orientation errors
//     const double position_error = delta_twist.vel.Norm();
//     const double orientation_error = ik_solver.isPositionOnly() ? 0 : delta_twist.rot.Norm();
//     const double delta_twist_norm = std::max(position_error, orientation_error);
//     if (delta_twist_norm <= epsilon_)
//     {
//       success = true;
//       break;
//     }

//     if (delta_twist_norm >= last_delta_twist_norm)
//     {
//       // if the error increased, we are close to a singularity -> reduce step size
//       double old_step_size = step_size;
//       step_size *= std::min(0.2, last_delta_twist_norm / delta_twist_norm);  // reduce scale;
//       KDL::Multiply(delta_q, step_size / old_step_size, delta_q);
//       RCLCPP_DEBUG(LOGGER, "      error increased: %f -> %f, scale: %f", last_delta_twist_norm, delta_twist_norm,
//                    step_size);
//       q_out = q_backup;  // restore previous unclipped joint values
//     }
//     else
//     {
//       q_backup = q_out;  // remember joint values of last successful step
//       step_size = 1.0;   // reset step size
//       last_delta_twist_norm = delta_twist_norm;

//       ik_solver.CartToJnt(q_out, delta_twist, delta_q, extra_joint_weights * joint_weights.array(), cartesian_weights);
//     }

//     clipToJointLimits(q_out, delta_q, extra_joint_weights);

//     const double delta_q_norm = delta_q.data.lpNorm<1>();
//     RCLCPP_DEBUG(LOGGER, "[%3d] pos err: %f  rot err: %f  delta_q: %f", i, position_error, orientation_error,
//                  delta_q_norm);
//     if (delta_q_norm < epsilon_)  // stuck in singularity
//     {
//       if (step_size < epsilon_)  // cannot reach target
//         break;
//       // wiggle joints
//       last_delta_twist_norm = DBL_MAX;
//       delta_q.data.setRandom();
//       delta_q.data *= std::min(0.1, delta_twist_norm);
//       clipToJointLimits(q_out, delta_q, extra_joint_weights);
//       extra_joint_weights.setOnes();
//     }

//     KDL::Add(q_out, delta_q, q_out);

//     RCLCPP_DEBUG_STREAM(LOGGER, "      delta_q: " << delta_q);
//     RCLCPP_DEBUG_STREAM(LOGGER, "      q: " << q_out);
//   }

//   int result = (i == max_iter) ? -3 : (success ? 0 : -2);
//   RCLCPP_DEBUG_STREAM(LOGGER, "Result " << result << " after " << i << " iterations: " << q_out);

//   return result;
// }

void URKinematicsPlugin::clipToJointLimits(const KDL::JntArray& q, KDL::JntArray& q_delta,
                                            Eigen::ArrayXd& weighting) const
{
  weighting.setOnes();
  for (std::size_t i = 0; i < q.rows(); ++i)
  {
    const double delta_max = joint_max_(i) - q(i);
    const double delta_min = joint_min_(i) - q(i);
    if (q_delta(i) > delta_max)
      q_delta(i) = delta_max;
    else if (q_delta(i) < delta_min)
      q_delta(i) = delta_min;
    else
      continue;

    weighting[mimic_joints_[i].map_index] = 0.01;
  }
}

bool URKinematicsPlugin::getPositionFK(const std::vector<std::string>& link_names,
                                        const std::vector<double>& joint_angles,
                                        std::vector<geometry_msgs::msg::Pose>& poses) const
{
  if (!initialized_)
  {
    RCLCPP_ERROR(LOGGER, "kinematics solver not initialized");
    return false;
  }
  poses.resize(link_names.size());
  if (joint_angles.size() != dimension_)
  {
    RCLCPP_ERROR(LOGGER, "Joint angles vector must have size: %d", dimension_);
    return false;
  }

  KDL::Frame p_out;
  KDL::JntArray jnt_pos_in(dimension_);
  jnt_pos_in.data = Eigen::Map<const Eigen::VectorXd>(joint_angles.data(), joint_angles.size());

  bool valid = true;
  for (unsigned int i = 0; i < poses.size(); ++i)
  {
    if (fk_solver_->JntToCart(jnt_pos_in, p_out) >= 0)
    {
      poses[i] = tf2::toMsg(p_out);
    }
    else
    {
      RCLCPP_ERROR(LOGGER, "Could not compute FK for %s", link_names[i].c_str());
      valid = false;
    }
  }
  return valid;
}

const std::vector<std::string>& URKinematicsPlugin::getJointNames() const
{
  return solver_info_.joint_names;
}

const std::vector<std::string>& URKinematicsPlugin::getLinkNames() const
{
  return solver_info_.link_names;
}

//add start
int URKinematicsPlugin::getJointIndex(const std::string &name) const
{
  for (unsigned int i=0; i < ik_chain_info_.joint_names.size(); i++) {
    if (ik_chain_info_.joint_names[i] == name)
      return i;
  }
  return -1;
}
//add end

}  // namespace ur_kinematics

// register KDLKinematics as a KinematicsBase implementation
#include <class_loader/class_loader.hpp>
CLASS_LOADER_REGISTER_CLASS(ur_kinematics::URKinematicsPlugin, kinematics::KinematicsBase)
