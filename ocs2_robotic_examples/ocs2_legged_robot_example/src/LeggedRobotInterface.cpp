/******************************************************************************
Copyright (c) 2021, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include <ocs2_legged_robot_example/LeggedRobotInterface.h>

#include <ocs2_pinocchio_interface/urdf.h>

namespace ocs2 {
namespace legged_robot {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
LeggedRobotInterface::LeggedRobotInterface(const std::string& taskFileFolderName, const ::urdf::ModelInterfaceSharedPtr& urdfTree) {
  pinocchioInterfacePtr_.reset(new PinocchioInterface(buildPinocchioInterface(urdfTree)));

  // Load the task file
  std::string taskFolder = ros::package::getPath("ocs2_legged_robot_example") + "/config/" + taskFileFolderName;
  taskFile_ = taskFolder + "/task.info";
  std::cerr << "Loading task file: " << taskFile_ << std::endl;

  // load setting from loading file
  loadSettings(taskFile_);

  // MPC
  setupOptimizer(taskFile_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void LeggedRobotInterface::loadSettings(const std::string& taskFile) {
  ddpSettings_ = ocs2::ddp::loadSettings(taskFile);
  mpcSettings_ = ocs2::mpc::loadSettings(taskFile);
  rolloutSettings_ = ocs2::rollout::loadSettings(taskFile, "rollout");
  modelSettings_ = loadModelSettings(taskFile);
  std::cerr << std::endl;

  // Gait Schedule
  std::cerr << std::endl;
  const auto initModeSchedule = loadModeSchedule(taskFile, "initialModeSchedule", false);
  const auto defaultModeSequenceTemplate = loadModeSequenceTemplate(taskFile, "defaultModeSequenceTemplate", false);
  const auto defaultGait = [&] {
    Gait gait{};
    gait.duration = defaultModeSequenceTemplate.switchingTimes.back();
    // Events: from time -> phase
    std::for_each(defaultModeSequenceTemplate.switchingTimes.begin() + 1, defaultModeSequenceTemplate.switchingTimes.end() - 1,
                  [&](double eventTime) { gait.eventPhases.push_back(eventTime / gait.duration); });
    // Modes:
    gait.modeSequence = defaultModeSequenceTemplate.modeSequence;
    return gait;
  }();

  auto gaitSchedule =
      std::make_shared<GaitSchedule>(initModeSchedule, defaultModeSequenceTemplate, modelSettings().phaseTransitionStanceTime_);

  // Swing trajectory planner
  const auto swingTrajectorySettings = loadSwingTrajectorySettings(taskFile);
  std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPlanner(new SwingTrajectoryPlanner(swingTrajectorySettings));

  // Mode schedule manager
  modeScheduleManagerPtr_ = std::make_shared<SwitchedModelModeScheduleManager>(std::move(gaitSchedule), std::move(swingTrajectoryPlanner));

  // Display
  std::cerr << "\nInitial Modes Schedule: \n" << initModeSchedule << std::endl;
  std::cerr << "\nDefault Modes Sequence Template: \n" << defaultModeSequenceTemplate << std::endl;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void LeggedRobotInterface::setupOptimizer(const std::string& taskFile) {
  /*
   * Initialization
   */
  initializerPtr_.reset(new LeggedRobotInitializer(*modeScheduleManagerPtr_));
  initialState_.setZero(centroidalModelInfo.stateDim);
  loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);

  pinocchioMappingPtr_.reset(new CentroidalModelPinocchioMapping<scalar_t>(centroidalModelInfo));
  pinocchioMappingAdPtr_.reset(new CentroidalModelPinocchioMapping<ad_scalar_t>(centroidalModelInfoAd));

  /*
   * Cost function
   */
  costPtr_.reset(new LeggedRobotCost(*modeScheduleManagerPtr_, *pinocchioInterfacePtr_, *pinocchioMappingPtr_, taskFile_));

  /*
   * Constraints
   */
  bool useAnalyticalGradientsConstraints = false;
  loadData::loadCppDataType(taskFile_, "legged_robot_interface.useAnalyticalGradientsConstraints", useAnalyticalGradientsConstraints);
  if (useAnalyticalGradientsConstraints) {
    throw std::runtime_error("[LeggedRobotInterface::setupOptimizer] The analytical constraint class is not yet implemented.");
  } else {
    constraintsPtr_.reset(new LeggedRobotConstraintAD(*modeScheduleManagerPtr_, *modeScheduleManagerPtr_->getSwingTrajectoryPlanner(),
                                                      *pinocchioInterfacePtr_, *pinocchioMappingAdPtr_, modelSettings_));
  }

  /*
   * Dynamics
   */
  bool useAnalyticalGradientsDynamics = false;
  loadData::loadCppDataType(taskFile_, "legged_robot_interface.useAnalyticalGradientsDynamics", useAnalyticalGradientsDynamics);
  if (useAnalyticalGradientsDynamics) {
    throw std::runtime_error("[LeggedRobotInterface::setupOptimizer] The analytical dynamics class is not yet implemented.");
  } else {
    dynamicsPtr_.reset(new LeggedRobotDynamicsAD(*pinocchioInterfacePtr_, *pinocchioMappingAdPtr_, "dynamics", "/tmp/ocs2",
                                                 modelSettings_.recompileLibraries_, true));
  }

  /*
   * Rollout
   */
  rolloutPtr_.reset(new TimeTriggeredRollout(*dynamicsPtr_, rolloutSettings_));

  /*
   * Solver
   */
  mpcPtr_.reset(new ocs2::MPC_DDP(rolloutPtr_.get(), dynamicsPtr_.get(), constraintsPtr_.get(), costPtr_.get(), initializerPtr_.get(),
                                  ddpSettings_, mpcSettings_));
  mpcPtr_->getSolverPtr()->setModeScheduleManager(modeScheduleManagerPtr_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioInterface LeggedRobotInterface::buildPinocchioInterface(const std::string& urdfPath) {
  // Add 6 DoF for the floating base
  pinocchio::JointModelComposite jointComposite(2);
  jointComposite.addJoint(pinocchio::JointModelTranslation());
  jointComposite.addJoint(pinocchio::JointModelSphericalZYX());

  return ocs2::getPinocchioInterfaceFromUrdfFile(urdfPath, jointComposite);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioInterface LeggedRobotInterface::buildPinocchioInterface(const ::urdf::ModelInterfaceSharedPtr& urdfTree) {
  // Add 6 DoF for the floating base
  pinocchio::JointModelComposite jointComposite(2);
  jointComposite.addJoint(pinocchio::JointModelTranslation());
  jointComposite.addJoint(pinocchio::JointModelSphericalZYX());

  // Remove extraneous joints from urdf
  ::urdf::ModelInterfaceSharedPtr newModel = std::make_shared<::urdf::ModelInterface>(*urdfTree);
  for (std::pair<const std::string, std::shared_ptr<::urdf::Joint>>& jointPair : newModel->joints_) {
    if (std::find(JOINT_NAMES_.begin(), JOINT_NAMES_.end(), jointPair.first) == JOINT_NAMES_.end()) {
      jointPair.second->type = urdf::Joint::FIXED;
    }
  }

  return getPinocchioInterfaceFromUrdfModel(newModel, jointComposite);
}

}  // namespace legged_robot
}  // namespace ocs2
