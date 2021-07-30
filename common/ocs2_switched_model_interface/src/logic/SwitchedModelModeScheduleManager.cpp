//
// Created by rgrandia on 18.03.20.
//

#include "ocs2_switched_model_interface/logic/SwitchedModelModeScheduleManager.h"

#include "ocs2_switched_model_interface/core/MotionPhaseDefinition.h"

namespace switched_model {

SwitchedModelModeScheduleManager::SwitchedModelModeScheduleManager(std::unique_ptr<GaitSchedule> gaitSchedule,
                                                                   std::unique_ptr<SwingTrajectoryPlanner> swingTrajectory,
                                                                   std::unique_ptr<TerrainModel> terrainModel)
    : gaitSchedule_(std::move(gaitSchedule)),
      swingTrajectoryPtr_(std::move(swingTrajectory)),
      terrainModel_(std::move(terrainModel)),
      activeDynamicsParameters_(),
      newDynamicsParameters_(std::unique_ptr<ComKinoSystemDynamicsParameters<scalar_t>>(
          new ComKinoSystemDynamicsParameters<scalar_t>(activeDynamicsParameters_))) {}

contact_flag_t SwitchedModelModeScheduleManager::getContactFlags(scalar_t time) const {
  return modeNumber2StanceLeg(this->getModeSchedule().modeAtTime(time));
}

void SwitchedModelModeScheduleManager::modifyReferences(scalar_t initTime, scalar_t finalTime, const vector_t& initState,
                                                        ocs2::TargetTrajectories& targetTrajectories, ocs2::ModeSchedule& modeSchedule) {
  const auto timeHorizon = finalTime - initTime;
  {
    auto lockedGaitSchedulePtr = gaitSchedule_.lock();
    lockedGaitSchedulePtr->advanceToTime(initTime);
    modeSchedule = lockedGaitSchedulePtr->getModeSchedule(1.5 * timeHorizon);
  }

  // Transfer terrain ownership if a new terrain is available
  std::unique_ptr<TerrainModel> newTerrain;
  terrainModel_.swap(newTerrain);  // Thread-safe swap with lockable Terrain
  if (newTerrain) {
    swingTrajectoryPtr_->updateTerrain(std::move(newTerrain));
  }

  // Prepare swing motions
  swingTrajectoryPtr_->updateSwingMotions(initTime, finalTime, initState, targetTrajectories, extractContactTimingsPerLeg(modeSchedule));

  {
    auto lockedDynamicsParameterPtr = newDynamicsParameters_.lock();
    activeDynamicsParameters_ = *lockedDynamicsParameterPtr;  // Copy external parameters to the active parameter set
  }
}

}  // namespace switched_model
