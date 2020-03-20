#include "ocs2_switched_model_interface/initialization/ComKinoOperatingPointsBase.h"

#include "ocs2_switched_model_interface/core/MotionPhaseDefinition.h"

namespace switched_model {

ComKinoOperatingPointsBase::ComKinoOperatingPointsBase(const com_model_t& comModel,
                                                       std::shared_ptr<const SwitchedModelModeScheduleManager> modeScheduleManagerPtr)
    : Base(), comModelPtr_(comModel.clone()), modeScheduleManagerPtr_(std::move(modeScheduleManagerPtr)) {
  if (!modeScheduleManagerPtr_) {
    throw std::runtime_error("[ComKinoOperatingPointsBase] Mode schedule manager cannot be a nullptr");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ComKinoOperatingPointsBase::ComKinoOperatingPointsBase(const ComKinoOperatingPointsBase& rhs)
    : Base(rhs), comModelPtr_(rhs.comModelPtr_->clone()), modeScheduleManagerPtr_(rhs.modeScheduleManagerPtr_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ComKinoOperatingPointsBase* ComKinoOperatingPointsBase::clone() const {
  return new ComKinoOperatingPointsBase(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void ComKinoOperatingPointsBase::computeInputOperatingPoints(contact_flag_t contactFlags, input_vector_t& inputs) {
  // Distribute total mass equally over active stance legs.
  inputs.setZero();

  const scalar_t totalMass = comModelPtr_->totalMass() * 9.81;
  size_t numStanceLegs(0);

  for (size_t i = 0; i < NUM_CONTACT_POINTS; i++) {
    if (contactFlags[i]) {
      ++numStanceLegs;
    }
  }

  if (numStanceLegs > 0) {
    for (size_t i = 0; i < NUM_CONTACT_POINTS; i++) {
      if (contactFlags[i]) {
        inputs(3 * i + 2) = totalMass / numStanceLegs;
      }
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void ComKinoOperatingPointsBase::getSystemOperatingTrajectories(const state_vector_t& initialState, const scalar_t& startTime,
                                                                const scalar_t& finalTime, scalar_array_t& timeTrajectory,
                                                                state_vector_array_t& stateTrajectory,
                                                                input_vector_array_t& inputTrajectory, bool concatOutput /*= false*/) {
  const auto midTime = 0.5 * (startTime + finalTime);
  const auto contactFlags = modeScheduleManagerPtr_->getContactFlags(midTime);

  Base::stateOperatingPoint_ = initialState;
  computeInputOperatingPoints(contactFlags, Base::inputOperatingPoint_);

  if (!concatOutput) {
    timeTrajectory.clear();
    stateTrajectory.clear();
    inputTrajectory.clear();
  }

  timeTrajectory.push_back(startTime);
  timeTrajectory.push_back(finalTime);

  stateTrajectory.push_back(Base::stateOperatingPoint_);
  stateTrajectory.push_back(Base::stateOperatingPoint_);

  inputTrajectory.push_back(Base::inputOperatingPoint_);
  inputTrajectory.push_back(Base::inputOperatingPoint_);
}

}  // namespace switched_model
