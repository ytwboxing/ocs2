/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <iostream>
#include <string>

#include "ocs2_cartpole/CartPoleInterface.h"
#include "ocs2_cartpole/dynamics/CartPoleSystemDynamics.h"
#include "ocs2_cartpole/StateInputIneqConstraint.h"

#include <ocs2_core/augmented_lagrangian/AugmentedLagrangian.h>

#include <ocs2_core/soft_constraint/StateInputSoftBoxConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>

#include <ocs2_core/constraint/LinearStateConstraint.h>
#include <ocs2_core/constraint/LinearStateInputConstraint.h>

#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/penalties/Penalties.h>

// Boost
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

namespace ocs2 {
namespace cartpole {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
CartPoleInterface::CartPoleInterface(const std::string& taskFile, const std::string& libraryFolder, bool verbose) {
  // check that task file exists
  boost::filesystem::path taskFilePath(taskFile);
  if (boost::filesystem::exists(taskFilePath)) {
    std::cerr << "[CartPoleInterface] Loading task file: " << taskFilePath << "\n";
  } else {
    throw std::invalid_argument("[CartPoleInterface] Task file not found: " + taskFilePath.string());
  }
  // create library folder if it does not exist
  boost::filesystem::path libraryFolderPath(libraryFolder);
  boost::filesystem::create_directories(libraryFolderPath);
  std::cerr << "[CartPoleInterface] Generated library path: " << libraryFolderPath << "\n";

  // Default initial condition
  loadData::loadEigenMatrix(taskFile, "initialState", initialState_);
  loadData::loadEigenMatrix(taskFile, "x_final", xFinal_);
  if (verbose) {
    std::cerr << "x_init:   " << initialState_.transpose() << "\n";
    std::cerr << "x_final:  " << xFinal_.transpose() << "\n";
  }

  ddpSettings_ = ddp::loadSettings(taskFile, "ddp", verbose);
  sqpSettings_ = sqp::loadSettings(taskFile, "sqp", verbose);
  ipmSettings_ = ipm::loadSettings(taskFile, "ipm", verbose);
  slpSettings_ = slp::loadSettings(taskFile, "slp", verbose);

  mpcSettings_ = mpc::loadSettings(taskFile, "mpc", verbose);

  /*
   * Optimal control problem
   */
  // Cost
  matrix_t Q(STATE_DIM, STATE_DIM);
  matrix_t R(INPUT_DIM, INPUT_DIM);
  matrix_t Qf(STATE_DIM, STATE_DIM);
  loadData::loadEigenMatrix(taskFile, "Q", Q);
  loadData::loadEigenMatrix(taskFile, "R", R);
  loadData::loadEigenMatrix(taskFile, "Q_final", Qf);
  if (verbose) {
    std::cerr << "Q:  \n" << Q << "\n";
    std::cerr << "R:  \n" << R << "\n";
    std::cerr << "Q_final:\n" << Qf << "\n";
  }

  problem_.costPtr->add("cost", std::make_unique<QuadraticStateInputCost>(Q, R));
  problem_.finalCostPtr->add("finalCost", std::make_unique<QuadraticStateCost>(Qf));

  // Dynamics
  CartPoleParameters cartPoleParameters;
  cartPoleParameters.loadSettings(taskFile, "cartpole_parameters", verbose);
  problem_.dynamicsPtr.reset(new CartPoleSytemDynamics(cartPoleParameters, libraryFolder, verbose));

  // Rollout
  auto rolloutSettings = rollout::loadSettings(taskFile, "rollout", verbose);
  rolloutPtr_.reset(new TimeTriggeredRollout(*problem_.dynamicsPtr, rolloutSettings));

  auto getPenalty = [&]() {
    // one can use either augmented::SlacknessSquaredHingePenalty or augmented::ModifiedRelaxedBarrierPenalty
    using penalty_type = augmented::SlacknessSquaredHingePenalty;
    penalty_type::Config boundsConfig;
    loadData::loadPenaltyConfig(taskFile, "bounds_penalty_config", boundsConfig, verbose);
    return penalty_type::create(boundsConfig);
  };
  auto getConstraint = [&]() {
    constexpr size_t numIneqConstraint = 4;
    vector_t e(numIneqConstraint);
    e << x_max, -x_min, u_max, -u_min;
    vector_t D(numIneqConstraint);
    D << 0.0, 0.0, -1.0, 1.0;
    matrix_t C = matrix_t::Zero(numIneqConstraint, STATE_DIM);
    C(0, 0) = -1.0;
    C(1, 0) = 1.0;
    return std::make_unique<LinearStateInputConstraint>(e, C, D);
  };

  ////// augmented lagrangian deals with ineq constr(hard), only for DDP
  // problem_.inequalityLagrangianPtr->add("bound_constraint", ocs2::create(getConstraint(), getPenalty()));

  ////// relax barrier deals with ineq constr(soft), for DDP/SQP/IPM/SLP
  problem_.softConstraintPtr->add("bound_constraint", getStateInputLimitSoftConstraint(taskFile));

  ////// IPM deals with ineq constr(hard)
  // problem_.inequalityConstraintPtr->add("bound_constraint", std::make_unique<StateInputIneqConstraint>(x_min, x_max, u_min, u_max));

  // Initialization
  cartPoleInitializerPtr_.reset(new DefaultInitializer(INPUT_DIM));
}

std::unique_ptr<StateInputCost> CartPoleInterface::getStateInputLimitSoftConstraint(const std::string& taskFile) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(taskFile, pt);
  
  scalar_t muStateLimits = 1e-2;
  scalar_t deltaStateLimits = 1e-3;
  loadData::loadPtreeValue(pt, muStateLimits, "soft_constraint.muStateLimits", true);
  loadData::loadPtreeValue(pt, deltaStateLimits, "soft_constraint.deltaStateLimits", true);

  scalar_t muInputLimits = 1e-2;
  scalar_t deltaInputLimits = 1e-3;
  loadData::loadPtreeValue(pt, muInputLimits, "soft_constraint.muInputLimits", true);
  loadData::loadPtreeValue(pt, deltaInputLimits, "soft_constraint.deltaInputLimits", true);

  std::vector<StateInputSoftBoxConstraint::BoxConstraint> stateLimits;
  stateLimits.reserve(1);
  StateInputSoftBoxConstraint::BoxConstraint state_box_constr;
  state_box_constr.index = 0;
  state_box_constr.lowerBound = x_min;
  state_box_constr.upperBound = x_max;
  state_box_constr.penaltyPtr.reset(new RelaxedBarrierPenalty({muStateLimits, deltaStateLimits}));
  stateLimits.push_back(std::move(state_box_constr));

  std::vector<StateInputSoftBoxConstraint::BoxConstraint> inputLimits;
  inputLimits.reserve(1);
  StateInputSoftBoxConstraint::BoxConstraint input_box_constr;
  input_box_constr.index = 0;
  input_box_constr.lowerBound = u_min;
  input_box_constr.upperBound = u_max;
  input_box_constr.penaltyPtr.reset(new RelaxedBarrierPenalty({muInputLimits, deltaInputLimits}));
  inputLimits.push_back(std::move(input_box_constr));

  auto boxConstraints = std::make_unique<StateInputSoftBoxConstraint>(stateLimits, inputLimits);
  boxConstraints->initializeOffset(0.0, vector_t::Zero(cartpole::STATE_DIM), vector_t::Zero(cartpole::INPUT_DIM));

  return boxConstraints;
}
}  // namespace cartpole
}  // namespace ocs2
