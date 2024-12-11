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

#include <cmath>
#include <iostream>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <ocs2_core/augmented_lagrangian/AugmentedLagrangian.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_ddp/ILQR.h>
#include <ocs2_ddp/SLQ.h>
#include <ocs2_sqp/SqpSolver.h>
#include <ocs2_oc/synchronized_module/SolverObserver.h>

#include "ocs2_cartpole/CartPoleInterface.h"
#include "ocs2_cartpole/package_path.h"

#include <time.h>

using namespace ocs2;
using namespace cartpole;

const double timeHorizon = 5.0;
const double dt = 0.05;
const int N = 100;

enum class SolverType { SLQ, ILQR, SQP };

class Timer {
public:
    explicit Timer() { start(); }
    double getMs() { return static_cast<double>(getNs()) / 1.e6; }
    double getSeconds() { return static_cast<double>(getNs()) / 1.e9; }
private:
    void start() { clock_gettime(CLOCK_MONOTONIC, &_startTime); }
    int64_t getNs() {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        return (int64_t)(now.tv_nsec - _startTime.tv_nsec) +
            1000000000 * (now.tv_sec - _startTime.tv_sec);
    }
    struct timespec _startTime;
};

class TestCartpole {
public:
    TestCartpole() = default;

    TestCartpole(const SolverType& solver_type) {
        // interface
        taskFile = ocs2::cartpole::getPath() + "/config/mpc/task.info";
        const std::string libFolder = " ";
        cartPoleInterfacePtr.reset(new CartPoleInterface(taskFile, libFolder, true /*verbose*/));

        const std::string finalCostName = "finalCost";
        if (!cartPoleInterfacePtr->optimalControlProblem().finalCostPtr->erase(finalCostName)) {
            throw std::runtime_error("[TestCartpole::TestCartpole]: " + finalCostName + " was not found!");
        }
        auto createFinalCost = [&]() {
            matrix_t Qf(STATE_DIM, STATE_DIM);
            loadData::loadEigenMatrix(taskFile, "Q_final", Qf);
        //   Qf *= (timeHorizon / cartPoleInterfacePtr->mpcSettings().timeHorizon_);  // scale cost
            return std::make_unique<QuadraticStateCost>(Qf);
        };
        cartPoleInterfacePtr->optimalControlProblem().finalCostPtr->add(finalCostName, createFinalCost());

        initTargetTrajectories.timeTrajectory.push_back(0.0);
        initTargetTrajectories.stateTrajectory.push_back(cartPoleInterfacePtr->getInitialTarget());
        initTargetTrajectories.inputTrajectory.push_back(vector_t::Zero(ocs2::cartpole::INPUT_DIM));

        setAlgorithm(solver_type);

        if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
            ddpPtr->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
        } else if (solver_type == SolverType::SQP) {
            sqpPtr->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
        }
        
    }

    void setAlgorithm(const SolverType& solver_type) {
        auto ddpSettings = cartPoleInterfacePtr->ddpSettings();
        auto sqpSettings = cartPoleInterfacePtr->sqpSettings();
        
        ddpSettings.displayInfo_ = false;
        ddpSettings.displayShortSummary_ = true;

        switch (solver_type) {
            case SolverType::SLQ: {
                ddpSettings.algorithm_ = ddp::Algorithm::SLQ;
                ddpPtr = std::make_unique<SLQ>(std::move(ddpSettings), 
                                                cartPoleInterfacePtr->getRollout(),
                                                cartPoleInterfacePtr->getOptimalControlProblem(),
                                                cartPoleInterfacePtr->getInitializer());
            }
            break;

            case SolverType::ILQR: {
                ddpSettings.algorithm_ = ddp::Algorithm::ILQR;
                ddpPtr = std::make_unique<ILQR>(std::move(ddpSettings), 
                                                cartPoleInterfacePtr->getRollout(),
                                                cartPoleInterfacePtr->getOptimalControlProblem(),
                                                cartPoleInterfacePtr->getInitializer());
            }
            break;

            case SolverType::SQP: {
                sqpPtr = std::make_unique<SqpSolver>(std::move(sqpSettings), 
                                                     cartPoleInterfacePtr->getOptimalControlProblem(),
                                                     cartPoleInterfacePtr->getInitializer());
            }
            break;

            default:
                throw std::runtime_error("[TestCartpole::getAlgorithm] undefined algorithm");
        }
    }

    void printSolution(const PrimalSolution& primalSolution, bool display) const {
        if (display) {
            std::cerr << "\n";
            // time
            std::cerr << "timeTrajectory = [";
            for (const auto& t : primalSolution.timeTrajectory_) {
                std::cerr << t << "; ";
            }
            std::cerr << "];\n";
            // state
            std::cerr << "stateTrajectory = [";
            for (const auto& x : primalSolution.stateTrajectory_) {
                std::cerr << x.transpose() << "; ";
            }
            std::cerr << "];\n";
            // input
            std::cerr << "inputTrajectory = [";
            for (const auto& u : primalSolution.inputTrajectory_) {
                std::cerr << u.transpose() << "; ";
            }
            std::cerr << "];\n";
        }
    }

    //   void testInputLimitsViolation(const scalar_array_t& timeTrajectory, const std::vector<LagrangianMetricsConstRef>& termMetrics) const {
    //     for (size_t i = 0; i < timeTrajectory.size(); i++) {
    //       const vector_t constraintViolation = termMetrics[i].constraint.cwiseMin(0.0);
    //     }
    //   }

    //   void testFinalState(const PrimalSolution& primalSolution) const {
    //     const auto& finalState = primalSolution.stateTrajectory_.back();
    //     const auto& desiredState = cartPoleInterfacePtr->getInitialTarget();
    //   }
    TargetTrajectories initTargetTrajectories;
    std::unique_ptr<CartPoleInterface> cartPoleInterfacePtr;
    std::string taskFile;
    std::unique_ptr<GaussNewtonDDP> ddpPtr;
    std::unique_ptr<SqpSolver> sqpPtr;
};

int main() {
    SolverType solver_type = SolverType::SLQ;

    TestCartpole cartpole_test(solver_type);

    Timer sol_timer;
    // run solver
    if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
        cartpole_test.ddpPtr->run(0.0, cartpole_test.cartPoleInterfacePtr->getInitialState(), timeHorizon);
    } else if (solver_type == SolverType::SQP) {
        cartpole_test.sqpPtr->run(0.0, cartpole_test.cartPoleInterfacePtr->getInitialState(), timeHorizon);
    }

    std::cout << "solve time(ms): " << sol_timer.getMs() << std::endl;

    int num_point;
    if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
        num_point = cartpole_test.ddpPtr->primalSolution(timeHorizon).timeTrajectory_.size() - 1;
    } else if (solver_type == SolverType::SQP) {
        num_point = cartpole_test.sqpPtr->primalSolution(timeHorizon).timeTrajectory_.size() - 1;
    }
    Eigen::MatrixXd u_sol(num_point, 1);

    for (int i = 0; i < num_point + 1; ++i) {
        if (i < num_point) {
            if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
                u_sol.row(i) = cartpole_test.ddpPtr->primalSolution(timeHorizon).inputTrajectory_[i].transpose();
            } else if (solver_type == SolverType::SQP) {
                u_sol.row(i) = cartpole_test.sqpPtr->primalSolution(timeHorizon).inputTrajectory_[i].transpose();
            }
        }
    }

    // std::cout << "u_sol\n" << u_sol << std::endl;

    // // std::cout << "================= " << cartpole_test.ddpPtr->primalSolution(0.0).inputTrajectory_.size() << std::endl;
    // std::cout << "time length: " << cartpole_test.ddpPtr->primalSolution(timeHorizon).timeTrajectory_.size() << std::endl;
    // std::cout << "last time: " << cartpole_test.ddpPtr->primalSolution(timeHorizon).timeTrajectory_.back() << std::endl;
    if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
        cartpole_test.printSolution(cartpole_test.ddpPtr->primalSolution(timeHorizon), true);
    } else if (solver_type == SolverType::SQP) {
        cartpole_test.printSolution(cartpole_test.sqpPtr->primalSolution(timeHorizon), true);
    }
    

    return 0;
}
