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
#include <numeric>

#include <gtest/gtest.h>

#include <ocs2_core/augmented_lagrangian/AugmentedLagrangian.h>
#include <ocs2_core/cost/QuadraticStateCost.h>
#include <ocs2_core/penalties/Penalties.h>
#include <ocs2_ddp/ILQR.h>
#include <ocs2_ddp/SLQ.h>
#include <ocs2_sqp/SqpSolver.h>
#include <ocs2_ipm/IpmSolver.h>
#include <ocs2_oc/synchronized_module/SolverObserver.h>

#include "ocs2_cartpole/CartPoleInterface.h"
#include "ocs2_cartpole/package_path.h"

#include <time.h>
#include <fstream>

using namespace ocs2;
using namespace cartpole;

const double timeHorizon = 5.0;
const double dt = 0.05;
const int N = 100;
enum class SolverType { SLQ, ILQR, SQP, IPM };

const std::string log_path = "log.csv";

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
        std::ofstream out_file;
        out_file.open(log_path, std::ios::trunc);
        out_file.close();

        // interface
        taskFile = ocs2::cartpole::getPath() + "/config/mpc/task.info";
        const std::string libFolder = " ";
        cartPoleInterfacePtr.reset(new CartPoleInterface(taskFile, libFolder, false /*verbose*/));

        // const std::string finalCostName = "finalCost";
        // if (!cartPoleInterfacePtr->optimalControlProblem().finalCostPtr->erase(finalCostName)) {
        //     throw std::runtime_error("[TestCartpole::TestCartpole]: " + finalCostName + " was not found!");
        // }
        // auto createFinalCost = [&]() {
        //     matrix_t Qf(STATE_DIM, STATE_DIM);
        //     loadData::loadEigenMatrix(taskFile, "Q_final", Qf);
        // //   Qf *= (timeHorizon / cartPoleInterfacePtr->mpcSettings().timeHorizon_);  // scale cost
        //     return std::make_unique<QuadraticStateCost>(Qf);
        // };
        // cartPoleInterfacePtr->optimalControlProblem().finalCostPtr->add(finalCostName, createFinalCost());

        initTargetTrajectories.timeTrajectory.push_back(0.0);
        initTargetTrajectories.stateTrajectory.push_back(cartPoleInterfacePtr->getInitialTarget());
        initTargetTrajectories.inputTrajectory.push_back(vector_t::Zero(ocs2::cartpole::INPUT_DIM));

        setAlgorithm(solver_type);

        if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
            ddpPtr->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
        } else if (solver_type == SolverType::SQP) {
            sqpPtr->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
        } else if (solver_type == SolverType::IPM) {
            ipmPtr->getReferenceManager().setTargetTrajectories(initTargetTrajectories);
        }
        
    }

    void setAlgorithm(const SolverType& solver_type) {
        auto ddpSettings = cartPoleInterfacePtr->ddpSettings();
        auto sqpSettings = cartPoleInterfacePtr->sqpSettings();
        auto ipmSettings = cartPoleInterfacePtr->ipmSettings();

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

            case SolverType::IPM: {
                ipmPtr = std::make_unique<IpmSolver>(std::move(ipmSettings), 
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

    TargetTrajectories initTargetTrajectories;
    std::unique_ptr<CartPoleInterface> cartPoleInterfacePtr;
    std::string taskFile;
    std::unique_ptr<GaussNewtonDDP> ddpPtr;
    std::unique_ptr<SqpSolver> sqpPtr;
    std::unique_ptr<IpmSolver> ipmPtr;
};

int main() {
    SolverType solver_type = SolverType::IPM;

    TestCartpole cartpole_test(solver_type);

    const int bench_num = 100;
    int bench_iter = 0;
    std::vector<double> solve_time_vec;

    while (bench_iter < bench_num) {
        Timer sol_timer;
        
        if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
            cartpole_test.ddpPtr->reset();
            cartpole_test.ddpPtr->run(0.0, cartpole_test.cartPoleInterfacePtr->getInitialState(), timeHorizon);
        } else if (solver_type == SolverType::SQP) {
            cartpole_test.sqpPtr->reset();
            cartpole_test.sqpPtr->run(0.0, cartpole_test.cartPoleInterfacePtr->getInitialState(), timeHorizon);
        } else if (solver_type == SolverType::IPM) {
            cartpole_test.ipmPtr->reset();
            cartpole_test.ipmPtr->run(0.0, cartpole_test.cartPoleInterfacePtr->getInitialState(), timeHorizon);
        }

        solve_time_vec.emplace_back(sol_timer.getMs());
        std::cout << "\n++++++++++++++++++++++++++++++++++++++solve time(ms): " << sol_timer.getMs() << std::endl;

        bench_iter++;

        if (bench_iter == bench_num) {
            std::cout << "\n\n";
            std::cout << "Total run " << bench_num << " benchmarks" << std::endl;
            std::cout << "ocs2" 
                      << "\n    max time(ms): " << *std::max_element(solve_time_vec.begin(), solve_time_vec.end())
                      << "\n    min time(ms): " << *std::min_element(solve_time_vec.begin(), solve_time_vec.end())
                      << "\n    avg time(ms): " << (std::accumulate(solve_time_vec.begin(), solve_time_vec.end(), 0.0) / solve_time_vec.size());
            std::cout << std::endl;

            int num_point;
            if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
                num_point = cartpole_test.ddpPtr->primalSolution(timeHorizon).timeTrajectory_.size();
            } else if (solver_type == SolverType::SQP) {
                num_point = cartpole_test.sqpPtr->primalSolution(timeHorizon).timeTrajectory_.size();
            } else if (solver_type == SolverType::IPM) {
                num_point = cartpole_test.ipmPtr->primalSolution(timeHorizon).timeTrajectory_.size();
            }
            Eigen::MatrixXd u_sol(num_point, 1);
            u_sol.setZero();
            Eigen::MatrixXd x_sol(num_point, 4);

            for (int i = 0; i < num_point; ++i) {
                if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
                    x_sol.row(i) = cartpole_test.ddpPtr->primalSolution(timeHorizon).stateTrajectory_[i].transpose();
                    if (i < num_point - 1) {
                        u_sol.row(i) = cartpole_test.ddpPtr->primalSolution(timeHorizon).inputTrajectory_[i].transpose();
                    }
                    
                } else if (solver_type == SolverType::SQP) {
                    x_sol.row(i) = cartpole_test.sqpPtr->primalSolution(timeHorizon).stateTrajectory_[i].transpose();
                    if (i < num_point - 1) {
                        u_sol.row(i) = cartpole_test.sqpPtr->primalSolution(timeHorizon).inputTrajectory_[i].transpose();
                    }
                } else if (solver_type == SolverType::IPM) {
                    x_sol.row(i) = cartpole_test.ipmPtr->primalSolution(timeHorizon).stateTrajectory_[i].transpose();
                    if (i < num_point - 1) {
                        u_sol.row(i) = cartpole_test.ipmPtr->primalSolution(timeHorizon).inputTrajectory_[i].transpose();
                    }
                }
            }
            
            std::ofstream out_file;
            out_file.open(log_path, std::ios::app);
           
            scalar_array_t time_traj;
            if (solver_type == SolverType::SLQ || solver_type == SolverType::ILQR) {
                time_traj = cartpole_test.ddpPtr->primalSolution(timeHorizon).timeTrajectory_;
            } else if (solver_type == SolverType::SQP) {
                time_traj = cartpole_test.sqpPtr->primalSolution(timeHorizon).timeTrajectory_;
            } else if (solver_type == SolverType::IPM) {
                time_traj = cartpole_test.ipmPtr->primalSolution(timeHorizon).timeTrajectory_;
            }
    
            for (int i = 0; i < num_point; ++i) {
                // const double index = inc * i;
                out_file << time_traj[i] << ",";
                out_file << x_sol(i, 0) << "," << x_sol(i, 1) << "," 
                        << x_sol(i, 2) << "," << x_sol(i, 3) << ",";
                out_file << u_sol(i) << ",";

                out_file << 0;
                out_file << std::endl;
            }
            out_file.close();
        }

    } 

    return 0;
}
