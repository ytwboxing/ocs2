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

#include "ocs2_oc/pre_condition/Scaling.h"

#include <atomic>
#include <functional>
#include <numeric>

namespace ocs2 {

// Internal helper functions
namespace {

scalar_t limitScaling(const scalar_t& v) {
  if (v < 1e-4) {
    return 1.0;
  } else if (v > 1e+4) {
    return 1e+4;
  } else {
    return v;
  }
}

template <typename T>
vector_t matrixInfNormRows(const Eigen::MatrixBase<T>& mat) {
  if (mat.rows() == 0 || mat.cols() == 0) {
    return vector_t(0);
  } else {
    return mat.rowwise().template lpNorm<Eigen::Infinity>();
  }
}

template <typename T, typename... Rest>
vector_t matrixInfNormRows(const Eigen::MatrixBase<T>& mat, const Eigen::MatrixBase<Rest>&... rest) {
  vector_t temp = matrixInfNormRows(rest...);
  if (mat.rows() == 0 || mat.cols() == 0) {
    return temp;
  } else if (temp.rows() != 0) {
    return mat.rowwise().template lpNorm<Eigen::Infinity>().cwiseMax(temp);
  } else {
    return mat.rowwise().template lpNorm<Eigen::Infinity>();
  }
}

template <typename T>
vector_t matrixInfNormCols(const Eigen::MatrixBase<T>& mat) {
  if (mat.rows() == 0 || mat.cols() == 0) {
    return vector_t(0);
  } else {
    return mat.colwise().template lpNorm<Eigen::Infinity>().transpose();
  }
}

template <typename T, typename... Rest>
vector_t matrixInfNormCols(const Eigen::MatrixBase<T>& mat, const Eigen::MatrixBase<Rest>&... rest) {
  vector_t temp = matrixInfNormCols(rest...);
  if (mat.rows() == 0 || mat.cols() == 0) {
    return temp;
  } else if (temp.rows() != 0) {
    return mat.colwise().template lpNorm<Eigen::Infinity>().transpose().cwiseMax(temp);
  } else {
    return mat.colwise().template lpNorm<Eigen::Infinity>().transpose();
  }
}

template <typename T>
void scaleMatrixInPlace(const vector_t* rowScale, const vector_t* colScale, Eigen::MatrixBase<T>& mat) {
  if (rowScale != nullptr) {
    mat.array().colwise() *= rowScale->array();
  }
  if (colScale != nullptr) {
    mat *= colScale->asDiagonal();
  }
}

void invSqrtInfNormInParallel(ThreadPool& threadPool, const std::vector<VectorFunctionLinearApproximation>& dynamics,
                              const std::vector<ScalarFunctionQuadraticApproximation>& cost, const vector_array_t& scalingVectors,
                              const int N, vector_array_t& D, vector_array_t& E) {
  // Helper function
  auto invSqrt = [](const vector_t& v) -> vector_t { return v.unaryExpr(std::ref(limitScaling)).array().sqrt().inverse(); };

  D[0] = invSqrt(matrixInfNormCols(cost[0].dfduu, dynamics[0].dfdu));
  E[0] = invSqrt(matrixInfNormRows(dynamics[0].dfdu, scalingVectors[0].asDiagonal().toDenseMatrix()));

  std::atomic_int timeStamp{1};
  auto task = [&](int workerId) {
    int k;
    while ((k = timeStamp++) < N) {
      D[2 * k - 1] =
          invSqrt(matrixInfNormCols(cost[k].dfdxx, cost[k].dfdux, scalingVectors[k - 1].asDiagonal().toDenseMatrix(), dynamics[k].dfdx));
      D[2 * k] = invSqrt(matrixInfNormCols(cost[k].dfdux.transpose().eval(), cost[k].dfduu, dynamics[k].dfdu));
      E[k] = invSqrt(matrixInfNormRows(dynamics[k].dfdx, dynamics[k].dfdu, scalingVectors[k].asDiagonal().toDenseMatrix()));
    }
  };
  threadPool.runParallel(std::move(task), threadPool.numThreads() + 1U);

  D[2 * N - 1] = invSqrt(matrixInfNormCols(cost[N].dfdxx, scalingVectors[N - 1].asDiagonal().toDenseMatrix()));
}

void scaleDataOneStepInPlaceInParallel(ThreadPool& threadPool, const vector_array_t& D, const vector_array_t& E, const int N,
                                       std::vector<VectorFunctionLinearApproximation>& dynamics,
                                       std::vector<ScalarFunctionQuadraticApproximation>& cost, std::vector<vector_t>& scalingVectors) {
  scaleMatrixInPlace(&(D[0]), nullptr, cost.front().dfdu);
  scaleMatrixInPlace(&(D[0]), &(D[0]), cost.front().dfduu);
  scaleMatrixInPlace(&(D[0]), nullptr, cost.front().dfdux);

  std::atomic_int timeStamp{1};
  auto scaleCost = [&](int workerId) {
    int k;
    while ((k = timeStamp++) <= N) {
      scaleMatrixInPlace(&(D[2 * k - 1]), nullptr, cost[k].dfdx);
      scaleMatrixInPlace(&(D[2 * k - 1]), &(D[2 * k - 1]), cost[k].dfdxx);
      if (cost[k].dfdu.size() > 0) {
        scaleMatrixInPlace(&(D[2 * k]), nullptr, cost[k].dfdu);
        scaleMatrixInPlace(&(D[2 * k]), &(D[2 * k]), cost[k].dfduu);
        scaleMatrixInPlace(&(D[2 * k]), &(D[2 * k - 1]), cost[k].dfdux);
      }
    }
  };
  threadPool.runParallel(std::move(scaleCost), threadPool.numThreads() + 1U);

  // Scale G & g
  /**
   * \f$
   * \tilde{B} = E * B * D,
   * scaling = E * I * D,
   * \tilde{g} = E * g,
   * \f$
   */
  scaleMatrixInPlace(&(E[0]), nullptr, dynamics[0].f);
  scaleMatrixInPlace(&(E[0]), nullptr, dynamics[0].dfdx);
  scalingVectors[0].array() *= E[0].array() * D[1].array();
  if (dynamics[0].dfdu.size() > 0) {
    scaleMatrixInPlace(&(E[0]), &(D[0]), dynamics[0].dfdu);
  }

  timeStamp = 1;
  auto scaleConstraints = [&](int workerId) {
    int k;
    while ((k = timeStamp++) < N) {
      scaleMatrixInPlace(&(E[k]), nullptr, dynamics[k].f);
      scaleMatrixInPlace(&(E[k]), &(D[2 * k - 1]), dynamics[k].dfdx);
      scalingVectors[k].array() *= E[k].array() * D[2 * k + 1].array();
      if (dynamics[k].dfdu.size() > 0) {
        scaleMatrixInPlace(&(E[k]), &(D[2 * k]), dynamics[k].dfdu);
      }
    }
  };
  threadPool.runParallel(std::move(scaleConstraints), threadPool.numThreads() + 1U);
}

vector_t matrixInfNormRows(const Eigen::SparseMatrix<scalar_t>& mat) {
  vector_t infNorm;
  infNorm.setZero(mat.rows());
  for (int j = 0; j < mat.outerSize(); ++j) {
    for (Eigen::SparseMatrix<scalar_t>::InnerIterator it(mat, j); it; ++it) {
      int i = it.row();
      infNorm(i) = std::max(infNorm(i), std::abs(it.value()));
    }
  }
  return infNorm;
}

vector_t matrixInfNormCols(const Eigen::SparseMatrix<scalar_t>& mat) {
  vector_t infNorm;
  infNorm.setZero(mat.cols());
  for (int j = 0; j < mat.outerSize(); ++j) {
    for (Eigen::SparseMatrix<scalar_t>::InnerIterator it(mat, j); it; ++it) {
      infNorm(j) = std::max(infNorm(j), std::abs(it.value()));
    }
  }
  return infNorm;
}

void scaleMatrixInPlace(const vector_t& rowScale, const vector_t& colScale, Eigen::SparseMatrix<scalar_t>& mat) {
  for (int j = 0; j < mat.outerSize(); ++j) {
    for (Eigen::SparseMatrix<scalar_t>::InnerIterator it(mat, j); it; ++it) {
      if (it.row() > rowScale.size() - 1 || it.col() > colScale.size() - 1) {
        throw std::runtime_error("[scaleMatrixInPlace] it.row() > rowScale.size() - 1 || it.col() > colScale.size() - 1");
      }
      it.valueRef() *= rowScale(it.row()) * colScale(j);
    }
  }
}

}  // anonymous namespace

void preConditioningInPlaceInParallel(ThreadPool& threadPool, const vector_t& x0, const OcpSize& ocpSize, const int iteration,
                                      std::vector<VectorFunctionLinearApproximation>& dynamics,
                                      std::vector<ScalarFunctionQuadraticApproximation>& cost, vector_array_t& DOut, vector_array_t& EOut,
                                      vector_array_t& scalingVectors, scalar_t& cOut) {
  const int N = ocpSize.numStages;
  if (N < 1) {
    throw std::runtime_error("[preConditioningInPlaceInParallel] The number of stages cannot be less than 1.");
  }

  // Init output
  cOut = 1.0;
  DOut.resize(2 * N);
  EOut.resize(N);
  scalingVectors.resize(N);
  for (int i = 0; i < N; i++) {
    DOut[2 * i].setOnes(ocpSize.numInputs[i]);
    DOut[2 * i + 1].setOnes(ocpSize.numStates[i + 1]);
    EOut[i].setOnes(ocpSize.numStates[i + 1]);
    scalingVectors[i].setOnes(ocpSize.numStates[i + 1]);
  }

  const auto numDecisionVariables = std::accumulate(ocpSize.numInputs.begin(), ocpSize.numInputs.end(), 0) +
                                    std::accumulate(ocpSize.numStates.begin() + 1, ocpSize.numStates.end(), 0);

  vector_array_t D(2 * N), E(N);
  for (int i = 0; i < iteration; i++) {
    invSqrtInfNormInParallel(threadPool, dynamics, cost, scalingVectors, N, D, E);
    scaleDataOneStepInPlaceInParallel(threadPool, D, E, N, dynamics, cost, scalingVectors);

    for (int k = 0; k < DOut.size(); k++) {
      DOut[k].array() *= D[k].array();
    }
    for (int k = 0; k < EOut.size(); k++) {
      EOut[k].array() *= E[k].array();
    }

    const auto infNormOfh = [&cost, &x0]() {
      scalar_t out = (cost.front().dfdu + cost.front().dfdux * x0).lpNorm<Eigen::Infinity>();
      for (int k = 1; k < cost.size(); ++k) {
        out = std::max(out, cost[k].dfdx.lpNorm<Eigen::Infinity>());
        if (cost[k].dfdu.size() > 0) {
          out = std::max(out, cost[k].dfdu.lpNorm<Eigen::Infinity>());
        }
      }
      return out;
    }();

    const auto sumOfInfNormOfH = [&cost]() {
      scalar_t out = matrixInfNormCols(cost[0].dfduu).derived().sum();
      for (int k = 1; k < cost.size(); k++) {
        if (cost[k].dfduu.size() == 0) {
          out += matrixInfNormCols(cost[k].dfdxx).derived().sum();
        } else {
          out += matrixInfNormCols(cost[k].dfdxx, cost[k].dfdux).derived().sum();
          out += matrixInfNormCols(cost[k].dfdux.transpose().eval(), cost[k].dfduu).derived().sum();
        }
      }
      return out;
    }();

    const auto averageOfInfNormOfH = sumOfInfNormOfH / static_cast<scalar_t>(numDecisionVariables);

    const auto gamma = 1.0 / limitScaling(std::max(averageOfInfNormOfH, infNormOfh));

    for (int k = 0; k <= N; ++k) {
      cost[k].dfdxx *= gamma;
      cost[k].dfduu *= gamma;
      cost[k].dfdux *= gamma;
      cost[k].dfdx *= gamma;
      cost[k].dfdu *= gamma;
    }

    cOut *= gamma;
  }
}

void preConditioningSparseMatrixInPlace(int iteration, Eigen::SparseMatrix<scalar_t>& H, vector_t& h, Eigen::SparseMatrix<scalar_t>& G,
                                        vector_t& g, vector_t& DOut, vector_t& EOut, scalar_t& cOut) {
  const int nz = H.rows();
  const int nc = G.rows();

  // Init output
  cOut = 1.0;
  DOut.setOnes(nz);
  EOut.setOnes(nc);

  for (int i = 0; i < iteration; i++) {
    vector_t D, E;
    const vector_t infNormOfHCols = matrixInfNormCols(H);
    const vector_t infNormOfGCols = matrixInfNormCols(G);

    D = infNormOfHCols.array().max(infNormOfGCols.array());
    E = matrixInfNormRows(G);

    for (int i = 0; i < D.size(); i++) {
      if (D(i) > 1e+4) D(i) = 1e+4;
      if (D(i) < 1e-4) D(i) = 1.0;
    }

    for (int i = 0; i < E.size(); i++) {
      if (E(i) > 1e+4) E(i) = 1e+4;
      if (E(i) < 1e-4) E(i) = 1.0;
    }

    D = D.array().sqrt().inverse();
    E = E.array().sqrt().inverse();

    scaleMatrixInPlace(D, D, H);
    scaleMatrixInPlace(E, D, G);
    h.array() *= D.array();
    g.array() *= E.array();

    DOut = DOut.cwiseProduct(D);
    EOut = EOut.cwiseProduct(E);

    const scalar_t infNormOfh = h.lpNorm<Eigen::Infinity>();
    const vector_t infNormOfHColsUpdated = matrixInfNormCols(H);
    const scalar_t gamma = 1.0 / limitScaling(std::max(infNormOfHColsUpdated.mean(), infNormOfh));

    H *= gamma;
    h *= gamma;

    cOut *= gamma;
  }
}

void scaleDataInPlace(const OcpSize& ocpSize, const vector_t& D, const vector_t& E, const scalar_t c,
                      std::vector<VectorFunctionLinearApproximation>& dynamics, std::vector<ScalarFunctionQuadraticApproximation>& cost,
                      std::vector<vector_t>& scalingVectors) {
  const int N = ocpSize.numStages;
  if (N < 1) {
    throw std::runtime_error("[PIPG] The number of stages cannot be less than 1.");
  }

  scalingVectors.resize(N);

  // Scale H & h
  const int nu_0 = ocpSize.numInputs.front();
  // Scale row - Pre multiply D
  cost.front().dfduu.array().colwise() *= c * D.head(nu_0).array();
  // Scale col - Post multiply D
  cost.front().dfduu *= D.head(nu_0).asDiagonal();

  cost.front().dfdu.array() *= c * D.head(nu_0).array();

  cost.front().dfdux.array().colwise() *= c * D.head(nu_0).array();

  int currRow = nu_0;
  for (int k = 1; k < N; ++k) {
    const int nx_k = ocpSize.numStates[k];
    const int nu_k = ocpSize.numInputs[k];
    auto& Q = cost[k].dfdxx;
    auto& R = cost[k].dfduu;
    auto& P = cost[k].dfdux;
    auto& q = cost[k].dfdx;
    auto& r = cost[k].dfdu;

    Q.array().colwise() *= c * D.segment(currRow, nx_k).array();
    Q *= D.segment(currRow, nx_k).asDiagonal();
    q.array() *= c * D.segment(currRow, nx_k).array();

    P *= D.segment(currRow, nx_k).asDiagonal();
    currRow += nx_k;

    R.array().colwise() *= c * D.segment(currRow, nu_k).array();
    R *= D.segment(currRow, nu_k).asDiagonal();
    r.array() *= c * D.segment(currRow, nu_k).array();

    P.array().colwise() *= c * D.segment(currRow, nu_k).array();

    currRow += nu_k;
  }

  const int nx_N = ocpSize.numStates[N];
  cost[N].dfdxx.array().colwise() *= c * D.tail(nx_N).array();
  cost[N].dfdxx *= D.tail(nx_N).asDiagonal();
  cost[N].dfdx.array() *= c * D.tail(nx_N).array();

  // Scale G & g
  auto& B_0 = dynamics[0].dfdu;
  auto& C_0 = scalingVectors[0];
  const int nx_1 = ocpSize.numStates[1];

  //  \tilde{B} = E * B * D,
  //  scaling = E * D,
  //  \tilde{g} = E * g
  B_0.array().colwise() *= E.head(nx_1).array();
  B_0 *= D.head(nu_0).asDiagonal();

  C_0 = E.head(nx_1).array() * D.segment(nu_0, nx_1).array();

  dynamics[0].dfdx.array().colwise() *= E.head(nx_1).array();
  dynamics[0].f.array() *= E.head(nx_1).array();

  currRow = nx_1;
  int currCol = nu_0;
  for (int k = 1; k < N; ++k) {
    const int nu_k = ocpSize.numInputs[k];
    const int nx_k = ocpSize.numStates[k];
    const int nx_next = ocpSize.numStates[k + 1];
    auto& A_k = dynamics[k].dfdx;
    auto& B_k = dynamics[k].dfdu;
    auto& b_k = dynamics[k].f;
    auto& C_k = scalingVectors[k];

    A_k.array().colwise() *= E.segment(currRow, nx_next).array();
    A_k *= D.segment(currCol, nx_k).asDiagonal();

    B_k.array().colwise() *= E.segment(currRow, nx_next).array();
    B_k *= D.segment(currCol + nx_k, nu_k).asDiagonal();

    C_k = E.segment(currRow, nx_next).array() * D.segment(currCol + nx_k + nu_k, nx_next).array();

    b_k.array() *= E.segment(currRow, nx_next).array();

    currRow += nx_next;
    currCol += nx_k + nu_k;
  }
}

}  // namespace ocs2