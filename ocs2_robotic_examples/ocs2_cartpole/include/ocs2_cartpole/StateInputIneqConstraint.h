#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>

namespace ocs2 {
namespace cartpole {
/*
It seems that SQP only can deal with ineq constr by relax barrier func (soft constr)
*/
class StateInputIneqConstraint final : public StateInputConstraint {
public:
    StateInputIneqConstraint(scalar_t x_min, scalar_t x_max, scalar_t u_min, scalar_t u_max)
    : StateInputConstraint(ConstraintOrder::Linear), 
    x_min_(x_min), x_max_(x_max), u_min_(u_min), u_max_(u_max) { }

    ~StateInputIneqConstraint() override = default;
    StateInputIneqConstraint* clone() const override { 
        return new StateInputIneqConstraint(*this); 
    }

    size_t getNumConstraints(scalar_t time) const override { return 4; }

    vector_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) const override {
        vector_t e(4);
        e << (state(0) - x_min_), 
             (x_max_  - state(0)), 
             (input(0) - u_min_), 
             (u_max_ - input(0));
        
        return e;
    }

    VectorFunctionLinearApproximation getLinearApproximation(scalar_t time, const vector_t& state, const vector_t& input,
                                                           const PreComputation& preComp) const override {
        VectorFunctionLinearApproximation jac(4, state.size(), input.size());
        jac.f = getValue(time, state, input, preComp);
        jac.dfdx << 1, 0, 0, 0,
                   -1, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, 0, 0;
        jac.dfdu << 0, 0, 1, -1;
        
        return jac;
    }

private:
    scalar_t x_min_, x_max_, u_min_, u_max_;
};

}  // namespace cartpole
}  // namespace ocs2
