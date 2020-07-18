/** Author: Sandhya Saisubramanian, UMass Amherst **/

#include "../../../include/domains/navigation/NavProblem.h"
#include "../../../include/domains/navigation/NavState.h"
#include "../../../include/domains/navigation/NavAction.h"

namespace mlmobj
{

NavState::NavState(int x, int y, int curr_speed, bool human, bool water, mlcore::Problem* problem)
{
    x_ = x;
    y_ = y;
    curr_speed_ = curr_speed;
    water_ = water;
    human_ = human;
    problem_ = problem;
    mobjCost_ = std::vector<double> (((MOProblem *) problem_)->size());
    MOProblem* aux = (MOProblem *) problem_;
    for (int i = 0; i < aux->size(); i++) {
        if (!aux->heuristics().empty())
            mobjCost_[i] = aux->heuristics()[i]->cost(this);
        else
            mobjCost_[i] = 0.0;
    }

    /* Adding a successor entry for each action */
    for (int i = 0; i < 8; i++) {
        allSuccessors_.push_back(std::list<mlcore::Successor> ());
    }
    /* Adding a successor entry for each action */
    for (int i = 0; i < 8; i++) {
        approval_.push_back(-1.0);
    }

//    std::vector<double> weights(problem->size(), 0.0);
//    resetCost(weights, -1);
}

std::ostream& NavState::print(std::ostream& os) const
{
    os << "(" << x_  << ", " << y_ << ", " << curr_speed_ << ", "
                    << human_ << ", " << water_ << ")";
    return os;
}
bool NavState::equals(mlcore::State* other) const
{
    NavState* state = (NavState*) other;
    return *this ==  *state;
}

int NavState::hashValue() const
{
    int hval = x_ + 31 * (y_ + 31 * (curr_speed_ + 31 * (human_ + 31 * (water_))));
    return hval;
}
}

