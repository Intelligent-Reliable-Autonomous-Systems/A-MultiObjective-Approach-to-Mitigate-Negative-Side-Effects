/** Author: Sandhya Saisubramanian, UMass Amherst **/

#include "../../../include/domains/boxpushing/BoxProblem.h"
#include "../../../include/domains/boxpushing/BoxState.h"
#include "../../../include/domains/boxpushing/BoxAction.h"

namespace mlmobj
{

BoxState::BoxState(int x, int y, bool loaded, bool hole, bool fragile, mlcore::Problem* problem)
{
    x_ = x;
    y_ = y;
    loaded_ = loaded;
    hole_ = hole;
    fragile_ = fragile;
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
    for (int i = 0; i < 5; i++) {
        allSuccessors_.push_back(std::list<mlcore::Successor> ());
    }
    /* Adding a successor entry for each action */
    for (int i = 0; i < 5; i++) {
        approval_.push_back(-1.0);
    }

//    std::vector<double> weights(problem->size(), 0.0);
//    resetCost(weights, -1);
}

std::ostream& BoxState::print(std::ostream& os) const
{
    BoxProblem* bps = (BoxProblem*) problem_;
    os << "(" << x_  << ", " << y_ << ", " << loaded_ << ", " << hole_ << ", " << fragile_ << ")";
    return os;
}
bool BoxState::equals(mlcore::State* other) const
{
    BoxState* state = (BoxState*) other;
    return *this ==  *state;
}

int BoxState::hashValue() const
{
    int hval = x_ + 31 * (y_ + 31 * (loaded_ + 31 * (hole_ + 31 * (fragile_))));
    return hval;
}
}

