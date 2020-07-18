/**Author: Sandhya Saisubramanian, UMass Amherst **/
#ifndef MDPLIB_MOPROBLEM_H
#define MDPLIB_MOPROBLEM_H

#include "Problem.h"
#include "State.h"
#include "Action.h"

#include "util/general.h"
#include <map>
#include <vector>

namespace mlmobj
{

/**
 * An abstract class for Stochastic Shortest Path Problems with multiple objectives.
 *
 * This class inherits the core methods from mlcore::Problem and provides three
 * additional methods. One method, provides a multi-objective cost function that
 * returns a vector of values (ordered in decreasing order of preference for LSSPs).
 * The other two methods are modified versions of the goal condition and transition
 * function that receive an index indicating with respect to which value function the
 * operation must be performed.
 */
class MOProblem : public mlcore::Problem
{
protected:
    int size_;
    int QueryCount_; // number of queries to the oracle
    /* A term that relaxes any lexicographical preferences */
    std::vector<double> slack_;

    /* Weights to to use if a linear combination of the objectives is desired */
    std::vector<double> weights_;
    std::map<std::string, double> learned_reward_;

    /*if this is true, then the learned values are used for planning*/
    bool use_learnedReward_ = false;

    /** Deviation score is based on
    ** "Penalizing side effects using stepwise relative reachability".
    **  If this is set to true, then the agent optimizes a scalarized objective: r(s,a) - beta * d(s,b)
    **  r(s,a) is the reward, d(s,b) is the deviation from baseline and beta is the scalarization parameter.
    **  In this work, only irreversible NSE are considered. Therefore, d(s,b) is boolean.
    **/
    bool deviation_score_ = false;

    /**
     * A vector of heuristics for all value functions on this problem (ordered in the
     * same lexicographical order).
     */
    std::vector<mlcore::Heuristic*> heuristics_;

    int state_dimensions_;

public:
    MOProblem() : slack_(0.0)  { }
    virtual ~MOProblem() {}

    void use_learnedReward(bool val) {use_learnedReward_ = val;}

    /** Returns if the human accepts performing action in state for the lexi level.
    * The reward (cost) for the action will then depend on the response.
    * @return Cost of action wrt NSE.
    */
    virtual double HumanFeedback(mlcore::State* s, mlcore::Action* a, int index) const = 0;

    /** Returns True if the human accepts performing action in state for the lexi level.
    * The reward (cost) for the action will then depend on the response.
    * @return yes/ no action wrt NSE.
    */
    virtual bool HumanClassification(mlcore::State* s, mlcore::Action* a, int index) const = 0;
    /**
     * Returns the heuristics vector used for this problem.
     */
    std::vector<mlcore::Heuristic*> & heuristics() { return heuristics_; }

    /** Returns True if the action is applicable in the ith level.**/
    virtual bool applicable(mlcore::State* s, mlcore::Action* a, int index) const = 0;
    /**
     * Sets the heuristics vector to be used for this problem.
     */
    void heuristics(std::vector<mlcore::Heuristic*> & h) { heuristics_ = h; }

    /**
     * Returns the number of value functions for this lexicographical problem.
     *
     * @return The number of value functions for this lexicographical problem.
     */
    int size() const { return size_; }

    /** Returns the query count to oracle **/
    int QueryCount() const {return QueryCount_;}

    /** Increments the query count by 1**/
    void IncrementQueryCount() { QueryCount_ += 1;}

    /** Returns the state dimensions for function approximation in RL **/
    int state_dimensions() {return state_dimensions_;}

    /** Sets the state dimensions for function approximation in RL **/
    void StateDimensions(int sd) {state_dimensions_ = sd; }

    /**
     * Returns the slack to use for this lexicographical problem.
     *
     * @return The slack of the problem.
     */
    //double slack() const { return slack_; }
    std::vector<double> slack() const { return slack_;}

    /** Returns True if optimizing scalarized objective with deviation score. **/
    bool deviationScore() {return deviation_score_;}

    void deviationScore(bool val) { deviation_score_ = val;}


    virtual bool isLow_NSE(mlcore::State* s, mlcore::Action* a)const = 0;
    /** Returns true if (s,a) result in high impact NSE **/
    virtual bool isHigh_NSE(mlcore::State* s, mlcore::Action* a) const = 0;


    /**
     * Sets the slack to use for this lexicographical problem.
     *
     * @param slack The slack of the problem.
     */
    //void slack(double slack) { slack_ = slack; }
    void slack(std::vector<double> slack) { slack_ =  slack;}

    /**
     * Returns the weights used for linearly combining the objectives.
     *
     * @return The weights used for linearly combining the objectives.
     */
    std::vector<double> weights() const { return weights_; }

    /**
     * Sets the weights used for linearly combining the objectives.
     *
     * @param w The weights used for linearly combining the objectives.
     */
    void weights(std::vector<double> w) { weights_ = w; }

    /**
     * Multiobjective cost function for the problem.
     *
     * Returns themobj_problem.h:169 cost of applying the given action to the given state
     * according to the i-th cost function.
     *
     * @return The cost of applying the given action to the given state according
     *        to the specified value function.
     */
    virtual double cost(mlcore::State* s, mlcore::Action* a, int i) const = 0;

    /**
     * Linear combination of the cost function in the problem.
     *
     * Returns a linear combination of the cost functions evaluated at the given state
     * and action pair. The weights for the linear combination are passed as parameters.
     *
     * @return The linear combination of the cost functions.
     */
    double cost(mlcore::State* s, mlcore::Action* a, const std::vector<double>& weights) const
    {
        double lcCost = 0.0;
        for (int i = 0; i < size_; i++) {
            lcCost += weights.at(i) * cost(s, a, i);
        }
        return lcCost;
    }

    /**
     * Lexicographical goal check.
     *
     * Checks if the state given as parameter is a goal or not under the value function
     * specified by the given index.
     *
     * @return true if the given state is a goal under the given value function.
     */
    virtual bool goal(mlcore::State* s, int index) const = 0;

    /**
     * Lexicographical transition function for the problem.
     *
     * Returns a list with all successors of the given state when the given action
     * is applied. The index indicates with respect to which value function it the
     * transition to be computed. Note that this index should only modify the behavior
     * transition function for goal states, so that they become terminal at the right
     * level.
     *
     * @return A list of succcessors of the given state after applying the
     *        given action.
     */
    virtual mlcore::SuccessorsList transition(mlcore::State* s, mlcore::Action* a, int index) = 0;

    /**
     * Overrides method from mlcore::Problem.
     */
    virtual bool applicable(mlcore::State* s, mlcore::Action* a) const = 0;

    /**
     * Overrides method from mlcore::Problem.
     */
    virtual bool goal(mlcore::State* s) const
    {
        return goal(s, 0);
    }

    /**
     * Overrides method from mlcore::Problem.
     */
    virtual mlcore::SuccessorsList transition(mlcore::State* s, mlcore::Action* a)
    {
        return transition(s, a, 0);
    }

    /**
     * Overrides method from mlcore::Problem.
     */
    virtual double cost(mlcore::State* s, mlcore::Action* a) const
    {
        return cost(s, a, weights_);
    }

};

}

#endif // MDPLIB_MOPROBLEM_H
