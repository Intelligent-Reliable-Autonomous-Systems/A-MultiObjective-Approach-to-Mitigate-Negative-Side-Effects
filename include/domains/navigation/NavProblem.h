/** Author: Sandhya Saisubramanian, UMass Amherst **/

#ifndef MDPLIB_NavProblem_H
#define MDPLIB_NavProblem_H

#include <unordered_map>
#include <vector>
#include <list>

#include "../../Action.h"
#include "../../Problem.h"
#include "../../State.h"
#include "../../MObjProblem.h"
#include "../../MObjState.h"
#include "../../util/general.h"
#include "NavAction.h"
#include "NavState.h"

#include <unordered_set>

namespace AVNavigation
{
    const unsigned char LEFT_SLOW = 0;
    const unsigned char LEFT_FAST = 1;
    const unsigned char RIGHT_SLOW = 2;
    const unsigned char RIGHT_FAST = 3;
    const unsigned char UP_SLOW = 4;
    const unsigned char UP_FAST = 5;
    const unsigned char DOWN_SLOW = 6;
    const unsigned char DOWN_FAST = 7;
}

/**
 * A class implementing the navigation problem.
 * Accelerating over stagnant water results in a splash which is a mild NSE
 * If there is a human nearby, then the splash is considered as a severe NSE.
 */
 namespace mlmobj
{
class NavProblem : public MOProblem
{
private:
    int width_;
    int height_;
    int x0_;
    int y0_;
    PairDoubleMap* goals_;
    mlcore::State* absorbing;
    IntPairSet walls;
    IntPairSet humanLocs;
    IntPairSet water;
    int isize_;
    void addSuccessor(NavState* state, std::list<mlcore::Successor>& successors,
        int newx, int newy, int newspeed, bool newhuman, bool newwater, double prob);


    /* Type of cost functions to use */
    const char* costFunctionType_;
    /** This only allows for actions that have no NSE in each state **/
    bool impactFreeTrans_ =  false;

    void addAllActions();

public:

    /**
     * Constructs a box pushing problem instance from the given file.
     */
    NavProblem(const char* filename, int isize, const char* costFunction, bool useLearnedvalues=false, std::string reward_file="");

    virtual ~NavProblem() {}

    /** states where box pushing can cause undesirable effects **/
    IntPairSet getwater(){return water;}

    IntPairSet getwalls(){return walls;}

    IntPairSet gethumanLocs(){return humanLocs;}

    int isize() { return isize_;}

    void ResetCostFunction() {costFunctionType_ = "optimal";}

    void Set_impactFreeTrans( bool impactFreeTrans) { impactFreeTrans_ = impactFreeTrans;}

    /** Loads the learned rewards from the given file and stores in a hashmap **/
    virtual void LoadLearnedReward(std::string reward_file);
    /**
     * Overrides method from Problem.
     */
    virtual bool goal(mlcore::State* s, int index) const;

    /**
     * Overrides method from MObjProblem.
     */
    virtual std::list<mlcore::Successor> transition(mlcore::State* s,
                                                    mlcore::Action* a, int index);

    /** Overrides method from MObjProblem.
    */
    virtual double HumanFeedback(mlcore::State* s, mlcore::Action* a, int index) const;

    /** Overrides method from MObjProblem.
    */
    virtual bool HumanClassification(mlcore::State* s, mlcore::Action* a, int index) const;

    /** Returns true if (s,a) result in low impact NSE **/
    virtual bool isLow_NSE(mlcore::State* s, mlcore::Action* a) const;
    /** Returns true if (s,a) result in high impact NSE **/
    virtual bool isHigh_NSE(mlcore::State* s, mlcore::Action* a) const;

    /**
     * Overrides method from Problem.
     */
    virtual double cost(mlcore::State* s, mlcore::Action* a) const;
    virtual double cost(mlcore::State* s, mlcore::Action* a, int i) const;
    virtual double cost_reduced(mlcore::State* s, mlcore::Action* a, int i) const;

    /**
     * Overrides method from Problem.
     */
    virtual bool applicable(mlcore::State* s, mlcore::Action* a) const;

    virtual bool applicable(mlcore::State* s, mlcore::Action* a, int index) const;
};
}
#endif // MDPLIB_NavProblem_H
