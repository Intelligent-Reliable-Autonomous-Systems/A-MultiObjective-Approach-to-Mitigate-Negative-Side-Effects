/** Author: Sandhya Saisubramanian, UMass Amherst **/

#ifndef MDPLIB_BOXPROBLEM_H
#define MDPLIB_BOXPROBLEM_H

#include <unordered_map>
#include <vector>
#include <list>

#include "../../Action.h"
#include "../../Problem.h"
#include "../../State.h"
#include "../../MObjProblem.h"
#include "../../MObjState.h"
#include "../../util/general.h"
#include "BoxAction.h"
#include "BoxState.h"

#include <unordered_set>

namespace box
{
    const unsigned char UP = 0;
    const unsigned char DOWN = 1;
    const unsigned char LEFT = 2;
    const unsigned char RIGHT = 3;
    const unsigned char PICK = 4;
}

/**
 * A class implementing the box pushing problem.
 * Pushing the box over error states results in negative side effects.
 * boxInit denotes the initial location of the box and S denotes the
 * start location of the agent. The primary objective is to move the
 * box from init location to the goal (G) location.
 */
 namespace mlmobj
{
class BoxProblem : public MOProblem
{
private:
    int bx_0_, by_0_; // init box location
    int width_;
    int height_;
    int x0_;
    int y0_;
    PairDoubleMap* goals_;
    mlcore::State* absorbing;
    IntPairSet walls;
    IntPairSet holes;
    IntPairSet fragiles;
    int isize_;
    void addSuccessor(BoxState* state, std::list<mlcore::Successor>& successors, int val,
                      int limit, int newx, int newy, bool newloaded, bool hole, bool fragile, double prob);


    void addAllActions();
    /* Type of cost functions to use */
    const char* costFunctionType_;
    /** This only allows for actions that have no NSE in each state **/
    bool impactFreeTrans_ =  false;


public:

    /**
     * Constructs a box pushing problem instance from the given file.
     */
    BoxProblem(const char* filename, int isize, const char* costFunction, bool useLearnedvalues=false, std::string reward_file="");

    virtual ~BoxProblem() {}

    /** states where box pushing can cause undesirable effects **/
    IntPairSet getholes(){return holes;}

    IntPairSet getwalls(){return walls;}

    IntPairSet getfragile(){return fragiles;}

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

   /** Overrides method from MObjProblem.
    */
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
#endif // MDPLIB_BOXPROBLEM_H
