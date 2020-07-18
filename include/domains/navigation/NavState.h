#ifndef MDPLIB_NavState_H
#define MDPLIB_NavState_H

#include "../../Problem.h"
#include "../../State.h"
#include "../../MObjProblem.h"
#include "../../MObjState.h"


namespace mlmobj
{

class NavState : public MOState
{
private:
    int x_;
    int y_;
    int curr_speed_;
    bool human_;
    bool water_;

    /* A cache of all successors (for all actions) of this state */
    std::vector<mlcore::SuccessorsList> allSuccessors_;

     /* A cache of all human approval (for all actions) of this state */
    std::vector<double> approval_;

    virtual std::ostream& print(std::ostream& os) const;

public:
    /**
     * Creates a state for the navigation problem with the given (x,y) position
     * vx and vy denote the current acceleration of the vehicle.
     * water indicates if there is stagnant water, splash indicates if the car splashes water.
     * When human and spplash are true, the car splashes water on the human.
     */
    NavState(int x, int y, int curr_speed, bool human, bool water, mlcore::Problem* problem);

    virtual ~NavState() {}

    int x() const { return x_; }

    int y() const { return y_; }

    int curr_speed() const { return curr_speed_; }

    bool water() const { return water_;}

    bool human() const { return human_; }

    /** Copy constructor **/
    NavState(const NavState &state)
	{
        x_ = state.x_;
        y_ = state.y_;
        curr_speed_ = state.curr_speed_;
        water_ = state.water_;
        human_ = state.human_;
  	}


    virtual mlcore::State& operator=(const mlcore::State& rhs)
    {
        if (this == &rhs)
            return *this;

        NavState* ns = (NavState *)  & rhs;
        x_ =  ns->x_;
        y_=  ns->y_;
        curr_speed_ = ns->curr_speed_;
        water_ = ns->water_;
        human_ = ns->human_;
        return *this;
    }

    virtual bool operator==(const mlcore::State& rhs) const
    {
        NavState* ns = (NavState *)  &rhs;
        return x_ == ns->x_ && y_ == ns->y_ && curr_speed_ == ns->curr_speed_
                    && human_ == ns->human_ && water_ == ns->water_;
    }

    /**
     * Returns a pointer to the successor cache of this state.
     */
    std::vector<mlcore::SuccessorsList>* allSuccessors()
    {
        return &allSuccessors_;
    }
    /**
     * Returns a pointer to the approval cache of this state.
     */
    std::vector<double>* allApproval()
    {
        return &approval_;
    }

    /**
     * Overrides method from State.
     */
    virtual bool equals(mlcore::State* other) const;

    /**
     * Overrides method from State.
     */
    virtual int hashValue() const;
};
}
#endif // MDPLIB_NavState_H
