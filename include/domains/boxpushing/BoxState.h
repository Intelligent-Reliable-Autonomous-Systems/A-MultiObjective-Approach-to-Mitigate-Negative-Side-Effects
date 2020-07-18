#ifndef MDPLIB_BOXSTATE_H
#define MDPLIB_BOXSTATE_H

#include "../../Problem.h"
#include "../../State.h"
#include "../../MObjProblem.h"
#include "../../MObjState.h"

namespace mlmobj
{

class BoxState : public MOState
{
private:
    int x_;
    int y_;
    bool loaded_;
    bool hole_;
    bool fragile_;
    int nseCount_;

    /* A cache of all successors (for all actions) of this state */
    std::vector<mlcore::SuccessorsList> allSuccessors_;

     /* A cache of all human approval (for all actions) of this state */
    std::vector<double> approval_;

    virtual std::ostream& print(std::ostream& os) const;

public:
    /**
     * Creates a state for the box pushing problem with the given (x,y) position
     * and (loaded) indicates if the agent has the box with it, and assigned to the given index.
     *
     * Every tuple (x, y, loaded) should be assigned to a unique index.
     */
    BoxState(int x, int y, bool loaded, bool hole, bool fragile, mlcore::Problem* problem);

    virtual ~BoxState() {}

    int x() const { return x_; }

    int y() const { return y_; }

    bool loaded() const { return loaded_;}

    bool hole() const { return hole_; }

    bool fragile() const { return fragile_; }

    /** Copy constructor **/
    BoxState(const BoxState &state)
	{
        x_ = state.x_;
        y_ = state.y_;
        loaded_ = state.loaded_;
        hole_ = state.hole_;
        fragile_ = state.fragile_;
  	}


    virtual mlcore::State& operator=(const mlcore::State& rhs)
    {
        if (this == &rhs)
            return *this;

        BoxState* bps = (BoxState *)  & rhs;
        x_ =  bps->x_;
        y_=  bps->y_;
        loaded_ = bps->loaded_;
        hole_ = bps->hole_;
        fragile_ = bps->fragile_;
        return *this;
    }

    virtual bool operator==(const mlcore::State& rhs) const
    {
        BoxState* bps = (BoxState *)  &rhs;
        return x_ == bps->x_ && y_ == bps->y_ && loaded_ == bps->loaded_ && hole_ == bps->hole_ && fragile_ == bps->fragile_ ;
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
#endif // MDPLIB_BOXSTATE_H
