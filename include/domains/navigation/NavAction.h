#ifndef MDPLIB_NavAction_H
#define MDPLIB_NavAction_H

#include <cassert>

#include "../../Action.h"
#include "../../Problem.h"
#include "../../State.h"


/**
 * A class implementing actions in the AV navigation domain
 * There are 4 possible actions: moving at low speed, medium sped, high speed, and halt.
 */
class NavAction : public mlcore::Action
{
private:
    unsigned char dir_;
    virtual std::ostream& print(std::ostream& os) const;

public:
    NavAction() : dir_(-1) {}

    NavAction(const unsigned char dir) : dir_(dir) {}

    /**
     * Overriding method from Action.
     */
    virtual mlcore::Action& operator=(const mlcore::Action& rhs)
    {
        if (this == &rhs)
            return *this;

        NavAction* action = (NavAction*)  & rhs;
        dir_ =  action->dir_;
        return *this;
    }

    /**
     * Overriding method from Action.
     */
    virtual int hashValue() const
    {
        return (int) dir_;
    }

    unsigned char dir() const
    {
        return dir_;
    }
};
#endif // MDPLIB_CATCHERACTION_H
