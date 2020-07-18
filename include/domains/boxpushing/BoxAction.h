#ifndef MDPLIB_BOXACTION_H
#define MDPLIB_BOXACTION_H

#include <cassert>

#include "../../Action.h"
#include "../../Problem.h"
#include "../../State.h"


/**
 * A class implementing actions in the box pushing domain.
 * There are 6 possible actions: moving in all four directions, a pick-up
 * and a drop-off action forthe box.
 */
class BoxAction : public mlcore::Action
{
private:
    unsigned char dir_;
    virtual std::ostream& print(std::ostream& os) const;

public:
    BoxAction() : dir_(-1) {}

    BoxAction(const unsigned char dir) : dir_(dir) {}

    /**
     * Overriding method from Action.
     */
    virtual mlcore::Action& operator=(const mlcore::Action& rhs)
    {
        if (this == &rhs)
            return *this;

        BoxAction* action = (BoxAction*)  & rhs;
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

#endif // MDPLIB_BOXACTION_H
