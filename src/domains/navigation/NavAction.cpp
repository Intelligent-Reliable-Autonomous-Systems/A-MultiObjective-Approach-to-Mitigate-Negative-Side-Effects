#include "../../../include/domains/navigation/NavAction.h"
#include "../../../include/domains/navigation/NavProblem.h"

 std::ostream& NavAction::print(std::ostream& os) const
    {
        os << "action ";
        if (dir_ == AVNavigation::LEFT_SLOW)
            os << "left slow";
        if (dir_ == AVNavigation::LEFT_FAST)
            os << "left fast";
        if (dir_ == AVNavigation::RIGHT_SLOW)
            os << "right slow";
        if (dir_ == AVNavigation::RIGHT_FAST)
            os << "right fast";
        if (dir_ == AVNavigation::UP_SLOW)
            os << "up slow";
        if (dir_ == AVNavigation::UP_FAST)
            os << "up fast";
        if (dir_ == AVNavigation::DOWN_SLOW)
            os << "down slow";
        if (dir_ == AVNavigation::DOWN_FAST)
            os << "down fast";
        return os;
    }
